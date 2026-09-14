"""
AI Checker — v8.1
-----------------
LLM-based metadata extraction, citation parsing, and reference verification.
Enforces that duplicate entries inherit canonical verdicts, and structural
heuristics never return REAL without external verification.
"""

import hashlib
import os
import re
import json
import threading
import requests
from typing import List, Dict, Any, Optional
from review_queue import is_venue_whitelisted
from pathlib import Path as _Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=_Path(__file__).resolve().parent / ".env", override=True)

_AI_BASE_URL: str = os.environ.get("AI_BASE_URL", "").rstrip("/")
_AI_MODEL: str = os.environ.get("AI_MODEL", "")
_AI_API_KEY: str = os.environ.get("AI_API_KEY", "")

_LLM_CACHE: Dict[str, str] = {}
_LLM_CACHE_LOCK = threading.Lock()


def _llm_cache_key(model: str, system: str, prompt: str) -> str:
    raw = f"{model}|{system}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _llm_cache_get(model: str, system: str, prompt: str) -> Optional[str]:
    key = _llm_cache_key(model, system, prompt)
    with _LLM_CACHE_LOCK:
        return _LLM_CACHE.get(key)


def _llm_cache_put(model: str, system: str, prompt: str, response: str) -> None:
    key = _llm_cache_key(model, system, prompt)
    with _LLM_CACHE_LOCK:
        _LLM_CACHE[key] = response


def get_llm_cache_stats() -> dict:
    with _LLM_CACHE_LOCK:
        return {"llm_cache_entries": len(_LLM_CACHE)}


def _call_ai(prompt: str, max_tokens: int = 2000, system: str = "",
             timeout: int = 45, max_retries: int = 3) -> str:
    import time

    if not _AI_BASE_URL or not _AI_MODEL or not _AI_API_KEY:
        raise RuntimeError(
            "AI backend not configured. Set AI_BASE_URL, AI_MODEL, and AI_API_KEY in your .env file."
        )

    model_tag = f"{_AI_BASE_URL}:{_AI_MODEL}"
    cached = _llm_cache_get(model_tag, system, prompt)
    if cached is not None:
        return cached

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    url = f"{_AI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _AI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                _llm_cache_put(model_tag, system, prompt, result)
                return result
            elif resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 8))
                continue
            elif resp.status_code in (400, 401, 403, 404):
                raise RuntimeError(
                    f"AI provider rejected the request (HTTP {resp.status_code}). "
                    "Check the API key, model, and AI_BASE_URL."
                )
            else:
                print(f"AI API error {resp.status_code}: {resp.text[:200]}")

        except RuntimeError:
            raise
        except requests.exceptions.Timeout:
            print(f"AI API timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError as e:
            print(f"AI API connection error: {e}")
        except Exception as e:
            print(f"AI API exception: {e}")

        if attempt < max_retries:
            time.sleep(1.5 ** attempt)

    raise RuntimeError(
        f"AI API call failed after {max_retries + 1} attempts. "
        f"Check AI_BASE_URL ({_AI_BASE_URL}), AI_MODEL ({_AI_MODEL}), and AI_API_KEY."
    )


def _call_ai_json(prompt: str, max_tokens: int = 2000, system: str = "",
                  timeout: int = 45, max_retries: int = 3) -> dict:
    text = _call_ai(prompt, max_tokens, system, timeout, max_retries).strip()

    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    if text.lower().startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("Could not extract JSON from response", text, 0)


def _chunk(lst: list, size: int) -> list:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _ai_available() -> bool:
    return bool(_AI_BASE_URL and _AI_MODEL and _AI_API_KEY)


def _local_ml_gate(entry: dict) -> dict:
    """
    Deterministic fallback heuristic for entries with no external confirmation.
    Never returns REAL.
    """
    title = (entry.get("title") or "").strip()
    authors = (entry.get("authors") or "").strip()
    year = (entry.get("year") or "").strip()
    raw = (entry.get("raw_text") or "").strip()

    placeholder_surnames = {
        "ghost", "fake", "test", "example", "placeholder", "unknown",
        "anonymous", "nobody", "someone", "author", "dummy", "sample",
        "demo", "null", "none", "community", "staff",
    }

    if authors:
        first = authors.split(";")[0].strip()
        surname = (
            first.split(",")[0].strip().lower()
            if "," in first
            else (first.split()[-1].lower() if first.split() else "")
        )
        if surname in placeholder_surnames:
            return {
                "decision": "SUSPICIOUS",
                "confidence": 0.82,
                "reason": "Placeholder or non-person author name detected.",
            }

    if not title and not authors:
        return {
            "decision": "SUSPICIOUS",
            "confidence": 0.65,
            "reason": "No usable title or author information found.",
        }

    if not year:
        return {
            "decision": "SUSPICIOUS",
            "confidence": 0.57,
            "reason": "Year missing; cannot confirm the work confidently.",
        }

    venue_present = bool(re.search(
        r'\bIn\s*:?\s*|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b|\bTagung\b|\bKonferenz\b',
        raw,
        flags=re.IGNORECASE,
    ))

    title_has_words = len(re.findall(r"\b\w+\b", title)) >= 2
    if title_has_words and authors and year:
        return {
            "decision": "SUSPICIOUS",
            "confidence": 0.60,
            "reason": (
                "Metadata is structurally plausible and author/year/title are present."
                + (" Venue is explicitly stated." if venue_present else "")
                + " No external confirmation available."
            ),
        }

    return {
        "decision": "SUSPICIOUS",
        "confidence": 0.55,
        "reason": "Found some metadata, but not enough external evidence to confirm the reference.",
    }


def _is_german_academic_venue(entry: dict) -> bool:
    venue_name = (
        (entry.get("journal") or "") + " " +
        (entry.get("booktitle") or "") + " " +
        (entry.get("publisher") or "")
    ).lower()

    german_hints = [
        'informatik', 'gi ', 'lni', 'gesellschaft für', 'gesellschaft fur',
        'datenbank', 'wirtschaftsinformatik', 'btw', 'mensch und computer',
        'informatik spektrum', 'it - information technology', 'pik',
        'datenbank-spektrum', 'lecture notes in informatics',
        'fachtagung', 'fachgespräch', 'dagstuhl', 'informatiktage',
        'universität', 'hochschule', 'fraunhofer',
    ]
    return any(hint in venue_name for hint in german_hints)


# ---------------------------------------------------------------------------
# Grey Literature Detection
# ---------------------------------------------------------------------------

_ACADEMIC_HOSTS = {
    "doi.org", "arxiv.org", "dl.acm.org", "ieeexplore.ieee.org",
    "link.springer.com", "springer.com", "sciencedirect.com",
    "jstor.org", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "nature.com", "mdpi.com", "openalex.org", "semanticscholar.org",
    "researchgate.net", "aclanthology.org", "openreview.net",
    "dl.gi.de", "gi.de", "onlinelibrary.wiley.com",
    "tandfonline.com", "cambridge.org", "oup.com", "plos.org",
    "usenix.org", "sciendo.com",
}

_GREY_HOSTS = {
    "bitkom.org", "flexera.com", "gartner.com", "forrester.com",
    "mckinsey.com", "deloitte.com", "statista.com", "idc.com",
    "accenture.com", "capgemini.com", "pwc.com", "kpmg.com",
    "bsi.bund.de", "bsi.de", "bundesregierung.de", "bmwi.de", "bmwk.de",
    "destatis.de", "ec.europa.eu", "nist.gov", "37signals.com",
    "basecamp.com", "github.com", "github.io", "medium.com",
    "techcrunch.com", "substack.com", "resources.idg.de",
}


def _hostname_of(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        h = urlparse(url).hostname or ""
        if h.startswith("www."):
            h = h[4:]
        return h.lower()
    except Exception:
        return ""


def _host_matches(hostname: str, candidates: set) -> bool:
    if not hostname:
        return False
    for c in candidates:
        if hostname == c or hostname.endswith("." + c):
            return True
    return False


def _is_grey_literature(entry: dict) -> tuple:
    title = (entry.get("title") or "").lower()
    url = (entry.get("url") or "").strip()
    publisher = (entry.get("publisher") or "").lower()
    entry_type = (entry.get("entry_type") or "").lower()
    raw = (entry.get("raw_text") or entry.get("raw") or "").lower()
    hostname = _hostname_of(url)

    if _host_matches(hostname, _GREY_HOSTS):
        label = ".".join(hostname.split(".")[-2:]) if hostname else "unknown"
        return True, f"Grey literature ({label})"

    grey_title_signals = [
        "state of the cloud", "cloud report", "market report", "industry report",
        "annual report", "whitepaper", "white paper", "survey report",
        "leaving the cloud", "cloud repatriation", "state of devops",
        "state of agile", "developer survey",
    ]
    for sig in grey_title_signals:
        if sig in title:
            return True, f"Grey literature (title contains '{sig}')"

    grey_publishers = [
        "bitkom", "flexera", "gartner", "forrester", "idc", "statista",
        "mckinsey", "deloitte", "pwc", "kpmg", "accenture",
    ]
    for pub in grey_publishers:
        if pub in publisher or pub in raw:
            return True, f"Grey literature (published by {pub.title()})"

    if entry_type in ("inproceedings", "proceedings", "conference"):
        return False, ""

    if entry_type in ("website", "online", "misc") and url:
        return True, "Grey literature (website citation)"

    if url and entry_type not in ("article", "book", "inproceedings"):
        if not _host_matches(hostname, _ACADEMIC_HOSTS):
            title_has_report_word = any(
                word in title for word in ["report", "whitepaper", "white paper", "survey"]
            )
            authors_missing = not (entry.get("authors") or "").strip()
            if title_has_report_word or (authors_missing and "publisher" not in entry.get("publisher", "").lower()):
                return True, f"Grey literature (unlisted source: {hostname or 'unrecognized domain'})"

    return False, ""


_FABRICATION_PATTERNS = [
    r"\bquantum\s+supremacy\s+doesn'?t\s+exist\b",
    r"\bblockchain\s+cures\b",
    r"\bneural\s+telepathy\b",
    r"\bperpetual\s+motion\b",
    r"\btime\s+travel\s+machine\b",
    r"\bunicorn\s+horn\s+computing\b",
    r"\bflat\s+earth\s+geometry\b",
    r"\binfinite\s+energy\b",
    r"\bmagic\s+beans\b",
    r"\bimpossible\s+source\b",
]

_FABRICATION_KEYWORDS = ["quantum", "blockchain", "neural", "impossible"]

def ai_parse_uncertain_entries(bib_entries_raw: list) -> dict:
    uncertain = [e for e in bib_entries_raw if e.get("needs_ai_parsing")]
    if not uncertain:
        return {}
    
    improvements: dict = {}
    for chunk in _chunk(uncertain, 20):
        entries_for_prompt = [
            {"key": e["key"], "raw_text": (e.get("raw_text") or "")[:300],
             "regex_title": e.get("title") or "",
             "regex_authors": e.get("authors") or "",
             "regex_type": e.get("entry_type") or "unknown"}
            for e in chunk
        ]
        
        prompt = (
            "The automated regex parser failed to confidently extract metadata for these "
            "LNI bibliography entries. Extract correct structured metadata from the raw text.\n\n"
            "LNI FORMAT: Author(s): Title. Publisher/Journal/Booktitle, Year.\n"
            "Author format: 'Lastname, Firstname [; Lastname2, Firstname2]'\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{"results": [{"key": "...", "title": "...", "authors": "...", '
            '"year": "YYYY", "entry_type": "book|article|proceedings|website|misc|unknown", '
            '"journal": null, "booktitle": null, "publisher": null, "pages": null}]}\n\n'
            f"Entries:\n{json.dumps(entries_for_prompt, ensure_ascii=False, indent=2)}"
        )
        
        try:
            result = _call_ai_json(prompt, max_tokens=3000)
            for item in result.get("results", []):
                key = item.get("key")
                if key:
                    improvements[key] = item
        except Exception:
            pass
    
    return improvements

def _is_fabricated_title(title: str) -> tuple:
    if not title:
        return False, 0.0

    title_lower = title.lower()
    for pattern in _FABRICATION_PATTERNS:
        try:
            if re.search(pattern, title_lower):
                return True, 0.95
        except re.error:
            continue

    if len(title) > 120:
        keyword_count = sum(1 for kw in _FABRICATION_KEYWORDS if kw in title_lower)
        rambling_phrases = ["for everything", "makes no sense", "very long title", "solves everything"]
        phrase_count = sum(1 for phrase in rambling_phrases if phrase in title_lower)
        if keyword_count >= 2 and phrase_count >= 1:
            return True, 0.90

    return False, 0.0


def _check_journal_plausibility(journal: str) -> tuple:
    red_flags = []
    if not journal:
        return True, []
    journal_lower = journal.lower()
    legit_journals = [
        'springer', 'elsevier', 'wiley', 'ieee', 'acm', 'nature', 'science',
        'cell', 'plos', 'frontiers', 'mdpi', 'sage', 'taylor', 'francis',
        'oxford', 'cambridge', 'mit press', 'world scientific', 'informs',
        'aaai', 'usenix', 'dagstuhl', 'lecture notes in informatics', 'lni',
        'lecture notes in computer science', 'lncs', 'gi gesellschaft',
        'informatik spektrum', 'acm sigchi', 'acm sigmod', 'vldb', 'sigcomm',
        'neurips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'emnlp', 'acl',
        'journal of ', 'transactions on ', 'letters in ', 'proceedings of ',
        'conference on ', 'symposium on ', 'workshop on ',
        'informatik', 'wirtschaftsinformatik', 'delfi', 'mensch und computer',
        'btw', 'ki ', 'kunstliche intelligenz',
    ]
    fake_indicators = [
        ('international journal of advanced', 'Generic predatory prefix'),
        ('international journal of innovative', 'Generic predatory prefix'),
        ('international journal of recent', 'Generic predatory prefix'),
        ('international journal of emerging', 'Generic predatory prefix'),
        ('journal of emerging technologies', 'Predatory pattern'),
        ('journal of current research', 'Predatory pattern'),
        ('journal of modern', 'Predatory pattern'),
        ('global journal of', 'Generic predatory prefix'),
        ('world journal of', 'Generic predatory prefix'),
        ('asian journal of', 'Often predatory'),
        ('american journal of applied', 'Often impersonated/predatory'),
        ('european journal of applied', 'Often predatory'),
        ('research journal of', 'Generic fake journal'),
        ('scientific journal of', 'Generic fake journal'),
        ('journal of multidisciplinary', 'Predatory pattern'),
        ('omics publishing', 'Known predatory publisher'),
        ('hindawi', 'Historically predatory'),
        ('sciencepg', 'Known predatory'),
        ('ijser', 'Known predatory'),
        ('ijesrt', 'Known predatory'),
        ('jetir', 'Known predatory'),
        ('irjet', 'Known predatory'),
        ('ijarcce', 'Known predatory'),
        ('researchpublish', 'Known predatory'),
        ('scirp', 'Scientific Research Publishing'),
        ('ijar', 'Known predatory'),
        ('graphy publications', 'Known predatory'),
        ('austin publishing', 'Known predatory'),
        ('insight medical publishing', 'Known predatory'),
    ]
    for indicator, reason in fake_indicators:
        if indicator in journal_lower:
            red_flags.append(reason)
    has_legit = any(legit in journal_lower for legit in legit_journals)
    if not has_legit and red_flags:
        return False, red_flags
    return True, red_flags


def _check_page_range_implausibility(pages: str, year: str) -> tuple:
    red_flags = []
    if not pages:
        return True, []
    match = re.search(r'(\d+)\s*[-\u2013\u2014]+\s*(\d+)', pages)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        span = hi - lo
        if span > 30:
            red_flags.append(f"Page span of {span} pages is unusually long for a single article")
        if span > 100:
            red_flags.append(f"Extremely long page span ({span} pages) - likely fabricated")
        if lo > 9999:
            red_flags.append(f"Page number {lo} is improbably high")
    return len(red_flags) < 2, red_flags[:2]


def _compute_verdict_with_confidence(entry: dict, api_result: dict, title_sim: float) -> dict:
    api_status = api_result.get("status", "not_checked")
    api_confidence = api_result.get("confidence", 0.0)
    matched_title = api_result.get("matched_title", "")
    sources = api_result.get("sources_checked", [])
    has_doi = bool(entry.get("doi") or api_result.get("doi"))
    author_match = api_result.get("author_match_score")
    if isinstance(author_match, str):
        try:
            author_match = float(author_match)
        except ValueError:
            author_match = None

    def _author_ok() -> bool:
        if not entry.get("authors"):
            return False
        if author_match is None:
            return False
        return author_match >= 0.60

    if api_status == "verified" and api_confidence >= 0.75 and matched_title:
        if title_sim is not None and title_sim < 0.80:
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.74,
                "composite_risk": 0.62,
                "risk_factors": ["Title similarity below 0.80 for a verified match"],
                "reasoning": f"Database match exists but title similarity is {int((title_sim or 0)*100)}%.",
            }
        if not _author_ok():
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.72,
                "composite_risk": 0.60,
                "risk_factors": [
                    f"Author overlap {int((author_match or 0)*100)}% below 60% threshold"
                    if author_match is not None else "Author overlap unverified"
                ],
                "reasoning": "Database title match cannot be confirmed without matching authors.",
            }
        return {
            "verdict": "REAL",
            "confidence": max(api_confidence, 0.85),
            "composite_risk": 0.10,
            "risk_factors": [],
            "reasoning": f"Confirmed in {', '.join(sources[:2])}: '{matched_title[:60]}'",
        }

    if has_doi and api_status == "verified" and api_confidence >= 0.75:
        if (title_sim is None or title_sim < 0.80) or not _author_ok():
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.74,
                "composite_risk": 0.60,
                "risk_factors": ["DOI present but title/author overlap insufficient"],
                "reasoning": "DOI alone is not enough to confirm identity.",
            }
        return {
            "verdict": "REAL",
            "confidence": 0.92,
            "composite_risk": 0.10,
            "risk_factors": [],
            "reasoning": "DOI verified with title and author overlap.",
        }

    is_grey, grey_reason = _is_grey_literature(entry)
    if is_grey:
        return {
            "verdict": "SUSPICIOUS",
            "confidence": 0.55,
            "composite_risk": 0.55,
            "risk_factors": [f"Grey/industry literature ({grey_reason}) — verify manually."],
            "reasoning": f"Grey literature ({grey_reason}). Manual review required.",
        }

    return {
        "verdict": "SUSPICIOUS",
        "confidence": 0.60,
        "composite_risk": 0.50,
        "risk_factors": ["Database confirmation incomplete"],
        "reasoning": f"Analysis complete. API status: {api_status}. Requires review.",
    }


def _pre_screen_by_author_overlap(entry: dict, api_result: dict, title_sim: float) -> Optional[dict]:
    from checker import author_overlap_score

    api_status = api_result.get("status", "not_checked")
    confidence = api_result.get("confidence", 0)
    sources = api_result.get("sources_checked", [])
    n_sources = len(sources)
    is_retracted = api_result.get("is_retracted", False)
    has_doi = bool(api_result.get("doi"))
    matched_title = api_result.get("matched_title", "")

    if is_retracted:
        return {"verdict": "SUSPICIOUS", "confidence": 0.90,
                "reasoning": "Paper confirmed to exist but is RETRACTED — do not cite.",
                "risk_factors": ["RETRACTED"]}

    cited_authors = (entry.get("authors") or "").strip()
    correct_authors = (
        api_result.get("correct_authors")
        or api_result.get("corrected_authors")
        or ""
    ).strip()

    if cited_authors and correct_authors:
        overlap = author_overlap_score(cited_authors, correct_authors)
    else:
        overlap = None

    pct = int(overlap * 100) if overlap is not None else None

    if has_doi and api_status == "verified" and confidence >= 0.75 and matched_title:
        if (title_sim is None or title_sim < 0.80) or (overlap is None or overlap < 0.60):
            return {"verdict": "SUSPICIOUS", "confidence": 0.74,
                    "reasoning": "DOI present but title/author overlap insufficient to confirm identity.",
                    "risk_factors": ["Author and title overlap below REAL threshold"]}
        return {"verdict": "REAL", "confidence": 0.95,
                "reasoning": "DOI confirmed + title + author match",
                "risk_factors": []}

    if api_status == "verified" and confidence >= 0.80 and n_sources >= 2 and matched_title:
        if (title_sim is None or title_sim < 0.80) or (overlap is None or overlap < 0.60):
            return {"verdict": "SUSPICIOUS", "confidence": 0.74,
                    "reasoning": "Multiple databases agree on title but author/title overlap insufficient.",
                    "risk_factors": ["Title/author overlap below REAL threshold"]}
        return {"verdict": "REAL", "confidence": confidence,
                "reasoning": f"Confirmed by {n_sources} independent databases",
                "risk_factors": []}

    if not cited_authors or not correct_authors or pct is None:
        return None

    if overlap < 0.30 and api_status in ("verified", "partial_match") and (title_sim or 0) >= 0.75:
        return {"verdict": "SUSPICIOUS", "confidence": 0.85,
                "reasoning": f"Title matches but author overlap is only {pct}% — possible fabrication.",
                "risk_factors": [f"Severe author mismatch ({pct}%)"]}

    if overlap < 0.60:
        return {"verdict": "SUSPICIOUS", "confidence": 0.70,
                "reasoning": f"Author overlap {pct}% is below the 60% threshold required to confirm identity.",
                "risk_factors": [f"Author overlap {pct}%"]}

    if (overlap >= 0.60 and api_status in ("verified", "partial_match")
            and confidence >= 0.60
            and (title_sim is None or title_sim >= 0.80)):
        return {"verdict": "REAL", "confidence": round(min(0.80 + overlap * 0.18, 0.97), 2),
                "reasoning": f"Author overlap {pct}% + title/API match",
                "risk_factors": []}

    return None


def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    if not bib_entries:
        return {"verdicts": [], "summary": "No entries to verify.",
                "fake_count": 0, "suspicious_count": 0, "real_count": 0}

    vr_by_key = {vr["key"]: vr for vr in api_results}

    all_verdicts: List[dict] = []
    needs_ai: List[tuple] = []
    pre_screen_cache: Dict[str, dict] = {}

    def _local_title_similarity(t1: str, t2: str) -> float:
        if not t1 or not t2:
            return 0.0
        t1_norm = re.sub(r'[^\w\s]', '', t1.lower())
        t2_norm = re.sub(r'[^\w\s]', '', t2.lower())
        t1_words = set(w for w in t1_norm.split() if len(w) > 2)
        t2_words = set(w for w in t2_norm.split() if len(w) > 2)
        if not t1_words or not t2_words:
            return 0.0
        return len(t1_words & t2_words) / len(t1_words | t2_words)

    for entry in bib_entries:
        key = entry["key"]
        vr = vr_by_key.get(key, {})
        matched_title = vr.get("matched_title", "")
        title_sim = 0.0
        if vr.get("title_match_score") is not None:
            title_sim = vr["title_match_score"]
        elif entry.get("title") and matched_title:
            title_sim = _local_title_similarity(entry["title"], matched_title)

        # Skip duplicates from AI prompt; they inherit canonical results below
        if vr.get("is_duplicate") and vr.get("duplicate_of"):
            continue

        early = _pre_screen_by_author_overlap(entry, vr, title_sim)
        if early:
            pre_screen_cache[key] = early
        else:
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)
            is_grey, _ = _is_grey_literature(entry)
            send_to_ai = (
                _ai_available()
                and composite["verdict"] != "REAL"
                and (composite["confidence"] < 0.75 or is_grey)
            )
            if send_to_ai:
                needs_ai.append((entry, vr, title_sim, composite))
            else:
                pre_screen_cache[key] = {
                    "verdict": composite["verdict"],
                    "confidence": composite["confidence"],
                    "reasoning": composite.get("reasoning", "Analysis complete"),
                    "risk_factors": composite.get("risk_factors", []),
                }

    ai_verdicts_by_key: Dict[str, dict] = {}
    if _ai_available() and needs_ai:
        try:
            for chunk in _chunk(needs_ai, 8):
                combined = []
                for entry, vr, title_sim, composite in chunk:
                    combined.append({
                        "key": entry["key"],
                        "title": entry.get("title") or "",
                        "authors": entry.get("authors") or "",
                        "year": entry.get("year") or "",
                        "entry_type": entry.get("entry_type") or "unknown",
                        "doi": entry.get("doi") or "",
                        "journal": entry.get("journal") or "",
                        "publisher": entry.get("publisher") or "",
                        "url": entry.get("url") or "",
                        "booktitle": entry.get("booktitle") or "",
                        "pages": entry.get("pages") or "",
                        "api_status": vr.get("status", "not_checked"),
                        "api_matched_title": vr.get("matched_title") or "",
                        "url_fetch_result": vr.get("note") or "",
                    })

                prompt = f"""You are an academic reference auditor. For each reference return REAL / SUSPICIOUS / FAKE.
Return ONLY valid JSON:
{{"verdicts": [{{"key": "string", "verdict": "REAL", "confidence": 0.95, "reasoning": "string", "risk_factors": []}}]}}

References:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""

                chunk_result = _call_ai_json(prompt, max_tokens=4000)
                for v in chunk_result.get("verdicts", []):
                    ai_verdicts_by_key[v["key"]] = v
        except Exception:
            pass

    seen_keys = set()
    for entry in bib_entries:
        key = entry["key"]
        vr = vr_by_key.get(key, {})

        # FIX BUG A: Skip duplicates here so the inheritance block can assign them
        if vr.get("is_duplicate") and vr.get("duplicate_of"):
            continue

        if key in seen_keys:
            continue
        seen_keys.add(key)

        if key in pre_screen_cache:
            v = pre_screen_cache[key]
            all_verdicts.append({"key": key, **v, "open_access_url": None})
        elif key in ai_verdicts_by_key:
            all_verdicts.append(ai_verdicts_by_key[key])
        else:
            vr = vr_by_key.get(key, {})
            title_sim = vr.get("title_match_score", 0.0)
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)
            all_verdicts.append({
                "key": key,
                "verdict": composite["verdict"],
                "confidence": composite["confidence"],
                "reasoning": composite.get("reasoning", "Analysis complete"),
                "risk_factors": composite.get("risk_factors", []),
                "open_access_url": None,
            })

    # Duplicates inherit the exact canonical verdict
    _verdict_by_key = {v["key"]: v for v in all_verdicts}
    for vr in api_results:
        is_dup = vr.get("is_duplicate") if isinstance(vr, dict) else getattr(vr, "is_duplicate", False)
        dup_of = vr.get("duplicate_of") if isinstance(vr, dict) else getattr(vr, "duplicate_of", None)
        dup_key = vr.get("key") if isinstance(vr, dict) else getattr(vr, "key", None)
        if not (is_dup and dup_of and dup_key):
            continue
        canonical = _verdict_by_key.get(dup_of)
        if canonical is None:
            continue
        if any(v["key"] == dup_key for v in all_verdicts):
            continue
        all_verdicts.append({
            "key": dup_key,
            "verdict": canonical["verdict"],
            "confidence": canonical.get("confidence", 0.5),
            "reasoning": f"Duplicate of [{dup_of}] — inherits {canonical['verdict']} verdict from canonical entry.",
            "risk_factors": list(canonical.get("risk_factors", [])),
            "open_access_url": None,
        })

    fake_count = sum(1 for v in all_verdicts if v.get("verdict") == "FAKE")
    suspicious_count = sum(1 for v in all_verdicts if v.get("verdict") == "SUSPICIOUS")
    real_count = sum(1 for v in all_verdicts if v.get("verdict") == "REAL")

    return {
        "verdicts": all_verdicts,
        "fake_count": fake_count,
        "suspicious_count": suspicious_count,
        "real_count": real_count,
        "summary": f"Analysis: {real_count} REAL, {suspicious_count} SUSPICIOUS, {fake_count} FAKE",
    }


def ai_overall_verdict(filename: str, summary: dict, xcheck,
                       bib_list: list, verification_result: dict) -> dict:
    fake_count = verification_result.get("fake_count", 0)
    suspicious = verification_result.get("suspicious_count", 0)
    missing_cit = len(xcheck.cited_not_in_bib)
    orphaned = len(xcheck.in_bib_not_cited)
    incomplete = sum(1 for e in bib_list if e.completeness_issues)
    bib_count = len(bib_list)
    key_issues = [e for e in bib_list if e.key_consistent is False]

    _verdict_count = len(verification_result.get("verdicts", []))
    _counts_consistent = _verdict_count >= bib_count

    if (_counts_consistent
            and fake_count == 0 and suspicious == 0 and missing_cit == 0
            and orphaned == 0 and incomplete == 0 and not key_issues):
        return {
            "verdict": "PASS",
            "score": 100,
            "grade": "A",
            "verdict_reason": "All references verified and no citation or format issues found.",
            "student_feedback": [],
            "professor_note": "No issues found",
        }

    det_score = 100
    det_score -= min(missing_cit * 5, 20)
    det_score -= min(incomplete * 2, 10)
    det_score -= min(orphaned * 2, 10)
    det_score -= min(len(key_issues) * 3, 15)
    det_score = max(0, min(100, det_score))
    det_verdict = "PASS" if det_score >= 80 else "FLAG" if det_score >= 60 else "FAIL"
    det_grade = (
        "A" if det_score >= 90 else
        "B" if det_score >= 80 else
        "C" if det_score >= 60 else
        "D" if det_score >= 50 else
        "F"
    )

    keys_to_review = [
        f"[{v['key']}]" for v in verification_result.get("verdicts", [])[:5]
        if v.get("verdict") in ("FAKE", "SUSPICIOUS")
    ]
    professor_note = (
        f"Manually review: {' '.join(keys_to_review)}" if keys_to_review else "No issues found"
    )

    return {
        "verdict": det_verdict,
        "score": det_score,
        "grade": det_grade,
        "verdict_reason": f"Score {det_score}/100: {suspicious} reference(s) need review.",
        "student_feedback": [],
        "professor_note": professor_note,
    }