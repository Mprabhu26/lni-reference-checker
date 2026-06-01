"""
AI Checker — v6.3 (FIXED)
--------------------------
FIXES:
  - Deduplication of verification results (no duplicate entries)
  - Removed misleading "api_status=error" - replaced with user-friendly messages
  - Removed confusing "composite_risk" from frontend display
  - Better consistency between risk factors and verdict
  - Local DB checked FIRST before API calls
"""

import hashlib
import os
import re
import json
import threading
import requests
from typing import List, Dict, Any, Optional
from review_queue import is_venue_whitelisted

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-1.5-flash:generateContent")

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


def _call_ai(prompt: str, max_tokens: int = 2000, system: str = "") -> str:
    """Call AI backend with automatic retry + exponential backoff."""
    import time
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    model_tag = f"groq:{GROQ_MODEL}" if groq_key else "gemini:1.5-flash"
    cached = _llm_cache_get(model_tag, system, prompt)
    if cached is not None:
        return cached
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        result = None
        used_model = None
        if groq_key:
            try:
                resp = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {groq_key}",
                             "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL, "messages": messages,
                          "max_tokens": max_tokens, "temperature": 0.1},
                    timeout=35,
                )
                if resp.status_code == 200:
                    result = resp.json()["choices"][0]["message"]["content"].strip()
                    used_model = f"groq:{GROQ_MODEL}"
                elif resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(wait, 8))
                    continue
            except Exception:
                pass
        if result is None and gemini_key:
            try:
                full_prompt = (system + "\n\n" + prompt) if system else prompt
                resp = requests.post(
                    f"{GEMINI_URL}?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": full_prompt}]}],
                          "generationConfig": {"maxOutputTokens": max_tokens,
                                               "temperature": 0.1}},
                    timeout=35,
                )
                if resp.status_code == 200:
                    parts = resp.json()["candidates"][0]["content"]["parts"]
                    result = "".join(p.get("text", "") for p in parts).strip()
                    used_model = "gemini:1.5-flash"
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
            except Exception:
                pass
        if result is not None:
            _llm_cache_put(used_model or model_tag, system, prompt, result)
            return result
        if attempt < MAX_RETRIES:
            time.sleep(1.5 ** attempt)
    missing = [k for k, v in [("GROQ_API_KEY", groq_key), ("GEMINI_API_KEY", gemini_key)] if not v]
    raise RuntimeError(f"No AI API key configured. Set {' or '.join(missing)}. "
                       "Groq: console.groq.com (free) | Gemini: aistudio.google.com (free)")


def _call_ai_json(prompt: str, max_tokens: int = 2000, system: str = "") -> dict:
    text = _call_ai(prompt, max_tokens, system).strip()
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
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise


def _chunk(lst: list, size: int) -> list:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _ai_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# 1. LLM-based reference extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are a bibliography metadata extractor. "
    "Output ONLY a JSON array — no prose, no markdown, no explanation.\n\n"
    "Each element must have these fields (use null for unknown):\n"
    '  {"raw": "full citation text", "authors": "Lastname, First [; ...]", '
    '"title": "...", "year": "YYYY", "journal": "...", '
    '"booktitle": "...", "publisher": "...", "pages": "...", '
    '"doi": "...", "url": "...", "isbn": "..."}\n\n'
    "Rules:\n"
    "- Extract EVERY bibliographic entry you find\n"
    "- Handle numbered [1], author-year, BibLaTeX, and plain text formats\n"
    "- Use the EXACT title text — never paraphrase or shorten\n"
    "- LNI author format is 'Lastname, Firstname' — preserve it\n"
    "- Skip non-reference text (equations, figures, section headings)\n"
    "- If no references exist, return an empty array: []"
)


def ai_extract_references_from_text(bib_text: str) -> List[Dict[str, Any]]:
    if not bib_text or not bib_text.strip():
        return []
    if not _ai_available():
        return []
    chunks = []
    if len(bib_text) > 6000:
        lines = bib_text.split('\n')
        buf, size = [], 0
        for line in lines:
            buf.append(line)
            size += len(line)
            if size >= 3000:
                chunks.append('\n'.join(buf))
                buf, size = [], 0
        if buf:
            chunks.append('\n'.join(buf))
    else:
        chunks = [bib_text]
    all_refs: List[Dict[str, Any]] = []
    for chunk in chunks:
        prompt = (
            "Extract all bibliographic references from the following text. "
            "Return a JSON array as specified.\n\n"
            f"BIBLIOGRAPHY TEXT:\n{chunk}"
        )
        try:
            data = _call_ai_json(prompt, max_tokens=3000, system=_EXTRACT_SYSTEM)
            if isinstance(data, list):
                all_refs.extend(data)
        except Exception:
            pass
    return all_refs


def merge_ai_extractions_into_bib_list(ai_refs: List[Dict], bib_list: list) -> list:
    if not ai_refs or not bib_list:
        return bib_list
    def _norm(t: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (t or "").lower())).strip()
    ai_by_title: Dict[str, dict] = {}
    for ar in ai_refs:
        t = _norm(ar.get("title", ""))
        if t:
            ai_by_title[t] = ar
    for entry in bib_list:
        entry_title_norm = _norm(entry.title or "")
        ai = ai_by_title.get(entry_title_norm)
        if not ai:
            for ar in ai_refs:
                raw = (ar.get("raw") or "")[:200]
                if entry.raw_text[:80].lower() in raw.lower() or raw[:80].lower() in entry.raw_text.lower():
                    ai = ar
                    break
        if not ai:
            continue
        if not entry.title and ai.get("title"): entry.title = ai["title"]
        if not entry.authors and ai.get("authors"): entry.authors = ai["authors"]
        if not entry.year and ai.get("year"): entry.year = str(ai["year"])
        if not entry.journal and ai.get("journal"): entry.journal = ai["journal"]
        if not entry.booktitle and ai.get("booktitle"): entry.booktitle = ai["booktitle"]
        if not entry.publisher and ai.get("publisher"): entry.publisher = ai["publisher"]
        if not entry.pages and ai.get("pages"): entry.pages = ai["pages"]
        if not entry.doi and ai.get("doi"): entry.doi = ai["doi"]
        if not entry.url and ai.get("url"): entry.url = ai["url"]
        if not entry.isbn and ai.get("isbn"): entry.isbn = ai["isbn"]
        if not entry.entry_type or entry.entry_type == "unknown":
            if ai.get("journal"):
                entry.entry_type = "article"
            elif ai.get("booktitle"):
                entry.entry_type = "proceedings"
            elif ai.get("publisher") and not ai.get("journal"):
                entry.entry_type = "book"
    return bib_list


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


# ---------------------------------------------------------------------------
# 3. ENHANCED: Composite Fake Detection Signals
# ---------------------------------------------------------------------------

def _check_journal_plausibility(journal: str) -> tuple:
    """Check if a journal name sounds plausible or fake."""
    red_flags = []
    if not journal:
        return False, ["No journal name provided"]
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
        ('international journal of engineering sciences', 'Generic phrase'),
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
        ('journal of engineering and technology', 'Generic phrase'),
        ('international research journal', 'Generic phrase'),
        ('journal of multidisciplinary', 'Predatory pattern'),
        ('advances in science', 'Generic predatory prefix'),
        ('science and technology journal', 'Generic phrase'),
        ('omics publishing', 'Known predatory publisher'),
        ('hindawi', 'Historically predatory'),
        ('sciencepg', 'Known predatory'),
        ('ijser', 'Known predatory (ijser.org)'),
        ('ijesrt', 'Known predatory'),
        ('jetir', 'Known predatory'),
        ('irjet', 'Known predatory'),
        ('ijarcce', 'Known predatory'),
        ('researchpublish', 'Known predatory'),
        ('scirp', 'Scientific Research Publishing - predatory'),
        ('ijar', 'Known predatory (ijar.info)'),
        ('graphy publications', 'Known predatory'),
        ('austin publishing', 'Known predatory'),
        ('insight medical publishing', 'Known predatory'),
        ('journal of computational and theoretical', 'AI-generated pattern'),
        ('international journal of computer applications', 'Overused generic title'),
        ('journal of software engineering and applications', 'Overused generic title'),
    ]
    for indicator, reason in fake_indicators:
        if indicator in journal_lower:
            red_flags.append(reason)
    has_legit = any(legit in journal_lower for legit in legit_journals)
    if not has_legit and red_flags:
        return False, red_flags
    return True, red_flags


def _check_author_name_plausibility(authors: str) -> tuple:
    """Check if author names seem plausible or AI-generated."""
    red_flags = []
    if not authors:
        return False, ["No authors provided"]
    ai_name_patterns = [
        (r'^[A-Z][a-z]{1,3}\s+[A-Z][a-z]{1,3}$', 'Suspiciously short name (e.g., "J Smith")'),
        (r'^[A-Z]\.\s+[A-Z]\.\s+[A-Z][a-z]+$', 'Initials-only first name pattern'),
        (r'(?:AI|GPT|LLM|Transformer|Neural|Deep|Learning)\s+[A-Z][a-z]+', 'AI-themed author name'),
    ]
    for pattern, reason in ai_name_patterns:
        if re.search(pattern, authors, re.IGNORECASE):
            red_flags.append(reason)
    if ';' in authors:
        name_parts = [n.strip() for n in authors.split(';')]
        if len(name_parts) >= 3:
            first_names = [n.split(',')[0].strip() if ',' in n else n.split()[0] for n in name_parts]
            if len(set(first_names)) == 1:
                red_flags.append("All authors share the same surname - unusual")
    return len(red_flags) < 2, red_flags[:2]


def _check_page_range_implausibility(pages: str, year: str) -> tuple:
    """Check if page range is implausible for the publication type."""
    red_flags = []
    if not pages:
        return True, []
    match = re.search(r'(\d+)\s*[-–—]+\s*(\d+)', pages)
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


def _check_conference_plausibility(booktitle: str) -> tuple:
    """Check if a conference name seems plausible."""
    red_flags = []
    if not booktitle:
        return True, []
    bt_lower = booktitle.lower()
    legit_conferences = [
        'acm', 'ieee', 'neurips', 'icml', 'iclr', 'cvpr', 'eccv', 'iccv',
        'acl', 'emnlp', 'naacl', 'sigir', 'kdd', 'www', 'sosp', 'osdi',
        'usenix', 'chi', 'uist', 'siggraph', 'ismar', 'iros', 'icra',
        'organized by', 'proceedings of the', 'international conference on',
        'european conference on', 'annual meeting of'
    ]
    fake_indicators = [
        ('international conference of', 'Missing "on" - common fake pattern'),
        ('world congress on', 'Often predatory'),
        ('global summit on', 'Generic fake conference'),
        ('annual conference on', 'Vague description'),
        ('international symposium of', 'Awkward phrasing'),
    ]
    has_legit = any(legit in bt_lower for legit in legit_conferences)
    for indicator, reason in fake_indicators:
        if indicator in bt_lower:
            red_flags.append(reason)
    if not has_legit and red_flags:
        return False, red_flags
    return True, red_flags


def _get_user_friendly_message(status: str, details: str = "") -> str:
    """Convert technical status to user-friendly message."""
    if status == "verified":
        return "Found in academic database"
    elif status == "partial_match":
        return "Partial match found - verify manually"
    elif status == "not_found":
        return "Not found in any database"
    elif status == "error":
        return "Could not complete verification"
    elif status == "retracted":
        return "⚠️ PAPER HAS BEEN RETRACTED"
    elif "DOI" in details:
        return "DOI format issue - check the DOI"
    elif "year" in details.lower():
        return "Year seems incorrect"
    else:
        return "Verification attempted"


def _is_grey_literature(entry: dict) -> tuple:
    """
    Detect industry reports, government docs, blog posts — valid sources
    not indexed in CrossRef/Semantic Scholar.
    Returns (is_grey: bool, reason: str).
    """
    title     = (entry.get("title") or "").lower()
    url       = (entry.get("url") or "").lower()
    publisher = (entry.get("publisher") or "").lower()
    entry_type = (entry.get("entry_type") or "").lower()
    raw       = (entry.get("raw_text") or entry.get("raw") or "").lower()

    grey_domains = [
        "bitkom.org", "flexera.com", "info.flexera.com", "gartner.com",
        "forrester.com", "mckinsey.com", "deloitte.com", "statista.com",
        "idc.com", "accenture.com", "capgemini.com", "pwc.com", "kpmg.com",
        "bsi.bund.de", "bsi.de", "bundesregierung.de", "bmwi.de", "bmwk.de",
        "destatis.de", "ec.europa.eu", "nist.gov", "37signals.com",
        "basecamp.com", "github.com", "github.io", "medium.com",
        "techcrunch.com", "substack.com", "resources.idg.de",
    ]
    for domain in grey_domains:
        if domain in url:
            return True, f"Industry/government source ({domain})"

    grey_title_signals = [
        "state of the cloud", "cloud report", "market report", "industry report",
        "annual report", "whitepaper", "white paper", "survey report",
        "leaving the cloud", "cloud repatriation", "state of devops",
        "state of agile", "developer survey",
    ]
    for sig in grey_title_signals:
        if sig in title:
            return True, f"Industry/grey literature ('{sig}')"

    grey_publishers = [
        "bitkom", "flexera", "gartner", "forrester", "idc", "statista",
        "mckinsey", "deloitte", "pwc", "kpmg", "accenture",
    ]
    for pub in grey_publishers:
        if pub in publisher or pub in raw:
            return True, f"Known industry publisher: {pub.title()}"

    if entry_type in ("website", "online", "misc") and url:
        return True, "Online/misc source with URL — not in academic databases"

    return False, ""


def _compute_verdict_with_confidence(entry: dict, api_result: dict, title_sim: float) -> dict:
    """Compute composite fake detection score from multiple signals."""
    # Grey literature: industry/org reports not in academic DBs → SUSPICIOUS, never FAKE
    is_grey, grey_reason = _is_grey_literature(entry)
    api_status_raw = api_result.get("status", "not_checked")
    if is_grey and api_status_raw in ("not_found", "not_checked", "error", "partial_match"):
        return {
            "verdict": "SUSPICIOUS",
            "confidence": 0.55,
            "composite_risk": 0.55,
            "risk_factors": [
                f"Grey/industry literature ({grey_reason}) — not indexed in academic databases. Verify URL manually."
            ],
        }
    if is_grey and api_status_raw == "verified":
        return {
            "verdict": "REAL",
            "confidence": 0.88,
            "composite_risk": 0.12,
            "risk_factors": [],
        }

    signals = []
    total_weight = 0
    weighted_sum = 0
    
    # Signal 1: API status
    api_status = api_result.get("status", "not_checked")
    status_scores = {
        "verified": 0.0,
        "retracted": 0.1,
        "partial_match": 0.4,
        "not_checked": 0.5,
        "not_found": 0.7,
        "error": 0.5
    }
    status_risk = status_scores.get(api_status, 0.5)
    signals.append({"name": "database_match", "risk": status_risk, "weight": 0.35, "friendly": _get_user_friendly_message(api_status)})
    weighted_sum += status_risk * 0.35
    total_weight += 0.35
    
    # Signal 2: Title similarity (inverse)
    # Only add this signal when there was an actual match candidate returned from an API.
    # title_sim=0.0 because NO database returned a result is NOT evidence of a fake —
    # it just means the paper wasn't found (already captured by Signal 1).
    matched_title = api_result.get("matched_title") or ""
    if title_sim is not None and matched_title:
        title_risk = 1.0 - title_sim
        signals.append({"name": "title_match", "risk": title_risk, "weight": 0.25, "friendly": f"Title match: {int(title_sim*100)}%"})
        weighted_sum += title_risk * 0.25
        total_weight += 0.25
    
    # Signal 3: Missing required fields
    required_fields = ["authors", "title", "year"]
    missing_count = sum(1 for f in required_fields if not entry.get(f))
    missing_risk = min(missing_count / 3, 0.5)
    if missing_count > 0:
        signals.append({"name": "missing_fields", "risk": missing_risk, "weight": 0.1, "friendly": f"Missing {missing_count} required field(s)"})
        weighted_sum += missing_risk * 0.1
        total_weight += 0.1
    
    # Signal 4: Key consistency
    if entry.get("key_consistent") is False:
        signals.append({"name": "key_mismatch", "risk": 0.6, "weight": 0.1, "friendly": "Key doesn't match author/year"})
        weighted_sum += 0.6 * 0.1
        total_weight += 0.1
    
    # Signal 5: Journal plausibility
    journal = entry.get("journal", "")
    if journal:
        journal_plausible, journal_flags = _check_journal_plausibility(journal)
        if not journal_plausible:
            signals.append({"name": "journal", "risk": 0.55, "weight": 0.1, "friendly": "Journal name appears suspicious"})
            weighted_sum += 0.55 * 0.1
            total_weight += 0.1
    
    # Signal 6: Page range
    pages = entry.get("pages", "")
    if pages:
        pages_plausible, _ = _check_page_range_implausibility(pages, entry.get("year", ""))
        if not pages_plausible:
            signals.append({"name": "page_range", "risk": 0.4, "weight": 0.05, "friendly": "Page range seems unusual"})
            weighted_sum += 0.4 * 0.05
            total_weight += 0.05
    
    # Signal 7: DOI format validity
    doi = entry.get("doi", "") or ""
    if doi:
        from checker import _validate_doi_format
        doi_valid, doi_reason = _validate_doi_format(doi)
        if not doi_valid:
            signals.append({"name": "invalid_doi_format", "risk": 0.65, "weight": 0.08, "friendly": f"Invalid DOI format"})
            weighted_sum += 0.65 * 0.08
            total_weight += 0.08
    
    # Signal 8: Future-year detection
    year = entry.get("year", "") or ""
    if year:
        from checker import _check_year_plausibility
        year_ok, year_reason = _check_year_plausibility(str(year))
        if not year_ok:
            signals.append({"name": "implausible_year", "risk": 0.75, "weight": 0.08, "friendly": f"Year seems incorrect"})
            weighted_sum += 0.75 * 0.08
            total_weight += 0.08
    
    # Normalize
    composite_risk = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # Determine verdict based on composite risk
    if composite_risk >= 0.75:
        verdict = "FAKE"
        confidence = min(0.7 + (composite_risk - 0.75) * 1.2, 0.95)
    elif composite_risk >= 0.55:
        verdict = "SUSPICIOUS"
        confidence = 0.5 + (composite_risk - 0.55) * 1.5
    else:
        verdict = "REAL"
        confidence = 0.7 + (1.0 - composite_risk) * 0.3
    
    # Collect user-friendly risk factors
    risk_factors = []
    for s in signals:
        if s["risk"] >= 0.4:
            risk_factors.append(s["friendly"])
    
    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "composite_risk": round(composite_risk, 2),
        "risk_factors": risk_factors[:4]
    }


def _pre_screen_by_author_overlap(entry: dict, api_result: dict, title_sim: float) -> Optional[dict]:
    """Deterministic pre-screen using author overlap + multi-source evidence."""
    from checker import author_overlap_score
    api_status = api_result.get("status", "not_checked")
    confidence = api_result.get("confidence", 0)
    sources = api_result.get("sources_checked", [])
    n_sources = len(sources)
    is_retracted = api_result.get("is_retracted", False)
    has_doi = bool(api_result.get("doi"))
    has_oa_url = bool(api_result.get("open_access_url"))
    has_version_note = bool(api_result.get("version_note"))

    # Grey literature: never pre-screen as FAKE; defer to _compute_verdict
    is_grey, _ = _is_grey_literature(entry)
    if is_grey:
        return None

    if is_retracted:
        return {"verdict": "REAL", "confidence": 0.99,
                "reasoning": "Paper confirmed to exist but RETRACTED — do not cite",
                "risk_factors": ["RETRACTED"]}
    if has_doi and api_status == "verified" and confidence >= 0.80:
        return {"verdict": "REAL", "confidence": 0.95,
                "reasoning": f"DOI confirmed + title {confidence:.0%} match",
                "risk_factors": []}
    if has_oa_url and confidence >= 0.75:
        return {"verdict": "REAL", "confidence": 0.91,
                "reasoning": f"Open-access copy retrieved and verified",
                "risk_factors": []}
    if has_version_note and confidence >= 0.65:
        return {"verdict": "REAL", "confidence": 0.88,
                "reasoning": f"Preprint found: {api_result.get('version_note','')}",
                "risk_factors": []}
    if api_status == "verified" and confidence >= 0.88 and n_sources >= 2:
        return {"verdict": "REAL", "confidence": confidence,
                "reasoning": f"Confirmed by {n_sources} independent databases",
                "risk_factors": []}
    
    cited_authors = (entry.get("authors") or "").strip()
    correct_authors = (api_result.get("correct_authors") or
                       api_result.get("corrected_authors") or "").strip()
    if not cited_authors or not correct_authors:
        if api_status == "verified" and confidence >= 0.82:
            return {"verdict": "REAL", "confidence": confidence,
                    "reasoning": f"Verified in database — no author data to cross-check",
                    "risk_factors": []}
        return None
    
    overlap = author_overlap_score(cited_authors, correct_authors)
    if overlap is None:
        return None
    pct = int(overlap * 100)
    if overlap < 0.25 and api_status in ("verified", "partial_match") and title_sim >= 0.40:
        return {"verdict": "FAKE", "confidence": 0.88,
                "reasoning": f"Title matches but author overlap is only {pct}% — possible fabrication",
                "risk_factors": [f"Author mismatch ({pct}%)"]}
    if overlap >= 0.60 and api_status in ("verified", "partial_match") and confidence >= 0.65:
        return {"verdict": "REAL", "confidence": round(min(0.80 + overlap * 0.18, 0.97), 2),
                "reasoning": f"Author overlap {pct}% + title match",
                "risk_factors": []}
    return None


def _llm_reverify_medium_confidence(entry: dict, vr: dict, matched_title: str) -> Optional[dict]:
    """Re-verify a medium-confidence database match using the LLM."""
    if not _ai_available():
        return None
    title = entry.get("title", "")
    authors = entry.get("authors", "")
    year = entry.get("year", "")
    doi = entry.get("doi", "")
    api_confidence = round(vr.get("confidence", 0), 2)
    api_sources = vr.get("sources_checked", [])
    prompt = f"""You are an expert academic librarian doing a second-opinion check.

A citation checker found a medium-confidence database match. Determine if this is:
A) A real paper cited with minor metadata errors (e.g. different edition, arXiv vs published version)
B) A suspicious citation that needs professor review
C) A likely fabricated reference

CITED IN PAPER:
  Title:   {title}
  Authors: {authors}
  Year:    {year}
  DOI:     {doi or "none"}

DATABASE FOUND (confidence {api_confidence}):
  Title:   {matched_title}
  Sources: {", ".join(api_sources)}

Respond ONLY with valid JSON:
{{"verdict": "REAL", "confidence": 0.82, "reasoning": "one sentence", "is_version_mismatch": false, "version_note": ""}}
"""
    try:
        raw = _call_ai(prompt, max_tokens=200)
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        verdict = result.get("verdict", "SUSPICIOUS").upper()
        if verdict not in ("REAL", "SUSPICIOUS", "FAKE"):
            verdict = "SUSPICIOUS"
        return {
            "verdict": verdict,
            "confidence": float(result.get("confidence", 0.6)),
            "reasoning": result.get("reasoning", "LLM re-verification"),
            "version_note": result.get("version_note", ""),
            "is_version_mismatch": bool(result.get("is_version_mismatch", False)),
        }
    except Exception:
        return None


def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    """
    Determine REAL / SUSPICIOUS / FAKE for each reference.
    Fixed: deduplication, user-friendly messages, consistent verdicts.
    """
    from review_queue import is_venue_whitelisted
    
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
        vr = vr_by_key.get(entry["key"], {})
        matched_title = vr.get("matched_title", "")
        title_sim = 0.0
        if entry.get("title") and matched_title:
            title_sim = _local_title_similarity(entry["title"], matched_title)
        
        venue = entry.get("journal") or entry.get("booktitle") or ""
        whitelist_check = is_venue_whitelisted(venue)
        
        early = _pre_screen_by_author_overlap(entry, vr, title_sim)
        if early:
            pre_screen_cache[entry["key"]] = early
        else:
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)
            
            if whitelist_check.get("whitelisted") and composite["verdict"] == "FAKE":
                composite["verdict"] = "SUSPICIOUS"
                composite["confidence"] = 0.6
                composite["reasoning"] = f"Venue is whitelisted ({whitelist_check.get('venue')})"
            
            if composite["verdict"] != "SUSPICIOUS" and composite["confidence"] >= 0.75:
                pre_screen_cache[entry["key"]] = {
                    "verdict": composite["verdict"],
                    "confidence": composite["confidence"],
                    "reasoning": f"Analysis complete",
                    "risk_factors": composite.get("risk_factors", []),
                }
            else:
                matched_title = vr.get("matched_title", "")
                api_conf = vr.get("confidence", 0)
                if matched_title and 0.45 <= api_conf <= 0.74:
                    reverify = _llm_reverify_medium_confidence(entry, vr, matched_title)
                    if reverify and reverify["verdict"] == "REAL":
                        pre_screen_cache[entry["key"]] = {
                            "verdict": "REAL",
                            "confidence": reverify["confidence"],
                            "reasoning": f"LLM re-verified: {reverify['reasoning']}",
                            "risk_factors": [],
                            "version_note": reverify.get("version_note", ""),
                        }
                        if reverify.get("is_version_mismatch"):
                            pre_screen_cache[entry["key"]]["reasoning"] += " (version mismatch — paper is real)"
                        continue
                    elif reverify and reverify["verdict"] == "FAKE":
                        pre_screen_cache[entry["key"]] = {
                            "verdict": "FAKE",
                            "confidence": reverify["confidence"],
                            "reasoning": f"LLM: {reverify['reasoning']}",
                            "risk_factors": ["llm_reverify_fake"],
                        }
                        continue
                needs_ai.append((entry, vr, title_sim, composite))
    
    ai_verdicts_by_key: Dict[str, dict] = {}
    fake_count = suspicious_count = real_count = 0
    
    for chunk in _chunk(needs_ai, 15):
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
                "key_consistent": entry.get("key_consistent"),
                "api_status": vr.get("status", "not_checked"),
                "api_confidence": round(vr.get("confidence", 0), 2),
                "api_matched_title": vr.get("matched_title") or "",
                "api_sources": vr.get("sources_checked", []),
                "open_access_url": vr.get("open_access_url") or "",
                "web_evidence": vr.get("web_evidence") or "",
                "arxiv_version_note": vr.get("version_note") or "",
                "composite_risk": composite.get("composite_risk", 0.5),
            })
        
        prompt = f"""You are a senior academic librarian and integrity specialist.
Analyze each reference below and return REAL / SUSPICIOUS / FAKE.

━━━ VERDICTS ━━━
REAL       — Paper demonstrably exists. Confirmed by database or strong evidence.
SUSPICIOUS — Cannot confirm, cannot falsify. Flag for professor manual review.
FAKE       — Positive evidence of fabrication. Only use when confident.

━━━ CALIBRATION (read carefully) ━━━
CONSERVATIVE BIAS: Calling a real paper FAKE harms a student unjustly.
Only mark FAKE when ≥2 independent signals confirm fabrication.
"Not found in database" alone is NOT enough to call FAKE.
GREY LITERATURE RULE: Industry reports (Bitkom, Flexera, Gartner, etc.), government publications, podcasts, and company whitepapers are valid sources but NOT indexed in CrossRef or Semantic Scholar. If a reference has a known publisher or URL and its only issue is not being in an academic database, mark SUSPICIOUS (not FAKE).

Return ONLY valid JSON (no markdown, no explanation outside JSON):
{{
  "verdicts": [
    {{"key": "string", "verdict": "REAL", "confidence": 0.95,
      "reasoning": "one concise sentence explaining the verdict",
      "risk_factors": ["list", "of", "specific", "flags"]}}
  ],
  "fake_count": 0, "suspicious_count": 0, "real_count": 0,
  "summary": "2-3 sentence overall assessment of the submission quality"
}}

References (with pre-computed signals from API sources):
{json.dumps(combined, ensure_ascii=False, indent=2)}"""
        
        try:
            chunk_result = _call_ai_json(prompt, max_tokens=4000)
            for v in chunk_result.get("verdicts", []):
                ai_verdicts_by_key[v["key"]] = v
        except Exception as e:
            for entry, vr, title_sim, composite in chunk:
                verdict = composite["verdict"]
                ai_verdicts_by_key[entry["key"]] = {
                    "key": entry["key"],
                    "verdict": verdict,
                    "confidence": composite["confidence"],
                    "reasoning": composite.get("reasoning", f"Analysis: {int(composite['composite_risk']*100)}% risk"),
                    "risk_factors": composite.get("risk_factors", []),
                }
    
    # Collect all verdicts with deduplication by key
    seen_keys = set()
    for entry in bib_entries:
        key = entry["key"]
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
            matched_title = vr.get("matched_title", "")
            title_sim = 0.0
            if entry.get("title") and matched_title:
                title_sim = _local_title_similarity(entry["title"], matched_title)
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)
            venue = entry.get("journal") or entry.get("booktitle") or ""
            whitelist_check = is_venue_whitelisted(venue)
            if whitelist_check.get("whitelisted") and composite["verdict"] == "FAKE":
                composite["verdict"] = "SUSPICIOUS"
                composite["confidence"] = 0.6
            all_verdicts.append({
                "key": key,
                "verdict": composite["verdict"],
                "confidence": composite["confidence"],
                "reasoning": f"Analysis: {int(composite['composite_risk']*100)}% risk",
                "risk_factors": composite.get("risk_factors", []),
                "open_access_url": None,
            })
    
        # Count verdicts directly from the all_verdicts array
    fake_count = sum(1 for v in all_verdicts if v.get("verdict") == "FAKE")
    suspicious_count = sum(1 for v in all_verdicts if v.get("verdict") == "SUSPICIOUS")
    real_count = sum(1 for v in all_verdicts if v.get("verdict") == "REAL")
    
    return {
        "verdicts": all_verdicts,
        "fake_count": fake_count,
        "suspicious_count": suspicious_count,
        "real_count": real_count,
        "summary": f"Analysis complete: {fake_count} FAKE, {suspicious_count} SUSPICIOUS, {real_count} REAL references identified.",
    }


def ai_overall_verdict(filename: str, summary: dict, xcheck,
                       bib_list: list, verification_result: dict) -> dict:
    fake_count = verification_result.get("fake_count", 0)
    suspicious = verification_result.get("suspicious_count", 0)
    missing_cit = len(xcheck.cited_not_in_bib)
    orphaned = len(xcheck.in_bib_not_cited)
    incomplete = sum(1 for e in bib_list if e.completeness_issues)
    bib_count = len(bib_list)
    cited_count = len(xcheck.correctly_used) + missing_cit
    key_issues = [e for e in bib_list if e.key_consistent is False]
    
    key_issue_details = [
        f"[{e.key}]: " + "; ".join(i for i in e.completeness_issues if "key" in i.lower())
        for e in key_issues
    ][:6]
    incomplete_details = [
        f"[{e.key}]: {'; '.join(e.completeness_issues[:2])}"
        for e in bib_list if e.completeness_issues
    ][:8]
    fake_details = [
        f"[{v['key']}] {v.get('reasoning','')}"
        for v in verification_result.get("verdicts", []) if v.get("verdict") == "FAKE"
    ]
    suspicious_details = [
        f"[{v['key']}] {v.get('reasoning','')}"
        for v in verification_result.get("verdicts", []) if v.get("verdict") == "SUSPICIOUS"
    ][:5]
    
    prompt = f"""You are a professor reviewing a student submission to a GI LNI conference.
Automated checks have run on "{filename}". Synthesise into a final assessment.

STATISTICS:
- Bibliography entries: {bib_count}
- In-text citations: {cited_count}
- Cited but missing from bibliography: {missing_cit}
- In bibliography but never cited (orphaned): {orphaned}
- Entries with missing required fields: {incomplete}
- LNI key-vs-metadata mismatches (deterministic): {len(key_issues)}
- FAKE references (AI+composite verdict): {fake_count}
- SUSPICIOUS references: {suspicious}
- Duplicates: {summary.get('duplicates', 0)}
- Self-citations flagged: {summary.get('self_citations', 0)}

KEY INCONSISTENCIES:
{chr(10).join(key_issue_details) or "None"}

INCOMPLETE ENTRIES (sample):
{chr(10).join(incomplete_details) or "None"}

FAKE REFERENCES:
{chr(10).join(fake_details) or "None"}

SUSPICIOUS (sample):
{chr(10).join(suspicious_details) or "None"}

VERDICT RULES:
PASS  — references appear legitimate, only minor formatting issues
FLAG  — suspicious refs or key mismatches require professor spot-check
FAIL  — multiple likely fake references OR critical structural failures

Return ONLY valid JSON, no markdown:
{{
  "verdict": "PASS",
  "score": 85,
  "grade": "B",
  "verdict_reason": "1-2 sentences",
  "student_feedback": [],
  "professor_note": "One sentence on what to manually review."
}}"""
    
    try:
        return _call_ai_json(prompt, max_tokens=800)
    except Exception as e:
        score = 100
        score -= min(fake_count * 20, 60)
        score -= min(missing_cit * 10, 30)
        score -= min(incomplete * 5, 20)
        score -= min(orphaned * 3, 12)
        score -= min(len(key_issues) * 5, 15)
        score = max(0, min(100, score))
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 45 else "F"
        verdict = "FAIL" if fake_count >= 2 or score < 45 else "FLAG" if fake_count >= 1 or len(key_issues) >= 2 or score < 75 else "PASS"
        
        # Simplified professor note - just the keys to review
        keys_to_review = []
        for v in verification_result.get("verdicts", [])[:3]:
            if v.get("verdict") in ("FAKE", "SUSPICIOUS"):
                keys_to_review.append(f"[{v['key']}]")
        professor_note = f"Manually review references: {' '.join(keys_to_review)}" if keys_to_review else "Review flagged references manually"
        
        return {
            "verdict": verdict,
            "score": score,
            "grade": grade,
            "verdict_reason": f"Computed from check results (AI unavailable: {str(e)[:50]})",
            "student_feedback": [],
            "professor_note": professor_note,
            "error": str(e),
        }