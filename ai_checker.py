"""
AI Checker — v7.2 (ENV-based model config)
--------------------------
CHANGES v7.2:
  - Removed hardcoded Groq/Gemini URLs and model names
  - Single OpenAI-compatible backend driven by AI_BASE_URL + AI_MODEL + AI_API_KEY
  - Preserved ALL original logic and fallback handling
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
_AI_MODEL: str    = os.environ.get("AI_MODEL", "")
_AI_API_KEY: str  = os.environ.get("AI_API_KEY", "")

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
    """Call AI backend (OpenAI-compatible) with bounded retries and backoff."""
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
        result = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                _llm_cache_put(model_tag, system, prompt, result)
                return result
            elif resp.status_code == 429:
                # Rate limit: respect Retry-After header if present
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 8))
                continue
            elif resp.status_code in (400, 401, 403, 404):
                raise RuntimeError(
                    f"AI provider rejected the request (HTTP {resp.status_code}). "
                    "Check the API key, enabled Generative Language API, model, "
                    "and AI_BASE_URL."
                )
            else:
                # Other HTTP error
                print(f"AI API error {resp.status_code}: {resp.text[:200]}")
                
        except RuntimeError:
            raise
        except requests.exceptions.Timeout:
            print(f"AI API timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError as e:
            print(f"AI API connection error: {e}")
        except Exception as e:
            print(f"AI API exception: {e}")

        # Retry with exponential backoff
        if attempt < max_retries:
            time.sleep(1.5 ** attempt)

    # All retries exhausted
    raise RuntimeError(
        f"AI API call failed after {max_retries + 1} attempts. "
        f"Check AI_BASE_URL ({_AI_BASE_URL}), AI_MODEL ({_AI_MODEL}), and AI_API_KEY."
    )


def _call_ai_json(prompt: str, max_tokens: int = 2000, system: str = "",
                  timeout: int = 45, max_retries: int = 3) -> dict:
    text = _call_ai(prompt, max_tokens, system, timeout, max_retries).strip()
    
    # Strip markdown code blocks
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()
    
    # Strip leading "json" label
    if text.lower().startswith("json"):
        text = text[4:].strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Fallback: find JSON object or array in text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    
    # If all parsing fails, raise
    raise json.JSONDecodeError("Could not extract JSON from response", text, 0)


def _chunk(lst: list, size: int) -> list:
    """Split list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _ai_available() -> bool:
    """Check if AI backend is properly configured."""
    return bool(_AI_BASE_URL and _AI_MODEL and _AI_API_KEY)


# ---------------------------------------------------------------------------
# German venue detection helper (for INFORMATION only, not used for leniency)
# ---------------------------------------------------------------------------

def _is_german_academic_venue(entry: dict) -> bool:
    """Detect if this is a German academic venue (GI/LNI related).
    This is for INFORMATION only - no leniency is applied based on this."""
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
        'universität', 'hochschule', 'fraunhofer'
    ]
    
    return any(hint in venue_name for hint in german_hints)


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
        
        # Merge fields from AI extraction
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
# 2. Grey Literature Detection
# ---------------------------------------------------------------------------

def _is_grey_literature(entry: dict) -> tuple:
    """
    Detect industry reports, government docs, blog posts — valid sources
    not indexed in CrossRef/Semantic Scholar.
    Returns (is_grey: bool, reason: str).

    Strategy: rather than relying solely on an exhaustive hardcoded list of
    known non-academic publishers (which will always miss new ones — e.g. a
    report from a firm we've never enumerated), we first check a much
    smaller ALLOWLIST of known academic/publisher domains. Anything with a
    URL that is *not* on that allowlist, and doesn't otherwise look like a
    formal published work, is treated as "likely grey literature" even if
    we don't recognize the specific publisher. The named-list checks below
    still run first purely so the reason string is more specific when we do
    recognize the source.
    """
    title     = (entry.get("title") or "").lower()
    url       = (entry.get("url") or "").lower()
    publisher = (entry.get("publisher") or "").lower()
    entry_type = (entry.get("entry_type") or "").lower()
    raw       = (entry.get("raw_text") or entry.get("raw") or "").lower()

    # Domains that ARE legitimate academic/publisher venues — a reference
    # hosted here should never be treated as grey literature just because
    # it also has a URL (many real papers cite their DOI/publisher link).
    academic_allowlist = [
        "doi.org", "arxiv.org", "dl.acm.org", "ieeexplore.ieee.org",
        "link.springer.com", "springer.com", "sciencedirect.com",
        "jstor.org", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
        "nature.com", "mdpi.com", "openalex.org", "semanticscholar.org",
        "researchgate.net", "aclanthology.org", "openreview.net",
        "dl.gi.de", "gi.de/service", "onlinelibrary.wiley.com",
        "tandfonline.com", "cambridge.org", "oup.com", "plos.org",
        "usenix.org", "sciendo.com",
    ]

    # Check domains first - these are the most reliable
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
            domain_name = domain.split('.')[0]
            return True, f"Grey literature ({domain_name})"
    
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
    
    # FIXED v8.9: Don't treat conference proceedings as grey literature
    # Conference papers (inproceedings) from academic conferences are NOT grey lit
    if entry_type in ("inproceedings", "proceedings", "conference"):
        return False, ""
    
    if entry_type in ("website", "online", "misc") and url:
        return True, "Grey literature (website citation)"
    
    # FIXED v8.9: Be more conservative with the generic fallback
    # Many papers from PDFs will have non-standard URLs but are still real
    # Only flag as grey if it's clearly NOT an academic/publisher URL
    # AND the entry doesn't have enough metadata to be a real paper
    if url and entry_type not in ("article", "book", "inproceedings"):
        if not any(dom in url for dom in academic_allowlist):
            # Only flag as grey if title/authors are missing or look like grey lit
            title_has_report_word = any(word in title for word in 
                                       ['report', 'whitepaper', 'white paper', 'survey'])
            authors_missing = not (entry.get("authors") or "").strip()
            
            if title_has_report_word or (authors_missing and "publisher" not in entry.get("publisher", "").lower()):
                try:
                    domain_guess = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    domain_label = domain_guess.group(1) if domain_guess else "unrecognized domain"
                except Exception:
                    domain_label = "unrecognized domain"
                return True, f"Grey literature (unlisted source: {domain_label})"

    
    return False, ""


# Configurable fabrication detection patterns (can be extended via config file)
_FABRICATION_PATTERNS = os.getenv("LNI_FABRICATION_PATTERNS", "").split("|") if os.getenv("LNI_FABRICATION_PATTERNS") else []
_FABRICATION_KEYWORDS = os.getenv("LNI_FABRICATION_KEYWORDS", "").split("|") if os.getenv("LNI_FABRICATION_KEYWORDS") else []

# Fallback patterns if no env config
if not _FABRICATION_PATTERNS:
    _FABRICATION_PATTERNS = [
        r"quantum.*supremacy.*doesn't exist",
        r"solving p=np",
        r"blockchain.*cures",
        r"neural telepathy",
        r"perpetual motion",
        r"time.*travel.*machine",
        r"unicorn horn.*computing",
        r"flat.*earth.*geometry",
        r"infinite energy",
        r"ai.*becomes.*god",
        r"magic beans",
        r"impossible.*source",
    ]

if not _FABRICATION_KEYWORDS:
    _FABRICATION_KEYWORDS = ["quantum", "blockchain", "ai", "neural", "impossible"]


def _is_fabricated_title(title: str) -> tuple:
    """
    Detect fabricated/nonsense paper titles using configurable patterns.
    Returns (is_fabricated: bool, confidence: float).
    
    Patterns can be customized via:
      - LNI_FABRICATION_PATTERNS env var (pipe-separated regex patterns)
      - LNI_FABRICATION_KEYWORDS env var (pipe-separated keywords for rambling check)
    """
    if not title:
        return False, 0.0
    
    title_lower = title.lower()
    
    # Check explicit patterns first
    for pattern in _FABRICATION_PATTERNS:
        try:
            if re.search(pattern, title_lower):
                return True, 0.95
        except re.error:
            continue  # Invalid regex, skip
    
    # Check for rambling structure + vague keywords
    if len(title) > 120:
        keyword_count = sum(1 for kw in _FABRICATION_KEYWORDS if kw in title_lower)
        
        # Also check for rambling phrases that indicate nonsense
        rambling_phrases = ["for everything", "makes no sense", "very long title", "solves everything"]
        phrase_count = sum(1 for phrase in rambling_phrases if phrase in title_lower)
        
        if keyword_count >= 2 and phrase_count >= 1:
            return True, 0.90
    
    return False, 0.0


# ---------------------------------------------------------------------------
# 3. Composite Fake Detection Signals (FIXED: Better thresholds)
# ---------------------------------------------------------------------------

def _check_journal_plausibility(journal: str) -> tuple:
    """Check if a journal name sounds plausible or fake."""
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
        return True, []
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
        return "Title match confirmed in academic database"
    elif status == "partial_match":
        return "Partial/approximate match found — review the matched title"
    elif status == "not_found":
        return "Not found in any searched database"
    elif status == "error":
        return "Verification could not complete (network/API error)"
    elif status == "retracted":
        return "⚠️ PAPER HAS BEEN RETRACTED — do not cite"
    elif "DOI" in details:
        return "DOI format issue — check the DOI value"
    elif "year" in details.lower():
        return "Year appears incorrect or implausible"
    else:
        return "Verification attempted"


def _compute_verdict_with_confidence(entry: dict, api_result: dict, title_sim: float) -> dict:
    """
    Compute verdict based on available evidence.
    FIXED v7.1: Lower thresholds for REAL detection to catch more papers.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Get API status
    # ─────────────────────────────────────────────────────────────────────────
    api_status = api_result.get("status", "not_checked")
    api_confidence = api_result.get("confidence", 0.0)
    matched_title = api_result.get("matched_title", "")
    sources = api_result.get("sources_checked", [])
    has_doi = bool(entry.get("doi") or api_result.get("doi"))
    has_url_confirmed = bool(api_result.get("open_access_url") and api_status == "verified")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: If API confirmed it, return REAL immediately (LOWERED THRESHOLD)
    # ─────────────────────────────────────────────────────────────────────────
    if api_status == "verified" and api_confidence >= 0.75 and matched_title:
        return {
            "verdict": "REAL",
            "confidence": max(api_confidence, 0.85),
            "composite_risk": 0.10,
            "risk_factors": [],
            "reasoning": f"Confirmed in {', '.join(sources[:2])}: '{matched_title[:60]}'",
        }

    # Partial match that was promoted to verified confidence via author overlap
    if api_status == "partial_match" and api_confidence >= 0.80 and matched_title:
        return {
            "verdict": "REAL",
            "confidence": api_confidence,
            "composite_risk": 0.15,
            "risk_factors": ["Title differs — may have format issues"],
            "reasoning": f"Strong author+title evidence in {', '.join(sources[:2])}: '{matched_title[:60]}'",
        }
    
    # DOI match is strongest evidence
    if has_doi and api_status == "verified" and api_confidence >= 0.75:
        return {
            "verdict": "REAL",
            "confidence": 0.92,
            "composite_risk": 0.10,
            "risk_factors": [],
            "reasoning": f"DOI verified in CrossRef: {entry.get('doi')}",
        }
    
    # URL confirmed (HTTP 200 with title match)
    if has_url_confirmed:
        return {
            "verdict": "REAL",
            "confidence": 0.88,
            "composite_risk": 0.12,
            "risk_factors": [],
            "reasoning": f"URL verified: page title matches cited reference",
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2b: URL was attempted but failed or title didn't match
    # A dead/mismatched URL is concrete negative evidence — cannot be REAL.
    # ─────────────────────────────────────────────────────────────────────────
    if api_status == "url_blocked":
        url_note = api_result.get("note", "")
        return {
            "verdict": "SUSPICIOUS",
            "confidence": 0.70,
            "composite_risk": 0.75,
            "risk_factors": [f"URL check failed: {url_note[:120]}"],
            "reasoning": f"URL check failed — {url_note[:120]}",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Grey literature detection (valid sources not in academic DBs)
    # ─────────────────────────────────────────────────────────────────────────
    is_grey, grey_reason = _is_grey_literature(entry)
    
    if is_grey:
        has_url = bool(entry.get("url"))
        # Signal to caller that AI should review this — return a sentinel
        # composite_risk of 0.55 puts it in "needs_ai" territory when AI is available.
        # When AI is NOT available we fall through to SUSPICIOUS as before.
        if _ai_available():
            # Return a result that will be picked up by needs_ai (confidence < 0.75, not REAL)
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.55,
                "composite_risk": 0.55,
                "risk_factors": [
                    f"Grey/industry literature ({grey_reason}) — verify the URL manually."
                ],
                "reasoning": f"Grey literature ({grey_reason}). Manual review required.",
            }
        # AI not available — deterministic fallback
        if has_url:
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.55,
                "composite_risk": 0.55,
                "risk_factors": [
                    f"Grey/industry literature ({grey_reason}) — verify the URL manually."
                ],
                "reasoning": "Industry/government report. Not expected in academic databases. AI analysis failed: No AI API key configured. Set AI_API_KEY in your .env file.",
            }
        return {
            "verdict": "SUSPICIOUS",
            "confidence": 0.50,
            "composite_risk": 0.60,
            "risk_factors": [
                f"Grey/industry literature ({grey_reason}) with no URL to verify."
            ],
            "reasoning": "Manual verification required. No AI API key configured.",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: No API confirmation → Compute risk signals (FIXED: better thresholds)
    # ─────────────────────────────────────────────────────────────────────────
    signals = []
    total_weight = 0
    weighted_sum = 0
    
    # Signal 1: API status
    status_scores = {
        "verified": 0.0,
        "retracted": 0.1,
        "partial_match": 0.4,
        "not_checked": 0.5,
        "not_found": 0.50,
        "suspicious": 0.60,
        "error": 0.55,
        "url_blocked": 0.75,   # URL fetch failed or title mismatch — strong negative signal
    }
    status_risk = status_scores.get(api_status, 0.50)
    signals.append({"name": "database_match", "risk": status_risk, "weight": 0.35, 
                    "friendly": _get_user_friendly_message(api_status)})
    weighted_sum += status_risk * 0.35
    total_weight += 0.35
    
    # Signal 2: Title similarity (only if we have a match from API)
    if title_sim is not None and matched_title:
        title_risk = 1.0 - min(title_sim, 0.95)
        signals.append({"name": "title_match", "risk": title_risk, "weight": 0.30, 
                        "friendly": f"Title match: {int(title_sim*100)}%"})
        weighted_sum += title_risk * 0.30
        total_weight += 0.30
    
    # Signal 3: Missing required fields
    required_fields = ["authors", "title", "year"]
    missing_count = sum(1 for f in required_fields if not entry.get(f))
    missing_risk = min(missing_count / 3, 0.5)
    if missing_count > 0:
        signals.append({"name": "missing_fields", "risk": missing_risk, "weight": 0.15, 
                        "friendly": f"Missing {missing_count} required field(s)"})
        weighted_sum += missing_risk * 0.15
        total_weight += 0.15
    
    # Signal 4: Key consistency
    if entry.get("key_consistent") is False:
        signals.append({"name": "key_mismatch", "risk": 0.5, "weight": 0.10, 
                        "friendly": "Key doesn't match author/year"})
        weighted_sum += 0.5 * 0.10
        total_weight += 0.10
    
    # Signal 5: Journal plausibility (informational only)
    journal = entry.get("journal", "")
    if journal:
        journal_plausible, journal_flags = _check_journal_plausibility(journal)
        if not journal_plausible:
            signals.append({"name": "journal", "risk": 0.5, "weight": 0.0,
                            "friendly": "Journal name appears suspicious"})

    # Signal 6a: Suspicious / placeholder author name — strong weighted signal
    _fake_author_surnames = {
        "ghost", "fake", "test", "example", "placeholder", "unknown",
        "anonymous", "nobody", "someone", "author", "dummy", "sample",
        "demo", "null", "none",
    }
    authors_str = (entry.get("authors") or "").strip()
    if authors_str:
        first_author = authors_str.split(";")[0].strip()
        if "," in first_author:
            surname_part = first_author.split(",")[0].strip().lower()
        else:
            parts = first_author.split()
            surname_part = parts[-1].lower() if parts else ""
        if surname_part in _fake_author_surnames:
            signals.append({
                "name": "fake_author",
                "risk": 0.90,
                "weight": 0.30,
                "friendly": f"Author name '{first_author}' looks like a placeholder",
            })
            weighted_sum += 0.90 * 0.30
            total_weight += 0.30
    
    # Signal 6: Page range (informational only)
    pages = entry.get("pages", "")
    if pages:
        pages_plausible, page_flags = _check_page_range_implausibility(pages, entry.get("year", ""))
        if not pages_plausible:
            risk_val = 0.6 if any("Extremely long" in f for f in page_flags) else 0.4
            signals.append({"name": "page_range", "risk": risk_val, "weight": 0.0,
                            "friendly": page_flags[0] if page_flags else "Page range seems unusual"})
    
    # Normalize
    composite_risk = weighted_sum / total_weight if total_weight > 0 else 0.4
    
    # ─────────────────────────────────────────────────────────────────────────
    # FIXED v7.1: Lower thresholds for better REAL detection
    # FAKE: >= 0.85 (only for obvious fakes)
    # SUSPICIOUS: 0.45-0.84
    # REAL: < 0.45 (lowered from 0.55)
    # ─────────────────────────────────────────────────────────────────────────
    if composite_risk >= 0.85:
        verdict = "FAKE"
        confidence = min(0.7 + (composite_risk - 0.85) * 2.0, 0.95)
    elif composite_risk >= 0.45:
        verdict = "SUSPICIOUS"
        confidence = 0.5 + (composite_risk - 0.45) * 0.8
    else:
        # Low risk - mark as REAL if we have some evidence
        if api_status == "not_found" and missing_count >= 2:
            verdict = "SUSPICIOUS"
            confidence = 0.52
        else:
            verdict = "REAL"
            confidence = 0.70 + (0.45 - composite_risk) * 0.5
    
    # Collect user-friendly risk factors (only medium/high risk)
    risk_factors = []
    for s in signals:
        if s["risk"] >= 0.5 and s["weight"] > 0:
            risk_factors.append(s["friendly"])
    
    return {
        "verdict": verdict,
        "confidence": round(min(confidence, 0.98), 2),
        "composite_risk": round(composite_risk, 2),
        "risk_factors": risk_factors[:4],
        "reasoning": f"Analysis: {int(composite_risk*100)}% risk score -> {verdict}. API status: {api_status}.",
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
    matched_title = api_result.get("matched_title", "")

    # Grey literature: never pre-screen as FAKE, but dead URLs are SUSPICIOUS immediately
    is_grey, grey_reason = _is_grey_literature(entry)
    if is_grey:
        if api_status == "url_blocked":
            url_note = api_result.get("note", "URL check failed")
            return {
                "verdict": "SUSPICIOUS",
                "confidence": 0.75,
                "reasoning": f"Grey literature with failed URL check: {url_note[:120]}",
                "risk_factors": [f"URL check failed: {url_note[:80]}"],
            }
        return None

    # Suspicious / placeholder author: never pre-screen as REAL
    _fake_author_surnames = {
        "ghost", "fake", "test", "example", "placeholder", "unknown",
        "anonymous", "nobody", "someone", "author", "dummy", "sample",
        "demo", "null", "none",
    }
    authors_str = (entry.get("authors") or "").strip()
    if authors_str:
        first_author = authors_str.split(";")[0].strip()
        surname_part = (
            first_author.split(",")[0].strip().lower()
            if "," in first_author
            else (first_author.split()[-1].lower() if first_author.split() else "")
        )
        if surname_part in _fake_author_surnames:
            return None  # force full analysis; don't shortcut to REAL

    if is_retracted:
        return {"verdict": "REAL", "confidence": 0.99,
                "reasoning": "Paper confirmed to exist but RETRACTED — do not cite",
                "risk_factors": ["RETRACTED"]}
    
    if has_doi and api_status == "verified" and confidence >= 0.75 and matched_title:
        return {"verdict": "REAL", "confidence": 0.95,
                "reasoning": f"DOI confirmed + title match",
                "risk_factors": []}
    
    if has_oa_url and confidence >= 0.70 and matched_title:
        return {"verdict": "REAL", "confidence": 0.91,
                "reasoning": "Open-access copy retrieved and verified",
                "risk_factors": []}
    
    if has_version_note and confidence >= 0.65:
        return {"verdict": "REAL", "confidence": 0.88,
                "reasoning": f"Preprint found: {api_result.get('version_note','')}",
                "risk_factors": []}
    
    if api_status == "verified" and confidence >= 0.80 and n_sources >= 2 and matched_title:
        return {"verdict": "REAL", "confidence": confidence,
                "reasoning": f"Confirmed by {n_sources} independent databases",
                "risk_factors": []}
    
    cited_authors = (entry.get("authors") or "").strip()
    correct_authors = (api_result.get("correct_authors") or
                       api_result.get("corrected_authors") or "").strip()
    if not cited_authors or not correct_authors:
        if api_status == "verified" and confidence >= 0.75 and (has_doi or has_oa_url) and matched_title:
            return {"verdict": "REAL", "confidence": confidence,
                    "reasoning": "Verified in database with DOI/URL",
                    "risk_factors": []}
        return None
    
    overlap = author_overlap_score(cited_authors, correct_authors)
    if overlap is None:
        return None
    pct = int(overlap * 100)
    
    # Lower thresholds for REAL detection
    if overlap < 0.20 and api_status in ("verified", "partial_match") and title_sim >= 0.60:
        return {"verdict": "FAKE", "confidence": 0.85,
                "reasoning": f"Title matches but author overlap is only {pct}% — possible fabrication",
                "risk_factors": [f"Author mismatch ({pct}%)"]}
    
    if overlap >= 0.50 and api_status in ("verified", "partial_match") and confidence >= 0.60:
        return {"verdict": "REAL", "confidence": round(min(0.80 + overlap * 0.18, 0.97), 2),
                "reasoning": f"Author overlap {pct}% + title/API match",
                "risk_factors": []}

    # Strong author match alone (even with low title sim due to bad formatting)
    if overlap >= 0.80 and api_status in ("verified", "partial_match", "suspicious") and matched_title:
        return {"verdict": "REAL", "confidence": round(min(0.75 + overlap * 0.20, 0.95), 2),
                "reasoning": f"Strong author overlap ({pct}%) confirms identity despite format issues",
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
A) A real paper cited with minor metadata errors
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
    FIXED v7.1: Better integration with API results.
    """
    from review_queue import is_venue_whitelisted
    
    if not bib_entries:
        return {"verdicts": [], "summary": "No entries to verify.",
                "fake_count": 0, "suspicious_count": 0, "real_count": 0}
    
    vr_by_key = {vr["key"]: vr for vr in api_results}

    def _is_authoritatively_verified(vr: dict) -> bool:
        """True only if the result came from a direct, high-confidence source."""
        if vr.get("status") != "verified":
            return False
        note = (vr.get("note") or "").lower()
        # Promoted partial matches are explicitly flagged in their note
        if "promoted from partial" in note:
            return False
        # Lower threshold to 0.75 to skip AI verification for confident API results
        # (APIs now early-stop at 0.85, so this is safe)
        if vr.get("confidence", 0) < 0.75:
            return False
        return True

    if (len(api_results) == len(bib_entries)
            and all(_is_authoritatively_verified(vr) for vr in api_results)):
        verdicts = [
            {
                "key": entry["key"],
                "verdict": "REAL",
                "confidence": vr_by_key[entry["key"]].get("confidence", 0.95),
                "reasoning": "Confirmed by the reference verification stage.",
                "risk_factors": [],
            }
            for entry in bib_entries
            if entry["key"] in vr_by_key
        ]
        return {
            "verdicts": verdicts,
            "fake_count": 0,
            "suspicious_count": 0,
            "real_count": len(verdicts),
            "summary": f"Analysis: {len(verdicts)} REAL, 0 SUSPICIOUS, 0 FAKE",
        }

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
        
        # FIXED: Try to get title similarity from the API result first
        if vr.get("title_match_score") is not None:
            title_sim = vr["title_match_score"]
        elif entry.get("title") and matched_title:
            title_sim = _local_title_similarity(entry["title"], matched_title)
        
        # FIXED: Check if this is a duplicate entry
        if vr.get("is_duplicate") and vr.get("duplicate_of"):
            canonical_key = vr["duplicate_of"]
            all_verdicts.append({
                "key": key,
                "verdict": "REAL",
                "confidence": vr.get("confidence", 0.9),
                "reasoning": f"Duplicate of [{canonical_key}] — same paper",
                "risk_factors": [],
            })
            continue
        
        early = _pre_screen_by_author_overlap(entry, vr, title_sim)
        if early:
            pre_screen_cache[key] = early
        else:
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)

            # Grey literature and low-confidence results go to AI when available
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
    ai_failed = False
    ai_error_message = ""
    
    # Process AI for uncertain entries
    if _ai_available() and needs_ai:
        try:
            # Process in smaller chunks for better results
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
                        "key_consistent": entry.get("key_consistent"),
                        "api_status": vr.get("status", "not_checked"),
                        "api_confidence": round(vr.get("confidence", 0), 2),
                        "api_matched_title": vr.get("matched_title") or "",
                        "api_sources": vr.get("sources_checked", []),
                        "open_access_url": vr.get("open_access_url") or "",
                        "web_evidence": vr.get("web_evidence") or "",
                        "arxiv_version_note": vr.get("version_note") or "",
                        "composite_risk": composite.get("composite_risk", 0.5),
                        # Concrete URL fetch outcome — treat this as ground truth
                        "url_fetch_result": vr.get("note") or "",
                    })
                
                prompt = f"""You are a strict academic librarian verifying bibliography entries. For each reference return REAL / SUSPICIOUS / FAKE.

Each entry includes a field "url_fetch_result" — this is the ACTUAL outcome of fetching the URL. Treat it as ground truth.
Each entry includes "api_status" — treat "url_blocked" as meaning the URL check failed or title did not match.

RULES — apply ALL, no exceptions:

REAL requires ALL of:
  - Specific, plausible title (not generic)
  - Named real person as author — NOT "Medium Staff", "Web Author", "Unknown", "Community", "StackOverflow Community", "GitHub Contributors", or any team/org name
  - Year present and plausible
  - For URL-only entries: api_status must be "verified" (HTTP 200 + title matched). If api_status is "url_blocked", the URL either returned a non-200 status or the page title did not match the cited title — this entry CANNOT be REAL.
  - For academic entries: database match with matching title AND author AND year (±1 year)

SUSPICIOUS when any of:
  - api_status is "url_blocked" — URL is dead, returned an error, or page title does not match
  - Author is a team/org name, not a specific person
  - Year missing
  - Partial database match with differing author or year
  - Insufficient evidence to confirm

FAKE when:
  - Title is nonsensical or fabricated
  - Authors are clearly placeholders
  - Year is impossible

NEVER mark REAL based on:
  - Domain looking legitimate (gi.de, medium.com, github.com are not evidence alone)
  - URL path looking plausible
  - Title sounding reasonable
  - Organization being well-known
  - Your own knowledge that a domain exists

Your reasoning must reference specific fields from the entry data (url_fetch_result, api_status, author name, year) — not general impressions.

Return ONLY valid JSON:
{{
  "verdicts": [
    {{"key": "string", "verdict": "REAL", "confidence": 0.95,
      "reasoning": "cite the specific field values that support this verdict", "risk_factors": []}}
  ]
}}

References:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""
                
                chunk_result = _call_ai_json(prompt, max_tokens=4000)
                for v in chunk_result.get("verdicts", []):
                    # Automatic AI output is evidence for review, not a
                    # professor-confirmed fake verdict.
                    if str(v.get("verdict", "")).upper() == "FAKE":
                        v["verdict"] = "SUSPICIOUS"
                    ai_verdicts_by_key[v["key"]] = v
        except Exception as e:
            ai_failed = True
            ai_error_message = f"AI analysis failed: {str(e)[:200]}"
    
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
            # AI was not run or failed - use composite result with explanation
            vr = vr_by_key.get(key, {})
            matched_title = vr.get("matched_title", "")
            title_sim = 0.0
            if entry.get("title") and matched_title:
                title_sim = _local_title_similarity(entry["title"], matched_title)
            composite = _compute_verdict_with_confidence(entry, vr, title_sim)
            
            # Add explanation about AI status
            if not _ai_available():
                composite["reasoning"] += " AI was not available (no API key configured)."
            elif ai_failed and key not in ai_verdicts_by_key:
                composite["reasoning"] += f" {ai_error_message}"
            elif needs_ai and key not in ai_verdicts_by_key:
                composite["reasoning"] += " AI analysis was attempted but did not return a verdict for this entry."
            
            all_verdicts.append({
                "key": key,
                "verdict": composite["verdict"],
                "confidence": composite["confidence"],
                "reasoning": composite.get("reasoning", ""),
                "risk_factors": composite.get("risk_factors", []),
                "open_access_url": None,
            })
    
    # FIXED: Ensure duplicate entries get REAL status if they're duplicates of REAL entries
    for vr in api_results:
        if getattr(vr, "is_duplicate", False) and getattr(vr, "duplicate_of", None):
            dup_key = vr.key
            canonical_key = vr.duplicate_of
            # Find the canonical verdict
            for v in all_verdicts:
                if v["key"] == canonical_key and v["verdict"] == "REAL":
                    # Update the duplicate verdict
                    for v2 in all_verdicts:
                        if v2["key"] == dup_key:
                            v2["verdict"] = "REAL"
                            v2["confidence"] = v["confidence"]
                            v2["reasoning"] = f"Duplicate of [{canonical_key}] — same paper"
                            v2["risk_factors"] = []
                            break
                    break
    
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
    cited_count = len(xcheck.correctly_used) + missing_cit
    key_issues = [e for e in bib_list if e.key_consistent is False]

    # A clean, fully verified submission needs no second LLM call. This avoids
    # waiting on provider retries after the substantive checks already passed.
    if (fake_count == 0 and suspicious == 0 and missing_cit == 0
            and orphaned == 0 and incomplete == 0 and not key_issues):
        return {
            "verdict": "PASS",
            "score": 100,
            "grade": "A",
            "verdict_reason": "All references verified and no citation or format issues found.",
            "student_feedback": [],
            "professor_note": "No issues found",
        }
    
    # Deterministic score/verdict — the AI may only write feedback text, not change these
    # Suspicious findings require professor review and do not reduce the score.
    det_score = 100
    det_score -= min(missing_cit * 5, 20)
    det_score -= min(incomplete * 2, 10)
    det_score -= min(orphaned * 2, 10)
    det_score -= min(len(key_issues) * 3, 15)
    det_score = max(0, min(100, det_score))
    det_verdict = "PASS" if det_score >= 80 else "FLAG" if det_score >= 60 else "FAIL"
    det_grade = "A" if det_score >= 90 else "B" if det_score >= 80 else "C" if det_score >= 60 else "D" if det_score >= 50 else "F"

    keys_to_review = [
        f"[{v['key']}]" for v in verification_result.get("verdicts", [])[:5]
        if v.get("verdict") in ("FAKE", "SUSPICIOUS")
    ]
    professor_note = f"Manually review: {' '.join(keys_to_review)}" if keys_to_review else "No issues found"

    return {
        "verdict": det_verdict,
        "score": det_score,
        "grade": det_grade,
        "verdict_reason": f"Score {det_score}/100: {suspicious} suspicious reference(s), {missing_cit} missing citation(s).",
        "student_feedback": [],
        "professor_note": professor_note,
    }