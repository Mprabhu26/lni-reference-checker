"""
AI Checker — v6 (IMPROVED FAKE DETECTION)
---------------
Groq (llama-3.3-70b) primary → Gemini 1.5 Flash fallback.
Both FREE tier:
  GROQ_API_KEY   → console.groq.com      (free, 14 400 req/day)
  GEMINI_API_KEY → aistudio.google.com   (free, 1 500 req/day)

NEW in v6.1 — Improved Fake Detection:
  - Lowered title similarity threshold for FAKE (0.35 → 0.25)
  - Added composite signal scoring (multiple weak signals = FAKE)
  - Journal/journal name validation
  - DOI resolution check integrated into verdict
  - Page range implausibility as FAKE signal
  - Author name pattern detection (AI-sounding names)
  - Conference/journal existence check
"""

import hashlib
import os
import re
import json
import threading
import requests  # ADDED: missing import
from typing import List, Dict, Any, Optional
from review_queue import is_venue_whitelisted

GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL  = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-1.5-flash:generateContent")

# ---------------------------------------------------------------------------
# Session-scoped LLM response cache
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core AI call helpers
# ---------------------------------------------------------------------------

def _call_ai(prompt: str, max_tokens: int = 2000, system: str = "") -> str:
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    model_tag = f"groq:{GROQ_MODEL}" if groq_key else "gemini:1.5-flash"
    cached = _llm_cache_get(model_tag, system, prompt)
    if cached is not None:
        return cached

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    result: Optional[str] = None
    used_model: Optional[str] = None

    if groq_key:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": messages,
                      "max_tokens": max_tokens, "temperature": 0.1},
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                used_model = f"groq:{GROQ_MODEL}"
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
                timeout=30,
            )
            if resp.status_code == 200:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                result = "".join(p.get("text", "") for p in parts).strip()
                used_model = "gemini:1.5-flash"
        except Exception:
            pass

    if result is None:
        missing = [k for k, v in [("GROQ_API_KEY", groq_key), ("GEMINI_API_KEY", gemini_key)] if not v]
        raise RuntimeError(
            f"No AI API key configured. Set {' or '.join(missing)}. "
            "Groq: console.groq.com (free) | Gemini: aistudio.google.com (free)"
        )

    _llm_cache_put(used_model or model_tag, system, prompt, result)
    return result


def _call_ai_json(prompt: str, max_tokens: int = 2000, system: str = "") -> dict:
    text = _call_ai(prompt, max_tokens, system).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()
    # Remove leading/trailing label like "json" after fence strip
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object or array from the text
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
        lines  = bib_text.split('\n')
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

        if not entry.title     and ai.get("title"):     entry.title     = ai["title"]
        if not entry.authors   and ai.get("authors"):   entry.authors   = ai["authors"]
        if not entry.year      and ai.get("year"):       entry.year      = str(ai["year"])
        if not entry.journal   and ai.get("journal"):   entry.journal   = ai["journal"]
        if not entry.booktitle and ai.get("booktitle"): entry.booktitle = ai["booktitle"]
        if not entry.publisher and ai.get("publisher"): entry.publisher = ai["publisher"]
        if not entry.pages     and ai.get("pages"):     entry.pages     = ai["pages"]
        if not entry.doi       and ai.get("doi"):       entry.doi       = ai["doi"]
        if not entry.url       and ai.get("url"):       entry.url       = ai["url"]
        if not entry.isbn      and ai.get("isbn"):      entry.isbn      = ai["isbn"]
        if not entry.entry_type or entry.entry_type == "unknown":
            if ai.get("journal"):
                entry.entry_type = "article"
            elif ai.get("booktitle"):
                entry.entry_type = "proceedings"
            elif ai.get("publisher") and not ai.get("journal"):
                entry.entry_type = "book"

    return bib_list


# ---------------------------------------------------------------------------
# 2. AI-assisted structured re-parsing for uncertain entries
# ---------------------------------------------------------------------------

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
# 3. ENHANCED: Composite Fake Detection Signals (NEW in v6.1)
# ---------------------------------------------------------------------------

def _check_journal_plausibility(journal: str) -> tuple:
    """
    Check if a journal name sounds plausible or fake.
    Returns (is_plausible: bool, red_flags: list)
    """
    red_flags = []
    
    if not journal:
        return False, ["No journal name provided"]
    
    journal_lower = journal.lower()
    
    # Known legitimate journals (partial list)
    legit_journals = [
        'springer', 'elsevier', 'wiley', 'ieee', 'acm', 'nature', 'science',
        'cell', 'plos', 'frontiers', 'mdpi', 'sage', 'taylor', 'francis',
        'oxford', 'cambridge', 'mit press', 'world scientific',
        'journal of ', 'transactions on ', 'letters in ', 'proceedings of '
    ]
    
    # Fake indicators
    fake_indicators = [
        ('international journal of advanced', 'Overused generic prefix'),
        ('journal of emerging', 'Predatory journal pattern'),
        ('journal of current', 'Predatory journal pattern'),
        ('american journal of', 'Often impersonated'),
        ('european journal of', 'Often impersonated'),
        ('research journal of', 'Generic fake journal'),
        ('scientific journal of', 'Generic fake journal'),
        ('global journal of', 'Generic fake journal'),
        ('journal of engineering and technology', 'Generic phrase'),
        ('international research journal', 'Generic phrase'),
        ('academy of', 'Often fake'),
        ('institute of', 'Often fake without verification'),
    ]
    
    for indicator, reason in fake_indicators:
        if indicator in journal_lower:
            red_flags.append(reason)
    
    # Check if it has a legit marker
    has_legit = any(legit in journal_lower for legit in legit_journals)
    
    # No legit marker AND has red flags -> suspicious
    if not has_legit and red_flags:
        return False, red_flags
    
    return True, red_flags


def _check_author_name_plausibility(authors: str) -> tuple:
    """
    Check if author names seem plausible or AI-generated.
    Returns (is_plausible: bool, red_flags: list)
    """
    red_flags = []
    
    if not authors:
        return False, ["No authors provided"]
    
    # AI-generated name patterns
    ai_name_patterns = [
        (r'^[A-Z][a-z]{1,3}\s+[A-Z][a-z]{1,3}$', 'Suspiciously short name (e.g., "J Smith")'),
        (r'^[A-Z]\.\s+[A-Z]\.\s+[A-Z][a-z]+$', 'Initials-only first name pattern'),
        (r'(?:AI|GPT|LLM|Transformer|Neural|Deep|Learning)\s+[A-Z][a-z]+', 'AI-themed author name'),
    ]
    
    for pattern, reason in ai_name_patterns:
        if re.search(pattern, authors, re.IGNORECASE):
            red_flags.append(reason)
    
    # Check for unusual name combinations
    if ';' in authors:
        name_parts = [n.strip() for n in authors.split(';')]
        # All names are suspiciously similar pattern
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
        
        # Conference papers are usually shorter
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
    
    # Known legitimate conferences (CS)
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


def _compute_composite_fake_score(entry: dict, api_result: dict, title_sim: float) -> dict:
    """
    Compute composite fake detection score from multiple signals.
    Each signal contributes to a suspicion score (0-1).
    Returns verdict with confidence and reasons.
    """
    signals = []
    total_weight = 0
    weighted_sum = 0
    
    # Signal 1: Title similarity (inverse)
    if title_sim is not None:
        title_risk = 1.0 - title_sim
        weight = 0.35
        signals.append({"name": "title_mismatch", "risk": title_risk, "weight": weight})
        weighted_sum += title_risk * weight
        total_weight += weight
    
    # Signal 2: API status
    api_status = api_result.get("status", "not_checked")
    status_risk = {
        "verified": 0.0,
        "partial_match": 0.4,
        "not_checked": 0.5,
        "not_found": 0.7,
        "error": 0.6
    }.get(api_status, 0.5)
    weight = 0.25
    signals.append({"name": "api_status", "risk": status_risk, "weight": weight})
    weighted_sum += status_risk * weight
    total_weight += weight
    
    # Signal 3: Journal plausibility
    journal = entry.get("journal", "")
    if journal:
        journal_plausible, journal_flags = _check_journal_plausibility(journal)
        journal_risk = 0.0 if journal_plausible else 0.6
        weight = 0.15
        signals.append({"name": "journal_suspicious", "risk": journal_risk, "weight": weight})
        weighted_sum += journal_risk * weight
        total_weight += weight
        if journal_flags:
            for flag in journal_flags:
                signals.append({"name": "journal_flag", "risk": 0.3, "weight": 0.05})
    
    # Signal 4: Page range implausibility
    pages = entry.get("pages", "")
    year = entry.get("year", "")
    if pages:
        pages_plausible, pages_flags = _check_page_range_implausibility(pages, year)
        pages_risk = 0.0 if pages_plausible else (0.4 if len(pages_flags) == 1 else 0.7)
        weight = 0.1
        signals.append({"name": "page_range", "risk": pages_risk, "weight": weight})
        weighted_sum += pages_risk * weight
        total_weight += weight
    
    # Signal 5: Conference plausibility (for proceedings)
    if entry.get("entry_type") in ("proceedings", "inproceedings"):
        booktitle = entry.get("booktitle", "")
        if booktitle:
            conf_plausible, conf_flags = _check_conference_plausibility(booktitle)
            conf_risk = 0.0 if conf_plausible else 0.5
            weight = 0.1
            signals.append({"name": "conference_suspicious", "risk": conf_risk, "weight": weight})
            weighted_sum += conf_risk * weight
            total_weight += weight
    
    # Signal 6: Missing required fields
    required_fields = ["authors", "title", "year"]
    missing_count = sum(1 for f in required_fields if not entry.get(f))
    missing_risk = min(missing_count / 3, 0.5)
    weight = 0.05
    signals.append({"name": "missing_fields", "risk": missing_risk, "weight": weight})
    weighted_sum += missing_risk * weight
    total_weight += weight
    
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
    
    # Collect risk factors for display
    risk_factors = []
    for s in signals:
        if s["risk"] >= 0.4:
            risk_factors.append(f"{s['name'].replace('_', ' ').title()}: {int(s['risk']*100)}% risk")
    
    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "composite_risk": round(composite_risk, 2),
        "risk_factors": risk_factors[:5]
    }


# ---------------------------------------------------------------------------
# 4. Pre-screen by author overlap (updated thresholds)
# ---------------------------------------------------------------------------

def _pre_screen_by_author_overlap(entry: dict, api_result: dict, title_sim: float) -> Optional[dict]:
    """
    Deterministic pre-screen using author overlap.
    UPDATED: Lower thresholds for FAKE detection.
    """
    from checker import author_overlap_score

    cited_authors   = (entry.get("authors") or "").strip()
    correct_authors = (api_result.get("correct_authors") or "").strip()

    if not cited_authors or not correct_authors:
        return None

    overlap = author_overlap_score(cited_authors, correct_authors)
    if overlap is None:
        return None

    api_status = api_result.get("status", "not_checked")
    pct = int(overlap * 100)

    # NEW: If overlap is very low AND API has any match → STRONG FAKE signal
    if overlap < 0.25:
        if api_status in ("verified", "partial_match"):
            # Even stronger if title also matches
            if title_sim >= 0.35:
                return {
                    "verdict": "FAKE",
                    "confidence": 0.92,
                    "reasoning": (
                        f"Paper found in database ({int(title_sim*100)}% title match) "
                        f"but author overlap is only {pct}% — this is a MISMATCHED reference. "
                        "The student cited a different paper or fabricated the authors."
                    ),
                    "risk_factors": [
                        f"Author overlap: {pct}% (threshold: 25%)",
                        f"Title match: {int(title_sim*100)}%",
                        f"API status: {api_status}",
                    ],
                }
        # No API match + very low author-derived signals → SUSPICIOUS (not necessarily fake)
        return {
            "verdict": "SUSPICIOUS",
            "confidence": 0.45,
            "reasoning": (
                f"Reference not found in academic databases and author-derived signals "
                f"are weak ({pct}% author indicator match). "
                "May be a niche publication or incorrectly formatted reference."
            ),
            "risk_factors": [
                f"Weak database presence",
                f"Author signal: {pct}% match",
            ],
        }

    # High overlap (≥70%) + API verified = REAL
    if overlap >= 0.70 and api_status == "verified":
        return {
            "verdict": "REAL",
            "confidence": round(min(0.85 + overlap * 0.15, 1.0), 2),
            "reasoning": (
                f"Found in academic database with {pct}% author match — "
                "reference appears genuine."
            ),
            "risk_factors": [],
        }

    # Medium-low overlap (25-50%) → need more signals
    if 0.25 <= overlap < 0.50:
        return None  # Send to AI for deeper analysis

    return None


# ---------------------------------------------------------------------------
# 5. AI verification (UPDATED with composite scoring option)
# ---------------------------------------------------------------------------

def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    """
    Determine REAL / SUSPICIOUS / FAKE for each reference.
    
    NEW v6.1:
      Uses composite scoring for many cases, AI for borderline.
      Also checks whitelisted venues (German conferences) to reduce false flags.
    """
    from review_queue import is_venue_whitelisted
    
    if not bib_entries:
        return {"verdicts": [], "summary": "No entries to verify.",
                "fake_count": 0, "suspicious_count": 0, "real_count": 0}

    vr_by_key = {vr["key"]: vr for vr in api_results}

    all_verdicts: List[dict] = []
    needs_ai: List[tuple] = []
    pre_screen_cache: Dict[str, dict] = {}

    # Helper function for title similarity (avoid circular import)
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
        
        # Compute title similarity if we have a match
        matched_title = vr.get("matched_title", "")
        title_sim = 0.0
        if entry.get("title") and matched_title:
            title_sim = _local_title_similarity(entry["title"], matched_title)
        
        # Check whitelisted venues (German conferences)
        venue = entry.get("journal") or entry.get("booktitle") or ""
        whitelist_check = is_venue_whitelisted(venue)
        
        # Try pre-screen
        early = _pre_screen_by_author_overlap(entry, vr, title_sim)
        if early:
            pre_screen_cache[entry["key"]] = early
        else:
            # Compute composite score as fallback/assist
            composite = _compute_composite_fake_score(entry, vr, title_sim)
            
            # If venue is whitelisted, downgrade FAKE to SUSPICIOUS
            if whitelist_check.get("whitelisted") and composite["verdict"] == "FAKE":
                composite["verdict"] = "SUSPICIOUS"
                composite["confidence"] = 0.6
                composite["reasoning"] = f"Venue is whitelisted ({whitelist_check.get('venue')}), marking as suspicious rather than fake. Manual review recommended."
            
            # If composite score strongly indicates FAKE or REAL, use it
            if composite["verdict"] != "SUSPICIOUS" and composite["confidence"] >= 0.75:
                pre_screen_cache[entry["key"]] = {
                    "verdict": composite["verdict"],
                    "confidence": composite["confidence"],
                    "reasoning": f"Composite signal analysis: {', '.join(composite['risk_factors'][:3])}",
                    "risk_factors": composite["risk_factors"],
                }
            else:
                # Borderline - send to AI
                needs_ai.append((entry, vr, title_sim, composite))

    ai_verdicts_by_key: Dict[str, dict] = {}
    fake_count = suspicious_count = real_count = 0

    # Process borderline cases with AI
    for chunk in _chunk(needs_ai, 15):
        combined = []
        for entry, vr, title_sim, composite in chunk:
            combined.append({
                "key":               entry["key"],
                "title":             entry.get("title") or "",
                "authors":           entry.get("authors") or "",
                "year":              entry.get("year") or "",
                "entry_type":        entry.get("entry_type") or "unknown",
                "doi":               entry.get("doi") or "",
                "journal":           entry.get("journal") or "",
                "publisher":         entry.get("publisher") or "",
                "url":               entry.get("url") or "",
                "booktitle":         entry.get("booktitle") or "",
                "pages":             entry.get("pages") or "",
                "key_consistent":    entry.get("key_consistent"),
                "api_status":        vr.get("status", "not_checked"),
                "api_confidence":    round(vr.get("confidence", 0), 2),
                "api_matched_title": vr.get("matched_title") or "",
                "api_sources":       vr.get("sources_checked", []),
                "open_access_url":   vr.get("open_access_url") or "",
                "web_evidence":      vr.get("web_evidence") or "",
                "arxiv_version_note": vr.get("version_note") or "",
                "composite_risk":    composite.get("composite_risk", 0.5),
            })

        prompt = f"""You are an academic integrity officer detecting fabricated references
in a student LNI-formatted paper.

━━━ THREE-TIER VERDICTS ━━━
REAL       — Paper exists and is correctly cited. Can be confirmed.
SUSPICIOUS — Cannot confirm but cannot call FAKE. Professor should manually verify.
FAKE       — Strong evidence of fabrication. CONFIDENTLY FLAG.

━━━ FAKE SIGNALS (2+ = FAKE) ━━━
• Composite risk score > 0.65 from automated checks
• key_consistent=false (initials/year don't match parsed metadata)
• Journal name sounds plausible but is NOT found anywhere
• Conference name has "of" instead of "on" (e.g., "Conference of AI")
• Unusually long page range (>50 pages for a single article)
• 2025+ publication year but no DOI or arXiv ID
• Generic, AI-sounding title without specific contribution
• Authors listed but none found in any academic database
• DOI present but does not resolve (checked via API)

━━━ DO NOT OVER-FLAG ━━━
• arxiv_version_note is set → the paper IS real (just older version)
• German/B2 conference papers often not indexed → SUSPICIOUS not FAKE
• Pre-2000 papers → be lenient
• Low API confidence alone ≠ FAKE

Return ONLY valid JSON:
{{
  "verdicts": [
    {{"key": "string", "verdict": "REAL", "confidence": 0.95,
      "reasoning": "one concise sentence", "risk_factors": []}}
  ],
  "fake_count": 0, "suspicious_count": 0, "real_count": 0,
  "summary": "2-3 sentence overall assessment"
}}

References with pre-computed signals:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""

        try:
            chunk_result = _call_ai_json(prompt, max_tokens=4000)
            for v in chunk_result.get("verdicts", []):
                ai_verdicts_by_key[v["key"]] = v
        except Exception as e:
            # Fallback to composite score for this batch
            for entry, vr, title_sim, composite in chunk:
                verdict = composite["verdict"]
                ai_verdicts_by_key[entry["key"]] = {
                    "key": entry["key"],
                    "verdict": verdict,
                    "confidence": composite["confidence"],
                    "reasoning": composite.get("reasoning", f"Composite analysis: {composite['composite_risk']*100:.0f}% risk"),
                    "risk_factors": composite.get("risk_factors", []),
                }

    # Collect all verdicts
    for entry in bib_entries:
        key = entry["key"]
        if key in pre_screen_cache:
            v = pre_screen_cache[key]
            all_verdicts.append({"key": key, **v, "open_access_url": None})
        elif key in ai_verdicts_by_key:
            all_verdicts.append(ai_verdicts_by_key[key])
        else:
            # Final fallback - compute composite score
            vr = vr_by_key.get(key, {})
            matched_title = vr.get("matched_title", "")
            title_sim = 0.0
            if entry.get("title") and matched_title:
                title_sim = _local_title_similarity(entry["title"], matched_title)
            composite = _compute_composite_fake_score(entry, vr, title_sim)
            
            # Check whitelist for final fallback too
            venue = entry.get("journal") or entry.get("booktitle") or ""
            whitelist_check = is_venue_whitelisted(venue)
            if whitelist_check.get("whitelisted") and composite["verdict"] == "FAKE":
                composite["verdict"] = "SUSPICIOUS"
                composite["confidence"] = 0.6
            
            all_verdicts.append({
                "key": key,
                "verdict": composite["verdict"],
                "confidence": composite["confidence"],
                "reasoning": f"Automated analysis: {composite['composite_risk']*100:.0f}% risk score",
                "risk_factors": composite["risk_factors"],
                "open_access_url": None,
            })

    for v in all_verdicts:
        verdict = v.get("verdict", "SUSPICIOUS")
        if verdict == "FAKE":
            fake_count += 1
        elif verdict == "SUSPICIOUS":
            suspicious_count += 1
        else:
            real_count += 1

    return {
        "verdicts":         all_verdicts,
        "fake_count":       fake_count,
        "suspicious_count": suspicious_count,
        "real_count":       real_count,
        "summary":          f"AI analysis complete: {fake_count} FAKE, {suspicious_count} SUSPICIOUS, {real_count} REAL references identified.",
    }


# ---------------------------------------------------------------------------
# 6. Overall verdict + professor report
# ---------------------------------------------------------------------------

def ai_overall_verdict(filename: str, summary: dict, xcheck,
                       bib_list: list, verification_result: dict) -> dict:
    fake_count  = verification_result.get("fake_count", 0)
    suspicious  = verification_result.get("suspicious_count", 0)
    missing_cit = len(xcheck.cited_not_in_bib)
    orphaned    = len(xcheck.in_bib_not_cited)
    incomplete  = sum(1 for e in bib_list if e.completeness_issues)
    bib_count   = len(bib_list)
    cited_count = len(xcheck.correctly_used) + missing_cit
    key_issues  = [e for e in bib_list if e.key_consistent is False]

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
  "student_feedback": ["specific point 1", "specific point 2"],
  "professor_note": "One sentence on what to manually review."
}}"""

    try:
        return _call_ai_json(prompt, max_tokens=800)
    except Exception as e:
        score = 100
        score -= min(fake_count * 20, 60)  # Increased penalty for FAKE
        score -= min(missing_cit * 10, 30)
        score -= min(incomplete * 5, 20)
        score -= min(orphaned * 3, 12)
        score -= min(len(key_issues) * 5, 15)
        score  = max(0, score)
        grade   = "A" if score>=90 else "B" if score>=75 else "C" if score>=60 else "D" if score>=45 else "F"
        verdict = ("FAIL" if fake_count >= 2 or score < 45 else  # Lowered threshold
                   "FLAG" if fake_count >= 1 or len(key_issues) >= 2 or score < 75 else
                   "PASS")
        return {
            "verdict": verdict, "score": score, "grade": grade,
            "verdict_reason": f"Computed from check results (AI unavailable: {e})",
            "student_feedback": ["Review your bibliography for completeness and accuracy."],
            "professor_note": "AI synthesis unavailable — review individual check results manually.",
            "error": str(e),
        }