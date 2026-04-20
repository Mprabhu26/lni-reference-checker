"""
AI Checker — v5
---------------
Groq (llama-3.3-70b) primary → Gemini 1.5 Flash fallback.
Both FREE tier:
  GROQ_API_KEY   → console.groq.com      (free, 14 400 req/day)
  GEMINI_API_KEY → aistudio.google.com   (free, 1 500 req/day)

WHAT USES AI:
  1. ai_extract_references_from_text()  — LLM-based bibliography extraction
       Parses ANY reference format (numeric [1], author-year, BibLaTeX, raw text)
       into structured dicts. Runs BEFORE parser.py regex for best coverage.
       Falls back silently to regex extraction if AI is unavailable.  ← NEW v5

  2. ai_parse_uncertain_entries()       — re-parses entries flagged needs_ai_parsing=True
       (regex extraction was uncertain). Runs BEFORE API lookups.

  3. ai_verify_references()             — THREE-TIER verdict per reference:
       REAL / SUSPICIOUS / FAKE
       Now includes author overlap pre-filter (from checker.author_overlap_score):
         - overlap < 0.30 → FAKE  (no AI token spent)         ← NEW v5
         - overlap >= 0.70 → REAL (no AI token spent)         ← NEW v5
         - otherwise → send to AI
       Also includes key_consistent signal from parser.

  4. ai_overall_verdict()               — PASS / FLAG / FAIL + student feedback

WHAT DOES NOT USE AI (100% deterministic):
  - LNI key format validation
  - Key-vs-metadata consistency (initials + year)
  - Required field presence per entry type
  - In-text citation extraction
  - Cross-check cited vs listed
  - Duplicate detection
  - Page range / author order / LNI style checks
  - Author overlap computation
  - Score computation
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional

GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL  = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-1.5-flash:generateContent")


# ---------------------------------------------------------------------------
# Core AI call helpers
# ---------------------------------------------------------------------------

def _call_ai(prompt: str, max_tokens: int = 2000, system: str = "") -> str:
    """Call Groq first, fall back to Gemini. Raises RuntimeError if both unavailable."""
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

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
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if gemini_key:
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
                return "".join(p.get("text", "") for p in parts).strip()
        except Exception:
            pass

    missing = [k for k, v in [("GROQ_API_KEY", groq_key), ("GEMINI_API_KEY", gemini_key)] if not v]
    raise RuntimeError(
        f"No AI API key configured. Set {' or '.join(missing)}. "
        "Groq: console.groq.com (free) | Gemini: aistudio.google.com (free)"
    )


def _call_ai_json(prompt: str, max_tokens: int = 2000, system: str = "") -> dict:
    text = _call_ai(prompt, max_tokens, system).strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return json.loads(text.strip())


def _chunk(lst: list, size: int) -> list:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _ai_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# 1. LLM-based reference extraction  (NEW v5)
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
    """
    Use the LLM to extract structured reference records from raw bibliography text.

    This handles ANY citation format (numbered, author-year, BibLaTeX, mixed),
    including layouts that defeat the regex parser (missing punctuation, multi-line
    entries without consistent delimiters, scanned PDFs with OCR artefacts).

    Returns a list of dicts. Each dict has at minimum:
        raw, title, authors, year
    and optionally: journal, booktitle, publisher, pages, doi, url, isbn.

    Falls back to empty list silently when AI is unavailable — the caller
    then uses the pure-regex parser result instead.
    """
    if not bib_text or not bib_text.strip():
        return []

    if not _ai_available():
        return []

    # Chunk large bibliographies (> 6 000 chars) into ~3 000-char pieces
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
            pass  # fall through; caller uses regex result

    return all_refs


def merge_ai_extractions_into_bib_list(ai_refs: List[Dict], bib_list: list) -> list:
    """
    Merge AI-extracted metadata into the existing regex-parsed bib_list.
    For each entry in bib_list, if there is a close AI match (by raw_text
    substring or title similarity), fill in any fields that the regex left None.

    This is non-destructive: existing non-None values are kept.
    """
    if not ai_refs or not bib_list:
        return bib_list

    def _norm(t: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (t or "").lower())).strip()

    # Index AI refs by normalised title
    ai_by_title: Dict[str, dict] = {}
    for ar in ai_refs:
        t = _norm(ar.get("title", ""))
        if t:
            ai_by_title[t] = ar

    for entry in bib_list:
        # Try to find a matching AI ref
        entry_title_norm = _norm(entry.title or "")
        ai = ai_by_title.get(entry_title_norm)

        if not ai:
            # Fuzzy match by raw_text
            for ar in ai_refs:
                raw = (ar.get("raw") or "")[:200]
                if entry.raw_text[:80].lower() in raw.lower() or raw[:80].lower() in entry.raw_text.lower():
                    ai = ar
                    break

        if not ai:
            continue

        # Merge: only fill fields that are currently None or empty
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
            # Infer type from AI fields
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
    """
    For entries flagged needs_ai_parsing=True, ask the AI to extract
    structured metadata from raw bibliography text.
    Runs BEFORE the API lookup loop so improved titles/authors flow into
    CrossRef, DBLP, Semantic Scholar etc. for better matching.
    Returns dict keyed by citation key with improved fields.
    """
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
# 3. Author overlap pre-filter  (new in v5 — runs before AI call)
# ---------------------------------------------------------------------------

def _pre_screen_by_author_overlap(entry: dict, api_result: dict) -> Optional[dict]:
    """
    Deterministic pre-screen using author overlap. Returns an early verdict
    dict {verdict, confidence, reasoning, risk_factors} or None (= send to AI).

    Rules:
      overlap < 0.30  → FAKE   (very few authors match — strong signal)
      overlap >= 0.70 → REAL   (most authors match — safe to skip AI)
      None or between → None   (insufficient data, let AI decide)
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

    if overlap < 0.30 and api_status in ("verified", "partial_match"):
        # Paper was found in a database, but cited authors barely match → hallucination
        pct = int(overlap * 100)
        return {
            "verdict":     "FAKE",
            "confidence":  round(1.0 - overlap, 2),
            "reasoning":   (
                f"Paper title found in academic database but only {pct}% of cited authors "
                "match the actual authors — strong sign of a fabricated or misattributed reference."
            ),
            "risk_factors": [
                f"Author overlap: {pct}% (threshold: 30%)",
                f"API status: {api_status}",
            ],
        }

    if overlap >= 0.70 and api_status == "verified":
        return {
            "verdict":     "REAL",
            "confidence":  round(min(0.8 + overlap * 0.2, 1.0), 2),
            "reasoning":   (
                f"Found in academic database with {int(overlap*100)}% author match — "
                "reference appears genuine."
            ),
            "risk_factors": [],
        }

    return None  # inconclusive — let AI decide


# ---------------------------------------------------------------------------
# 4. Reference authenticity / fake detection  (THREE-TIER verdict)
# ---------------------------------------------------------------------------

def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    """
    Determine REAL / SUSPICIOUS / FAKE for each reference.

    Pipeline per entry:
      1. Author overlap pre-filter (deterministic, free)  ← NEW v5
      2. If inconclusive → send to AI (batched, max 20/call)

    Three-tier verdict:
      REAL       — reference exists; high API confidence or strong author match
      SUSPICIOUS — cannot confirm but not enough evidence for FAKE
                   (old German workshop, book chapter, grey literature)
      FAKE       — strong evidence of fabrication (see FAKE SIGNALS in prompt)
    """
    if not bib_entries:
        return {"verdicts": [], "summary": "No entries to verify.",
                "fake_count": 0, "suspicious_count": 0, "real_count": 0}

    vr_by_key = {vr["key"]: vr for vr in api_results}

    all_verdicts:     List[dict] = []
    needs_ai:         List[dict] = []  # (entry, vr) pairs that need AI
    pre_screen_cache: Dict[str, dict] = {}  # key → early_verdict

    # ── Author overlap pre-screen ──────────────────────────────────────────
    for entry in bib_entries:
        vr = vr_by_key.get(entry["key"], {})
        early = _pre_screen_by_author_overlap(entry, vr)
        if early:
            pre_screen_cache[entry["key"]] = early
        else:
            needs_ai.append((entry, vr))

    # ── AI verification for remaining entries ──────────────────────────────
    ai_verdicts_by_key: Dict[str, dict] = {}
    fake_count = suspicious_count = real_count = 0
    summaries: List[str] = []

    for chunk in _chunk(needs_ai, 20):
        combined = []
        for entry, vr in chunk:
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
                "key_consistent":    entry.get("key_consistent"),
                "api_status":        vr.get("status", "not_checked"),
                "api_confidence":    round(vr.get("confidence", 0), 2),
                "api_matched_title": vr.get("matched_title") or "",
                "api_sources":       vr.get("sources_checked", []),
                "api_doi_found":     vr.get("doi") or "",
                "open_access_url":   vr.get("open_access_url") or "",
                "web_evidence":      vr.get("web_evidence") or "",
                "api_note":          vr.get("note") or "",
            })

        prompt = f"""You are an academic integrity officer detecting fabricated references
in a student LNI-formatted paper.

━━━ THREE-TIER VERDICTS ━━━
REAL       — Paper exists. High API confidence, matched title, or well-known publication.
             Low API confidence can still be REAL if mismatch is formatting/language/niche.
SUSPICIOUS — Cannot confirm but cannot call FAKE. Professor should manually verify.
             (old German workshop, book chapter, grey literature, pre-2000 papers)
FAKE       — Strong fabrication evidence — requires 2+ FAKE signals below.

━━━ FAKE SIGNALS (need 2 or more) ━━━
• key_consistent=false: author initials or year in [Key] don't match parsed metadata
• Recent (post-2010) English CS paper not found anywhere despite 6+ sources searched
• Journal/conference name sounds plausible but no evidence it actually exists
• Year is in the future
• Page span >200 pages for a single article
• Overly generic AI-sounding title: "A Comprehensive Survey of Deep Learning for X"
• DOI present but resolves to a completely different paper
• Author+year+venue combination not found even by web search

━━━ DO NOT OVER-FLAG ━━━
• German workshop/book chapter papers often not indexed — SUSPICIOUS not FAKE
• Websites are verified by URL only (not academic DBs) — don't flag missing DB entry
• Pre-2000 papers often not indexed — be lenient
• Low API confidence alone ≠ FAKE
• key_consistent=null means the check couldn't run — do not penalise

Focus ONLY on whether the publication EXISTS. Do NOT re-check missing fields
(title, year, pages) — that is already done deterministically.

Return ONLY valid JSON, no markdown:
{{
  "verdicts": [
    {{"key": "string", "verdict": "REAL", "confidence": 0.95,
      "reasoning": "one concise sentence", "risk_factors": [],
      "open_access_url": null}}
  ],
  "fake_count": 0, "suspicious_count": 0, "real_count": 0,
  "summary": "2-3 sentence overall assessment"
}}

References:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""

        try:
            chunk_result = _call_ai_json(prompt, max_tokens=4000)
            for v in chunk_result.get("verdicts", []):
                ai_verdicts_by_key[v["key"]] = v
            s = chunk_result.get("summary", "")
            if s:
                summaries.append(s)
        except Exception as e:
            # Fallback: map API status directly
            for entry, vr in chunk:
                status  = vr.get("status", "not_checked")
                verdict = "REAL" if status == "verified" else "SUSPICIOUS"
                ai_verdicts_by_key[entry["key"]] = {
                    "key": entry["key"], "verdict": verdict,
                    "confidence": round(vr.get("confidence", 0.5), 2),
                    "reasoning": f"AI unavailable — API result: {status}. {vr.get('note','')}",
                    "risk_factors": [], "open_access_url": vr.get("open_access_url"),
                }

    # ── Combine pre-screen and AI verdicts ─────────────────────────────────
    for entry in bib_entries:
        key = entry["key"]
        if key in pre_screen_cache:
            v = pre_screen_cache[key]
            all_verdicts.append({"key": key, **v, "open_access_url": None})
        elif key in ai_verdicts_by_key:
            all_verdicts.append(ai_verdicts_by_key[key])
        else:
            all_verdicts.append({"key": key, "verdict": "SUSPICIOUS", "confidence": 0.5,
                                  "reasoning": "Could not be verified", "risk_factors": [],
                                  "open_access_url": None})

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
        "summary":          " ".join(summaries) or "No AI integrity summary available.",
    }


# ---------------------------------------------------------------------------
# 5. Overall verdict + professor report
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
- FAKE references (AI+author-overlap verdict): {fake_count}
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

AI integrity summary: {verification_result.get('summary', '')}

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
        score -= min(fake_count * 15, 45)
        score -= min(missing_cit * 10, 30)
        score -= min(incomplete * 5, 20)
        score -= min(orphaned * 3, 12)
        score -= min(len(key_issues) * 5, 15)
        score  = max(0, score)
        grade   = "A" if score>=90 else "B" if score>=75 else "C" if score>=60 else "D" if score>=45 else "F"
        verdict = ("FAIL" if fake_count >= 3 or score < 45 else
                   "FLAG" if fake_count >= 1 or len(key_issues) >= 2 or score < 75 else
                   "PASS")
        return {
            "verdict": verdict, "score": score, "grade": grade,
            "verdict_reason": f"Computed from check results (AI unavailable: {e})",
            "student_feedback": ["Review your bibliography for completeness and accuracy."],
            "professor_note": "AI synthesis unavailable — review individual check results manually.",
            "error": str(e),
        }