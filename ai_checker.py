"""
AI Checker — Groq + Gemini (structured parsing, fake detection, overall verdict)
----------------------------------------------------------------------------------
Uses Groq (llama-3.3-70b) as primary, Gemini 1.5 Flash as fallback.
Both are FREE tier — set via environment variables:
    GROQ_API_KEY   -> console.groq.com      (free, 14 400 req/day)
    GEMINI_API_KEY -> aistudio.google.com   (free, 1 500 req/day)

WHAT USES AI (judgment / knowledge required):
  1. ai_parse_uncertain_entries() — re-parses entries flagged needs_ai_parsing=True
     to extract title/author/type when the regex failed. Runs BEFORE verification
     so better metadata flows into the API lookups.
  2. ai_verify_references()       — REAL / SUSPICIOUS / FAKE per reference.
     Now receives key_consistent flag so the AI has a deterministic signal.
  3. ai_overall_verdict()         — PASS / FLAG / FAIL + student feedback.

WHAT DOES NOT USE AI (100% deterministic, handled in parser.py / checker.py):
  - LNI citation key format validation
  - Key-vs-metadata consistency (initials + year)
  - Required field presence per entry type
  - In-text citation extraction
  - Cross-check cited vs listed
  - Duplicate detection (rapidfuzz)
  - Page range format / author order / LNI macro style
  - Score computation

FIXES vs v3:
  - New ai_parse_uncertain_entries(): batch-calls the AI for entries where the
    regex parser couldn't confidently extract title/authors. This significantly
    improves the data quality passed to the API lookups and fake detector.
  - ai_verify_references() prompt now includes key_consistent signal, author/year
    cross-check instruction, and clearer FAKE criteria for AI-hallucination patterns.
  - Prompts strip completeness-field checks (already done deterministically).
  - Batch size safety: large bibliographies split into chunks of 20 for AI calls.
"""

import os
import json
import requests

GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL  = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)


# ---------------------------------------------------------------------------
# Core AI call helpers
# ---------------------------------------------------------------------------

def _call_ai(prompt: str, max_tokens: int = 2000) -> str:
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if groq_key:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      GROQ_MODEL,
                    "messages":   [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if gemini_key:
        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.1,
                    },
                },
                timeout=30,
            )
            if resp.status_code == 200:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
        except Exception:
            pass

    missing = []
    if not groq_key:   missing.append("GROQ_API_KEY")
    if not gemini_key: missing.append("GEMINI_API_KEY")
    raise RuntimeError(
        f"No AI API key configured. Set {' or '.join(missing)} as environment "
        "variables. Groq: free key at console.groq.com | "
        "Gemini: free key at aistudio.google.com"
    )


def _call_ai_json(prompt: str, max_tokens: int = 2000) -> dict:
    text = _call_ai(prompt, max_tokens).strip()
    # Strip markdown fences if the model wrapped the JSON
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return json.loads(text.strip())


def _chunk(lst: list, size: int) -> list:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ---------------------------------------------------------------------------
# 1. AI-assisted structured parsing (new in v4)
# ---------------------------------------------------------------------------

def ai_parse_uncertain_entries(bib_entries_raw: list) -> dict:
    """
    For entries where the regex parser flagged needs_ai_parsing=True,
    ask the AI to extract structured metadata from the raw bibliography text.

    Returns a dict keyed by citation key with improved metadata fields.
    This runs BEFORE the API verification loop so the improved titles/authors
    flow into CrossRef, Semantic Scholar, etc. for better matching.

    bib_entries_raw: list of dicts with at least 'key', 'raw_text', and
    optionally partial 'title', 'authors', 'year', 'entry_type'.
    """
    uncertain = [e for e in bib_entries_raw if e.get("needs_ai_parsing")]
    if not uncertain:
        return {}

    improvements: dict = {}

    for chunk in _chunk(uncertain, 20):
        entries_for_prompt = [
            {
                "key":        e["key"],
                "raw_text":   e.get("raw_text", "")[:300],
                "regex_title":   e.get("title") or "",
                "regex_authors": e.get("authors") or "",
                "regex_type":    e.get("entry_type") or "unknown",
            }
            for e in chunk
        ]

        prompt = f"""You are a bibliography metadata extractor for LNI (Lecture Notes in Informatics) formatted reference lists.

The automated regex parser failed to confidently extract metadata for the following bibliography entries. Please extract the correct structured metadata from the raw text.

LNI FORMAT RULES:
- Author order: "Lastname, Firstname [; Lastname2, Firstname2]:" followed by title
- Entry types: book, article (journal), proceedings (conference/workshop), website, misc
- Books: Author: Title. Publisher, Year.
- Articles: Author: Title. Journal, Vol. X, No. Y, pp. A--B, Year.
- Proceedings: Author: Title. In: Booktitle (Eds.), pp. A--B. Publisher, Year.

For each entry, return your best extraction. If a field cannot be determined, use null.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "results": [
    {{
      "key": "string",
      "title": "string or null",
      "authors": "Lastname, Firstname [; Lastname2, Firstname2] or null",
      "year": "YYYY or null",
      "entry_type": "book|article|proceedings|website|misc|unknown",
      "journal": "string or null",
      "booktitle": "string or null",
      "publisher": "string or null",
      "pages": "string or null"
    }}
  ]
}}

Entries:
{json.dumps(entries_for_prompt, ensure_ascii=False, indent=2)}"""

        try:
            result = _call_ai_json(prompt, max_tokens=3000)
            for item in result.get("results", []):
                key = item.get("key")
                if key:
                    improvements[key] = item
        except Exception:
            # If AI parsing fails, we proceed with whatever the regex got.
            pass

    return improvements


# ---------------------------------------------------------------------------
# 2. Reference authenticity / fake detection
# ---------------------------------------------------------------------------

def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    """
    AI decides REAL / SUSPICIOUS / FAKE for each reference by combining
    parsed metadata (including the new key_consistent flag) with evidence
    from external API lookups.

    Key improvements vs v3:
    - key_consistent field now passed in and the prompt instructs AI to treat
      a key/metadata mismatch as a strong FAKE signal
    - Prompt no longer asks AI to re-check field completeness (done in code)
    - Clearer guidance on when NOT to flag (old German papers, niche venues)
    - Batch safety: split into chunks of 20 entries
    """
    if not bib_entries:
        return {
            "verdicts": [], "summary": "No entries to verify.",
            "fake_count": 0, "suspicious_count": 0, "real_count": 0,
        }

    vr_by_key = {vr["key"]: vr for vr in api_results}
    all_verdicts = []
    fake_count = suspicious_count = real_count = 0
    summaries = []

    for chunk in _chunk(bib_entries, 20):
        combined = []
        for entry in chunk:
            vr = vr_by_key.get(entry["key"], {})
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
                # key_consistent: True/False/None from deterministic check
                "key_consistent":    entry.get("key_consistent"),
                # API evidence
                "api_status":        vr.get("status", "not_checked"),
                "api_confidence":    round(vr.get("confidence", 0), 2),
                "api_matched_title": vr.get("matched_title") or "",
                "api_sources":       vr.get("sources_checked", []),
                "api_doi_found":     vr.get("doi") or "",
                "open_access_url":   vr.get("open_access_url") or "",
                "web_evidence":      vr.get("web_evidence") or "",
                "api_note":          vr.get("note") or "",
            })

        prompt = f"""You are an academic integrity officer helping a professor detect fabricated references in a student LNI-formatted paper.

For each reference you receive: parsed metadata + results from CrossRef, Semantic Scholar, OpenAlex, arXiv, Open Library, and web search.

━━━ VERDICTS ━━━
REAL       — Reference exists. High API confidence, matched title, or well-known publication.
             Low API confidence can still be REAL if the mismatch is due to title variation,
             language differences, or the paper is old/niche.
SUSPICIOUS — Cannot confirm but not enough evidence to call fake. Plausible but absent from
             all databases (old German workshop, book chapter, grey literature). The professor
             should manually verify.
FAKE       — Strong evidence of fabrication. Multiple FAKE signals present (see below).

━━━ FAKE SIGNALS (need 2+ signals to call FAKE) ━━━
• key_consistent=false: the author initials or year in [Key] don't match the metadata —
  this is a deterministic mismatch computed from the parsed data, not an API opinion
• Title appears nowhere in any database despite being a recent (post-2010) English-language
  CS or AI paper
• Journal/conference sounds plausible but no evidence it exists
• Year is in the future
• Page span >200 pages for a single article
• Overly generic AI-sounding title: "A Survey of Machine Learning Approaches for X"
• DOI present but doesn't resolve OR resolves to a completely different paper
• Author name + year + venue combination found nowhere, not even via web search

━━━ DO NOT OVER-FLAG ━━━
• LNI papers frequently cite German workshop proceedings not indexed in CrossRef — SUSPICIOUS, not FAKE
• Websites are verified by URL check only, not academic DBs
• Pre-2000 papers are often not indexed — be lenient unless other signals exist
• Low API confidence alone is NOT enough for FAKE
• key_consistent=null means the check couldn't run — do not penalise

━━━ YOUR TASK ━━━
Do NOT re-check required fields (title, year, pages) — that is already done by the system.
Focus ONLY on whether the paper/book/website actually EXISTS and is legitimate.
Reason about author+year+venue plausibility, cross-check them against each other.

Return ONLY valid JSON, no markdown:
{{
  "verdicts": [
    {{
      "key": "string",
      "verdict": "REAL",
      "confidence": 0.95,
      "reasoning": "one concise sentence explaining your decision",
      "risk_factors": [],
      "open_access_url": null
    }}
  ],
  "fake_count": 0,
  "suspicious_count": 0,
  "real_count": 0,
  "summary": "2–3 sentence overall assessment of this bibliography's integrity"
}}

References:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""

        try:
            chunk_result = _call_ai_json(prompt, max_tokens=4000)
            all_verdicts.extend(chunk_result.get("verdicts", []))
            fake_count      += chunk_result.get("fake_count", 0)
            suspicious_count += chunk_result.get("suspicious_count", 0)
            real_count      += chunk_result.get("real_count", 0)
            s = chunk_result.get("summary", "")
            if s:
                summaries.append(s)
        except Exception as e:
            # Fallback: map API status directly
            for entry in chunk:
                vr     = vr_by_key.get(entry["key"], {})
                status = vr.get("status", "not_checked")
                verdict = "REAL" if status == "verified" else "SUSPICIOUS"
                all_verdicts.append({
                    "key":            entry["key"],
                    "verdict":        verdict,
                    "confidence":     round(vr.get("confidence", 0.5), 2),
                    "reasoning":      f"AI unavailable — API result: {status}. {vr.get('note', '')}",
                    "risk_factors":   [],
                    "open_access_url": vr.get("open_access_url"),
                })
                if verdict == "SUSPICIOUS":
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
# 3. Overall verdict + professor report
# ---------------------------------------------------------------------------

def ai_overall_verdict(
    filename: str,
    summary: dict,
    xcheck,
    bib_list: list,
    verification_result: dict,
) -> dict:
    """
    AI synthesises all deterministic check results into a final verdict.
    Prompt is tightened: no repetition of field-check info already in summary.
    """
    fake_count  = verification_result.get("fake_count", 0)
    suspicious  = verification_result.get("suspicious_count", 0)
    missing_cit = len(xcheck.cited_not_in_bib)
    orphaned    = len(xcheck.in_bib_not_cited)
    incomplete  = sum(1 for e in bib_list if e.completeness_issues)
    bib_count   = len(bib_list)
    cited_count = len(xcheck.correctly_used) + missing_cit

    # Key-consistency failures (deterministic signal)
    key_inconsistent = [
        e for e in bib_list if e.key_consistent is False
    ]
    key_inconsistent_details = [
        f"[{e.key}]: " + "; ".join(
            i for i in e.completeness_issues if "key inconsistency" in i.lower()
        )
        for e in key_inconsistent
    ][:6]

    incomplete_details = [
        f"[{e.key}]: {'; '.join(e.completeness_issues[:2])}"
        for e in bib_list if e.completeness_issues
    ][:8]

    fake_details = [
        f"[{v['key']}] {v.get('reasoning', '')}"
        for v in verification_result.get("verdicts", [])
        if v.get("verdict") == "FAKE"
    ]

    suspicious_details = [
        f"[{v['key']}] {v.get('reasoning', '')}"
        for v in verification_result.get("verdicts", [])
        if v.get("verdict") == "SUSPICIOUS"
    ][:5]

    prompt = f"""You are a professor reviewing a student submission to a GI LNI (Lecture Notes in Informatics) conference.

Automated checks have run on "{filename}". Synthesise into a final assessment.

STATISTICS:
- Bibliography entries: {bib_count}
- In-text citations: {cited_count}
- Cited but missing from bibliography: {missing_cit}
- In bibliography but never cited (orphaned): {orphaned}
- Entries with missing required fields: {incomplete}
- LNI key-vs-metadata mismatches (deterministic): {len(key_inconsistent)}
- FAKE references (AI verdict): {fake_count}
- SUSPICIOUS references: {suspicious}
- Duplicates: {summary.get('duplicates', 0)}
- Self-citations flagged: {summary.get('self_citations', 0)}

KEY INCONSISTENCIES (author initials or year in key ≠ parsed metadata):
{chr(10).join(key_inconsistent_details) or "None"}

INCOMPLETE ENTRIES (sample):
{chr(10).join(incomplete_details) or "None"}

FAKE REFERENCES:
{chr(10).join(fake_details) or "None"}

SUSPICIOUS REFERENCES (sample):
{chr(10).join(suspicious_details) or "None"}

AI integrity summary: {verification_result.get("summary", "")}

VERDICT RULES:
- PASS: References appear legitimate; only minor formatting issues
- FLAG: Significant issues requiring professor spot-check (suspicious refs, key mismatches)
- FAIL: Multiple likely fake references OR critical structural failures

Return ONLY valid JSON, no markdown:
{{
  "verdict": "PASS",
  "score": 85,
  "grade": "B",
  "verdict_reason": "1–2 sentences explaining the verdict",
  "student_feedback": ["specific actionable point 1", "specific actionable point 2"],
  "professor_note": "One sentence on what the professor should manually review."
}}"""

    try:
        return _call_ai_json(prompt, max_tokens=800)
    except Exception as e:
        # Deterministic fallback
        score = 100
        score -= min(fake_count * 15, 45)
        score -= min(missing_cit * 10, 30)
        score -= min(incomplete * 5, 20)
        score -= min(orphaned * 3, 12)
        score -= min(len(key_inconsistent) * 5, 15)
        score  = max(0, score)
        grade   = (
            "A" if score >= 90 else
            "B" if score >= 75 else
            "C" if score >= 60 else
            "D" if score >= 45 else "F"
        )
        verdict = (
            "FAIL" if fake_count >= 3 or score < 45 else
            "FLAG" if fake_count >= 1 or len(key_inconsistent) >= 2 or score < 75 else
            "PASS"
        )
        return {
            "verdict":          verdict,
            "score":            score,
            "grade":            grade,
            "verdict_reason":   f"Computed from check results (AI unavailable: {e})",
            "student_feedback": ["Review your bibliography for completeness and accuracy."],
            "professor_note":   "AI synthesis unavailable — review individual check results manually.",
            "error":            str(e),
        }