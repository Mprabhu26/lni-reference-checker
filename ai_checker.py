"""
AI Checker — Groq + Gemini (fake detection & overall verdict only)
------------------------------------------------------------------
Uses Groq (llama-3.3-70b) as primary, Gemini 1.5 Flash as fallback.
Both are FREE tier — set via environment variables:
    GROQ_API_KEY   -> console.groq.com      (free, 14 400 req/day)
    GEMINI_API_KEY -> aistudio.google.com   (free, 1 500 req/day)

WHAT USES AI (judgment required — cannot be 100% correct with pure code):
  1. ai_verify_references()  — REAL / SUSPICIOUS / FAKE per reference
     (combines parsed metadata + multi-source API lookup evidence)
  2. ai_overall_verdict()    — PASS / FLAG / FAIL + student feedback

WHAT DOES NOT USE AI (100% deterministic, handled in parser.py / checker.py):
  - LNI citation key format validation     (strict regex rule)
  - Required field presence per entry type (lookup table)
  - In-text citation extraction            (regex)
  - Cross-check cited vs listed            (set difference)
  - Duplicate detection                    (rapidfuzz similarity)
  - Page range format check                (regex)
  - LNI macro style check                  (regex)
  - Score computation                      (arithmetic)
"""

import os
import json
import requests

GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL  = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-1.5-flash:generateContent")


def _call_ai(prompt: str, max_tokens: int = 2000) -> str:
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if groq_key:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
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
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
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
        f"No AI API key configured. Set {' or '.join(missing)} as environment variables. "
        "Groq: free key at console.groq.com | Gemini: free key at aistudio.google.com"
    )


def _call_ai_json(prompt: str, max_tokens: int = 2000) -> dict:
    text = _call_ai(prompt, max_tokens).strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return json.loads(text.strip())


# ── 1. Reference authenticity / fake detection ────────────────────────────────

def ai_verify_references(bib_entries: list, api_results: list) -> dict:
    """
    AI decides REAL / SUSPICIOUS / FAKE for each reference by combining
    parsed metadata with evidence from external API lookups.
    """
    if not bib_entries:
        return {"verdicts": [], "summary": "No entries to verify.",
                "fake_count": 0, "suspicious_count": 0, "real_count": 0}

    vr_by_key = {vr["key"]: vr for vr in api_results}

    combined = []
    for entry in bib_entries:
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

For each reference: parsed metadata + results from CrossRef, Semantic Scholar, OpenAlex, arXiv, and web search.

VERDICTS:
- REAL: Reference exists. High API confidence, matched title, or well-known publication. Low confidence can still be REAL if mismatch is due to title variation, language, or old/niche paper.
- SUSPICIOUS: Cannot confirm but not enough evidence to call fake. Plausible but absent from all DBs (old German workshop, book chapter, grey literature). Manual verification needed.
- FAKE: Strong evidence of fabrication — found nowhere, implausible metadata, AI-hallucination patterns.

FAKE SIGNALS:
- Title nowhere in databases despite being recent English-language CS work
- Journal/conference sounds plausible but doesn't exist
- Year in the future
- Page numbers impossible for a single article (e.g. pp. 1-500)
- Overly generic AI-sounding title ("A Survey of Machine Learning Methods")
- DOI present but doesn't match found record

DO NOT over-flag:
- LNI papers often cite German/small workshop papers not in CrossRef — these can be REAL
- Websites don't appear in academic DBs — check only URL status
- Pre-2000 papers often not indexed — be lenient
- Low API confidence alone is NOT enough for FAKE

Return ONLY valid JSON, no markdown:
{{
  "verdicts": [
    {{
      "key": "string",
      "verdict": "REAL",
      "confidence": 0.95,
      "reasoning": "one concise sentence",
      "risk_factors": [],
      "open_access_url": null
    }}
  ],
  "fake_count": 0,
  "suspicious_count": 0,
  "real_count": 0,
  "summary": "2-3 sentence overall assessment"
}}

References:
{json.dumps(combined, ensure_ascii=False, indent=2)}"""

    try:
        return _call_ai_json(prompt, max_tokens=4000)
    except Exception as e:
        verdicts = []
        for entry in bib_entries:
            vr = vr_by_key.get(entry["key"], {})
            status = vr.get("status", "not_checked")
            verdict = "REAL" if status == "verified" else "SUSPICIOUS"
            verdicts.append({
                "key": entry["key"],
                "verdict": verdict,
                "confidence": round(vr.get("confidence", 0.5), 2),
                "reasoning": f"AI unavailable — API result: {status}. {vr.get('note', '')}",
                "risk_factors": [],
                "open_access_url": vr.get("open_access_url"),
            })
        return {
            "verdicts": verdicts,
            "fake_count": 0,
            "suspicious_count": sum(1 for v in verdicts if v["verdict"] == "SUSPICIOUS"),
            "real_count": sum(1 for v in verdicts if v["verdict"] == "REAL"),
            "summary": f"AI unavailable ({e}). Showing raw API results only.",
            "error": str(e),
        }


# ── 2. Overall verdict + professor report ─────────────────────────────────────

def ai_overall_verdict(filename: str, summary: dict, xcheck,
                       bib_list: list, verification_result: dict) -> dict:
    """
    AI synthesises all deterministic check results into a final verdict.
    """
    fake_count  = verification_result.get("fake_count", 0)
    suspicious  = verification_result.get("suspicious_count", 0)
    missing_cit = len(xcheck.cited_not_in_bib)
    orphaned    = len(xcheck.in_bib_not_cited)
    incomplete  = sum(1 for e in bib_list if e.completeness_issues)
    bib_count   = len(bib_list)
    cited_count = len(xcheck.correctly_used) + missing_cit

    incomplete_details = [
        f"[{e.key}]: {'; '.join(e.completeness_issues[:2])}"
        for e in bib_list if e.completeness_issues
    ][:8]

    fake_details = [
        f"[{v['key']}] {v.get('reasoning','')}"
        for v in verification_result.get("verdicts", [])
        if v.get("verdict") == "FAKE"
    ]

    suspicious_details = [
        f"[{v['key']}] {v.get('reasoning','')}"
        for v in verification_result.get("verdicts", [])
        if v.get("verdict") == "SUSPICIOUS"
    ][:5]

    prompt = f"""You are a professor reviewing a student submission to a GI LNI (Lecture Notes in Informatics) conference.

Automated checks have run on "{filename}". Synthesise into a final assessment.

STATISTICS:
- Bibliography entries: {bib_count}
- In-text citations: {cited_count}
- Cited but missing from bibliography: {missing_cit}
- In bibliography but never cited: {orphaned}
- Incomplete entries (missing required fields): {incomplete}
- FAKE references (AI verdict): {fake_count}
- SUSPICIOUS references: {suspicious}
- Duplicates: {summary.get('duplicates', 0)}
- Self-citations flagged: {summary.get('self_citations', 0)}

INCOMPLETE ENTRIES (sample):
{chr(10).join(incomplete_details) or "None"}

FAKE REFERENCES:
{chr(10).join(fake_details) or "None"}

SUSPICIOUS (sample):
{chr(10).join(suspicious_details) or "None"}

Integrity summary: {verification_result.get('summary', '')}

VERDICT RULES:
- PASS: References appear legitimate, only minor issues
- FLAG: Significant issues needing professor spot-check
- FAIL: Multiple likely fake references OR critical structural failures

Return ONLY valid JSON, no markdown:
{{
  "verdict": "PASS",
  "score": 85,
  "grade": "B",
  "verdict_reason": "1-2 sentences",
  "student_feedback": ["specific point 1", "specific point 2"],
  "professor_note": "One sentence on what needs manual review."
}}"""

    try:
        return _call_ai_json(prompt, max_tokens=800)
    except Exception as e:
        score = 100
        score -= min(fake_count * 15, 45)
        score -= min(missing_cit * 10, 30)
        score -= min(incomplete * 5, 20)
        score -= min(orphaned * 3, 12)
        score = max(0, score)
        grade   = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 45 else "F"
        verdict = "FAIL" if fake_count >= 3 or score < 45 else "FLAG" if fake_count >= 1 or score < 75 else "PASS"
        return {
            "verdict":          verdict,
            "score":            score,
            "grade":            grade,
            "verdict_reason":   f"Computed from check results (AI unavailable: {e})",
            "student_feedback": ["Review your bibliography for completeness and accuracy."],
            "professor_note":   "AI synthesis unavailable — review individual check results manually.",
            "error":            str(e),
        }
