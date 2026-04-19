"""
Flask Web Server — LNI Reference Checker v3
============================================
Pipeline:
  EXTRACT   — pdfplumber/docx parses text, splits body vs bibliography
  PARSE     — regex extracts structured BibEntry objects (100% deterministic)
  CHECK     — all rule-based checks run in pure Python (100% deterministic):
                * LNI key format validation
                * Required field completeness per entry type
                * In-text citation extraction
                * Cross-check: cited keys vs bibliography keys
                * Duplicate detection (rapidfuzz)
                * Page range format, author order, LNI macro style
  API LOOKUP — CrossRef / Semantic Scholar / OpenAlex / arXiv / DDG (evidence)
  AI CHECK  — Groq (llama-3.3-70b) -> Gemini fallback for judgment tasks only:
                * ai_verify_references() — REAL / SUSPICIOUS / FAKE per entry
                * ai_overall_verdict()   — PASS / FLAG / FAIL + student feedback
  /ai-review — original manual AI audit button (unchanged, Groq -> Gemini)
"""

import os
import re
import json
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response

from extractor import extract
from parser import parse_bibliography, entries_to_dict
from checker import (
    extract_citations_from_body,
    extract_citation_contexts,
    detect_self_citations,
    cross_check,
    verify_all_references,
    check_lni_macros,
    find_duplicates,
    compute_score,
)
from ai_checker import ai_verify_references, ai_overall_verdict

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _run_full_check(main_path: str, bib_path: str = None,
                    verify: bool = True, filename: str = "") -> dict:
    """Core pipeline. Steps 1-4 are deterministic; step 5 uses AI."""

    # ── Step 1: Extract ──────────────────────────────────────────────────────
    sections = extract(main_path, bib_path)
    body     = sections["body"]
    bib_text = sections["bibliography"]
    fmt      = sections["format"]

    # ── Step 2: Parse bibliography (deterministic) ────────────────────────────
    bib_list = parse_bibliography(bib_text)
    bib_dict = entries_to_dict(bib_list)

    # ── Step 3: All rule-based checks (deterministic, 100% accurate) ─────────
    style_suggestions = check_lni_macros(body)

    cited_keys = extract_citations_from_body(body)
    if fmt == "latex":
        for group in re.findall(
            r'\\(?:cite|Cite|citet|Citet|citep)\{([^}]+)\}',
            sections.get("full_text", "")
        ):
            for k in group.split(','):
                cited_keys.add(k.strip())

    xcheck            = cross_check(bib_dict, cited_keys)
    citation_contexts = extract_citation_contexts(body)
    duplicates        = find_duplicates(bib_dict)
    author_candidates = re.findall(r'\b([A-ZÄÖÜ][a-zäöüß]{2,})\b', body[:800])
    self_citations    = detect_self_citations(bib_dict, author_candidates)

    # ── Step 4: External API lookups (evidence gathering, not AI) ────────────
    api_results_raw = []
    if verify and bib_dict:
        api_results_raw = verify_all_references(bib_dict)

    # Build plain dicts for passing to AI
    bib_dicts = [
        {
            "key":        e.key,
            "entry_type": e.entry_type or "unknown",
            "authors":    e.authors,
            "title":      e.title,
            "year":       e.year,
            "publisher":  e.publisher,
            "journal":    e.journal,
            "booktitle":  e.booktitle,
            "pages":      e.pages,
            "url":        e.url,
            "urldate":    e.urldate,
            "doi":        e.doi,
            "raw_text":   e.raw_text[:300],
        }
        for e in bib_list
    ]

    api_results_dicts = [
        {
            "key":             vr.key,
            "status":          vr.status,
            "confidence":      round(vr.confidence, 2),
            "matched_title":   vr.matched_title,
            "doi":             vr.doi,
            "open_access_url": vr.open_access_url,
            "note":            vr.note,
            "sources_checked": vr.sources_checked,
            "web_evidence":    vr.web_evidence,
        }
        for vr in api_results_raw
    ]

    # ── Step 5: AI checks (only where judgment is needed) ────────────────────
    # 5a. Fake detection — AI reasons about authenticity using all evidence
    verification_result = ai_verify_references(bib_dicts, api_results_dicts)

    # Build a summary dict for the overall verdict
    summary_for_ai = {
        "duplicates":     len(duplicates),
        "self_citations": len(self_citations),
        "style_issues":   len(style_suggestions),
    }

    # 5b. Overall verdict — AI synthesises everything
    overall = ai_overall_verdict(
        filename=filename or Path(main_path).name,
        summary=summary_for_ai,
        xcheck=xcheck,
        bib_list=bib_list,
        verification_result=verification_result,
    )

    # ── Assemble result (compatible with original frontend shape) ─────────────
    # Map AI verdicts back to the format the frontend expects
    ai_verdicts_by_key = {v["key"]: v for v in verification_result.get("verdicts", [])}

    verification_output = []
    for vr in api_results_raw:
        ai = ai_verdicts_by_key.get(vr.key, {})
        # Translate AI verdict to legacy status field so existing frontend works
        ai_verdict = ai.get("verdict", "SUSPICIOUS")
        if ai_verdict == "REAL":
            status = "verified"
        elif ai_verdict == "FAKE":
            status = "not_found"
        else:
            status = "partial_match"

        verification_output.append({
            "key":             vr.key,
            "title":           vr.title,
            # Original API fields
            "status":          status,
            "confidence":      round(ai.get("confidence", vr.confidence), 2),
            "matched_title":   vr.matched_title,
            "doi":             vr.doi or ai.get("open_access_url"),
            "open_access_url": ai.get("open_access_url") or vr.open_access_url,
            "note":            vr.note,
            "sources_checked": vr.sources_checked,
            "web_evidence":    vr.web_evidence,
            # New AI fields
            "ai_verdict":      ai_verdict,
            "ai_reasoning":    ai.get("reasoning", ""),
            "ai_risk_factors": ai.get("risk_factors", []),
        })

    # Also include entries that had no API result (websites, etc.)
    api_keys = {vr.key for vr in api_results_raw}
    for entry in bib_dicts:
        if entry["key"] not in api_keys:
            ai = ai_verdicts_by_key.get(entry["key"], {})
            ai_verdict = ai.get("verdict", "SUSPICIOUS")
            verification_output.append({
                "key":             entry["key"],
                "title":           entry.get("title") or "",
                "status":          "verified" if ai_verdict == "REAL" else "not_found" if ai_verdict == "FAKE" else "partial_match",
                "confidence":      ai.get("confidence", 0.5),
                "matched_title":   None,
                "doi":             entry.get("doi"),
                "open_access_url": ai.get("open_access_url") or entry.get("url"),
                "note":            ai.get("reasoning", ""),
                "sources_checked": [],
                "web_evidence":    None,
                "ai_verdict":      ai_verdict,
                "ai_reasoning":    ai.get("reasoning", ""),
                "ai_risk_factors": ai.get("risk_factors", []),
            })

    # Deterministic score (kept for batch/export compatibility)
    det_score = compute_score(bib_list, xcheck, api_results_raw,
                              style_suggestions, duplicates)

    # Use AI verdict/score if available, fall back to deterministic
    final_score = {
        "score":   overall.get("score", det_score["score"]),
        "grade":   overall.get("grade", det_score["grade"]),
        "verdict": overall.get("verdict", "FLAG"),
        "verdict_reason":   overall.get("verdict_reason", ""),
        "student_feedback": overall.get("student_feedback", []),
        "professor_note":   overall.get("professor_note", ""),
        "penalties":        det_score.get("penalties", []),
        "max_score":        100,
    }

    return {
        "filename": filename or Path(main_path).name,
        "format":   fmt.upper(),
        "stats": {
            "body_chars":      len(body),
            "bib_chars":       len(bib_text),
            "bib_entry_count": len(bib_dict),
            "citation_count":  len(cited_keys),
        },
        "bibliography": [
            {
                "key":                 e.key,
                "type":                e.entry_type or "unknown",
                "authors":             e.authors,
                "title":               e.title,
                "year":                e.year,
                "publisher":           e.publisher,
                "journal":             e.journal,
                "url":                 e.url,
                "doi":                 e.doi,
                "raw":                 e.raw_text[:250],
                "completeness_issues": e.completeness_issues,
            }
            for e in bib_list
        ],
        "cross_check": {
            "correctly_used":   xcheck.correctly_used,
            "cited_not_in_bib": xcheck.cited_not_in_bib,
            "in_bib_not_cited": xcheck.in_bib_not_cited,
        },
        "citation_contexts":  citation_contexts,
        "style_suggestions":  style_suggestions,
        "duplicates":         duplicates,
        "self_citations":     self_citations,
        "score":              final_score,
        "verification":       verification_output,
        "verification_ai_summary": verification_result.get("summary", ""),
        "summary": {
            "missing_from_bib":   len(xcheck.cited_not_in_bib),
            "uncited_entries":    len(xcheck.in_bib_not_cited),
            "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
            "fake_candidates":    verification_result.get("fake_count", 0),
            "suspicious":         verification_result.get("suspicious_count", 0),
            "verified":           sum(1 for v in verification_output if v["status"] == "verified"),
            "style_issues":       len(style_suggestions),
            "open_access":        sum(1 for v in verification_output if v.get("open_access_url")),
            "duplicates":         len(duplicates),
            "self_citations":     len(self_citations),
            "bib_entry_count":    len(bib_dict),
            "citation_count":     len(cited_keys),
        },
    }


@app.route("/check", methods=["POST"])
def check():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    main_file = request.files["file"]
    filename  = main_file.filename
    ext       = Path(filename).suffix.lower()
    verify    = request.form.get("verify", "true").lower() == "true"

    if ext not in {".pdf", ".docx", ".tex", ".latex"}:
        return jsonify({"error": f"Unsupported format '{ext}'. Supported: PDF, DOCX, TEX"}), 400

    tmpdir    = tempfile.mkdtemp()
    main_path = os.path.join(tmpdir, filename)
    main_file.save(main_path)

    bib_path = None
    if "bib" in request.files and ext in (".tex", ".latex"):
        bib_file = request.files["bib"]
        bib_path = os.path.join(tmpdir, bib_file.filename)
        bib_file.save(bib_path)

    try:
        result = _run_full_check(main_path, bib_path, verify=verify, filename=filename)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/ai-review", methods=["POST"])
def ai_review():
    """
    Manual AI audit button — unchanged from v2.
    Groq (llama-3.3-70b) -> Gemini 1.5 Flash fallback.
    Set GROQ_API_KEY or GEMINI_API_KEY env vars.
    """
    import requests as req

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No result data provided"}), 400

    s  = data.get("summary", {})
    sc = data.get("score", {})

    flagged = [
        v for v in data.get("verification", [])
        if v.get("status") in ("not_found", "partial_match", "error")
        or v.get("ai_verdict") in ("FAKE", "SUSPICIOUS")
    ]
    incomplete  = [e for e in data.get("bibliography", []) if e.get("completeness_issues")]
    dupes       = data.get("duplicates", [])
    self_cit    = data.get("self_citations", [])

    flagged_lines = "\n".join(
        f"  [{v['key']}] title=\"{v['title']}\" "
        f"ai_verdict={v.get('ai_verdict','?')} "
        f"reasoning=\"{v.get('ai_reasoning','')[:80]}\" "
        f"confidence={int(v['confidence']*100)}% "
        f"sources={','.join(v.get('sources_checked',[]))}"
        for v in flagged
    ) or "  None"

    incomplete_lines = "\n".join(
        f"  [{e['key']}] {e.get('title','?')} — missing: {', '.join(e['completeness_issues'])}"
        for e in incomplete
    ) or "  None"

    dupe_lines = "\n".join(
        f"  [{d['key_a']}] vs [{d['key_b']}] — {int(d['similarity']*100)}% similar"
        for d in dupes
    ) or "  None"

    self_lines = "\n".join(
        f"  [{sc_['key']}] matched author: {sc_['matched_author']}"
        for sc_ in self_cit
    ) or "  None"

    ai_summary = data.get("verification_ai_summary", "")

    prompt = f"""You are assisting a professor auditing a student's LNI-format academic paper reference list.

AUDIT SUMMARY:
- File: {data.get('filename','?')} | Score: {sc.get('score','?')}/100 | Grade: {sc.get('grade','?')} | Verdict: {sc.get('verdict','?')}
- Total bib entries: {s.get('bib_entry_count',0)} | In-text citations: {s.get('citation_count',0)}
- Missing from bib: {s.get('missing_from_bib',0)} | Never cited: {s.get('uncited_entries',0)}
- Incomplete entries: {s.get('incomplete_entries',0)} | Duplicates: {s.get('duplicates',0)}
- Self-citations: {s.get('self_citations',0)} | Style violations: {s.get('style_issues',0)}
- AI integrity summary: {ai_summary}

FLAGGED REFERENCES (AI verdict FAKE/SUSPICIOUS or low API confidence):
{flagged_lines}

INCOMPLETE ENTRIES:
{incomplete_lines}

DUPLICATE PAIRS:
{dupe_lines}

SELF-CITATIONS:
{self_lines}

TASK:
1. For each FLAGGED reference, give a one-line verdict:
   REAL — appears genuine (low confidence likely due to formatting/title variation)
   SUSPICIOUS — existence unconfirmed, professor should manually verify
   FAKE — strong signs of fabrication

2. OVERALL verdict: PASS / FLAG / FAIL
   PASS  = minor issues only, references appear legitimate
   FLAG  = suspicious references, professor should spot-check
   FAIL  = multiple likely fake references or severe structural problems

Reply in EXACT format:
REFERENCE VERDICTS:
[key]: VERDICT — reason

OVERALL: PASS/FLAG/FAIL — one sentence reason"""

    ai_text   = None
    ai_source = None
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if groq_key:
        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600, "temperature": 0.1},
                timeout=25
            )
            if resp.status_code == 200:
                ai_text   = resp.json()["choices"][0]["message"]["content"].strip()
                ai_source = "Groq (LLaMA 3.3 70B)"
        except Exception:
            pass

    if not ai_text and gemini_key:
        try:
            resp = req.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 600, "temperature": 0.1}},
                timeout=25
            )
            if resp.status_code == 200:
                parts     = resp.json()["candidates"][0]["content"]["parts"]
                ai_text   = "".join(p.get("text", "") for p in parts).strip()
                ai_source = "Gemini 1.5 Flash"
        except Exception:
            pass

    if not ai_text:
        missing = []
        if not groq_key:   missing.append("GROQ_API_KEY")
        if not gemini_key: missing.append("GEMINI_API_KEY")
        if missing:
            return jsonify({
                "error": f"No AI API key configured. Set {' or '.join(missing)} as environment variables.",
                "hint":  "Groq: free key at console.groq.com | Gemini: free key at aistudio.google.com"
            }), 503
        return jsonify({"error": "Both Groq and Gemini failed to respond."}), 503

    return jsonify({"verdict": ai_text, "ai_source": ai_source,
                    "flagged_count": len(flagged)})


@app.route("/batch", methods=["POST"])
def batch_check():
    uploaded = request.files.getlist("files")
    verify   = request.form.get("verify", "true").lower() == "true"

    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    tmpdir  = tempfile.mkdtemp()

    try:
        for main_file in uploaded:
            filename = main_file.filename
            ext      = Path(filename).suffix.lower()
            if ext not in {".pdf", ".docx", ".tex", ".latex"}:
                results.append({"filename": filename, "error": f"Unsupported format '{ext}'"})
                continue

            file_path = os.path.join(tmpdir, filename)
            main_file.save(file_path)

            try:
                result = _run_full_check(file_path, verify=verify, filename=filename)
                results.append({
                    "filename":     filename,
                    "format":       result["format"],
                    "score":        result["score"],
                    "summary":      result["summary"],
                    "flagged_refs": [
                        v["key"] for v in result["verification"]
                        if v.get("ai_verdict") in ("FAKE", "SUSPICIOUS")
                        or v.get("status") in ("not_found", "partial_match")
                    ],
                })
            except Exception as e:
                import traceback
                results.append({"filename": filename, "error": str(e),
                                 "trace": traceback.format_exc()})

        results.sort(key=lambda r: r.get("score", {}).get("score", -1), reverse=True)
        return jsonify({"files": results, "count": len(results)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/export", methods=["POST"])
def export_report():
    from flask import Response
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data"}), 400

    lines = []
    sc    = data.get("score", {})
    s     = data.get("summary", {})

    lines.append("=" * 70)
    lines.append("LNI REFERENCE CHECKER v3 — PROFESSOR REPORT")
    lines.append("=" * 70)
    lines.append(f"File    : {data.get('filename', '?')}")
    lines.append(f"Format  : {data.get('format', '?')}")
    lines.append(f"Score   : {sc.get('score', '?')}/100  Grade: {sc.get('grade', '?')}  Verdict: {sc.get('verdict', '?')}")
    lines.append(f"Reason  : {sc.get('verdict_reason', '')}")
    lines.append("")

    lines.append("── SUMMARY ──")
    lines.append(f"  Bibliography entries : {s.get('bib_entry_count', 0)}")
    lines.append(f"  In-text citations    : {s.get('citation_count', 0)}")
    lines.append(f"  Missing from bib     : {s.get('missing_from_bib', 0)}")
    lines.append(f"  Never cited          : {s.get('uncited_entries', 0)}")
    lines.append(f"  Incomplete entries   : {s.get('incomplete_entries', 0)}")
    lines.append(f"  FAKE (AI verdict)    : {s.get('fake_candidates', 0)}")
    lines.append(f"  SUSPICIOUS           : {s.get('suspicious', 0)}")
    lines.append(f"  Verified REAL        : {s.get('verified', 0)}")
    lines.append(f"  Duplicates           : {s.get('duplicates', 0)}")
    lines.append(f"  Self-citations       : {s.get('self_citations', 0)}")
    lines.append(f"  Style issues         : {s.get('style_issues', 0)}")
    lines.append(f"  Open-access links    : {s.get('open_access', 0)}")
    lines.append("")

    if sc.get("student_feedback"):
        lines.append("── FEEDBACK FOR STUDENT ──")
        for fb in sc["student_feedback"]:
            lines.append(f"  • {fb}")
        lines.append("")

    if sc.get("professor_note"):
        lines.append("── NOTE FOR PROFESSOR ──")
        lines.append(f"  {sc['professor_note']}")
        lines.append("")

    if sc.get("penalties"):
        lines.append("── SCORE BREAKDOWN ──")
        for p in sc["penalties"]:
            lines.append(f"  -{p['deduction']:2d}  {p['category']} ({p['count']}×)")
        lines.append("")

    xc = data.get("cross_check", {})
    if xc.get("cited_not_in_bib"):
        lines.append("── CITED BUT MISSING FROM BIBLIOGRAPHY ──")
        for k in xc["cited_not_in_bib"]:
            lines.append(f"  [MISSING] {k}")
        lines.append("")

    if xc.get("in_bib_not_cited"):
        lines.append("── IN BIBLIOGRAPHY BUT NEVER CITED ──")
        for k in xc["in_bib_not_cited"]:
            lines.append(f"  [ORPHAN]  {k}")
        lines.append("")

    fake_refs = [v for v in data.get("verification", []) if v.get("ai_verdict") == "FAKE"]
    if fake_refs:
        lines.append("── LIKELY FAKE REFERENCES (AI) ──")
        for v in fake_refs:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    AI reasoning: {v.get('ai_reasoning', '')}")
            for rf in v.get("ai_risk_factors", []):
                lines.append(f"    ⚠ {rf}")
            lines.append(f"    Sources checked: {', '.join(v.get('sources_checked', []))}")
        lines.append("")

    suspicious_refs = [v for v in data.get("verification", []) if v.get("ai_verdict") == "SUSPICIOUS"]
    if suspicious_refs:
        lines.append("── SUSPICIOUS REFERENCES ──")
        for v in suspicious_refs:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    {v.get('ai_reasoning', '')}")
        lines.append("")

    incomplete = [e for e in data.get("bibliography", []) if e.get("completeness_issues")]
    if incomplete:
        lines.append("── INCOMPLETE ENTRIES ──")
        for e in incomplete:
            lines.append(f"  [{e['key']}] {e.get('title', '?')}")
            for issue in e["completeness_issues"]:
                lines.append(f"    ⚠ {issue}")
        lines.append("")

    if data.get("duplicates"):
        lines.append("── DUPLICATE ENTRIES ──")
        for d in data["duplicates"]:
            lines.append(f"  [{d['key_a']}] vs [{d['key_b']}]  ({int(d['similarity']*100)}% similar)")
        lines.append("")

    if data.get("style_suggestions"):
        lines.append("── STYLE ISSUES ──")
        for s_ in data["style_suggestions"]:
            lines.append(f"  {s_['message']} ({s_['count']}×)")
        lines.append("")

    if data.get("verification_ai_summary"):
        lines.append("── AI INTEGRITY SUMMARY ──")
        lines.append(f"  {data['verification_ai_summary']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Generated by LNI Reference Checker v3")
    lines.append("Deterministic checks: key format, required fields, cross-check, duplicates, style")
    lines.append("AI checks (Groq/Gemini): fake detection, overall verdict")
    lines.append("=" * 70)

    report = "\n".join(lines)
    fname  = data.get("filename", "report").replace(" ", "_") + "_lni_report.txt"
    return Response(report, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


if __name__ == "__main__":
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │   LNI Reference Checker v3                      │")
    print("  │   http://localhost:5000                          │")
    print("  │                                                  │")
    print("  │   Deterministic: key · fields · xcheck · dupes  │")
    print("  │   AI (Groq→Gemini): fake detection · verdict    │")
    print("  │   APIs: CrossRef · SemanticScholar · OpenAlex   │")
    print("  │          arXiv · Google Scholar · DuckDuckGo    │")
    print("  └─────────────────────────────────────────────────┘\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
