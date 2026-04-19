"""
Flask Web Server — LNI Reference Checker v2
Handles PDF, DOCX, and LaTeX (.tex + optional .bib) uploads.

New in v2:
  - Parallel multi-source verification (CrossRef + Semantic Scholar + OpenAlex + Google Scholar + DuckDuckGo)
  - DOI-first lookup for definitive verification
  - Upgraded Groq model (llama-3.3-70b-versatile)
  - /export endpoint for PDF/CSV report download
  - sources_checked and web_evidence fields in verification output
  - Batch endpoint now includes per-file AI summary flag counts
"""

import os
import re
import json
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from extractor import extract
from parser import parse_bibliography, entries_to_dict
from checker import (
    extract_citations_from_body, extract_citation_contexts,
    detect_self_citations, cross_check, verify_all_references,
    check_lni_macros, find_duplicates, compute_score
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _run_full_check(main_path: str, bib_path: str = None,
                    verify: bool = True, filename: str = "") -> dict:
    """Core pipeline shared by /check and /batch."""
    sections = extract(main_path, bib_path)
    body     = sections["body"]
    bib_text = sections["bibliography"]
    fmt      = sections["format"]

    bib_list          = parse_bibliography(bib_text)
    bib_dict          = entries_to_dict(bib_list)
    style_suggestions = check_lni_macros(body)

    cited_keys = extract_citations_from_body(body)
    if fmt == "latex":
        latex_cites = re.findall(
            r'\\(?:cite|Cite|citet|Citet|citep)\{([^}]+)\}',
            sections.get("full_text", "")
        )
        for group in latex_cites:
            for k in group.split(','):
                cited_keys.add(k.strip())

    xcheck            = cross_check(bib_dict, cited_keys)
    citation_contexts = extract_citation_contexts(body)

    verification_results = []
    if verify and bib_dict:
        verification_results = verify_all_references(bib_dict)

    duplicates = find_duplicates(bib_dict)
    score      = compute_score(bib_list, xcheck, verification_results,
                                style_suggestions, duplicates)

    author_candidates = re.findall(r'\b([A-ZÄÖÜ][a-zäöüß]{2,})\b', body[:800])
    self_citations    = detect_self_citations(bib_dict, author_candidates)

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
        "score":              score,
        "verification": [
            {
                "key":             vr.key,
                "title":           vr.title,
                "status":          vr.status,
                "confidence":      round(vr.confidence, 2),
                "matched_title":   vr.matched_title,
                "doi":             vr.doi,
                "open_access_url": vr.open_access_url,
                "note":            vr.note,
                "sources_checked": vr.sources_checked,
                "web_evidence":    vr.web_evidence,
            }
            for vr in verification_results
        ],
        "summary": {
            "missing_from_bib":   len(xcheck.cited_not_in_bib),
            "uncited_entries":    len(xcheck.in_bib_not_cited),
            "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
            "fake_candidates":    sum(1 for vr in verification_results if vr.status == "not_found"),
            "verified":           sum(1 for vr in verification_results if vr.status == "verified"),
            "style_issues":       len(style_suggestions),
            "open_access":        sum(1 for vr in verification_results if vr.open_access_url),
            "duplicates":         len(duplicates),
            "self_citations":     len(self_citations),
        }
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
    Professor audit AI: re-evaluates ambiguous/flagged references using a free LLM.
    Tries Groq (llama-3.3-70b-versatile) first, falls back to Gemini 1.5 Flash.
    Both are free-tier APIs — set keys via env vars:
        GROQ_API_KEY   — get free at console.groq.com
        GEMINI_API_KEY — get free at aistudio.google.com
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
    ]
    incomplete = [
        e for e in data.get("bibliography", [])
        if e.get("completeness_issues")
    ]
    dupes    = data.get("duplicates", [])
    self_cit = data.get("self_citations", [])

    flagged_lines = "\n".join(
        f"  [{v['key']}] title=\"{v['title']}\" "
        f"matched=\"{v.get('matched_title','—')}\" "
        f"confidence={int(v['confidence']*100)}% "
        f"sources={','.join(v.get('sources_checked',[]))} "
        f"web_evidence=\"{(v.get('web_evidence') or '')[:80]}\""
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

    prompt = f"""You are assisting a professor auditing a student's LNI-format academic paper reference list.

AUDIT SUMMARY:
- File: {data.get('filename','?')} | Score: {sc.get('score','?')}/100 | Grade: {sc.get('grade','?')}
- Total bib entries: {s.get('bib_entry_count',0)} | In-text citations: {s.get('citation_count',0)}
- Missing from bib: {s.get('missing_from_bib',0)} | Never cited: {s.get('uncited_entries',0)}
- Incomplete entries: {s.get('incomplete_entries',0)} | Duplicates: {s.get('duplicates',0)}
- Self-citations: {s.get('self_citations',0)} | Style violations: {s.get('style_issues',0)}

FLAGGED REFERENCES (unverified / partial match / error — sources checked shown):
{flagged_lines}

INCOMPLETE ENTRIES:
{incomplete_lines}

DUPLICATE PAIRS:
{dupe_lines}

SELF-CITATIONS:
{self_lines}

TASK:
1. For each FLAGGED reference, give a one-line verdict using exactly:
   REAL — reference appears genuine (low confidence likely due to formatting/title variation)
   SUSPICIOUS — existence unconfirmed, professor should manually verify
   FAKE — strong signs of fabrication (no match anywhere, implausible metadata)

2. Give an OVERALL verdict: PASS / FLAG / FAIL
   PASS  = minor issues only, references appear legitimate
   FLAG  = several suspicious references, professor should spot-check
   FAIL  = multiple likely fake references or severe structural problems

Reply in this EXACT format:
REFERENCE VERDICTS:
[key]: VERDICT — reason

OVERALL: PASS/FLAG/FAIL — one sentence reason"""

    ai_text   = None
    ai_source = None
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Try Groq first — llama-3.3-70b is significantly more capable than 3.1-8b
    if groq_key:
        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type":  "application/json"
                },
                json={
                    "model":       "llama-3.3-70b-versatile",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  600,
                    "temperature": 0.1,
                },
                timeout=25
            )
            if resp.status_code == 200:
                ai_text   = resp.json()["choices"][0]["message"]["content"].strip()
                ai_source = "Groq (LLaMA 3.3 70B)"
            elif resp.status_code == 429:
                pass  # rate limited — fall through to Gemini
        except Exception:
            pass

    # Fallback: Gemini 1.5 Flash
    if not ai_text and gemini_key:
        try:
            resp = req.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 600, "temperature": 0.1}
                },
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

    return jsonify({
        "verdict":       ai_text,
        "ai_source":     ai_source,
        "flagged_count": len(flagged),
    })


@app.route("/batch", methods=["POST"])
def batch_check():
    """
    Accept multiple PDF/DOCX/TEX uploads and run the full check on each.
    Returns per-file results + aggregate leaderboard sorted by score.
    """
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
                # For batch, return summary + score only (not full bib list)
                results.append({
                    "filename": filename,
                    "format":   result["format"],
                    "score":    result["score"],
                    "summary":  result["summary"],
                    "flagged_refs": [
                        v["key"] for v in result["verification"]
                        if v["status"] in ("not_found", "partial_match")
                    ],
                })
            except Exception as e:
                import traceback
                results.append({
                    "filename": filename,
                    "error":    str(e),
                    "trace":    traceback.format_exc()
                })

        results.sort(
            key=lambda r: r.get("score", {}).get("score", -1),
            reverse=True
        )
        return jsonify({"files": results, "count": len(results)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/export", methods=["POST"])
def export_report():
    """
    Generate a plain-text professor report from check results.
    POST body: same JSON as /check returns.
    Returns: text/plain report suitable for printing / attaching to review notes.
    """
    from flask import Response
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data"}), 400

    lines = []
    sc    = data.get("score", {})
    s     = data.get("summary", {})

    lines.append("=" * 70)
    lines.append("LNI REFERENCE CHECKER — PROFESSOR REPORT")
    lines.append("=" * 70)
    lines.append(f"File   : {data.get('filename', '?')}")
    lines.append(f"Format : {data.get('format', '?')}")
    lines.append(f"Score  : {sc.get('score', '?')}/100  Grade: {sc.get('grade', '?')}")
    lines.append("")

    lines.append("── SUMMARY ──")
    lines.append(f"  Bibliography entries : {s.get('bib_entry_count', 0)}")
    lines.append(f"  In-text citations    : {s.get('citation_count', 0)}")
    lines.append(f"  Missing from bib     : {s.get('missing_from_bib', 0)}")
    lines.append(f"  Never cited          : {s.get('uncited_entries', 0)}")
    lines.append(f"  Incomplete entries   : {s.get('incomplete_entries', 0)}")
    lines.append(f"  Fake / unverified    : {s.get('fake_candidates', 0)}")
    lines.append(f"  Duplicates           : {s.get('duplicates', 0)}")
    lines.append(f"  Self-citations       : {s.get('self_citations', 0)}")
    lines.append(f"  Style issues         : {s.get('style_issues', 0)}")
    lines.append(f"  Open-access links    : {s.get('open_access', 0)}")
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

    not_found = [v for v in data.get("verification", []) if v["status"] == "not_found"]
    if not_found:
        lines.append("── UNVERIFIED / POTENTIALLY FAKE REFERENCES ──")
        for v in not_found:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"           Sources checked: {', '.join(v.get('sources_checked', []))}")
            if v.get("web_evidence"):
                lines.append(f"           Web evidence: {v['web_evidence'][:100]}")
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

    lines.append("=" * 70)
    lines.append("Generated by LNI Reference Checker v2")
    lines.append("=" * 70)

    report = "\n".join(lines)
    fname  = data.get("filename", "report").replace(" ", "_") + "_lni_report.txt"
    return Response(
        report,
        mimetype="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )


if __name__ == "__main__":
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │   LNI Reference Checker v2 — Web UI    │")
    print("  │   http://localhost:5000                 │")
    print("  │                                         │")
    print("  │   Sources: CrossRef · SemanticScholar   │")
    print("  │            OpenAlex · GoogleScholar     │")
    print("  │            DuckDuckGo web fingerprint   │")
    print("  │   AI: Groq llama-3.3-70b → Gemini      │")
    print("  └─────────────────────────────────────────┘\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)