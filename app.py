"""
Flask Web Server — LNI Reference Checker v5
============================================
Pipeline:
  EXTRACT       — pdfplumber/docx/LaTeX, last-match bib detection, DOCX table cells

  AI EXTRACT    — LLM parses bibliography text into structured records (NEW v5)
                  Handles any format: numbered [1], author-year, BibLaTeX, mixed
                  Falls back silently to regex extraction if no API key set

  PARSE         — regex extracts BibEntry objects (100% deterministic):
                  LNI key format, key-vs-metadata consistency, field completeness,
                  future-year, implausible page ranges, needs_ai_parsing flag

  AI RE-PARSE   — Groq/Gemini re-parses entries flagged needs_ai_parsing=True
                  Improved metadata flows into API lookups

  CHECK         — deterministic rule-based checks:
                  Citation extraction (LNI + numeric detection), cross-check,
                  duplicate detection, LNI macro/style checks

  API LOOKUP    — 10 parallel sources, all FREE:
                  CrossRef · Semantic Scholar · OpenAlex · arXiv
                  DBLP · ACL Anthology · OpenReview · Open Library
                  GitHub API · Google Scholar · DuckDuckGo
                  Results cached in memory + disk (LNI_CACHE_DIR)

  AI VERIFY     — Groq→Gemini: three-tier REAL/SUSPICIOUS/FAKE per reference
                  Author overlap pre-filter runs first (no AI tokens for clear cases)
                  key_consistent signal from parser included in AI prompt

  SSE STREAM    — /check streams progress events to browser in real time (NEW v5)
                  Each reference result is sent as a Server-Sent Event so the
                  professor sees live per-reference progress instead of a 60s wait

  /ai-review    — manual AI audit button (Groq→Gemini), unchanged from v4

Environment variables (all optional, all free):
  GROQ_API_KEY              → console.groq.com       (14 400 req/day free)
  GEMINI_API_KEY            → aistudio.google.com    (1 500 req/day free)
  SEMANTIC_SCHOLAR_API_KEY  → semanticscholar.org/product/api (free)
  GITHUB_TOKEN              → github.com/settings/tokens     (free)
  LNI_CACHE_DIR             → disk cache path  (default: .lni_cache)
  UNPAYWALL_EMAIL           → your email for Unpaywall polite pool
"""

import os
import re
import json
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

from extractor import extract
from parser import parse_bibliography, entries_to_dict
from checker import (
    extract_citations_from_body,
    extract_citation_contexts,
    detect_self_citations,
    cross_check,
    verify_reference,
    verify_all_references,
    check_lni_macros,
    find_duplicates,
    compute_score,
)
from ai_checker import (
    ai_extract_references_from_text,
    merge_ai_extractions_into_bib_list,
    ai_parse_uncertain_entries,
    ai_verify_references,
    ai_overall_verdict,
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Core pipeline (non-streaming, used by batch and export)
# ---------------------------------------------------------------------------

def _run_full_check(main_path: str, bib_path: str = None,
                    verify: bool = True, filename: str = "") -> dict:
    """Full check pipeline. Returns complete result dict."""

    # ── Step 1: Extract ──────────────────────────────────────────────────────
    sections = extract(main_path, bib_path)
    body     = sections["body"]
    bib_text = sections["bibliography"]
    fmt      = sections["format"]

    # ── Step 2: AI bibliography extraction (new v5) ───────────────────────────
    # Runs before regex parsing so LLM-extracted metadata can fill gaps.
    ai_extracted_refs = ai_extract_references_from_text(bib_text)

    # ── Step 3: Regex parse (deterministic) ──────────────────────────────────
    bib_list = parse_bibliography(bib_text)
    bib_dict = entries_to_dict(bib_list)

    # Merge AI extractions into regex-parsed entries
    if ai_extracted_refs:
        bib_list = merge_ai_extractions_into_bib_list(ai_extracted_refs, bib_list)
        bib_dict = entries_to_dict(bib_list)

    # ── Step 4: AI re-parse for uncertain entries ─────────────────────────────
    bib_dicts = _bib_to_dicts(bib_list)
    ai_parse_improvements = ai_parse_uncertain_entries(bib_dicts)
    if ai_parse_improvements:
        bib_list = _apply_ai_improvements(bib_list, ai_parse_improvements)
        bib_dict = entries_to_dict(bib_list)
        bib_dicts = _bib_to_dicts(bib_list)

    # ── Step 5: Rule-based checks (deterministic) ────────────────────────────
    style_suggestions = check_lni_macros(body)
    cited_keys = extract_citations_from_body(body)
    if fmt == "latex":
        for group in re.findall(
            r'\\(?:cite|Cite|citet|Citet|citep)\{([^}]+)\}',
            sections.get("full_text", ""),
        ):
            for k in group.split(','):
                cited_keys.add(k.strip())

    has_numeric = '__numeric_citations__' in cited_keys
    xcheck             = cross_check(bib_dict, cited_keys)
    citation_contexts  = extract_citation_contexts(body)
    duplicates         = find_duplicates(bib_dict)
    self_citations     = detect_self_citations(bib_dict, body)

    # ── Step 6: API lookups ──────────────────────────────────────────────────
    api_results_raw = []
    if verify and bib_dict:
        api_results_raw = verify_all_references(bib_dict)

    api_results_dicts = _vr_to_dicts(api_results_raw)

    # ── Step 7: AI verification ──────────────────────────────────────────────
    verification_result = ai_verify_references(bib_dicts, api_results_dicts)
    summary_for_ai = {"duplicates": len(duplicates), "self_citations": len(self_citations),
                      "style_issues": len(style_suggestions)}
    overall = ai_overall_verdict(
        filename=filename or Path(main_path).name,
        summary=summary_for_ai, xcheck=xcheck,
        bib_list=bib_list, verification_result=verification_result,
    )

    return _assemble_result(
        filename=filename or Path(main_path).name,
        fmt=fmt, body=body, bib_text=bib_text,
        bib_list=bib_list, bib_dict=bib_dict,
        cited_keys=cited_keys, has_numeric=has_numeric,
        xcheck=xcheck, citation_contexts=citation_contexts,
        duplicates=duplicates, self_citations=self_citations,
        style_suggestions=style_suggestions,
        api_results_raw=api_results_raw,
        verification_result=verification_result,
        overall=overall,
        ai_parse_improvements=ai_parse_improvements,
    )


# ---------------------------------------------------------------------------
# SSE streaming pipeline  (new v5)
# ---------------------------------------------------------------------------

def _run_streaming_check(main_path: str, bib_path: str = None,
                          verify: bool = True, filename: str = ""):
    """
    Generator that yields Server-Sent Event strings.
    Sends progress events as each reference is verified, then sends the
    final complete result as a 'done' event.

    Event format (JSON lines):
      event: progress
      data: {"step": "extract|parse|check|verify_N|ai|done", "message": "...", ...}

      event: done
      data: {<full result dict>}

      event: error
      data: {"error": "...", "trace": "..."}
    """
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        yield _sse("progress", {"step": "extract", "message": "Extracting text from document…"})

        sections = extract(main_path, bib_path)
        body     = sections["body"]
        bib_text = sections["bibliography"]
        fmt      = sections["format"]

        yield _sse("progress", {"step": "ai_extract",
                                  "message": "AI extracting bibliography structure…"})
        ai_extracted_refs = ai_extract_references_from_text(bib_text)

        yield _sse("progress", {"step": "parse",
                                  "message": "Parsing bibliography entries…"})
        bib_list = parse_bibliography(bib_text)
        bib_dict = entries_to_dict(bib_list)

        if ai_extracted_refs:
            bib_list = merge_ai_extractions_into_bib_list(ai_extracted_refs, bib_list)
            bib_dict = entries_to_dict(bib_list)

        bib_dicts = _bib_to_dicts(bib_list)
        ai_parse_improvements = ai_parse_uncertain_entries(bib_dicts)
        if ai_parse_improvements:
            bib_list = _apply_ai_improvements(bib_list, ai_parse_improvements)
            bib_dict = entries_to_dict(bib_list)
            bib_dicts = _bib_to_dicts(bib_list)

        yield _sse("progress", {"step": "check",
                                  "message": f"Running deterministic checks on {len(bib_list)} entries…"})
        style_suggestions = check_lni_macros(body)
        cited_keys = extract_citations_from_body(body)
        if fmt == "latex":
            for group in re.findall(
                r'\\(?:cite|Cite|citet|Citet|citep)\{([^}]+)\}',
                sections.get("full_text", ""),
            ):
                for k in group.split(','):
                    cited_keys.add(k.strip())

        has_numeric       = '__numeric_citations__' in cited_keys
        xcheck            = cross_check(bib_dict, cited_keys)
        citation_contexts = extract_citation_contexts(body)
        duplicates        = find_duplicates(bib_dict)
        self_citations    = detect_self_citations(bib_dict, body)

        # ── Per-reference verification with live SSE updates ──────────────────
        api_results_raw = []
        if verify and bib_dict:
            total = len(bib_dict)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from checker import verify_reference

            future_to_key = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                for key, entry in bib_dict.items():
                    future_to_key[executor.submit(verify_reference, entry)] = key

                done_count = 0
                for future in as_completed(future_to_key, timeout=120):
                    key = future_to_key[future]
                    try:
                        vr = future.result()
                    except Exception as e:
                        from checker import VerificationResult
                        vr = VerificationResult(key=key, title="", status="error",
                            confidence=0.0, note=f"Crashed: {e}", sources_checked=[])
                    api_results_raw.append(vr)
                    done_count += 1

                    yield _sse("progress", {
                        "step":    "verify",
                        "message": f"Verified {done_count}/{total}: [{vr.key}] → {vr.status}",
                        "key":     vr.key,
                        "status":  vr.status,
                        "confidence": round(vr.confidence, 2),
                        "done":    done_count,
                        "total":   total,
                    })

            # Restore original order
            key_order = list(bib_dict.keys())
            api_results_raw.sort(
                key=lambda r: key_order.index(r.key) if r.key in key_order else 999
            )

        yield _sse("progress", {"step": "ai_verify",
                                  "message": "AI hallucination check running…"})
        api_results_dicts   = _vr_to_dicts(api_results_raw)
        verification_result = ai_verify_references(bib_dicts, api_results_dicts)

        yield _sse("progress", {"step": "ai_verdict",
                                  "message": "AI generating overall verdict…"})
        summary_for_ai = {"duplicates": len(duplicates), "self_citations": len(self_citations),
                           "style_issues": len(style_suggestions)}
        overall = ai_overall_verdict(
            filename=filename or Path(main_path).name,
            summary=summary_for_ai, xcheck=xcheck,
            bib_list=bib_list, verification_result=verification_result,
        )

        result = _assemble_result(
            filename=filename or Path(main_path).name,
            fmt=fmt, body=body, bib_text=bib_text,
            bib_list=bib_list, bib_dict=bib_dict,
            cited_keys=cited_keys, has_numeric=has_numeric,
            xcheck=xcheck, citation_contexts=citation_contexts,
            duplicates=duplicates, self_citations=self_citations,
            style_suggestions=style_suggestions,
            api_results_raw=api_results_raw,
            verification_result=verification_result,
            overall=overall,
            ai_parse_improvements=ai_parse_improvements,
        )
        yield _sse("done", result)

    except Exception as e:
        import traceback
        yield _sse("error", {"error": str(e), "trace": traceback.format_exc()})
    finally:
        shutil.rmtree(Path(main_path).parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: build bib_dicts list
# ---------------------------------------------------------------------------

def _bib_to_dicts(bib_list: list) -> list:
    return [
        {"key": e.key, "entry_type": e.entry_type or "unknown",
         "authors": e.authors, "title": e.title, "year": e.year,
         "publisher": e.publisher, "journal": e.journal, "booktitle": e.booktitle,
         "pages": e.pages, "url": e.url, "urldate": e.urldate,
         "doi": e.doi, "isbn": e.isbn, "raw_text": e.raw_text[:300],
         "needs_ai_parsing": e.needs_ai_parsing, "key_consistent": e.key_consistent}
        for e in bib_list
    ]


def _vr_to_dicts(api_results_raw: list) -> list:
    return [
        {"key": vr.key, "status": vr.status, "confidence": round(vr.confidence, 2),
         "matched_title": vr.matched_title, "doi": vr.doi,
         "open_access_url": vr.open_access_url, "note": vr.note,
         "sources_checked": vr.sources_checked, "web_evidence": vr.web_evidence,
         "correct_authors": vr.correct_authors}
        for vr in api_results_raw
    ]


def _apply_ai_improvements(bib_list: list, improvements: dict) -> list:
    for entry in bib_list:
        imp = improvements.get(entry.key)
        if not imp:
            continue
        if not entry.title     and imp.get("title"):     entry.title     = imp["title"]
        if not entry.authors   and imp.get("authors"):   entry.authors   = imp["authors"]
        if not entry.year      and imp.get("year"):       entry.year      = imp["year"]
        if (not entry.entry_type or entry.entry_type == "unknown") and imp.get("entry_type"):
            entry.entry_type = imp["entry_type"]
        for field in ("journal", "booktitle", "publisher", "pages"):
            if not getattr(entry, field) and imp.get(field):
                setattr(entry, field, imp[field])
    return bib_list


# ---------------------------------------------------------------------------
# Helper: assemble the final result dict
# ---------------------------------------------------------------------------

def _assemble_result(
    filename, fmt, body, bib_text, bib_list, bib_dict,
    cited_keys, has_numeric, xcheck, citation_contexts,
    duplicates, self_citations, style_suggestions,
    api_results_raw, verification_result, overall, ai_parse_improvements,
):
    ai_verdicts_by_key = {v["key"]: v for v in verification_result.get("verdicts", [])}

    verification_output = []
    for vr in api_results_raw:
        ai = ai_verdicts_by_key.get(vr.key, {})
        ai_verdict = ai.get("verdict", "SUSPICIOUS")
        status = "verified" if ai_verdict == "REAL" else "not_found" if ai_verdict == "FAKE" else "partial_match"
        verification_output.append({
            "key":             vr.key,
            "title":           vr.title,
            "status":          status,
            "confidence":      round(ai.get("confidence", vr.confidence), 2),
            "matched_title":   vr.matched_title,
            "doi":             vr.doi or ai.get("open_access_url"),
            "open_access_url": ai.get("open_access_url") or vr.open_access_url,
            "note":            vr.note,
            "sources_checked": vr.sources_checked,
            "web_evidence":    vr.web_evidence,
            "ai_verdict":      ai_verdict,
            "ai_reasoning":    ai.get("reasoning", ""),
            "ai_risk_factors": ai.get("risk_factors", []),
        })

    api_keys = {vr.key for vr in api_results_raw}
    for entry in _bib_to_dicts(bib_list):
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

    ai_fake_count = verification_result.get("fake_count", 0)
    det_score = compute_score(bib_list, xcheck, api_results_raw,
                              style_suggestions, duplicates, ai_fake_count=ai_fake_count)
    final_score = {
        "score":            overall.get("score", det_score["score"]),
        "grade":            overall.get("grade", det_score["grade"]),
        "verdict":          overall.get("verdict", "FLAG"),
        "verdict_reason":   overall.get("verdict_reason", ""),
        "student_feedback": overall.get("student_feedback", []),
        "professor_note":   overall.get("professor_note", ""),
        "penalties":        det_score.get("penalties", []),
        "max_score":        100,
    }

    bib_output = [
        {"key": e.key, "type": e.entry_type or "unknown",
         "authors": e.authors, "title": e.title, "year": e.year,
         "publisher": e.publisher, "journal": e.journal, "url": e.url,
         "doi": e.doi, "isbn": e.isbn, "raw": e.raw_text[:250],
         "completeness_issues": e.completeness_issues,
         "key_consistent": e.key_consistent,
         "ai_reparsed": e.key in ai_parse_improvements}
        for e in bib_list
    ]

    real_cited = {k for k in cited_keys if not k.startswith('__')}

    return {
        "filename": filename, "format": fmt.upper(),
        "stats": {
            "body_chars": len(body), "bib_chars": len(bib_text),
            "bib_entry_count": len(bib_dict), "citation_count": len(real_cited),
            "numeric_citations_found": has_numeric,
        },
        "bibliography":    bib_output,
        "cross_check": {
            "correctly_used":   xcheck.correctly_used,
            "cited_not_in_bib": xcheck.cited_not_in_bib,
            "in_bib_not_cited": xcheck.in_bib_not_cited,
        },
        "citation_contexts":        citation_contexts,
        "style_suggestions":        style_suggestions,
        "duplicates":               duplicates,
        "self_citations":           self_citations,
        "score":                    final_score,
        "verification":             verification_output,
        "verification_ai_summary":  verification_result.get("summary", ""),
        "summary": {
            "missing_from_bib":   len(xcheck.cited_not_in_bib),
            "uncited_entries":    len(xcheck.in_bib_not_cited),
            "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
            "key_inconsistencies": sum(1 for e in bib_list if e.key_consistent is False),
            "fake_candidates":    verification_result.get("fake_count", 0),
            "suspicious":         verification_result.get("suspicious_count", 0),
            "verified":           sum(1 for v in verification_output if v["status"] == "verified"),
            "style_issues":       len(style_suggestions),
            "open_access":        sum(1 for v in verification_output if v.get("open_access_url")),
            "duplicates":         len(duplicates),
            "self_citations":     len(self_citations),
            "bib_entry_count":    len(bib_dict),
            "citation_count":     len(real_cited),
            "ai_reparsed_entries": len(ai_parse_improvements),
            "numeric_citations":  has_numeric,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/check", methods=["POST"])
def check():
    """
    SSE streaming endpoint.  The browser receives per-reference progress events
    in real time, then a final 'done' event with the full result.

    JavaScript usage:
        const es = new EventSource('/check');   // for GET demo
        // For POST with file use fetch + ReadableStream:
        const resp = await fetch('/check', {method:'POST', body: formData});
        const reader = resp.body.getReader();
        // read lines, parse 'event:' and 'data:' SSE lines
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    main_file = request.files["file"]
    filename  = main_file.filename
    ext       = Path(filename).suffix.lower()
    verify    = request.form.get("verify", "true").lower() == "true"
    streaming = request.form.get("stream", "true").lower() == "true"

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

    if streaming:
        # Return SSE stream — tmpdir is cleaned up inside the generator
        return Response(
            stream_with_context(
                _run_streaming_check(main_path, bib_path, verify=verify, filename=filename)
            ),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )
    else:
        # Non-streaming fallback (used by batch route internally)
        try:
            result = _run_full_check(main_path, bib_path, verify=verify, filename=filename)
            return jsonify(result)
        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/check-sync", methods=["POST"])
def check_sync():
    """Non-streaming fallback endpoint for clients that don't support SSE."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    main_file = request.files["file"]
    filename  = main_file.filename
    ext       = Path(filename).suffix.lower()
    verify    = request.form.get("verify", "true").lower() == "true"

    if ext not in {".pdf", ".docx", ".tex", ".latex"}:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

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
    """Manual AI audit button — Groq → Gemini fallback."""
    import requests as req

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No result data provided"}), 400

    s  = data.get("summary", {})
    sc = data.get("score", {})

    flagged = [v for v in data.get("verification", [])
               if v.get("status") in ("not_found", "partial_match", "error")
               or v.get("ai_verdict") in ("FAKE", "SUSPICIOUS")]
    incomplete = [e for e in data.get("bibliography", []) if e.get("completeness_issues")]
    key_issues = [e for e in data.get("bibliography", []) if e.get("key_consistent") is False]
    dupes    = data.get("duplicates", [])
    self_cit = data.get("self_citations", [])

    flagged_lines = "\n".join(
        f"  [{v['key']}] \"{v['title']}\" ai={v.get('ai_verdict','?')} "
        f"conf={int(v['confidence']*100)}% src={','.join(v.get('sources_checked',[]))}"
        for v in flagged
    ) or "  None"

    key_lines = "\n".join(
        f"  [{e['key']}]: " + "; ".join(i for i in e.get("completeness_issues",[]) if "key" in i.lower())
        for e in key_issues
    ) or "  None"

    prompt = f"""You are assisting a professor auditing a student's LNI reference list.

AUDIT SUMMARY:
- File: {data.get('filename','?')} | Score: {sc.get('score','?')}/100 | Verdict: {sc.get('verdict','?')}
- Bib entries: {s.get('bib_entry_count',0)} | Citations: {s.get('citation_count',0)}
- Missing from bib: {s.get('missing_from_bib',0)} | Never cited: {s.get('uncited_entries',0)}
- Incomplete: {s.get('incomplete_entries',0)} | Key mismatches: {s.get('key_inconsistencies',0)}
- Duplicates: {s.get('duplicates',0)} | Self-citations: {s.get('self_citations',0)}
- AI integrity: {data.get('verification_ai_summary','')}

LNI KEY-VS-METADATA MISMATCHES:
{key_lines}

FLAGGED REFERENCES:
{flagged_lines}

INCOMPLETE: {chr(10).join(f"  [{e['key']}] {e.get('title','?')} — {', '.join(e['completeness_issues'])}" for e in incomplete) or "  None"}
DUPLICATES: {chr(10).join(f"  [{d['key_a']}] vs [{d['key_b']}] {int(d['similarity']*100)}% similar" for d in dupes) or "  None"}
SELF-CITATIONS: {chr(10).join(f"  [{s_['key']}] {s_['matched_author']}" for s_ in self_cit) or "  None"}

TASK:
1. For each FLAGGED reference: REAL / SUSPICIOUS / FAKE — one-line reason
2. For each KEY MISMATCH: typo or fabrication?
3. OVERALL: PASS / FLAG / FAIL — one sentence

Format:
REFERENCE VERDICTS:
[key]: VERDICT — reason

KEY MISMATCHES:
[key]: comment

OVERALL: VERDICT — reason"""

    groq_key   = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    ai_text = ai_source = None

    if groq_key:
        try:
            resp = req.post(GROQ_URL if False else "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 700, "temperature": 0.1}, timeout=25)
            if resp.status_code == 200:
                ai_text   = resp.json()["choices"][0]["message"]["content"].strip()
                ai_source = "Groq (LLaMA 3.3 70B)"
        except Exception:
            pass

    if not ai_text and gemini_key:
        try:
            resp = req.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 700, "temperature": 0.1}}, timeout=25)
            if resp.status_code == 200:
                parts     = resp.json()["candidates"][0]["content"]["parts"]
                ai_text   = "".join(p.get("text","") for p in parts).strip()
                ai_source = "Gemini 1.5 Flash"
        except Exception:
            pass

    if not ai_text:
        missing = [k for k, v in [("GROQ_API_KEY", groq_key), ("GEMINI_API_KEY", gemini_key)] if not v]
        if missing:
            return jsonify({"error": f"Set {' or '.join(missing)} as env vars.",
                             "hint": "Groq: console.groq.com | Gemini: aistudio.google.com"}), 503
        return jsonify({"error": "Both Groq and Gemini failed."}), 503

    return jsonify({"verdict": ai_text, "ai_source": ai_source, "flagged_count": len(flagged)})


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
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data"}), 400

    sc = data.get("score", {})
    s  = data.get("summary", {})
    lines = [
        "=" * 70,
        "LNI REFERENCE CHECKER v5 — PROFESSOR REPORT",
        "=" * 70,
        f"File    : {data.get('filename','?')}",
        f"Format  : {data.get('format','?')}",
        f"Score   : {sc.get('score','?')}/100  Grade: {sc.get('grade','?')}  Verdict: {sc.get('verdict','?')}",
        f"Reason  : {sc.get('verdict_reason','')}",
        "",
        "── SUMMARY ──",
        f"  Bibliography entries    : {s.get('bib_entry_count',0)}",
        f"  In-text citations       : {s.get('citation_count',0)}",
        f"  Missing from bib        : {s.get('missing_from_bib',0)}",
        f"  Never cited (orphaned)  : {s.get('uncited_entries',0)}",
        f"  Incomplete entries      : {s.get('incomplete_entries',0)}",
        f"  Key-vs-metadata errors  : {s.get('key_inconsistencies',0)}",
        f"  FAKE (AI verdict)       : {s.get('fake_candidates',0)}",
        f"  SUSPICIOUS              : {s.get('suspicious',0)}",
        f"  Verified REAL           : {s.get('verified',0)}",
        f"  Duplicates              : {s.get('duplicates',0)}",
        f"  Self-citations          : {s.get('self_citations',0)}",
        f"  Style issues            : {s.get('style_issues',0)}",
        f"  Open-access links       : {s.get('open_access',0)}",
        f"  Entries AI-reparsed     : {s.get('ai_reparsed_entries',0)}",
    ]
    if s.get("numeric_citations"):
        lines.append("  ⚠ Numeric citations [1] detected — LNI requires [Author+Year]")
    lines.append("")

    if sc.get("student_feedback"):
        lines.append("── FEEDBACK FOR STUDENT ──")
        for fb in sc["student_feedback"]:
            lines.append(f"  • {fb}")
        lines.append("")

    if sc.get("professor_note"):
        lines += ["── NOTE FOR PROFESSOR ──", f"  {sc['professor_note']}", ""]

    if sc.get("penalties"):
        lines.append("── SCORE BREAKDOWN ──")
        for p in sc["penalties"]:
            lines.append(f"  -{p['deduction']:2d}  {p['category']} ({p['count']}×)")
        lines.append("")

    key_issues = [e for e in data.get("bibliography",[]) if e.get("key_consistent") is False]
    if key_issues:
        lines.append("── LNI KEY-VS-METADATA MISMATCHES ──")
        for e in key_issues:
            lines.append(f"  [{e['key']}] {e.get('title','?')}")
            for issue in e.get("completeness_issues",[]):
                if "key" in issue.lower():
                    lines.append(f"    ⚠ {issue}")
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

    fake_refs = [v for v in data.get("verification",[]) if v.get("ai_verdict") == "FAKE"]
    if fake_refs:
        lines.append("── LIKELY FAKE REFERENCES (AI) ──")
        for v in fake_refs:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    AI: {v.get('ai_reasoning','')}")
            for rf in v.get("ai_risk_factors",[]):
                lines.append(f"    ⚠ {rf}")
            lines.append(f"    Sources: {', '.join(v.get('sources_checked',[]))}")
        lines.append("")

    suspicious_refs = [v for v in data.get("verification",[]) if v.get("ai_verdict") == "SUSPICIOUS"]
    if suspicious_refs:
        lines.append("── SUSPICIOUS REFERENCES ──")
        for v in suspicious_refs:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    {v.get('ai_reasoning','')}")
        lines.append("")

    incomplete = [e for e in data.get("bibliography",[]) if e.get("completeness_issues")]
    if incomplete:
        lines.append("── INCOMPLETE ENTRIES ──")
        for e in incomplete:
            lines.append(f"  [{e['key']}] {e.get('title','?')}")
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
        for ss in data["style_suggestions"]:
            lines.append(f"  {ss['message']} ({ss['count']}×)")
        lines.append("")

    if data.get("verification_ai_summary"):
        lines += ["── AI INTEGRITY SUMMARY ──", f"  {data['verification_ai_summary']}", ""]

    lines += [
        "=" * 70,
        "Generated by LNI Reference Checker v5",
        "Deterministic: key format, key-consistency, fields, xcheck, dupes, style",
        "AI (Groq/Gemini): bibliography extraction, re-parsing, fake detection, verdict",
        "APIs: CrossRef · SS · OpenAlex · arXiv · DBLP · ACL Anthology",
        "      OpenReview · Open Library · GitHub · Scholar · DuckDuckGo",
        "=" * 70,
    ]

    report = "\n".join(lines)
    fname  = data.get("filename","report").replace(" ","_") + "_lni_report.txt"
    return Response(report, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


if __name__ == "__main__":
    print("\n  ┌──────────────────────────────────────────────────────────┐")
    print("  │   LNI Reference Checker v5                               │")
    print("  │   http://localhost:5000                                  │")
    print("  │                                                          │")
    print("  │   Deterministic: key · key-consistency · fields          │")
    print("  │                  xcheck · dupes · style                  │")
    print("  │   AI (Groq→Gemini): extract · re-parse · fake · verdict │")
    print("  │   APIs: CrossRef · SS · OpenAlex · arXiv · DBLP         │")
    print("  │          ACL · OpenReview · OpenLibrary · GitHub         │")
    print("  │   Cache: LNI_CACHE_DIR (disk) + in-memory               │")
    print("  └──────────────────────────────────────────────────────────┘\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)