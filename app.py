"""
Flask Web Server — LNI Reference Checker v6.1
============================================
IMPROVEMENTS in v6.1:
  - Better PDF extraction with fallback methods and user warnings
  - Improved error messages with actionable advice
  - Upload validation with immediate feedback
  - Session timeout handling for long-running checks
  - Graceful degradation when services unavailable
"""

import os
import re
import sys
from typing import Optional

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import tempfile
import shutil
import time
import signal
from pathlib import Path
from functools import wraps
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
    get_llm_cache_stats,
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
app.config["TIMEOUT"] = 180  # 3 minutes timeout for long operations

# Ensure both databases are initialized at startup
from local_db import init_cache_db
from review_queue import init_review_db
try:
    init_cache_db()
    init_review_db()
except Exception as _db_init_err:
    print(f"Warning: DB init error (non-fatal): {_db_init_err}")


# ---------------------------------------------------------------------------
# Timeout decorator for long-running operations
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass

def timeout(seconds=180):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function timed out after {seconds} seconds")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if sys.platform != 'win32':
                # Set timeout handler (Unix only)
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            else:
                # Windows doesn't support SIGALRM
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status", methods=["GET"])
def status():
    """Returns server status including session cache metrics and API availability."""
    from checker import _ARXIV_BIBTEX_MEM_CACHE, _MEM_CACHE
    llm_stats = get_llm_cache_stats()
    
    # Check API availability
    groq_available = bool(os.environ.get("GROQ_API_KEY"))
    gemini_available = bool(os.environ.get("GEMINI_API_KEY"))
    semantic_available = bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    
    return jsonify({
        "status": "ok",
        "version": "6.1",
        "cache": {
            "llm_response_cache_entries": llm_stats["llm_cache_entries"],
            "arxiv_bibtex_cache_entries": len(_ARXIV_BIBTEX_MEM_CACHE),
            "verification_result_cache_entries": len(_MEM_CACHE),
        },
        "ai_available": groq_available or gemini_available,
        "ai_provider": "groq" if groq_available else "gemini" if gemini_available else "none",
        "apis": {
            "groq": groq_available,
            "gemini": gemini_available,
            "semantic_scholar": semantic_available,
            "github": bool(os.environ.get("GITHUB_TOKEN")),
            "unpaywall": bool(os.environ.get("UNPAYWALL_EMAIL")),
        },
        "env": {
            "disk_cache_dir": os.environ.get("LNI_CACHE_DIR", ".lni_cache"),
        },
    })


# ---------------------------------------------------------------------------
# Core pipeline helpers
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



def _check_year_mismatch(vr, bib_dict: dict) -> Optional[int]:
    """Return year difference if cited year differs from CrossRef year by >1, else None."""
    corrected_year = getattr(vr, "corrected_year", None)
    entry = bib_dict.get(vr.key)
    if corrected_year and entry and entry.year:
        try:
            diff = int(corrected_year) - int(entry.year)
            return diff if abs(diff) > 1 else None
        except (ValueError, TypeError):
            pass
    return None


def _check_author_overlap(vr, bib_dict: dict) -> Optional[float]:
    """Return author overlap score (0-1) if corrected authors available, else None."""
    corrected_authors = getattr(vr, "corrected_authors", None)
    entry = bib_dict.get(vr.key)
    if corrected_authors and entry and entry.authors:
        from checker import author_overlap_score
        return author_overlap_score(entry.authors, corrected_authors)
    return None


def _vr_to_dicts(api_results_raw: list) -> list:
    return [
        {"key": vr.key, "status": vr.status, "confidence": round(vr.confidence, 2),
         "matched_title": vr.matched_title, "doi": vr.doi,
         "open_access_url": vr.open_access_url, "note": vr.note,
         "sources_checked": vr.sources_checked, "web_evidence": vr.web_evidence,
         "correct_authors": vr.correct_authors,
         "version_note": vr.version_note,
         "is_retracted": getattr(vr, "is_retracted", False),
         "retraction_doi": getattr(vr, "retraction_doi", None),
         "retraction_note": getattr(vr, "retraction_note", None),
         "corrected_title": getattr(vr, "corrected_title", None),
         "corrected_authors": getattr(vr, "corrected_authors", None),
         "corrected_year": getattr(vr, "corrected_year", None),
         "corrected_journal": getattr(vr, "corrected_journal", None),
         "corrected_publisher": getattr(vr, "corrected_publisher", None),
         "corrected_volume": getattr(vr, "corrected_volume", None),
         "corrected_pages": getattr(vr, "corrected_pages", None),
         "year_mismatch": _check_year_mismatch(vr, {}),
         "author_overlap": _check_author_overlap(vr, {}),
         }
        for vr in api_results_raw
    ]


def _apply_ai_improvements(bib_list: list, improvements: dict) -> list:
    for entry in bib_list:
        imp = improvements.get(entry.key)
        if not imp:
            continue
        if not entry.title and imp.get("title"):
            entry.title = imp["title"]
        if not entry.authors and imp.get("authors"):
            entry.authors = imp["authors"]
        if not entry.year and imp.get("year"):
            entry.year = imp["year"]
        if (not entry.entry_type or entry.entry_type == "unknown") and imp.get("entry_type"):
            entry.entry_type = imp["entry_type"]
        for field in ("journal", "booktitle", "publisher", "pages"):
            if not getattr(entry, field) and imp.get(field):
                setattr(entry, field, imp[field])
    return bib_list


def _assemble_result(
    filename, fmt, body, bib_text, bib_list, bib_dict,
    cited_keys, has_numeric, xcheck, citation_contexts,
    duplicates, self_citations, style_suggestions,
    api_results_raw, verification_result, overall, ai_parse_improvements,
    is_scanned=False,
):
    ai_verdicts_by_key = {v["key"]: v for v in verification_result.get("verdicts", [])}
    vr_by_key = {vr.key: vr for vr in api_results_raw}

    verification_output = []
    for vr in api_results_raw:
        ai = ai_verdicts_by_key.get(vr.key, {})
        ai_verdict = ai.get("verdict", "SUSPICIOUS")
        status = "verified" if ai_verdict == "REAL" else "not_found" if ai_verdict == "FAKE" else "partial_match"
        verification_output.append({
            "key": vr.key,
            "title": vr.title,
            "status": status,
            "confidence": round(ai.get("confidence", vr.confidence), 2),
            "matched_title": vr.matched_title,
            "doi": vr.doi or ai.get("open_access_url"),
            "open_access_url": ai.get("open_access_url") or vr.open_access_url,
            "note": vr.note,
            "sources_checked": vr.sources_checked,
            "web_evidence": vr.web_evidence,
            "ai_verdict": ai_verdict,
            "ai_reasoning": ai.get("reasoning", ""),
            "ai_risk_factors": ai.get("risk_factors", []),
            "version_note": vr.version_note,
            "is_retracted": getattr(vr, "is_retracted", False),
            "retraction_doi": getattr(vr, "retraction_doi", None),
            "retraction_note": getattr(vr, "retraction_note", None),
            "corrected_title": getattr(vr, "corrected_title", None),
            "corrected_authors": getattr(vr, "corrected_authors", None),
            "corrected_year": getattr(vr, "corrected_year", None),
            "corrected_journal": getattr(vr, "corrected_journal", None),
            "corrected_publisher": getattr(vr, "corrected_publisher", None),
            "corrected_volume": getattr(vr, "corrected_volume", None),
            "corrected_pages": getattr(vr, "corrected_pages", None),
        })

    api_keys = {vr.key for vr in api_results_raw}
    for entry in _bib_to_dicts(bib_list):
        if entry["key"] not in api_keys:
            ai = ai_verdicts_by_key.get(entry["key"], {})
            ai_verdict = ai.get("verdict", "SUSPICIOUS")
            verification_output.append({
                "key": entry["key"],
                "title": entry.get("title") or "",
                "status": "verified" if ai_verdict == "REAL" else "not_found" if ai_verdict == "FAKE" else "partial_match",
                "confidence": ai.get("confidence", 0.5),
                "matched_title": None,
                "doi": entry.get("doi"),
                "open_access_url": ai.get("open_access_url") or entry.get("url"),
                "note": ai.get("reasoning", ""),
                "sources_checked": [],
                "web_evidence": None,
                "ai_verdict": ai_verdict,
                "ai_reasoning": ai.get("reasoning", ""),
                "ai_risk_factors": ai.get("risk_factors", []),
                "version_note": None,
            })

    ai_fake_count = verification_result.get("fake_count", 0)

    # Count retracted, year mismatches, author mismatches from verification output
    retracted_count = 0
    year_mismatches = 0
    author_mismatches = 0
    for vr in api_results_raw:
        if getattr(vr, "is_retracted", False):
            retracted_count += 1
        # Year mismatch: cited year vs CrossRef corrected year, diff > 1
        corrected_year = getattr(vr, "corrected_year", None)
        bib_entry = bib_dict.get(vr.key)
        if corrected_year and bib_entry and bib_entry.year:
            try:
                diff = abs(int(corrected_year) - int(bib_entry.year))
                if diff > 1:
                    year_mismatches += 1
            except (ValueError, TypeError):
                pass
        # Author mismatch: overlap score below 0.4 when we have corrected authors
        corrected_authors = getattr(vr, "corrected_authors", None)
        if corrected_authors and bib_entry and bib_entry.authors:
            from checker import author_overlap_score
            overlap = author_overlap_score(bib_entry.authors, corrected_authors)
            if overlap is not None and overlap < 0.4:
                author_mismatches += 1

    det_score = compute_score(bib_list, xcheck, api_results_raw,
                              style_suggestions, duplicates,
                              ai_fake_count=ai_fake_count,
                              retracted_count=retracted_count,
                              year_mismatches=year_mismatches,
                              author_mismatches=author_mismatches)
    s = det_score["score"]
    det_verdict = "PASS" if s >= 75 else "FLAG" if s >= 50 else "FAIL"

    final_score = {
        "score": det_score["score"],
        "grade": det_score["grade"],
        "verdict": overall.get("verdict", det_verdict),
        "verdict_reason": overall.get("verdict_reason", ""),
        "student_feedback": overall.get("student_feedback", []),
        "professor_note": overall.get("professor_note", ""),
        "penalties": det_score.get("penalties", []),
        "max_score": 100,
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

    real_cited = set()
    for k in cited_keys:
        k_str = str(k)
        if not k_str.startswith('__'):
            real_cited.add(k_str)
        elif k_str.startswith('__NUM_'):
            num = k_str.replace('__NUM_', '').replace('__', '')
            real_cited.add(num)
    version_notes = [
        {"key": v["key"], "note": v["version_note"]}
        for v in verification_output if v.get("version_note")
    ]

    return {
        "filename": filename, "format": fmt.upper(),
        "stats": {
            "body_chars": len(body), "bib_chars": len(bib_text),
            "bib_entry_count": len(bib_dict), "citation_count": len(real_cited),
            "numeric_citations_found": has_numeric,
        },
        "bibliography": bib_output,
        "cross_check": {
            "correctly_used": xcheck.correctly_used,
            "cited_not_in_bib": xcheck.cited_not_in_bib,
            "in_bib_not_cited": xcheck.in_bib_not_cited,
        },
        "citation_contexts": citation_contexts,
        "style_suggestions": style_suggestions,
        "duplicates": duplicates,
        "self_citations": self_citations,
        "score": final_score,
        "verification": verification_output,
        "verification_ai_summary": verification_result.get("summary", ""),
        "arxiv_version_notes": version_notes,
        "is_scanned": bool(is_scanned),
        "summary": {
            "missing_from_bib": len(xcheck.cited_not_in_bib),
            "uncited_entries": len(xcheck.in_bib_not_cited),
            "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
            "key_inconsistencies": sum(1 for e in bib_list if e.key_consistent is False),
            "fake_candidates": verification_result.get("fake_count", 0),
            "suspicious": verification_result.get("suspicious_count", 0),
            "verified": sum(1 for v in verification_output if v["status"] == "verified"),
            "retracted": sum(1 for v in verification_output if v.get("is_retracted")),
            "style_issues": len(style_suggestions),
            "open_access": sum(1 for v in verification_output if v.get("open_access_url")),
            "duplicates": len(duplicates),
            "self_citations": len(self_citations),
            "bib_entry_count": len(bib_dict),
            "citation_count": len(real_cited),
            "ai_reparsed_entries": len(ai_parse_improvements),
            "numeric_citations": has_numeric,
            "arxiv_version_mismatches": len(version_notes),
        },
    }


# ---------------------------------------------------------------------------
# SSE streaming pipeline with improved error handling
# ---------------------------------------------------------------------------

def _run_streaming_check(main_path: str, bib_path: str = None,
                          verify: bool = True, filename: str = ""):
    """
    Generator that yields Server-Sent Event strings with progress updates.
    """
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    start_time = time.time()
    
    try:
        yield _sse("progress", {"step": "extract", "message": "📄 Extracting text from document..."})

        sections = extract(main_path, bib_path)
        body = sections.get("body", "")
        bib_text = sections.get("bibliography", "")
        fmt = sections.get("format", "unknown")
        
        # Check for PDF extraction warnings
        if sections.get("warning"):
            yield _sse("progress", {"step": "warning", "message": sections["warning"]})
        
        if sections.get("is_scanned"):
            yield _sse("progress", {"step": "warning", 
                "message": "⚠️ This appears to be a SCANNED PDF. Only text-based PDFs work. Please upload the original document if possible."})
        
        if len(body) < 200:
            yield _sse("progress", {"step": "warning",
                "message": "⚠️ Very little text extracted. This may be a scanned/image-only PDF, or the document uses unusual formatting. Results may be incomplete."})
            
            # Suggest alternative if body is extremely short
            if len(body) < 50:
                yield _sse("progress", {"step": "warning",
                    "message": "❌ Almost no text could be extracted. For best results, upload the original LaTeX source, Word document, or a text-based PDF (not scanned)."})
        
        # Show extraction stats
        yield _sse("progress", {"step": "extract_done", 
            "message": f"✓ Extracted {len(body):,} characters of body text, {len(bib_text):,} characters of bibliography"})
        
        if not bib_text.strip():
            yield _sse("progress", {"step": "warning",
                "message": "⚠️ No bibliography section detected. Make sure your document has a 'Literaturverzeichnis' or 'References' section."})
        
        yield _sse("progress", {"step": "ai_extract",
            "message": "🤖 AI extracting bibliography structure (may take 10-20 seconds)..."})
        ai_extracted_refs = ai_extract_references_from_text(bib_text)

        yield _sse("progress", {"step": "parse",
            "message": "📚 Parsing bibliography entries..."})
        bib_list = parse_bibliography(bib_text)
        bib_dict = entries_to_dict(bib_list)
        
        yield _sse("progress", {"step": "parse_done",
            "message": f"✓ Found {len(bib_list)} bibliography entries"})

        if ai_extracted_refs:
            bib_list = merge_ai_extractions_into_bib_list(ai_extracted_refs, bib_list)
            bib_dict = entries_to_dict(bib_list)

        bib_dicts = _bib_to_dicts(bib_list)
        ai_parse_improvements = ai_parse_uncertain_entries(bib_dicts)
        if ai_parse_improvements:
            bib_list = _apply_ai_improvements(bib_list, ai_parse_improvements)
            bib_dict = entries_to_dict(bib_list)
            bib_dicts = _bib_to_dicts(bib_list)
            yield _sse("progress", {"step": "parse_done",
                "message": f"✓ AI re-parsed {len(ai_parse_improvements)} uncertain entries"})

        yield _sse("progress", {"step": "check",
            "message": f"🔍 Running deterministic checks on {len(bib_list)} entries..."})
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
        xcheck = cross_check(bib_dict, cited_keys)
        citation_contexts = extract_citation_contexts(body)
        duplicates = find_duplicates(bib_dict)
        self_citations = detect_self_citations(bib_dict, body)

        # Report cross-check results
        missing_count = len(xcheck.cited_not_in_bib)
        orphaned_count = len(xcheck.in_bib_not_cited)
        if missing_count > 0:
            yield _sse("progress", {"step": "check_result",
                "message": f"⚠️ Found {missing_count} citation(s) missing from bibliography"})
        if orphaned_count > 0:
            yield _sse("progress", {"step": "check_result",
                "message": f"⚠️ Found {orphaned_count} bibliography entry(s) never cited"})

        # Per-reference verification with live SSE updates
        api_results_raw = []
        if verify and bib_dict:
            total = len(bib_dict)
            yield _sse("progress", {"step": "verify_start",
                "message": f"🔍 Verifying {total} references against academic databases (CrossRef, Semantic Scholar, arXiv, DBLP, etc.)..."})
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from checker import verify_reference, VerificationResult

            future_to_key = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                for key, entry in bib_dict.items():
                    future_to_key[executor.submit(verify_reference, entry)] = key

                done_count = 0
                verified_count = 0
                not_found_count = 0
                
                for future in as_completed(future_to_key, timeout=120):
                    key = future_to_key[future]
                    try:
                        vr = future.result()
                    except Exception as e:
                        vr = VerificationResult(key=key, title="", status="error",
                            confidence=0.0, note=f"Crashed: {e}", sources_checked=[])
                    api_results_raw.append(vr)
                    done_count += 1
                    
                    if vr.status == "verified":
                        verified_count += 1
                    elif vr.status in ("not_found", "partial_match"):
                        not_found_count += 1

                    progress_data = {
                        "step": "verify",
                        "message": f"Verifying references: {done_count}/{total}",
                        "key": vr.key,
                        "status": vr.status,
                        "confidence": round(vr.confidence, 2),
                        "done": done_count,
                        "total": total,
                        "verified_count": verified_count,
                        "not_found_count": not_found_count,
                    }
                    if vr.version_note:
                        progress_data["version_note"] = vr.version_note
                    yield _sse("progress", progress_data)

            key_order = list(bib_dict.keys())
            api_results_raw.sort(
                key=lambda r: key_order.index(r.key) if r.key in key_order else 999
            )
            
            yield _sse("progress", {"step": "verify_done",
                "message": f"✓ Verification complete: {verified_count} verified, {not_found_count} not found/partial"})

        yield _sse("progress", {"step": "ai_verify",
            "message": "🤖 AI hallucination check running..."})
        api_results_dicts = _vr_to_dicts(api_results_raw)
        verification_result = ai_verify_references(bib_dicts, api_results_dicts)
        
        fake_count = verification_result.get("fake_count", 0)
        if fake_count > 0:
            yield _sse("progress", {"step": "ai_result",
                "message": f"⚠️ AI identified {fake_count} likely FAKE reference(s)"})

        yield _sse("progress", {"step": "ai_verdict",
            "message": "📋 Generating final verdict..."})
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
            is_scanned=bool(sections.get("is_scanned")),
        )
        
        elapsed = time.time() - start_time
        result["processing_time_seconds"] = round(elapsed, 1)
        
        yield _sse("done", result)

    except TimeoutError:
        yield _sse("error", {"error": "Processing timed out after 3 minutes. The document may be too large or complex. Try a simpler format (plain PDF or DOCX)."})
    except Exception as e:
        import traceback
        error_msg = str(e)
        # Provide user-friendly error messages
        if "pdfplumber" in error_msg.lower():
            error_msg = "PDF parsing failed. The PDF may be corrupted or use an unsupported encoding. Try converting to PDF from Word or LaTeX."
        elif "memory" in error_msg.lower():
            error_msg = "Out of memory. The document is too large. Try splitting it into smaller files."
        elif "timeout" in error_msg.lower():
            error_msg = "Request timed out. The document may be too large or complex."
        
        yield _sse("error", {"error": error_msg, "trace": traceback.format_exc()})
    finally:
        shutil.rmtree(Path(main_path).parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/check", methods=["POST"])
def check():
    """SSE streaming endpoint. Returns per-reference progress events then a 'done' event."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a PDF, DOCX, or TEX file."}), 400

    main_file = request.files["file"]
    filename = main_file.filename
    ext = Path(filename).suffix.lower()
    verify = request.form.get("verify", "true").lower() == "true"
    streaming = request.form.get("stream", "true").lower() == "true"

    # Validate file type
    if ext not in {".pdf", ".docx", ".tex", ".latex"}:
        return jsonify({"error": f"Unsupported format '{ext}'. Supported formats: PDF, DOCX, TEX, LATEX"}), 400
    
    # Validate file size
    main_file.seek(0, 2)
    file_size = main_file.tell()
    main_file.seek(0)
    
    if file_size > 30 * 1024 * 1024:
        return jsonify({"error": f"File too large: {file_size / 1024 / 1024:.1f} MB. Maximum size is 30 MB."}), 400
    
    if file_size == 0:
        return jsonify({"error": "Empty file uploaded. Please select a valid document."}), 400

    tmpdir = tempfile.mkdtemp()
    main_path = os.path.join(tmpdir, filename)
    main_file.save(main_path)

    bib_path = None
    if "bib" in request.files and ext in (".tex", ".latex"):
        bib_file = request.files["bib"]
        if bib_file.filename and bib_file.filename.endswith('.bib'):
            bib_path = os.path.join(tmpdir, bib_file.filename)
            bib_file.save(bib_path)

    if streaming:
        return Response(
            stream_with_context(
                _run_streaming_check(main_path, bib_path, verify=verify, filename=filename)
            ),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        try:
            result = _run_full_check(main_path, bib_path, verify=verify, filename=filename)
            return jsonify(result)
        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_full_check(main_path: str, bib_path: str = None,
                    verify: bool = True, filename: str = "") -> dict:
    """Full check pipeline (non-streaming, for batch and export)."""
    sections = extract(main_path, bib_path)
    body = sections["body"]
    bib_text = sections["bibliography"]
    fmt = sections["format"]

    ai_extracted_refs = ai_extract_references_from_text(bib_text)

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
    xcheck = cross_check(bib_dict, cited_keys)
    citation_contexts = extract_citation_contexts(body)
    duplicates = find_duplicates(bib_dict)
    self_citations = detect_self_citations(bib_dict, body)

    api_results_raw = []
    if verify and bib_dict:
        api_results_raw = verify_all_references(bib_dict)

    api_results_dicts = _vr_to_dicts(api_results_raw)
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
        is_scanned=bool(sections.get("is_scanned")),
    )

@app.route("/check-stream", methods=["POST"])
def check_stream():
    """Alias for streaming check endpoint."""
    return check()

@app.route("/check-sync", methods=["POST"])
def check_sync():
    """Non-streaming fallback endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    main_file = request.files["file"]
    filename = main_file.filename
    ext = Path(filename).suffix.lower()
    verify = request.form.get("verify", "true").lower() == "true"

    if ext not in {".pdf", ".docx", ".tex", ".latex"}:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

    tmpdir = tempfile.mkdtemp()
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

    s = data.get("summary", {})
    sc = data.get("score", {})

    flagged = [v for v in data.get("verification", [])
               if v.get("status") in ("not_found", "partial_match", "error")
               or v.get("ai_verdict") in ("FAKE", "SUSPICIOUS")]
    incomplete = [e for e in data.get("bibliography", []) if e.get("completeness_issues")]
    key_issues = [e for e in data.get("bibliography", []) if e.get("key_consistent") is False]
    dupes = data.get("duplicates", [])
    self_cit = data.get("self_citations", [])

    flagged_lines = "\n".join(
        f"  [{v['key']}] \"{v['title']}\" ai={v.get('ai_verdict','?')} "
        f"conf={int(v['confidence']*100)}% src={','.join(v.get('sources_checked',[]))}"
        + (f"\n    ℹ {v['version_note']}" if v.get('version_note') else "")
        for v in flagged
    ) or "  None"

    key_lines = "\n".join(
        f"  [{e['key']}]: " + "; ".join(i for i in e.get("completeness_issues", []) if "key" in i.lower())
        for e in key_issues
    ) or "  None"

    prompt = f"""You are assisting a professor auditing a student's LNI reference list.

AUDIT SUMMARY:
- File: {data.get('filename','?')} | Score: {sc.get('score','?')}/100 | Verdict: {sc.get('verdict','?')}
- Bib entries: {s.get('bib_entry_count',0)} | Citations: {s.get('citation_count',0)}
- Missing from bib: {s.get('missing_from_bib',0)} | Never cited: {s.get('uncited_entries',0)}
- Incomplete: {s.get('incomplete_entries',0)} | Key mismatches: {s.get('key_inconsistencies',0)}
- Duplicates: {s.get('duplicates',0)} | Self-citations: {s.get('self_citations',0)}
- arXiv version mismatches (non-error): {s.get('arxiv_version_mismatches',0)}
- AI integrity: {data.get('verification_ai_summary','')}

LNI KEY-VS-METADATA MISMATCHES:
{key_lines}

FLAGGED REFERENCES:
{flagged_lines}

INCOMPLETE: {chr(10).join(f"  [{e['key']}] {e.get('title','?')} — {', '.join(e['completeness_issues'])}" for e in incomplete) or "  None"}
DUPLICATES: {chr(10).join(f"  [{d['key_a']}] vs [{d['key_b']}] {int(d['similarity']*100)}% similar" for d in dupes) or "  None"}
SELF-CITATIONS: {chr(10).join(f"  [{s_['key']}] {s_['matched_author']}" for s_ in self_cit) or "  None"}

Return JSON with verdict and reasoning."""

    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    ai_text = ai_source = None

    if groq_key:
        try:
            resp = req.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 700, "temperature": 0.1}, timeout=25)
            if resp.status_code == 200:
                ai_text = resp.json()["choices"][0]["message"]["content"].strip()
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
                parts = resp.json()["candidates"][0]["content"]["parts"]
                ai_text = "".join(p.get("text", "") for p in parts).strip()
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



@app.route("/api/export-bibtex", methods=["POST"])
def export_bibtex():
    """Generate corrected BibTeX for references that have database-corrected metadata."""
    data = request.get_json()
    verification = data.get("verification", [])
    bibliography = data.get("bibliography", [])

    bib_by_key = {b["key"]: b for b in bibliography}
    lines = [
        "% LNI Reference Checker — Corrected BibTeX Export",
        "% Generated for references where database metadata was found.",
        "% Review each entry — some fields may need manual adjustment.",
        "",
    ]

    exported = 0
    for v in verification:
        key = v.get("key", "")
        orig = bib_by_key.get(key, {})
        entry_type = orig.get("type") or orig.get("entry_type") or "misc"
        entry_type = entry_type.lower().replace("@", "")

        # Only export if we have corrected data or a retraction
        has_correction = any(v.get(f) for f in (
            "corrected_title", "corrected_authors", "corrected_year",
            "corrected_journal", "corrected_publisher", "corrected_volume", "corrected_pages"
        ))
        is_retracted = v.get("is_retracted", False)

        if not has_correction and not is_retracted:
            continue

        if is_retracted:
            lines.append(f"% ⚠ RETRACTED: {v.get('retraction_note', 'See CrossRef for retraction notice')}")

        title  = v.get("corrected_title")  or orig.get("title", "")
        authors= v.get("corrected_authors")or orig.get("authors", "")
        year   = v.get("corrected_year")   or orig.get("year", "")
        journal= v.get("corrected_journal")or orig.get("journal", "")
        publisher = v.get("corrected_publisher") or orig.get("publisher", "")
        volume = v.get("corrected_volume") or orig.get("volume", "")
        pages  = v.get("corrected_pages")  or orig.get("pages", "")
        doi    = v.get("doi") or orig.get("doi", "")
        url    = v.get("open_access_url") or orig.get("url", "")

        lines.append(f"@{entry_type}{{{key},")
        if title:    lines.append(f"  title     = {{{{{title}}}}},")
        if authors:  lines.append(f"  author    = {{{authors}}},")
        if year:     lines.append(f"  year      = {{{year}}},")
        if journal:  lines.append(f"  journal   = {{{{{journal}}}}},")
        if publisher:lines.append(f"  publisher = {{{{{publisher}}}}},")
        if volume:   lines.append(f"  volume    = {{{volume}}},")
        if pages:    lines.append(f"  pages     = {{{pages}}},")
        if doi:      lines.append(f"  doi       = {{{doi}}},")
        if url:      lines.append(f"  url       = {{{url}}},")
        lines.append("}\n")
        exported += 1

    if exported == 0:
        return jsonify({"error": "No entries with database-corrected metadata found. Run with 'Verify references online' enabled."}), 400

    bibtex_str = "\n".join(lines)
    from flask import Response
    return Response(
        bibtex_str,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=corrected_references.bib"}
    )


@app.route("/batch", methods=["POST"])
def batch_check():
    uploaded = request.files.getlist("files")
    verify = request.form.get("verify", "true").lower() == "true"

    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    if len(uploaded) > 50:
        return jsonify({"error": "Maximum 50 files per batch. Please reduce the number of files."}), 400

    results = []
    tmpdir = tempfile.mkdtemp()

    try:
        for idx, main_file in enumerate(uploaded):
            filename = main_file.filename
            ext = Path(filename).suffix.lower()
            if ext not in {".pdf", ".docx", ".tex", ".latex"}:
                results.append({"filename": filename, "error": f"Unsupported format '{ext}'"})
                continue

            file_path = os.path.join(tmpdir, f"{idx}_{filename}")
            main_file.save(file_path)

            try:
                result = _run_full_check(file_path, verify=verify, filename=filename)
                results.append({
                    "filename": filename,
                    "format": result["format"],
                    "score": result["score"],
                    "summary": result["summary"],
                    "flagged_refs": [
                        v["key"] for v in result["verification"]
                        if v.get("ai_verdict") in ("FAKE", "SUSPICIOUS")
                        or v.get("status") in ("not_found", "partial_match")
                    ],
                    "arxiv_version_notes": result.get("arxiv_version_notes", []),
                })
            except Exception as e:
                import traceback
                results.append({"filename": filename, "error": str(e)})

        results.sort(key=lambda r: r.get("score", {}).get("score", -1), reverse=True)
        return jsonify({"files": results, "count": len(results)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.route("/api/review", methods=["POST"])
def submit_review():
    """Professor submits a manual review decision"""
    from review_queue import add_review_decision, add_false_positive
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    key = data.get("key")
    decision = data.get("decision")  # 'verified' or 'rejected'
    title = data.get("title", "")
    url = data.get("url", "")
    note = data.get("note", "")
    
    # ai_verdict lets add_review_decision auto-record a false_positive
    ai_verdict = data.get("ai_verdict", "")   # "FAKE" / "SUSPICIOUS" from frontend
    authors    = data.get("authors", "")

    success = add_review_decision(
        title=title or f"Paper_{key}",
        authors=authors,
        decision=decision,
        note=note,
        verified_url=url,
        ai_had_said=ai_verdict,
    )

    return jsonify({"success": success})


@app.route("/api/review/stats", methods=["GET"])
def review_stats():
    from review_queue import get_review_stats, get_pending_reviews
    
    return jsonify({
        "stats": get_review_stats(),
        "pending": get_pending_reviews(10)
    })


@app.route("/api/inject_paper", methods=["POST"])
def inject_paper():
    """
    Professor manually injects a confirmed-real paper into the local SQLite DB.
    Only verified/real papers should be injected — suspicious entries are NOT stored
    automatically; the professor must explicitly call this after confirming.
    """
    from local_db import inject_confirmed_paper, get_cache_stats
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    title   = (data.get("title") or "").strip()
    authors = (data.get("authors") or "").strip()
    year    = (data.get("year") or "").strip()
    doi     = (data.get("doi") or "").strip()
    url     = (data.get("url") or "").strip()

    if not title:
        return jsonify({"error": "Title is required"}), 400

    ok = inject_confirmed_paper(title=title, authors=authors,
                                 year=year, doi=doi, url=url)
    if not ok:
        return jsonify({"error": "Failed to save to database"}), 500

    stats = get_cache_stats()
    return jsonify({
        "success": True,
        "message": f"'{title[:60]}' saved to local DB.",
        "db_total": stats["total_papers"],
        "db_size_kb": stats["db_size_kb"],
    })


@app.route("/api/db_stats", methods=["GET"])
def db_stats():
    """Return local SQLite DB statistics."""
    from local_db import get_cache_stats
    return jsonify(get_cache_stats())


@app.route("/export", methods=["POST"])
def export_report():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data"}), 400

    sc = data.get("score", {})
    s = data.get("summary", {})
    
    verdict_icon = "✅" if sc.get("verdict") == "PASS" else "⚠️" if sc.get("verdict") == "FLAG" else "❌"
    
    lines = [
        "=" * 80,
        "LNI REFERENCE CHECKER v6.1 — PROFESSOR REPORT",
        "=" * 80,
        f"File        : {data.get('filename', '?')}",
        f"Format      : {data.get('format', '?')}",
        f"Score       : {sc.get('score', '?')}/100  Grade: {sc.get('grade', '?')}",
        f"Verdict     : {verdict_icon} {sc.get('verdict', '?')}",
        f"Reason      : {sc.get('verdict_reason', 'No AI reasoning provided')}",
        "",
        "─" * 80,
        "SUMMARY STATISTICS",
        "─" * 80,
        f"  Bibliography entries        : {s.get('bib_entry_count', 0)}",
        f"  In-text citations           : {s.get('citation_count', 0)}",
        f"  Missing from bibliography   : {s.get('missing_from_bib', 0)}",
        f"  Never cited (orphaned)      : {s.get('uncited_entries', 0)}",
        f"  Incomplete entries          : {s.get('incomplete_entries', 0)}",
        f"  Key-vs-metadata errors      : {s.get('key_inconsistencies', 0)}",
        f"  FAKE (AI verdict)           : {s.get('fake_candidates', 0)}",
        f"  SUSPICIOUS references       : {s.get('suspicious', 0)}",
        f"  Verified REAL               : {s.get('verified', 0)}",
        f"  Duplicates                  : {s.get('duplicates', 0)}",
        f"  Self-citations              : {s.get('self_citations', 0)}",
        f"  Style issues                : {s.get('style_issues', 0)}",
        f"  Open-access links           : {s.get('open_access', 0)}",
        f"  Processing time             : {data.get('processing_time_seconds', '?')} seconds",
    ]
    
    if s.get("numeric_citations"):
        lines.append("  ⚠ Numeric citations [1] detected — LNI requires [Author+Year]")
    lines.append("")

    if sc.get("student_feedback"):
        lines.append("─" * 80)
        lines.append("FEEDBACK FOR STUDENT")
        lines.append("─" * 80)
        for fb in sc["student_feedback"]:
            lines.append(f"  • {fb}")
        lines.append("")

    if sc.get("professor_note"):
        lines.append("─" * 80)
        lines.append("NOTE FOR PROFESSOR")
        lines.append("─" * 80)
        lines.append(f"  {sc['professor_note']}")
        lines.append("")

    if sc.get("penalties"):
        lines.append("─" * 80)
        lines.append("SCORE BREAKDOWN")
        lines.append("─" * 80)
        for p in sc["penalties"]:
            lines.append(f"  -{p['deduction']:2d}  {p['category']} ({p['count']}×)")
        lines.append("")

    version_notes = data.get("arxiv_version_notes", [])
    if version_notes:
        lines.append("─" * 80)
        lines.append("ARXIV VERSION NOTES (informational — not penalised)")
        lines.append("─" * 80)
        for vn in version_notes:
            lines.append(f"  [{vn['key']}] {vn['note']}")
        lines.append("")

    fake_refs = [v for v in data.get("verification", []) if v.get("ai_verdict") == "FAKE"]
    if fake_refs:
        lines.append("─" * 80)
        lines.append("LIKELY FAKE REFERENCES — Requires Investigation")
        lines.append("─" * 80)
        for v in fake_refs:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    Reasoning: {v.get('ai_reasoning', 'No reasoning provided')}")
            for rf in v.get("ai_risk_factors", []):
                lines.append(f"    ⚠ {rf}")
            lines.append(f"    Sources: {', '.join(v.get('sources_checked', []))}")
        lines.append("")

    suspicious_refs = [v for v in data.get("verification", []) if v.get("ai_verdict") == "SUSPICIOUS"]
    if suspicious_refs:
        lines.append("─" * 80)
        lines.append("SUSPICIOUS REFERENCES — Manual Review Recommended")
        lines.append("─" * 80)
        for v in suspicious_refs[:10]:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    {v.get('ai_reasoning', 'No reasoning provided')}")
        if len(suspicious_refs) > 10:
            lines.append(f"  ... and {len(suspicious_refs) - 10} more")
        lines.append("")

    missing_refs = data.get("cross_check", {}).get("cited_not_in_bib", [])
    if missing_refs:
        lines.append("─" * 80)
        lines.append("CITED BUT MISSING FROM BIBLIOGRAPHY")
        lines.append("─" * 80)
        for k in missing_refs[:20]:
            lines.append(f"  [MISSING] {k}")
        if len(missing_refs) > 20:
            lines.append(f"  ... and {len(missing_refs) - 20} more")
        lines.append("")

    orphaned_refs = data.get("cross_check", {}).get("in_bib_not_cited", [])
    if orphaned_refs:
        lines.append("─" * 80)
        lines.append("IN BIBLIOGRAPHY BUT NEVER CITED")
        lines.append("─" * 80)
        for k in orphaned_refs[:20]:
            lines.append(f"  [ORPHAN]  {k}")
        if len(orphaned_refs) > 20:
            lines.append(f"  ... and {len(orphaned_refs) - 20} more")
        lines.append("")

    lines += [
        "",
        "=" * 80,
        "Generated by LNI Reference Checker v6.1",
        "Deterministic: key format, key-consistency, fields, xcheck, dupes, style",
        "AI (Groq/Gemini): bibliography extraction, re-parsing, fake detection, verdict",
        "APIs: CrossRef · SS · OpenAlex · arXiv BibTeX (versioned) · DBLP · ACL Anthology",
        "      OpenReview · Open Library · GitHub · Scholar · DuckDuckGo",
        "Cache: disk (LNI_CACHE_DIR) + in-memory verification + session LLM cache",
        "=" * 80,
    ]

    report = "\n".join(lines)
    fname = re.sub(r'[^\w\-.]', '_', data.get("filename", "report")) + "_lni_report.txt"
    return Response(report, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


if __name__ == "__main__":
    print("\n  ┌──────────────────────────────────────────────────────────────────────────────┐")
    print("  │   LNI Reference Checker v6.1                                                  │")
    print("  │   http://localhost:5000                                                        │")
    print("  │                                                                                │")
    print("  │   IMPROVEMENTS in v6.1:                                                        │")
    print("  │   • Better PDF extraction with multiple fallback methods                      │")
    print("  │   • Clear warnings for scanned PDFs and extraction issues                     │")
    print("  │   • Improved error messages with actionable advice                            │")
    print("  │   • File size and type validation before processing                           │")
    print("  │   • Processing timeout protection (3 minutes)                                 │")
    print("  │   • Enhanced progress messages with counts and status                         │")
    print("  │                                                                                │")
    print("  │   Deterministic: key · key-consistency · fields · xcheck · dupes · style      │")
    print("  │   AI (Groq→Gemini): extract · re-parse · fake · verdict                       │")
    print("  │   APIs: CrossRef · SS · OpenAlex · arXiv (versioned) · DBLP · ACL             │")
    print("  │          OpenReview · OpenLibrary · GitHub                                    │")
    print("  └──────────────────────────────────────────────────────────────────────────────┘\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)