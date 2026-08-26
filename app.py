"""
Flask Web Server — LNI Reference Checker v7.0
=============================================
CHANGES v7.0:
  - Removed ai_extract_references_from_text() full-text AI pass (wasteful).
    Only ai_parse_uncertain_entries() is called for entries the regex flagged.
  - Verification pipeline now follows strict 4-step order:
      1. Local DB  2. Academic APIs  3. URL fetch (suspicious only)  4. AI
  - AI final verdict never outputs FAKE — only REAL or SUSPICIOUS.
    FAKE is only set by professor manual action.
  - FIXED v8.2: Duplicate entries are now handled properly via verify_all_references()
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

import json
import tempfile
import shutil
import time
import signal
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

from extractor import extract
from parser import parse_bibliography, entries_to_dict, _validate_key_vs_metadata
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
from ai_checker import (
    ai_parse_uncertain_entries,
    ai_verify_references,
    ai_overall_verdict,
    get_llm_cache_stats,
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
app.config["TIMEOUT"] = 180

from local_db import init_cache_db
from review_queue import init_review_db
try:
    init_cache_db()
    init_review_db()
except Exception as _db_init_err:
    print(f"Warning: DB init error (non-fatal): {_db_init_err}")


# ---------------------------------------------------------------------------
# Timeout decorator
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
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            else:
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
    from checker import _ARXIV_BIBTEX_MEM_CACHE, _MEM_CACHE
    llm_stats = get_llm_cache_stats()
    from ai_checker import _ai_available, _AI_BASE_URL, _AI_MODEL
    groq_available = _ai_available()
    provider_label = _AI_BASE_URL.split("/")[2] if "//" in _AI_BASE_URL else (_AI_BASE_URL or "none")
    return jsonify({
        "status": "ok",
        "version": "7.0",
        "cache": {
            "llm_response_cache_entries": llm_stats["llm_cache_entries"],
            "arxiv_bibtex_cache_entries": len(_ARXIV_BIBTEX_MEM_CACHE),
            "verification_result_cache_entries": len(_MEM_CACHE),
        },
        "ai_available": groq_available,
        "ai_provider": provider_label if groq_available else "none",
        "apis": {
            "groq": groq_available,
            "github": bool(os.environ.get("GITHUB_TOKEN")),
            "unpaywall": bool(os.environ.get("UNPAYWALL_EMAIL")),
        },
        "env": {
            "disk_cache_dir": os.environ.get("LNI_CACHE_DIR", ".lni_cache"),
        },
    })


# ---------------------------------------------------------------------------
# Pipeline helpers
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
         "title_match_score": getattr(vr, "title_match_score", None),
         "author_match_score": getattr(vr, "author_match_score", None),
         "is_duplicate": getattr(vr, "is_duplicate", False),
         "duplicate_of": getattr(vr, "duplicate_of", None),
         }
        for vr in api_results_raw
    ]


def _verified_result_without_ai(bib_dicts: list, api_results_raw: list) -> dict:
    """Build the final verdict when every reference already passed verification."""
    by_key = {vr.key: vr for vr in api_results_raw}
    verdicts = []
    for entry in bib_dicts:
        vr = by_key.get(entry["key"])
        if vr is None:
            continue
        verdicts.append({
            "key": entry["key"],
            "verdict": "REAL",
            "confidence": vr.confidence,
            "reasoning": "Confirmed by the reference verification stage.",
            "risk_factors": [],
        })
    return {
        "verdicts": verdicts,
        "fake_count": 0,
        "suspicious_count": 0,
        "real_count": len(verdicts),
        "summary": f"Analysis: {len(verdicts)} REAL, 0 SUSPICIOUS, 0 FAKE",
    }


def _compute_metadata_warnings(entry_dict: dict, vr_dict: dict, bib_entry=None) -> list:
    """
    Compare cited metadata against what the database found.
    Returns a list of warning dicts: {type, cited, correct, severity}
    Severity: 'error' (clear wrong data) | 'warn' (minor/ambiguous)
    Covers: author mismatch, year mismatch, extra/fake authors,
            missing authors, publisher mismatch, title case, journal mismatch.
    """
    import re as _re
    from checker import author_overlap_score
    warnings = []

    cited_authors = (entry_dict.get("authors") or "").strip()
    correct_authors = (vr_dict.get("correct_authors") or
                       vr_dict.get("corrected_authors") or "").strip()
    cited_year = str(entry_dict.get("year") or "").strip()
    corrected_year = str(vr_dict.get("corrected_year") or "").strip()
    cited_title = (entry_dict.get("title") or "").strip()
    matched_title = (vr_dict.get("matched_title") or
                     vr_dict.get("corrected_title") or "").strip()
    cited_publisher = (entry_dict.get("publisher") or "").strip()
    corrected_publisher = (vr_dict.get("corrected_publisher") or "").strip()
    cited_journal = (entry_dict.get("journal") or "").strip()
    corrected_journal = (vr_dict.get("corrected_journal") or "").strip()

    # ── Author warnings ──────────────────────────────────────────────────────
    # Detect et al. — if present, author count comparisons are suppressed
    _has_et_al = bool(_re.search(r'\bet\.?\s*al\.?', cited_authors, _re.IGNORECASE))

    if cited_authors and correct_authors:
        overlap = author_overlap_score(cited_authors, correct_authors)
        if overlap is not None:
            # Count authors in each
            def _count_authors(s):
                parts = [p.strip() for p in _re.split(r';|\band\b|\bund\b', s, flags=_re.IGNORECASE) if p.strip()]
                return [p for p in parts if not _re.match(r'^et\.?\s*al\.?$', p.lower())]

            cited_list = _count_authors(cited_authors)
            correct_list = _count_authors(correct_authors)
            n_cited = len(cited_list)
            n_correct = len(correct_list)

            if overlap < 0.40:
                warnings.append({
                    "type": "author_mismatch",
                    "label": "Author mismatch",
                    "cited": cited_authors[:120],
                    "correct": correct_authors[:120],
                    "severity": "error",
                    "detail": f"Only {int(overlap*100)}% of cited authors match the database record"
                })
            elif overlap < 0.75:
                warnings.append({
                    "type": "author_mismatch",
                    "label": "Author mismatch",
                    "cited": cited_authors[:120],
                    "correct": correct_authors[:120],
                    "severity": "warn",
                    "detail": f"Partial author match ({int(overlap*100)}%) — verify author list"
                })
            elif not _has_et_al:
                # Good overlap but count differs → extra or missing authors
                # Skip if et al. is used (intentional truncation)
                if n_cited > n_correct:
                    warnings.append({
                        "type": "extra_authors",
                        "label": "Extra authors",
                        "cited": cited_authors[:120],
                        "correct": correct_authors[:120],
                        "severity": "warn",
                        "detail": f"Cited {n_cited} authors but database lists {n_correct} — possible fabricated co-author"
                    })
                elif n_correct > n_cited + 1:
                    warnings.append({
                        "type": "missing_authors",
                        "label": "Missing authors",
                        "cited": cited_authors[:120],
                        "correct": correct_authors[:120],
                        "severity": "warn",
                        "detail": f"Cited {n_cited} authors but database lists {n_correct} — some authors omitted"
                    })

    # ── Year warnings ────────────────────────────────────────────────────────
    if cited_year and corrected_year:
        try:
            m_c = _re.search(r'\d{4}', cited_year)
            m_d = _re.search(r'\d{4}', corrected_year)
            y_cited   = int(m_c.group()) if m_c else None
            y_correct = int(m_d.group()) if m_d else None
            if y_cited is None or y_correct is None:
                raise ValueError("no year found")
            diff = abs(y_cited - y_correct)
            if diff > 0:
                sev = "error" if diff > 2 else "warn"
                warnings.append({
                    "type": "year_mismatch",
                    "label": "Year mismatch",
                    "cited": cited_year,
                    "correct": corrected_year,
                    "severity": sev,
                    "detail": f"Cited year {cited_year} but publication year is {corrected_year} (off by {diff} year{'s' if diff > 1 else ''})"
                })
        except (ValueError, TypeError):
            pass

    # ── Publisher warnings ───────────────────────────────────────────────────
    if cited_publisher and corrected_publisher:
        cp_norm = cited_publisher.lower().replace(" ", "")
        db_norm = corrected_publisher.lower().replace(" ", "")
        if cp_norm not in db_norm and db_norm not in cp_norm:
            warnings.append({
                "type": "publisher_mismatch",
                "label": "Publisher mismatch",
                "cited": cited_publisher[:80],
                "correct": corrected_publisher[:80],
                "severity": "warn",
                "detail": f"Cited publisher '{cited_publisher}' differs from database record '{corrected_publisher}'"
            })

    # ── Journal warnings ─────────────────────────────────────────────────────
    if cited_journal and corrected_journal:
        cj_norm = cited_journal.lower().replace(" ", "")
        dj_norm = corrected_journal.lower().replace(" ", "")
        if cj_norm not in dj_norm and dj_norm not in cj_norm:
            warnings.append({
                "type": "journal_mismatch",
                "label": "Journal mismatch",
                "cited": cited_journal[:80],
                "correct": corrected_journal[:80],
                "severity": "warn",
                "detail": f"Cited journal name differs from database record"
            })

    return warnings


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
        entry.key_consistent = None
        entry.key_mismatch_detail = None
        _validate_key_vs_metadata(entry)
    return bib_list


def _build_match_breakdown(vr, ai: dict) -> dict:
    """
    Build a structured match-quality breakdown for display in the UI.

    The 'confidence' number on a reference card is MATCH QUALITY (how well the
    bibliography entry matched a database record), NOT a probability that the
    paper exists. This dict gives the UI enough information to show a clear,
    honest label like:

        Title match: 78% | Author match: 65% | Source: CrossRef
        Verdict: ⚠ Suspicious — needs manual review

    instead of the ambiguous bare "78% — no database found this".

    Fields
    ------
    title_match   : int|None   — title similarity %, None if no DB match attempted
    author_match  : int|None   — author overlap %, None if unavailable
    api_found     : bool       — True if any academic API returned a candidate
    sources       : list[str]  — which APIs were queried
    confidence_label : str     — short human label for the confidence band
    confidence_tooltip : str   — one-sentence explanation of what the number means
    """
    title_sim = getattr(vr, "title_match_score", None) if vr else None
    author_sim = getattr(vr, "author_match_score", None) if vr else None
    sources = (getattr(vr, "sources_checked", []) if vr else []) or []
    api_found = bool(getattr(vr, "matched_title", None) if vr else None)

    # Pull per-signal info surfaced by ai_checker if available
    ai_risk = ai.get("risk_factors", [])
    for rf in ai_risk:
        if isinstance(rf, str) and rf.startswith("Title match:"):
            try:
                pct = int(rf.split(":")[1].strip().rstrip("%"))
                if title_sim is None:
                    title_sim = pct / 100.0
            except (ValueError, IndexError):
                pass

    conf = ai.get("confidence") or (getattr(vr, "confidence", 0.0) if vr else 0.0)

    # Band label
    if conf >= 0.95:
        band = "Strong match"
    elif conf >= 0.80:
        band = "Good match"
    elif conf >= 0.70:
        band = "Partial match"
    elif conf >= 0.50:
        band = "Weak match"
    else:
        band = "No match found"

    tooltip = (
        "This percentage is match quality — how closely the bibliography entry "
        "matches a record in academic databases. "
        "≥95%: confirmed real. 70–94%: suspicious (may be real but poorly formatted). "
        "<70%: likely hallucinated or a very obscure work."
    )

    return {
        "title_match": round(title_sim * 100) if title_sim is not None else None,
        "author_match": round(author_sim * 100) if author_sim is not None else None,
        "api_found": api_found,
        "sources": sources,
        "confidence_label": band,
        "confidence_tooltip": tooltip,
    }


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
        # ── Check for duplicates for THIS entry ──────────────────────────────────
        dup_info = None
        if getattr(vr, "is_duplicate", False) and getattr(vr, "duplicate_of", None):
            dup_info = {"duplicate_of": vr.duplicate_of, "reason": "Same paper (deduplicated)"}
        else:
            for d in duplicates:
                if d.get("key_a") == vr.key:
                    dup_info = {"duplicate_of": d.get("key_b"), "reason": d.get("reason", "")}
                    break
                elif d.get("key_b") == vr.key:
                    dup_info = {"duplicate_of": d.get("key_a"), "reason": d.get("reason", "")}
                    break

        ai = ai_verdicts_by_key.get(vr.key, {})
        ai_verdict = ai.get("verdict", "SUSPICIOUS")
        # Map AI verdict to display status
        if ai_verdict == "REAL":
            status = "verified"
        elif ai_verdict == "FAKE":
            # AI never outputs FAKE directly now; this path is for professor-confirmed fakes
            status = "not_found"
        else:
            status = "suspicious"

        _raw = (bib_dict.get(vr.key) and bib_dict[vr.key].raw_text or "")[:300]
        _vr_title = vr.title or (bib_dict.get(vr.key) and bib_dict[vr.key].title) or ""
        ai_reasoning_text = ai.get("reasoning", "")

        verification_output.append({
            "key": vr.key,
            "title": _vr_title,
            "raw": _raw,
            "status": status,
            "confidence": round(ai.get("confidence", vr.confidence), 2),
            "matched_title": vr.matched_title,
            "doi": vr.doi or ai.get("open_access_url"),
            "open_access_url": ai.get("open_access_url") or vr.open_access_url,
            "note": ai_reasoning_text or vr.note,
            "api_note": vr.note,
            "sources_checked": vr.sources_checked,
            "web_evidence": vr.web_evidence,
            "ai_verdict": ai_verdict,
            "ai_reasoning": ai_reasoning_text,
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
            # match_breakdown: explicit per-signal scores so the UI can show
            # "Title match: 78% | Author match: 65%" instead of a bare confidence %.
            # confidence = match quality against DB records, NOT probability of existence.
            "match_breakdown": _build_match_breakdown(vr, ai),
            "is_duplicate": dup_info is not None,
            "duplicate_of": dup_info.get("duplicate_of") if dup_info else None,
            "duplicate_reason": dup_info.get("reason") if dup_info else None,
        })
        # Compute metadata warnings for this entry
        _entry_obj = bib_dict.get(vr.key)
        _entry_raw = {
            "authors": getattr(_entry_obj, "authors", "") or "",
            "year": getattr(_entry_obj, "year", "") or "",
            "title": getattr(_entry_obj, "title", "") or "",
            "publisher": getattr(_entry_obj, "publisher", "") or "",
            "journal": getattr(_entry_obj, "journal", "") or "",
        }
        _vr_raw = {
            "correct_authors": vr.correct_authors,
            "corrected_authors": getattr(vr, "corrected_authors", None),
            "corrected_year": getattr(vr, "corrected_year", None),
            "matched_title": vr.matched_title,
            "corrected_title": getattr(vr, "corrected_title", None),
            "corrected_publisher": getattr(vr, "corrected_publisher", None),
            "corrected_journal": getattr(vr, "corrected_journal", None),
        }
        verification_output[-1]["metadata_warnings"] = _compute_metadata_warnings(_entry_raw, _vr_raw)

    # Entries that never went through API verification (e.g. no title/DOI)
    api_keys = {vr.key for vr in api_results_raw}
    for entry in _bib_to_dicts(bib_list):
        if entry["key"] not in api_keys:
            ai = ai_verdicts_by_key.get(entry["key"], {})
            ai_verdict = ai.get("verdict") or "SUSPICIOUS"
            verification_output.append({
                "key": entry["key"],
                "title": entry.get("title") or "",
                "raw": entry.get("raw_text") or "",
                "status": "verified" if ai_verdict == "REAL" else "suspicious",
                "confidence": ai.get("confidence", 0.5),
                "matched_title": None,
                "doi": entry.get("doi"),
                "open_access_url": ai.get("open_access_url") or entry.get("url"),
                "note": ai.get("reasoning", ""),
                "api_note": ai.get("reasoning", ""),
                "web_evidence": None,
                "ai_verdict": ai_verdict,
                "ai_reasoning": ai.get("reasoning", ""),
                "ai_risk_factors": ai.get("risk_factors", []),
                "version_note": None,
                "metadata_warnings": [],
                "match_breakdown": _build_match_breakdown(None, ai),
            })

    # Score
    retracted_count = sum(1 for vr in api_results_raw if getattr(vr, "is_retracted", False))
    # Fakes only deducted after professor manually confirms via Mark as Fake — not auto.
    professor_confirmed_fakes = 0
    det_score = compute_score(
        bib_list, xcheck, api_results_raw,
        style_suggestions, duplicates,
        professor_confirmed_fakes=professor_confirmed_fakes,
        retracted_count=retracted_count,
        verification_results=verification_output,
    )
    s = det_score["score"]
    det_verdict = "PASS" if s >= 80 else "FLAG" if s >= 60 else "FAIL"

    _pen_parts = "; ".join(
        f"{p['count']} {p['category'].lower()} (-{p['deduction']}pts)"
        for p in det_score.get("penalties", [])
    ) or "No issues detected."
    _det_reason = f"Score {s}/100. {_pen_parts}"
    _ai_reason = overall.get("verdict_reason", "")
    _verdict_reason = _ai_reason if (_ai_reason and overall.get("verdict") == det_verdict) else _det_reason

    final_score = {
        "score": det_score["score"],
        "grade": det_score["grade"],
        "verdict": det_verdict,
        "verdict_reason": _verdict_reason,
        "student_feedback": overall.get("student_feedback", []),
        "professor_note": overall.get("professor_note", ""),
        "penalties": det_score.get("penalties", []),
        "max_score": 100,
    }

    # ── Map each entry to the duplicate group it belongs to (if any) ──────────
    # `duplicates` holds pairwise matches (key_a, key_b, reason); collapse
    # these into groups via the same union-find canonicalization used
    # elsewhere so entries can be flagged in the Bibliography Format Check
    # tab too, not just the Reference Verification tab.
    _dup_group_map: Dict[str, List[str]] = {}
    if duplicates:
        _parent = {e.key: e.key for e in bib_list}

        def _dup_find(x):
            while _parent.get(x, x) != x:
                x = _parent[x]
            return x

        def _dup_union(x, y):
            px, py = _dup_find(x), _dup_find(y)
            if px != py:
                _parent[py] = px

        for d in duplicates:
            ka, kb = d.get("key_a"), d.get("key_b")
            if ka in _parent and kb in _parent:
                _dup_union(ka, kb)

        _groups: Dict[str, List[str]] = {}
        for e in bib_list:
            root = _dup_find(e.key)
            _groups.setdefault(root, []).append(e.key)

        for root, members in _groups.items():
            if len(members) > 1:
                for m in members:
                    _dup_group_map[m] = [k for k in members if k != m]

    bib_output = [
        {"key": e.key, "type": e.entry_type or "unknown",
         "authors": e.authors, "title": e.title, "year": e.year,
         "publisher": e.publisher, "journal": e.journal, "url": e.url,
         "doi": e.doi, "isbn": e.isbn, "pages": e.pages, "raw": e.raw_text[:250],
         "completeness_issues": e.completeness_issues,
         "key_consistent": e.key_consistent,
         "is_duplicate": e.key in _dup_group_map,
         "duplicate_of": _dup_group_map.get(e.key, []),
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
# Core pipeline — streaming check (FIXED: no duplicate verification loop)
# ---------------------------------------------------------------------------

def _run_streaming_check(main_path: str, bib_path: str = None,
                          verify: bool = True, filename: str = ""):
    """Generator yielding SSE strings. Final event is 'done' with full result JSON."""
    import sys

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    start_time = time.time()
    
    # Diagnostic: log that we've started
    print(f"[DIAG] _run_streaming_check started for {filename}", file=sys.stderr, flush=True)

    try:
        sse_msg = _sse("progress", {"step": "extract", "message": "📄 Extracting text from document..."})
        print(f"[DIAG] Yielding extract message", file=sys.stderr, flush=True)
        yield sse_msg

        print(f"[DIAG] Calling extract({main_path})", file=sys.stderr, flush=True)
        sections = extract(main_path, bib_path)
        print(f"[DIAG] Extract returned successfully", file=sys.stderr, flush=True)
        body = sections.get("body", "")
        bib_text = sections.get("bibliography", "")
        fmt = sections.get("format", "unknown")

        if sections.get("warning"):
            yield _sse("progress", {"step": "warning", "message": sections["warning"]})
        if sections.get("is_scanned"):
            yield _sse("progress", {"step": "warning",
                "message": "⚠️ Scanned PDF detected. Text extraction may be incomplete."})
        if len(body) < 200:
            yield _sse("progress", {"step": "warning",
                "message": "⚠️ Very little text extracted — PDF may be image-only."})

        yield _sse("progress", {"step": "extract_done",
            "message": f"✓ Extracted {len(body):,} chars body, {len(bib_text):,} chars bibliography"})

        if not bib_text.strip():
            yield _sse("progress", {"step": "warning",
                "message": "⚠️ No bibliography section found. Add a 'Literaturverzeichnis' heading."})

        # ── Parse bibliography ────────────────────────────────────────────────
        print(f"[DIAG] Calling parse_bibliography", file=sys.stderr, flush=True)
        yield _sse("progress", {"step": "parse", "message": "📚 Parsing bibliography entries..."})
        bib_list = parse_bibliography(bib_text)
        print(f"[DIAG] parse_bibliography returned {len(bib_list)} entries", file=sys.stderr, flush=True)
        bib_dict = entries_to_dict(bib_list)
        yield _sse("progress", {"step": "parse_done",
            "message": f"✓ Found {len(bib_list)} bibliography entries"})

        # ── AI re-parse only uncertain entries (flagged by regex) ─────────────
        bib_dicts = _bib_to_dicts(bib_list)
        uncertain_count = sum(1 for e in bib_dicts if e.get("needs_ai_parsing"))
        if uncertain_count:
            yield _sse("progress", {"step": "ai_reparse",
                "message": f"🤖 AI re-parsing {uncertain_count} uncertain entries..."})
        ai_parse_improvements = ai_parse_uncertain_entries(bib_dicts)
        if ai_parse_improvements:
            bib_list = _apply_ai_improvements(bib_list, ai_parse_improvements)
            bib_dict = entries_to_dict(bib_list)
            bib_dicts = _bib_to_dicts(bib_list)
            yield _sse("progress", {"step": "parse_done",
                "message": f"✓ AI re-parsed {len(ai_parse_improvements)} uncertain entries"})

        # ── Deterministic checks ──────────────────────────────────────────────
        yield _sse("progress", {"step": "check",
            "message": f"🔍 Running checks on {len(bib_list)} entries..."})
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

        if len(xcheck.cited_not_in_bib) > 0:
            yield _sse("progress", {"step": "check_result",
                "message": f"⚠️ {len(xcheck.cited_not_in_bib)} citation(s) missing from bibliography"})
        if len(xcheck.in_bib_not_cited) > 0:
            yield _sse("progress", {"step": "check_result",
                "message": f"⚠️ {len(xcheck.in_bib_not_cited)} bibliography entry(s) never cited"})

        # ── Verification: SINGLE call to verify_all_references ────────────────
        # FIXED: Removed the duplicate manual verification loop that was
        # causing entries to be verified twice and overwriting results.
        api_results_raw = []
        if verify and bib_dict:
            total = len(bib_dict)
            print(f"[DIAG] Starting verify_all_references for {total} entries", file=sys.stderr, flush=True)
            yield _sse("progress", {"step": "verify_start",
                "message": f"🔍 Verifying {total} references (DB → APIs → URL → AI)..."})

            # verify_all_references now handles duplicates internally
            api_results_raw = verify_all_references(bib_dict)
            print(f"[DIAG] verify_all_references completed, got {len(api_results_raw)} results", file=sys.stderr, flush=True)

            verified_count = sum(1 for r in api_results_raw if r.status == "verified")
            suspicious_count = sum(1 for r in api_results_raw if r.status == "suspicious")
            
            # Send progress updates for each result
            for i, vr in enumerate(api_results_raw):
                progress_data = {
                    "step": "verify",
                    "message": f"Verifying: {i+1}/{total}",
                    "key": vr.key,
                    "status": vr.status,
                    "confidence": round(vr.confidence, 2),
                    "done": i+1,
                    "total": total,
                    "verified_count": verified_count,
                    "suspicious_count": suspicious_count,
                }
                if vr.version_note:
                    progress_data["version_note"] = vr.version_note
                yield _sse("progress", progress_data)
                time.sleep(0.05)  # Small delay for UI responsiveness

            yield _sse("progress", {"step": "verify_done",
                "message": f"✓ Verification done: {verified_count} verified, "
                           f"{suspicious_count} suspicious"})

        # ── AI final verdict pass (only suspicious entries) ───────────────────
        yield _sse("progress", {"step": "ai_verify",
            "message": "🤖 AI review of suspicious entries..."})
        api_results_dicts = _vr_to_dicts(api_results_raw)
        if (api_results_raw and len(api_results_raw) == len(bib_dicts)
                and all(vr.status == "verified" for vr in api_results_raw)):
            verification_result = _verified_result_without_ai(bib_dicts, api_results_raw)
        else:
            verification_result = ai_verify_references(bib_dicts, api_results_dicts)

        suspicious_ai = verification_result.get("suspicious_count", 0)
        if suspicious_ai > 0:
            yield _sse("progress", {"step": "ai_result",
                "message": f"⚠️ AI flagged {suspicious_ai} reference(s) as SUSPICIOUS"})

        yield _sse("progress", {"step": "ai_verdict", "message": "📋 Generating final verdict..."})
        summary_for_ai = {
            "duplicates": len(duplicates),
            "self_citations": len(self_citations),
            "style_issues": len(style_suggestions),
        }
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

        result["processing_time_seconds"] = round(time.time() - start_time, 1)
        yield _sse("done", result)

    except TimeoutError:
        yield _sse("error", {"error": "Processing timed out after 3 minutes."})
    except Exception as e:
        import traceback
        error_msg = str(e)
        if "pdfplumber" in error_msg.lower():
            error_msg = "PDF parsing failed. Try converting to a text-based PDF, or upload the original DOCX/LaTeX."
        elif "memory" in error_msg.lower():
            error_msg = "Out of memory. Document is too large."
        yield _sse("error", {"error": error_msg, "trace": traceback.format_exc()})
    finally:
        shutil.rmtree(Path(main_path).parent, ignore_errors=True)


def _run_full_check(main_path: str, bib_path: str = None,
                    verify: bool = True, filename: str = "") -> dict:
    """Non-streaming full pipeline (for batch and export)."""
    sections = extract(main_path, bib_path)
    body = sections["body"]
    bib_text = sections["bibliography"]
    fmt = sections["format"]

    bib_list = parse_bibliography(bib_text)
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
    summary_for_ai = {
        "duplicates": len(duplicates),
        "self_citations": len(self_citations),
        "style_issues": len(style_suggestions),
    }
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


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.route("/check", methods=["POST"])
def check():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    main_file = request.files["file"]
    filename = main_file.filename
    ext = Path(filename).suffix.lower()
    verify = request.form.get("verify", "true").lower() == "true"
    streaming = request.form.get("stream", "true").lower() == "true"

    if ext not in {".pdf", ".docx", ".tex", ".latex"}:
        return jsonify({"error": f"Unsupported format '{ext}'. Use PDF, DOCX, or TEX."}), 400

    main_file.seek(0, 2)
    file_size = main_file.tell()
    main_file.seek(0)
    if file_size > 30 * 1024 * 1024:
        return jsonify({"error": f"File too large ({file_size/1024/1024:.1f} MB). Max 30 MB."}), 400
    if file_size == 0:
        return jsonify({"error": "Empty file."}), 400

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


@app.route("/check-stream", methods=["POST"])
def check_stream():
    return check()


@app.route("/check-sync", methods=["POST"])
def check_sync():
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


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@app.route("/batch", methods=["POST"])
def batch_check():
    uploaded = request.files.getlist("files")
    verify = request.form.get("verify", "true").lower() == "true"
    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400
    if len(uploaded) > 50:
        return jsonify({"error": "Maximum 50 files per batch."}), 400

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
                        or v.get("status") == "suspicious"
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


# ---------------------------------------------------------------------------
# AI manual audit
# ---------------------------------------------------------------------------

@app.route("/ai-review", methods=["POST"])
def ai_review():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No result data provided"}), 400

    s = data.get("summary", {})
    sc = data.get("score", {})
    flagged = [v for v in data.get("verification", [])
               if v.get("status") == "suspicious"
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

    prompt = f"""You are assisting a professor auditing a student's LNI reference list.

AUDIT SUMMARY:
- File: {data.get('filename','?')} | Score: {sc.get('score','?')}/100 | Verdict: {sc.get('verdict','?')}
- Bib entries: {s.get('bib_entry_count',0)} | Citations: {s.get('citation_count',0)}
- Missing from bib: {s.get('missing_from_bib',0)} | Never cited: {s.get('uncited_entries',0)}
- Incomplete: {s.get('incomplete_entries',0)} | Key mismatches: {s.get('key_inconsistencies',0)}
- Duplicates: {s.get('duplicates',0)} | Self-citations: {s.get('self_citations',0)}

SUSPICIOUS REFERENCES:
{flagged_lines}

INCOMPLETE: {chr(10).join(f"  [{e['key']}] {e.get('title','?')} — {', '.join(e['completeness_issues'])}" for e in incomplete) or "  None"}
DUPLICATES: {chr(10).join(f"  [{d['key_a']}] vs [{d['key_b']}] {int(d['similarity']*100)}% similar" for d in dupes) or "  None"}
SELF-CITATIONS: {chr(10).join(f"  [{s_['key']}] {s_['matched_author']}" for s_ in self_cit) or "  None"}

Return JSON with verdict and reasoning."""

    from ai_checker import _call_ai, _ai_available, _AI_BASE_URL, _AI_MODEL
    if not _ai_available():
        return jsonify({"error": "AI unavailable — set AI_API_KEY, AI_BASE_URL, AI_MODEL in .env"}), 503
    try:
        ai_text = _call_ai(prompt, max_tokens=700)
        provider = _AI_BASE_URL.split("/")[2] if "//" in _AI_BASE_URL else _AI_BASE_URL
        return jsonify({"verdict": ai_text, "ai_source": f"{provider} ({_AI_MODEL})",
                        "flagged_count": len(flagged)})
    except Exception as e:
        return jsonify({"error": f"AI API failed: {str(e)}"}), 503


# ---------------------------------------------------------------------------
# Professor review / confirm / inject
# ---------------------------------------------------------------------------

@app.route("/api/review", methods=["POST"])
def submit_review():
    from review_queue import add_review_decision
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    success = add_review_decision(
        title=data.get("title", "") or f"Paper_{data.get('key','')}",
        authors=data.get("authors", ""),
        decision=data.get("decision", ""),
        note=data.get("note", ""),
        verified_url=data.get("url", ""),
        ai_had_said=data.get("ai_verdict", ""),
    )
    return jsonify({"success": success})


@app.route("/api/review/stats", methods=["GET"])
def review_stats():
    from review_queue import get_review_stats, get_pending_reviews
    return jsonify({"stats": get_review_stats(), "pending": get_pending_reviews(10)})


@app.route("/api/inject_paper", methods=["POST"])
def inject_paper():
    from local_db import inject_confirmed_paper, get_cache_stats
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400
    ok = inject_confirmed_paper(
        title=title,
        authors=(data.get("authors") or "").strip(),
        year=(data.get("year") or "").strip(),
        doi=(data.get("doi") or "").strip(),
        url=(data.get("url") or "").strip(),
    )
    if not ok:
        return jsonify({"error": "Failed to save to database"}), 500
    stats = get_cache_stats()
    return jsonify({"success": True,
                    "message": f"'{title[:60]}' saved to local DB.",
                    "db_total": stats["total_papers"],
                    "db_size_kb": stats["db_size_kb"]})


@app.route("/api/confirm_paper", methods=["POST"])
def confirm_paper():
    return inject_paper()


# ---------------------------------------------------------------------------
# DB browser
# ---------------------------------------------------------------------------

@app.route("/api/db_stats", methods=["GET"])
def db_stats():
    from local_db import get_cache_stats
    return jsonify(get_cache_stats())


@app.route("/api/db_contents", methods=["GET"])
def db_contents():
    from local_db import get_all_papers, get_cache_stats
    limit  = min(int(request.args.get("limit",  100)), 500)
    offset = int(request.args.get("offset", 0))
    search = request.args.get("search", "").strip()
    papers = get_all_papers(limit=limit, offset=offset, search=search)
    stats  = get_cache_stats()
    return jsonify({"papers": papers, "total": stats["total_papers"],
                    "limit": limit, "offset": offset, "search": search,
                    "by_source": stats.get("by_source", {}),
                    "db_size_kb": stats.get("db_size_kb", 0)})


@app.route("/api/db_delete", methods=["POST"])
def db_delete():
    from local_db import delete_paper, get_cache_stats
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title required", "success": False}), 400
    try:
        ok = delete_paper(title)
        stats = get_cache_stats()
        return jsonify({"success": ok, "db_total": stats["total_papers"],
                        "db_size_kb": stats.get("db_size_kb", 0),
                        "message": "Deleted" if ok else "Not found"})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/db_delete_all", methods=["POST"])
def db_delete_all():
    import sqlite3
    from local_db import CACHE_DB, _ensure_db, get_cache_stats
    try:
        _ensure_db()
        conn = sqlite3.connect(str(CACHE_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.execute("DELETE FROM verified_papers")
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        stats = get_cache_stats()
        return jsonify({"success": True, "deleted": deleted,
                        "db_total": stats["total_papers"],
                        "message": f"Deleted {deleted} entries"})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.route("/api/export-bibtex", methods=["POST"])
def export_bibtex():
    data = request.get_json()
    verification = data.get("verification", [])
    bibliography = data.get("bibliography", [])
    bib_by_key = {b["key"]: b for b in bibliography}
    lines = [
        "% LNI Reference Checker — Corrected BibTeX Export",
        "% Only entries with database-corrected metadata are included.",
        "",
    ]
    exported = 0
    for v in verification:
        key = v.get("key", "")
        orig = bib_by_key.get(key, {})
        entry_type = (orig.get("type") or orig.get("entry_type") or "misc").lower().replace("@", "")
        has_correction = any(v.get(f) for f in (
            "corrected_title", "corrected_authors", "corrected_year",
            "corrected_journal", "corrected_publisher", "corrected_volume", "corrected_pages"))
        if not has_correction and not v.get("is_retracted"):
            continue
        if v.get("is_retracted"):
            lines.append(f"% ⚠ RETRACTED: {v.get('retraction_note','See CrossRef')}")
        title  = v.get("corrected_title")   or orig.get("title", "")
        authors= v.get("corrected_authors") or orig.get("authors", "")
        year   = v.get("corrected_year")    or orig.get("year", "")
        journal= v.get("corrected_journal") or orig.get("journal", "")
        pub    = v.get("corrected_publisher") or orig.get("publisher", "")
        volume = v.get("corrected_volume")  or orig.get("volume", "")
        pages  = v.get("corrected_pages")   or orig.get("pages", "")
        doi    = v.get("doi") or orig.get("doi", "")
        url    = v.get("open_access_url") or orig.get("url", "")
        lines.append(f"@{entry_type}{{{key},")
        if title:   lines.append(f"  title     = {{{{{title}}}}},")
        if authors: lines.append(f"  author    = {{{authors}}},")
        if year:    lines.append(f"  year      = {{{year}}},")
        if journal: lines.append(f"  journal   = {{{{{journal}}}}},")
        if pub:     lines.append(f"  publisher = {{{{{pub}}}}},")
        if volume:  lines.append(f"  volume    = {{{volume}}},")
        if pages:   lines.append(f"  pages     = {{{pages}}},")
        if doi:     lines.append(f"  doi       = {{{doi}}},")
        if url:     lines.append(f"  url       = {{{url}}},")
        lines.append("}\n")
        exported += 1
    if exported == 0:
        return jsonify({"error": "No entries with corrected metadata found."}), 400
    from flask import Response as _R
    return _R("\n".join(lines), mimetype="text/plain",
              headers={"Content-Disposition": "attachment; filename=corrected_references.bib"})


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
        "LNI REFERENCE CHECKER v7.0 — PROFESSOR REPORT",
        "=" * 80,
        f"File        : {data.get('filename', '?')}",
        f"Format      : {data.get('format', '?')}",
        f"Score       : {sc.get('score', '?')}/100  Grade: {sc.get('grade', '?')}",
        f"Verdict     : {verdict_icon} {sc.get('verdict', '?')}",
        f"Reason      : {sc.get('verdict_reason', 'No AI reasoning provided')}",
        "",
        "─" * 80,
        "SUMMARY",
        "─" * 80,
        f"  Bibliography entries      : {s.get('bib_entry_count', 0)}",
        f"  In-text citations         : {s.get('citation_count', 0)}",
        f"  Missing from bibliography : {s.get('missing_from_bib', 0)}",
        f"  Never cited (orphaned)    : {s.get('uncited_entries', 0)}",
        f"  Incomplete entries        : {s.get('incomplete_entries', 0)}",
        f"  Key-vs-metadata errors    : {s.get('key_inconsistencies', 0)}",
        f"  SUSPICIOUS references     : {s.get('suspicious', 0)}",
        f"  Verified REAL             : {s.get('verified', 0)}",
        f"  Duplicates                : {s.get('duplicates', 0)}",
        f"  Self-citations            : {s.get('self_citations', 0)}",
        f"  Processing time           : {data.get('processing_time_seconds', '?')} seconds",
    ]
    if s.get("numeric_citations"):
        lines.append("  ⚠ Numeric citations detected — LNI requires [Author+Year]")

    suspicious_refs = [v for v in data.get("verification", [])
                       if v.get("status") == "suspicious" or v.get("ai_verdict") == "SUSPICIOUS"]
    if suspicious_refs:
        lines += ["", "─" * 80, "SUSPICIOUS REFERENCES — Manual Review Required", "─" * 80]
        for v in suspicious_refs[:20]:
            lines.append(f"  [{v['key']}] {v['title']}")
            lines.append(f"    {v.get('ai_reasoning', v.get('note', 'No reasoning'))}")

    missing_refs = data.get("cross_check", {}).get("cited_not_in_bib", [])
    if missing_refs:
        lines += ["", "─" * 80, "CITED BUT MISSING FROM BIBLIOGRAPHY", "─" * 80]
        for k in missing_refs[:20]:
            lines.append(f"  [MISSING] {k}")

    lines += ["", "=" * 80,
              "Generated by LNI Reference Checker v7.0",
              "Pipeline: DB → APIs → URL → AI (suspicious only)",
              "=" * 80]

    report = "\n".join(lines)
    fname = re.sub(r'[^\w\-.]', '_', data.get("filename", "report")) + "_lni_report.txt"
    from flask import Response as _R
    return _R(report, mimetype="text/plain",
              headers={"Content-Disposition": f'attachment; filename="{fname}"'})


if __name__ == "__main__":
    print("\n  LNI Reference Checker v7.0")
    print("  http://localhost:5000")
    print("  Pipeline: DB → APIs → URL → AI (suspicious only)\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)