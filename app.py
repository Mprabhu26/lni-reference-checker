"""
Flask Web Server — LNI Reference Checker
Handles PDF, DOCX, and LaTeX (.tex + optional .bib) uploads.
"""

import os
import json
import tempfile
import zipfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from extractor import extract
from parser import parse_bibliography, entries_to_dict
from checker import (
    extract_citations_from_body, cross_check, verify_all_references,
    check_lni_macros, find_duplicates, compute_score
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/check", methods=["POST"])
def check():
    files = request.files
    verify = request.form.get("verify", "true").lower() == "true"

    # Determine what was uploaded
    if "file" not in files:
        return jsonify({"error": "No file uploaded"}), 400

    main_file = files["file"]
    filename  = main_file.filename
    ext       = Path(filename).suffix.lower()

    allowed = {".pdf", ".docx", ".tex", ".latex"}
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format '{ext}'. Supported: PDF, DOCX, TEX"}), 400

    # Write to temp dir
    tmpdir = tempfile.mkdtemp()
    main_path = os.path.join(tmpdir, filename)
    main_file.save(main_path)

    # Optional .bib sidecar for LaTeX
    bib_path = None
    if "bib" in files and ext in (".tex", ".latex"):
        bib_file = files["bib"]
        bib_path = os.path.join(tmpdir, bib_file.filename)
        bib_file.save(bib_path)

    try:
        # ── Step 1: Extract ──────────────────────────────────────
        sections = extract(main_path, bib_path)
        body     = sections["body"]
        bib_text = sections["bibliography"]
        fmt      = sections["format"]

        # ── Step 2: Parse bibliography ───────────────────────────
        bib_list = parse_bibliography(bib_text)
        bib_dict = entries_to_dict(bib_list)
        style_suggestions = check_lni_macros(body)

        # ── Step 3: Cross-check ──────────────────────────────────
        # For LaTeX, also find \cite{key} style citations
        cited_keys = extract_citations_from_body(body)
        if fmt == "latex":
            import re
            latex_cites = re.findall(r'\\(?:cite|Cite|citet|Citet|citep)\{([^}]+)\}', sections["full_text"])
            for group in latex_cites:
                for k in group.split(','):
                    cited_keys.add(k.strip())

        xcheck = cross_check(bib_dict, cited_keys)

        # ── Step 4: Verify ───────────────────────────────────────
        verification_results = []
        if verify and bib_dict:
            verification_results = verify_all_references(bib_dict, delay=0.4)

        # ── Step 5: Duplicates ───────────────────────────────────
        duplicates = find_duplicates(bib_dict)

        # ── Step 6: Score ────────────────────────────────────────
        score = compute_score(bib_list, xcheck, verification_results,
                              style_suggestions, duplicates)

        # ── Build response ───────────────────────────────────────
        result = {
            "filename": filename,
            "format": fmt.upper(),
            "stats": {
                "body_chars": len(body),
                "bib_chars": len(bib_text),
                "bib_entry_count": len(bib_dict),
                "citation_count": len(cited_keys),
            },
            "bibliography": [
                {
                    "key": e.key,
                    "type": e.entry_type or "unknown",
                    "authors": e.authors,
                    "title": e.title,
                    "year": e.year,
                    "publisher": e.publisher,
                    "journal": e.journal,
                    "url": e.url,
                    "raw": e.raw_text[:250],
                    "completeness_issues": e.completeness_issues,
                }
                for e in bib_list
            ],
            "cross_check": {
                "correctly_used": xcheck.correctly_used,
                "cited_not_in_bib": xcheck.cited_not_in_bib,
                "in_bib_not_cited": xcheck.in_bib_not_cited,
            },
            "style_suggestions": style_suggestions,
            "duplicates": duplicates,
            "score": score,
            "verification": [
                {
                    "key": vr.key,
                    "title": vr.title,
                    "status": vr.status,
                    "confidence": round(vr.confidence, 2),
                    "matched_title": vr.matched_title,
                    "doi": vr.doi,
                    "open_access_url": vr.open_access_url,
                    "note": vr.note,
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
            }
        }
        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)




@app.route("/batch", methods=["POST"])
def batch_check():
    """
    Accept multiple PDF uploads (field name: 'files') and run the full
    check pipeline on each. Returns a list of per-file results plus an
    aggregate leaderboard sorted by score descending.
    """
    import re as _re
    import shutil

    uploaded = request.files.getlist("files")
    verify   = request.form.get("verify", "true").lower() == "true"

    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    results  = []
    tmpdir   = tempfile.mkdtemp()

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
                sections          = extract(file_path)
                body              = sections["body"]
                bib_text          = sections["bibliography"]
                fmt               = sections["format"]
                bib_list          = parse_bibliography(bib_text)
                bib_dict          = entries_to_dict(bib_list)
                style_suggestions = check_lni_macros(body)
                cited_keys        = extract_citations_from_body(body)
                xcheck            = cross_check(bib_dict, cited_keys)
                verification_results = verify_all_references(bib_dict, delay=0.3) if verify and bib_dict else []
                duplicates        = find_duplicates(bib_dict)
                score             = compute_score(bib_list, xcheck, verification_results,
                                                  style_suggestions, duplicates)
                results.append({
                    "filename": filename,
                    "format":   fmt.upper(),
                    "score":    score,
                    "summary": {
                        "bib_entry_count":    len(bib_dict),
                        "citation_count":     len(cited_keys),
                        "missing_from_bib":   len(xcheck.cited_not_in_bib),
                        "uncited_entries":    len(xcheck.in_bib_not_cited),
                        "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
                        "fake_candidates":    sum(1 for vr in verification_results if vr.status == "not_found"),
                        "duplicates":         len(duplicates),
                        "style_issues":       len(style_suggestions),
                    },
                })
            except Exception as e:
                import traceback
                results.append({"filename": filename, "error": str(e),
                                 "trace": traceback.format_exc()})

        # Sort by score descending (errors go to the bottom)
        results.sort(key=lambda r: r.get("score", {}).get("score", -1), reverse=True)
        return jsonify({"files": results, "count": len(results)})

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    print("\n  ┌─────────────────────────────────────┐")
    print("  │   LNI Reference Checker — Web UI    │")
    print("  │   http://localhost:5000             │")
    print("  └─────────────────────────────────────┘\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)