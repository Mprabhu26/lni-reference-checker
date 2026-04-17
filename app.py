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
from checker import extract_citations_from_body, cross_check, verify_all_references, check_lni_macros

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
                "missing_from_bib":  len(xcheck.cited_not_in_bib),
                "uncited_entries":   len(xcheck.in_bib_not_cited),
                "incomplete_entries": sum(1 for e in bib_list if e.completeness_issues),
                "fake_candidates":   sum(1 for vr in verification_results if vr.status == "not_found"),
                "verified":          sum(1 for vr in verification_results if vr.status == "verified"),
                "style_issues": len(style_suggestions),
                "open_access":       sum(1 for vr in verification_results if vr.open_access_url),
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


if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    print("\n  ┌─────────────────────────────────────┐")
    print("  │   LNI Reference Checker — Web UI    │")
    print("  │   http://localhost:5000             │")
    print("  └─────────────────────────────────────┘\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)