"""
Universal Extractor
-------------------
Extracts body text and bibliography from:
  - PDF  (.pdf)
  - Word (.docx)
  - LaTeX (.tex + optional .bib)
"""

import re
import zipfile
import os
from pathlib import Path


# ── Shared: split body vs bibliography ──────────────────────────────────────

BIB_HEADINGS = re.compile(
    r'(Literaturverzeichnis|References|Literatur\b|Bibliography)',
    re.IGNORECASE
)

def split_body_bib(full_text: str) -> dict:
    match = BIB_HEADINGS.search(full_text)
    if match:
        body = full_text[:match.start()].strip()
        bib  = full_text[match.start():].strip()
    else:
        body = full_text.strip()
        bib  = ""
    return {"full_text": full_text, "body": body, "bibliography": bib, "format": None}


# ── PDF extractor ────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> dict:
    import pdfplumber
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    result = split_body_bib(text)
    result["format"] = "pdf"
    return result


# ── DOCX extractor ───────────────────────────────────────────────────────────

def extract_docx(path: str) -> dict:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    result = split_body_bib(text)
    result["format"] = "docx"
    return result


# ── LaTeX extractor ──────────────────────────────────────────────────────────

def extract_latex(tex_path: str, bib_path: str = None) -> dict:
    """
    Extracts text from a .tex file and optionally reads a .bib sidecar.
    If .bib is present, uses it directly for bibliography parsing.
    """
    with open(tex_path, encoding="utf-8", errors="replace") as f:
        tex = f.read()

    # Strip LaTeX commands for body text
    body = _clean_latex(tex)

    # If .bib file available, read it raw for our bib parser
    bib_text = ""
    if bib_path and os.path.exists(bib_path):
        with open(bib_path, encoding="utf-8", errors="replace") as f:
            bib_text = f.read()
        bib_section = _bibtex_to_lni_text(bib_text)
    else:
        # Fall back: extract \bibliography citations from tex
        bib_section = _extract_tex_bib_section(tex)

    return {
        "full_text": body + "\n" + bib_section,
        "body": body,
        "bibliography": bib_section,
        "format": "latex",
        "raw_bibtex": bib_text,
    }


def _clean_latex(tex: str) -> str:
    """Remove LaTeX markup, keep readable text."""
    # Remove comments
    tex = re.sub(r'%.*', '', tex)
    # Remove common environments we don't need
    tex = re.sub(r'\\begin\{(figure|table|lstlisting|verbatim|equation|align)[^}]*\}.*?\\end\{\1\}', '', tex, flags=re.DOTALL)
    # Remove \command{arg} → arg
    tex = re.sub(r'\\(?:textbf|textit|emph|texttt|text|section|subsection|subsubsection|caption|label|ref|Cref|cref|url|href)\{([^}]*)\}', r'\1', tex)
    # Remove \cite{...} — keep for citation extraction
    # Remove remaining commands
    tex = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\[a-zA-Z]+\*?', '', tex)
    tex = re.sub(r'[{}]', '', tex)
    tex = re.sub(r'\s+', ' ', tex)
    return tex.strip()


def _bibtex_to_lni_text(bibtex: str) -> str:
    """
    Convert BibTeX entries to LNI-style [Key] text so our parser can read them.
    Example: @Book{Ez10, author={...}, title={...}} → [Ez10] author: title. year.
    """
    lines = ["Literaturverzeichnis\n"]
    entry_pattern = re.compile(
        r'@\w+\{(\w+),(.*?)\}(?=\s*@|\s*$)', re.DOTALL
    )
    field_pattern = re.compile(r'(\w+)\s*=\s*[\{"](.*?)[\}"]', re.DOTALL)

    for entry_match in entry_pattern.finditer(bibtex):
        key = entry_match.group(1)
        body = entry_match.group(2)
        fields = {}
        for fm in field_pattern.finditer(body):
            fields[fm.group(1).lower()] = re.sub(r'\s+', ' ', fm.group(2)).strip()

        author  = fields.get("author", "")
        title   = fields.get("title", "")
        year    = fields.get("year", "")
        pub     = fields.get("publisher", "")
        journal = fields.get("journal", "")
        pages   = fields.get("pages", "")
        url     = fields.get("url", "")
        urldate = fields.get("urldate", "")
        booktitle = fields.get("booktitle", "")

        parts = []
        if author:  parts.append(f"{author}:")
        if title:   parts.append(title + ".")
        if journal: parts.append(journal + ".")
        if booktitle and not journal: parts.append(f"In: {booktitle}.")
        if pub:     parts.append(pub + ".")
        if pages:   parts.append(f"S. {pages}.")
        if url:     parts.append(url)
        if urldate: parts.append(f"Stand: {urldate}")
        if year:    parts.append(year + ".")

        line = f"[{key}] {' '.join(parts)}"
        lines.append(line)

    return "\n".join(lines)


def _extract_tex_bib_section(tex: str) -> str:
    """Fallback: look for thebibliography environment inside .tex"""
    match = re.search(r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}', tex, re.DOTALL)
    if not match:
        return ""
    raw = match.group(1)
    lines = ["Literaturverzeichnis\n"]
    for item in re.finditer(r'\\bibitem\{(\w+)\}(.*?)(?=\\bibitem|\Z)', raw, re.DOTALL):
        key  = item.group(1)
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', item.group(2))
        text = re.sub(r'[{}\\]', '', text).strip()
        lines.append(f"[{key}] {text}")
    return "\n".join(lines)


# ── Router ───────────────────────────────────────────────────────────────────

def extract(file_path: str, bib_path: str = None) -> dict:
    """
    Auto-detect file type and extract text.
    Returns dict with keys: body, bibliography, format, full_text
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext in (".tex", ".latex"):
        return extract_latex(file_path, bib_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .tex")
