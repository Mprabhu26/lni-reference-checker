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

    # ── Normalize cross-page bibliography entry breaks ───────────────────────
    # 1. Rejoin words split by end-of-line hyphenation: "Refer-\nence" → "Reference"
    text = re.sub(r'-\n(\S)', r'\1', text)
    # 2. Within the bibliography section, a soft newline inside an entry is NOT
    #    a new entry (new entries always start with [Key]).  Collapse those so
    #    the entry regex in parser.py can match the full entry on one logical line.
    #    Strategy: replace any \n NOT followed by [ or a section heading with a space.
    bib_marker = re.search(
        r'(Literaturverzeichnis|References|Literatur\b|Bibliography)',
        text, re.IGNORECASE
    )
    if bib_marker:
        body_part = text[:bib_marker.start()]
        bib_part  = text[bib_marker.start():]
        # In bib section: collapse newlines that are NOT before a new [Key] entry
        bib_part = re.sub(r'\n(?!\[)', ' ', bib_part)
        # Re-introduce newlines before each [Key] so the parser can split entries
        bib_part = re.sub(r'\s+(\[[A-Za-z]{2,6}\d{2}[a-z]?\])', r'\n\1', bib_part)
        text = body_part + bib_part

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


def _parse_bibtex_fields(body: str) -> dict:
    """
    Parse BibTeX field assignments from an entry body, correctly handling
    nested braces such as title = {An {Introduction} to X}.
    Returns a dict of lowercase field name → cleaned string value.
    """
    fields = {}
    # Find each field start: word = { or word = "
    field_start = re.compile(r'(\w+)\s*=\s*([{"])', re.DOTALL)
    pos = 0
    while pos < len(body):
        m = field_start.search(body, pos)
        if not m:
            break
        field_name = m.group(1).lower()
        delimiter  = m.group(2)
        close      = '}' if delimiter == '{' else '"'
        content_start = m.end()

        if delimiter == '{':
            # Walk forward counting brace depth
            depth = 1
            i = content_start
            while i < len(body) and depth > 0:
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth -= 1
                i += 1
            value = body[content_start:i - 1]  # exclude closing brace
            pos = i
        else:
            # Quoted value: find closing " not preceded by backslash
            end = body.find('"', content_start)
            while end != -1 and body[end - 1] == '\\':
                end = body.find('"', end + 1)
            if end == -1:
                break
            value = body[content_start:end]
            pos = end + 1

        # Collapse inner braces used for case-protection: {Word} → Word
        value = re.sub(r'\{([^{}]*)\}', r'\1', value)
        fields[field_name] = re.sub(r'\s+', ' ', value).strip()

    return fields


def _bibtex_to_lni_text(bibtex: str) -> str:
    """
    Convert BibTeX entries to LNI-style [Key] text so our parser can read them.
    Example: @Book{Ez10, author={...}, title={...}} → [Ez10] author: title. year.
    Resolves crossref fields so child entries inherit missing fields from parent.
    Handles nested braces correctly via _parse_bibtex_fields().
    """
    lines = ["Literaturverzeichnis\n"]
    entry_pattern = re.compile(
        r'@\w+\{(\w+),(.*?)\}(?=\s*@|\s*$)', re.DOTALL
    )

    # ── Pass 1: parse all entries into a raw fields dict ────────────────────
    all_fields: dict[str, dict] = {}
    for entry_match in entry_pattern.finditer(bibtex):
        key    = entry_match.group(1)
        body   = entry_match.group(2)
        fields = _parse_bibtex_fields(body)
        all_fields[key] = fields

    # ── Pass 2: resolve crossref inheritance ────────────────────────────────
    for key, fields in all_fields.items():
        parent_key = fields.get("crossref", "").strip()
        if parent_key and parent_key in all_fields:
            parent = all_fields[parent_key]
            # Child inherits any field the parent has that the child is missing
            for field_name, value in parent.items():
                if field_name != "crossref" and field_name not in fields:
                    fields[field_name] = value

    # ── Pass 3: render to LNI text ──────────────────────────────────────────
    for key, fields in all_fields.items():
        author    = fields.get("author", "")
        title     = fields.get("title", "")
        year      = fields.get("year", "")
        pub       = fields.get("publisher", "")
        journal   = fields.get("journal", "")
        pages     = fields.get("pages", "")
        url       = fields.get("url", "")
        urldate   = fields.get("urldate", "")
        booktitle = fields.get("booktitle", "")

        parts = []
        if author:    parts.append(f"{author}:")
        if title:     parts.append(title + ".")
        if journal:   parts.append(journal + ".")
        if booktitle and not journal: parts.append(f"In: {booktitle}.")
        if pub:       parts.append(pub + ".")
        if pages:     parts.append(f"S. {pages}.")
        if url:       parts.append(url)
        if urldate:   parts.append(f"Stand: {urldate}")
        if year:      parts.append(year + ".")

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