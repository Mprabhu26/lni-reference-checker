"""
Universal Extractor
-------------------
Extracts body text and bibliography from:
  - PDF  (.pdf)
  - Word (.docx)
  - LaTeX (.tex + optional .bib)
"""

import re
import os
from pathlib import Path


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


def extract_pdf(path: str) -> dict:
    import pdfplumber
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    # Rejoin hyphenated line breaks
    text = re.sub(r'-\n(\S)', r'\1', text)

    # Normalize bibliography section: collapse soft newlines inside entries
    bib_marker = BIB_HEADINGS.search(text)
    if bib_marker:
        body_part = text[:bib_marker.start()]
        bib_part  = text[bib_marker.start():]
        bib_part  = re.sub(r'\n(?!\[)', ' ', bib_part)
        bib_part  = re.sub(r'\s+(\[[A-Za-z]{2,6}\d{2}[a-z]?\])', r'\n\1', bib_part)
        text = body_part + bib_part

    result = split_body_bib(text)
    result["format"] = "pdf"
    return result


def extract_docx(path: str) -> dict:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    result = split_body_bib(text)
    result["format"] = "docx"
    return result


def extract_latex(tex_path: str, bib_path: str = None) -> dict:
    with open(tex_path, encoding="utf-8", errors="replace") as f:
        tex = f.read()

    body = _clean_latex(tex)

    bib_text = ""
    if bib_path and os.path.exists(bib_path):
        with open(bib_path, encoding="utf-8", errors="replace") as f:
            bib_text = f.read()
        bib_section = _bibtex_to_lni_text(bib_text)
    else:
        bib_section = _extract_tex_bib_section(tex)

    return {
        "full_text":    body + "\n" + bib_section,
        "body":         body,
        "bibliography": bib_section,
        "format":       "latex",
        "raw_bibtex":   bib_text,
    }


def _clean_latex(tex: str) -> str:
    tex = re.sub(r'%.*', '', tex)
    tex = re.sub(
        r'\\begin\{(figure|table|lstlisting|verbatim|equation|align)[^}]*\}.*?\\end\{\1\}',
        '', tex, flags=re.DOTALL
    )
    tex = re.sub(
        r'\\(?:textbf|textit|emph|texttt|text|section|subsection|subsubsection|'
        r'caption|label|ref|Cref|cref|url|href)\{([^}]*)\}',
        r'\1', tex
    )
    tex = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\[a-zA-Z]+\*?', '', tex)
    tex = re.sub(r'[{}]', '', tex)
    tex = re.sub(r'\s+', ' ', tex)
    return tex.strip()


def _parse_bibtex_fields(body: str) -> dict:
    fields = {}
    field_start = re.compile(r'(\w+)\s*=\s*([{"])', re.DOTALL)
    pos = 0
    while pos < len(body):
        m = field_start.search(body, pos)
        if not m:
            break
        field_name = m.group(1).lower()
        delimiter  = m.group(2)
        content_start = m.end()

        if delimiter == '{':
            depth = 1
            i = content_start
            while i < len(body) and depth > 0:
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth -= 1
                i += 1
            value = body[content_start:i - 1]
            pos = i
        else:
            end = body.find('"', content_start)
            while end != -1 and body[end - 1] == '\\':
                end = body.find('"', end + 1)
            if end == -1:
                break
            value = body[content_start:end]
            pos = end + 1

        value = re.sub(r'\{([^{}]*)\}', r'\1', value)
        fields[field_name] = re.sub(r'\s+', ' ', value).strip()

    return fields


def _bibtex_to_lni_text(bibtex: str) -> str:
    lines = ["Literaturverzeichnis\n"]
    entry_pattern = re.compile(r'@\w+\{(\w+),(.*?)\}(?=\s*@|\s*$)', re.DOTALL)

    all_fields: dict = {}
    for entry_match in entry_pattern.finditer(bibtex):
        key    = entry_match.group(1)
        body   = entry_match.group(2)
        fields = _parse_bibtex_fields(body)
        all_fields[key] = fields

    # Resolve crossref inheritance
    for key, fields in all_fields.items():
        parent_key = fields.get("crossref", "").strip()
        if parent_key and parent_key in all_fields:
            parent = all_fields[parent_key]
            for fn, val in parent.items():
                if fn != "crossref" and fn not in fields:
                    fields[fn] = val

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
        doi       = fields.get("doi", "")

        parts = []
        if author:    parts.append(f"{author}:")
        if title:     parts.append(title + ".")
        if journal:   parts.append(journal + ".")
        if booktitle and not journal: parts.append(f"In: {booktitle}.")
        if pub:       parts.append(pub + ".")
        if pages:     parts.append(f"S. {pages}.")
        if doi:       parts.append(f"doi: {doi}")
        if url:       parts.append(url)
        if urldate:   parts.append(f"Stand: {urldate}")
        if year:      parts.append(year + ".")

        lines.append(f"[{key}] {' '.join(parts)}")

    return "\n".join(lines)


def _extract_tex_bib_section(tex: str) -> str:
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


def extract(file_path: str, bib_path: str = None) -> dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext in (".tex", ".latex"):
        return extract_latex(file_path, bib_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .tex")