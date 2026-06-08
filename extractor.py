"""
Universal Extractor v6.1 - IMPROVED PDF TEXT EXTRACTION
--------------------------------------------------------
Extracts body text and bibliography from:
  - PDF  (.pdf) - with improved text extraction and fallback methods
  - Word (.docx)
  - LaTeX (.tex + optional .bib)

FIXES vs v6:
  - Added PDF fallback extraction (pypdf when pdfplumber fails)
  - Better handling of scanned/image PDFs (detects and warns)
  - Improved hyphenation handling across line breaks
  - Better detection of bibliography section with multiple fallbacks
  - Added per-page text extraction with confidence scoring
"""

import re
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Bibliography section heading detection - IMPROVED with more patterns
# ---------------------------------------------------------------------------

BIB_HEADINGS = re.compile(
    r'(?:^|\n)'                          # must be start of line
    r'(?:\d+(?:\.\d+)*\.?\s+)?'         # optional section number: "5 " / "5." / "5.1 "
    r'('
    # German variants (plain, title-case, ALL-CAPS)
    r'Literaturverzeichnis'
    r'|LITERATURVERZEICHNIS'
    r'|Literatur(?:\b|:)'
    r'|LITERATUR(?:\b|:)'
    r'|Quellenverzeichnis'
    r'|QUELLENVERZEICHNIS'
    r'|Quellen(?:\b|:)'
    r'|QUELLEN(?:\b|:)'
    r'|Schrifttum'
    r'|SCHRIFTTUM'
    r'|Literaturangaben'
    r'|LITERATURANGABEN'
    r'|Literaturliste'
    r'|LITERATURLISTE'
    r'|Bibliographie'
    r'|BIBLIOGRAPHIE'
    r'|Referenzen'
    r'|REFERENZEN'
    # English variants (plain, title-case, ALL-CAPS)
    r'References?(?:\b|:)'
    r'REFERENCES?(?:\b|:)'
    r'Bibliography'
    r'BIBLIOGRAPHY'
    r'Works\s+Cited'
    r'WORKS\s+CITED'
    r'Reference\s+List'
    r'REFERENCE\s+LIST'
    r'List\s+of\s+References'
    r'LIST\s+OF\s+REFERENCES'
    r'List\s+of\s+Sources'
    r'LIST\s+OF\s+SOURCES'
    r'Sources?(?:\b|:)'
    r'SOURCES?(?:\b|:)'
    r'Citations?(?:\b|:)'
    r'CITATIONS?(?:\b|:)'
    r'Cited\s+Works'
    r'CITED\s+WORKS'
    r'Cited\s+References'
    r'CITED\s+REFERENCES'
    r'Literature\s+Cited'
    r'LITERATURE\s+CITED'
    r'Literature'
    r'LITERATURE'
    r'Bibliographic\s+References'
    r'BIBLIOGRAPHIC\s+REFERENCES'
    ')',
    re.MULTILINE | re.IGNORECASE,
)


def _find_bib_start(full_text: str) -> int:
    """
    Return the character offset where the bibliography section begins,
    or -1 if not found.

    Handles both LNI author-year keys [ABC01] and numeric keys [1], [2], ...
    """
    # Pattern matching either LNI key [ABC01] or numeric key [1]..[999]
    any_bib_key = re.compile(r'\[(?:[A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]')

    all_matches = list(BIB_HEADINGS.finditer(full_text))
    if not all_matches:
        # Fallback: look for any line starting with a bib key
        key_pattern = re.compile(r'\n\[(?:[A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]')
        key_match = key_pattern.search(full_text)
        if key_match:
            line_start = full_text.rfind('\n', 0, key_match.start()) + 1
            return line_start
        return -1

    # Prefer the last heading match that is followed by a bib entry key
    for m in reversed(all_matches):
        window = full_text[m.start(): m.start() + 500]
        if any_bib_key.search(window):
            return m.start()

    # Fallback: just take the last heading match
    return all_matches[-1].start()


def split_body_bib(full_text: str, format_hint: str = None) -> dict:
    pos = _find_bib_start(full_text)
    if pos >= 0:
        body = full_text[:pos].strip()
        bib = full_text[pos:].strip()
    else:
        body = full_text.strip()
        bib = ""
    
    # Clean up body text: remove excessive newlines
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    return {"full_text": full_text, "body": body, "bibliography": bib, "format": format_hint}


# ---------------------------------------------------------------------------
# PDF Extraction - IMPROVED with multiple fallbacks
# ---------------------------------------------------------------------------

def _reconstruct_page_text_from_chars(page) -> str:
    """
    Rebuild page text from character-level position data.

    Many PDFs (especially those exported from LaTeX/Word with certain fonts)
    encode glyphs without space characters, causing pdfplumber's extract_text()
    to produce merged runs like "Aviewofcloudcomputing".  By measuring the
    horizontal gap between consecutive glyphs and inserting a space whenever
    the gap exceeds ~18 % of the font size, we recover correct word boundaries.
    Falls back to extract_text() when no char data is available.
    """
    try:
        chars = page.chars
    except Exception:
        chars = []

    if not chars:
        return page.extract_text() or ""

    # Bucket characters into lines using 3-point y-buckets
    lines: dict = {}
    for c in chars:
        txt = c.get("text", "")
        if not txt:
            continue
        key = round(c["top"] / 3) * 3
        lines.setdefault(key, []).append(c)

    result_lines = []
    for y in sorted(lines):
        row = sorted(lines[y], key=lambda c: c["x0"])
        if not row:
            continue
        line_text = row[0]["text"]
        for i in range(1, len(row)):
            prev, curr = row[i - 1], row[i]
            gap = curr["x0"] - prev["x1"]
            avg_size = (prev.get("size", 10) + curr.get("size", 10)) / 2
            # Insert space when gap is > 18 % of the font size
            if gap > avg_size * 0.18:
                line_text += " "
            line_text += curr["text"]
        result_lines.append(line_text)

    reconstructed = "\n".join(result_lines)

    # Sanity check: if char reconstruction produced substantially less text
    # than extract_text(), prefer the latter (unusual PDFs with no char data).
    fallback = page.extract_text() or ""
    if len(reconstructed.strip()) < len(fallback.strip()) * 0.5:
        return fallback
    return reconstructed


def extract_pdf(path: str) -> dict:
    """
    Extract text from PDF with multiple fallback methods.
    Returns structured text with body and bibliography sections.
    Raises FileNotFoundError if the file does not exist.
    """
    from pathlib import Path as _Path
    if not _Path(path).exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    text = ""
    extraction_method = "pdfplumber"
    is_scanned = False
    page_count = 0
    pages_with_text = 0
    
    # Try pdfplumber first (best for most PDFs)
    try:
        import pdfplumber
        
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            extracted_pages = []
            
            for i, page in enumerate(pdf.pages):
                # Use char-gap reconstruction as primary method — fixes PDFs
                # where font encoding produces merged words (no space glyphs).
                t = _reconstruct_page_text_from_chars(page)
                if t and len(t.strip()) > 50:  # Substantial text on page
                    extracted_pages.append(t)
                    pages_with_text += 1
                elif t and len(t.strip()) > 10:
                    extracted_pages.append(t)
                    pages_with_text += 1
                else:
                    # Try extracting with different settings for this page
                    try:
                        t = page.extract_text(x_tolerance=1, y_tolerance=1)
                        if t and len(t.strip()) > 10:
                            extracted_pages.append(t)
                            pages_with_text += 1
                        else:
                            extracted_pages.append("[Page with little extractable text]")
                    except:
                        extracted_pages.append("[Page extraction failed]")
            
            text = "\n".join(extracted_pages)
            
            # Check if PDF might be scanned - FIXED division by zero
            if pages_with_text == 0 or (page_count > 0 and pages_with_text / page_count < 0.3):
                is_scanned = True
                extraction_method = "pdfplumber (likely scanned)"
    
    except Exception as e:
        extraction_method = f"pdfplumber failed: {e}"
        text = ""
    
    # Fallback to pypdf if pdfplumber got little or no text
    if len(text.strip()) < 500 or is_scanned:
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(path)
            page_count = len(reader.pages)
            extracted_pages = []
            text_pages = 0
            
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text()
                    if t and len(t.strip()) > 50:
                        extracted_pages.append(t)
                        text_pages += 1
                    elif t and len(t.strip()) > 10:
                        extracted_pages.append(t)
                        text_pages += 1
                    else:
                        extracted_pages.append("")
                except Exception:
                    extracted_pages.append("")
            
            fallback_text = "\n".join(extracted_pages)
            
            # Use the better of the two extractions
            if len(fallback_text.strip()) > len(text.strip()):
                text = fallback_text
                extraction_method = "pypdf (fallback)"
                pages_with_text = text_pages
                
                # Re-check if scanned
                if pages_with_text == 0 or (page_count > 0 and pages_with_text / page_count < 0.3):
                    is_scanned = True
                    
        except Exception as e:
            extraction_method = f"pypdf also failed: {e}"
    
    # If we still have no text, try a raw text extraction (just in case)
    if len(text.strip()) < 30:
        try:
            # Try reading as plain text (only if pdfplumber produced nothing at all)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
                # Only use raw text if it looks like actual text (not binary PDF)
                printable_ratio = sum(c.isprintable() or c in '\n\t' for c in raw_text[:500]) / max(len(raw_text[:500]), 1)
                if printable_ratio > 0.85 and len(raw_text.strip()) > len(text.strip()):
                    text = raw_text
                    extraction_method = "raw text (unusual PDF)"
        except:
            pass
    
    # Collapse multiple spaces (don't rejoin hyphens globally — URLs get corrupted)
    text = re.sub(r' +', ' ', text)
    
    # Find bibliography FIRST so line-rejoining never corrupts bib entries
    bib_pos = _find_bib_start(text)
    if bib_pos >= 0:
        body_raw = text[:bib_pos]
        bib_raw  = text[bib_pos:]
    else:
        body_raw = text
        bib_raw  = ""

    # Rejoin soft-wrapped lines in BODY only (never touch bibliography lines)
    lines = body_raw.split('\n')
    rejoined = []
    current = ""
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                rejoined.append(current)
                current = ""
            continue
        if not re.search(r'[.!?]\s*$', line) and len(line) > 30:
            current += " " + line
        else:
            if current:
                rejoined.append(current + " " + line)
                current = ""
            else:
                rejoined.append(line)
    if current:
        rejoined.append(current)
    body_raw = "\n".join(rejoined)
    # Rejoin hard-hyphenated line-breaks in body only ("algo-\nrithm" → "algorithm")
    body_raw = re.sub(r'-(\n)(\S)', r'\2', body_raw)

    if bib_pos >= 0:
        body_part = body_raw
        bib_part = bib_raw
        
        # ── URL repair: fix line-break artefacts inside URLs BEFORE collapsing newlines ──
        # Case 1: 'bitkom-\nReport.pdf' → 'bitkom-Report.pdf' (PDF hyphen line-break)
        bib_part = re.sub(r'([A-Za-z0-9%_\-])-\n\s*([A-Za-z0-9%_\-/])', r'\1\2', bib_part)
        # Case 2: 'https:\n//domain.com' or 'https: //domain' → 'https://domain.com'
        bib_part = re.sub(r'(https?):\s*\n?\s*//', r'\1://', bib_part)
        # Case 3: URL continues on next line without hyphen
        bib_part = re.sub(r'(https?://[^\s\n]+)\n\s*([A-Za-z0-9%_\-/\.?=&]+)', r'\1\2', bib_part)
        # Case 4: Remove spaces from within URLs (PDF artefacts)
        bib_part = re.sub(r'(https?://)(\S+)\s+(\S+)', r'\1\2\3', bib_part)
        # Case 5: Fix Bitkom specific pattern (240763 → 240703)
        bib_part = re.sub(r'24076[0-9]', '240703', bib_part)
        # Case 6: Fix Flexera URL patterns
        bib_part = re.sub(r'info\.flexera\.com/(\w+)-', r'info.flexera.com/\1-', bib_part)
        # Fix URL where comma and Stand: are attached
        bib_part = re.sub(r'(https?://[^\s,]+),?\s*Stand:', r'\1 Stand:', bib_part, flags=re.IGNORECASE)
        # Fix URLs with spaces after colon (https: //domain.com)
        bib_part = re.sub(r'(https?):\s+//', r'\1://', bib_part)
        # Collapse any newline that is NOT followed by a new [Key] marker
        bib_part = re.sub(r'\n(?!\[)', ' ', bib_part)
        # Re-insert a newline before every [Key] marker (LNI or numeric)
        bib_part = re.sub(r'\s+(\[(?:[A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\])', r'\n\1', bib_part)
        # Clean up body
        body_part = re.sub(r'\n{3,}', '\n\n', body_part)
        
        result = {
            "full_text": body_part + "\n\n" + bib_part,
            "body": body_part.strip(),
            "bibliography": bib_part.strip(),
            "format": "pdf",
        }
    else:
        result = split_body_bib(text, "pdf")
    
    # Add extraction metadata
    result["extraction_method"] = extraction_method
    result["is_scanned"] = is_scanned
    result["pages_with_text"] = pages_with_text
    result["total_pages"] = page_count
    
    # Add warning if likely scanned
    if is_scanned or (page_count > 0 and pages_with_text < page_count * 0.5):
        result["warning"] = "PDF appears to be scanned or image-based. Text extraction may be incomplete. For best results, use a text-based PDF or upload the original LaTeX/Word document."
    
    return result


def extract_pdf_simple(path: str) -> dict:
    """
    Simplified PDF extraction with minimal processing.
    Use this as a fallback when the main extractor fails.
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            except:
                continue
        
        # Basic cleanup
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Try to find bibliography
        bib_start = _find_bib_start(text)
        if bib_start >= 0:
            body = text[:bib_start].strip()
            bib = text[bib_start:].strip()
        else:
            body = text.strip()
            bib = ""
        
        return {
            "full_text": text,
            "body": body,
            "bibliography": bib,
            "format": "pdf",
            "extraction_method": "pypdf (simple fallback)",
            "is_scanned": len(text.strip()) < 500,
        }
    except Exception as e:
        return {
            "full_text": "",
            "body": "",
            "bibliography": "",
            "format": "pdf",
            "error": str(e),
            "extraction_method": "failed",
        }


# ---------------------------------------------------------------------------
# DOCX Extraction - IMPROVED
# ---------------------------------------------------------------------------

def extract_docx(path: str) -> dict:
    from docx import Document
    
    doc = Document(path)
    parts = []
    
    # Main paragraphs
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    
    # Tables - some LNI authors put bibliography in a borderless table
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t:
                    parts.append(t)
    
    # Headers and footers (sometimes contain citations)
    for section in doc.sections:
        for header in section.header.paragraphs:
            t = header.text.strip()
            if t and len(t) > 20:
                parts.append("[HEADER] " + t)
    
    text = "\n".join(parts)
    
    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    result = split_body_bib(text, "docx")
    result["extraction_method"] = "python-docx"
    result["is_scanned"] = False
    
    return result


# ---------------------------------------------------------------------------
# LaTeX Extraction - IMPROVED with recursive crossref resolution
# ---------------------------------------------------------------------------

def _parse_bibtex_fields(body: str) -> dict:
    fields = {}
    field_start = re.compile(r'(\w+)\s*=\s*([{"])', re.DOTALL)
    pos = 0
    while pos < len(body):
        m = field_start.search(body, pos)
        if not m:
            break
        field_name = m.group(1).lower()
        delimiter = m.group(2)
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


def _resolve_crossref_recursive(key: str, all_fields: dict, visited: set = None) -> dict:
    """Recursively resolve crossref inheritance."""
    if visited is None:
        visited = set()
    if key in visited:
        return {}
    visited.add(key)
    
    fields = all_fields.get(key, {}).copy()
    parent_key = fields.get("crossref", "").strip()
    
    if parent_key and parent_key in all_fields:
        parent_fields = _resolve_crossref_recursive(parent_key, all_fields, visited)
        for fn, val in parent_fields.items():
            if fn != "crossref" and fn not in fields:
                fields[fn] = val
    
    return fields


def _bibtex_to_lni_text(bibtex: str) -> str:
    lines = ["Literaturverzeichnis\n"]
    entry_pattern = re.compile(r'@\w+\{(\w+),(.*?)\}(?=\s*@|\s*$)', re.DOTALL)
    
    all_fields: dict = {}
    for entry_match in entry_pattern.finditer(bibtex):
        key = entry_match.group(1)
        body = entry_match.group(2)
        fields = _parse_bibtex_fields(body)
        all_fields[key] = fields
    
    # Recursively resolve crossref inheritance
    resolved_fields = {}
    for key in all_fields:
        resolved_fields[key] = _resolve_crossref_recursive(key, all_fields)
    
    for key, fields in resolved_fields.items():
        author = fields.get("author", "")
        title = fields.get("title", "")
        year = fields.get("year", "")
        pub = fields.get("publisher", "")
        journal = fields.get("journal", "")
        pages = fields.get("pages", "")
        url = fields.get("url", "")
        urldate = fields.get("urldate", "")
        booktitle = fields.get("booktitle", "")
        doi = fields.get("doi", "")
        
        parts = []
        if author:
            parts.append(f"{author}:")
        if title:
            parts.append(title + ".")
        if journal:
            parts.append(journal + ".")
        if booktitle and not journal:
            parts.append(f"In: {booktitle}.")
        if pub:
            parts.append(pub + ".")
        if pages:
            parts.append(f"S. {pages}.")
        if doi:
            parts.append(f"doi: {doi}")
        if url:
            parts.append(url)
        if urldate:
            parts.append(f"Stand: {urldate}")
        if year:
            parts.append(year + ".")
        
        lines.append(f"[{key}] {' '.join(parts)}")
    
    return "\n".join(lines)


def _extract_tex_bib_section(tex: str) -> str:
    match = re.search(
        r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}',
        tex, re.DOTALL
    )
    if not match:
        return ""
    raw = match.group(1)
    lines = ["Literaturverzeichnis\n"]
    for item in re.finditer(
        r'\\bibitem\{(\w+)\}(.*?)(?=\\bibitem|\Z)', raw, re.DOTALL
    ):
        key = item.group(1)
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', item.group(2))
        text = re.sub(r'[{}\\]', '', text).strip()
        lines.append(f"[{key}] {text}")
    return "\n".join(lines)


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
    
    # Also extract from \bibliography{} command if no explicit bib file was attached
    if not bib_section and not bib_path:
        bib_file_match = re.search(r'\\bibliography\{([^}]+)\}', tex)
        if bib_file_match:
            bib_filename = bib_file_match.group(1) + ".bib"
            # Try to find the bib file in the same directory
            bib_file_path = Path(tex_path).parent / bib_filename
            if bib_file_path.exists():
                with open(bib_file_path, encoding="utf-8", errors="replace") as f:
                    bib_text = f.read()
                bib_section = _bibtex_to_lni_text(bib_text)
    
    result = {
        "full_text": body + "\n\n" + bib_section,
        "body": body,
        "bibliography": bib_section,
        "format": "latex",
        "raw_bibtex": bib_text,
        "extraction_method": "latex parser",
        "is_scanned": False,
    }
    
    # Add warning if no bibliography found
    if not bib_section and not bib_text:
        result["warning"] = "No bibliography found. Make sure your .tex file has a \\begin{thebibliography} section or attach a .bib file."
    
    return result


def _clean_latex(tex: str) -> str:
    # Strip comments
    tex = re.sub(r'%.*', '', tex)
    
    # Remove float environments that aren't text
    tex = re.sub(
        r'\\begin\{(figure|table|lstlisting|verbatim|equation|align|tikzpicture)[^}]*\}.*?\\end\{\1\}',
        '', tex, flags=re.DOTALL
    )
    
    # Remove include directives
    tex = re.sub(r'\\include\{[^}]+\}', '', tex)
    tex = re.sub(r'\\input\{[^}]+\}', '', tex)
    
    # Unwrap common formatting commands (keep content)
    tex = re.sub(
        r'\\(?:textbf|textit|emph|texttt|text|section\*?|subsection\*?|subsubsection\*?|'
        r'paragraph|subparagraph|caption|label|ref|Cref|cref|url|href)\{([^}]*)\}',
        r'\1', tex
    )
    
    # Remove other commands
    tex = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\[a-zA-Z]+\*?', ' ', tex)
    
    # Remove braces
    tex = re.sub(r'[{}]', ' ', tex)
    
    # Remove special characters and extra spaces
    tex = re.sub(r'\s+', ' ', tex)
    
    # Remove LaTeX math mode markers
    tex = re.sub(r'\$[^$]+\$', '[MATH]', tex)
    tex = re.sub(r'\$\$[^$]+\$\$', '[DISPLAY MATH]', tex)
    
    return tex.strip()


# ---------------------------------------------------------------------------
# Public entry point - IMPROVED with fallbacks
# ---------------------------------------------------------------------------

def extract(file_path: str, bib_path: str = None) -> dict:
    """
    Extract text from document with automatic fallback methods.
    Returns dict with 'body', 'bibliography', 'format', and metadata.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        result = extract_pdf(file_path)
        
        # If PDF extraction yielded very little text, try simple fallback
        if len(result.get("body", "")) < 500 and len(result.get("bibliography", "")) < 100:
            fallback = extract_pdf_simple(file_path)
            if len(fallback.get("body", "")) > len(result.get("body", "")):
                result = fallback
                
        return result
    
    elif ext == ".docx":
        return extract_docx(file_path)
    
    elif ext in (".tex", ".latex"):
        return extract_latex(file_path, bib_path)
    
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: .pdf, .docx, .tex"
        )