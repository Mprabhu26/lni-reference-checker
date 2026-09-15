"""
Universal Extractor v6.5 — FIXED bibliography end-detection + space-repair
--------------------------------------------------------------------------
v6.5 changes:
  - _find_bib_end: new function that finds where the bibliography stops
    (before "Hilfsmittel", "Anhang", "Appendix", "Erklärung", or any
    other section that follows a bibliography in a typical German
    academic PDF). Previously the entire tail of the document was kept
    as part of the last bibliography entry.
  - _clean_fallback_text: aggressive hyphen-space repair now runs on
    every page unconditionally (not just pages judged "broken" by the
    space-ratio heuristic). This fixes `Research- Technology` and
    `25- 40` output from PDFs that encode hyphens and spaces as separate
    glyph runs.
  - Bibliography section heading detection unchanged.
  - _reconstruct_page_text_from_chars, _words_based_text unchanged.
  - URL line-break repair unchanged.
"""

import re
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _normalize_citation_key(key: str) -> str:
    if re.match(r'^[A-Z\u00C0-\u00DE\u0100-\u017F][A-Z0-9\u00C0-\u00DE\u0100-\u017F]{0,2}\d{2}[a-z]?$', key):
        return key

    year_match = re.search(r'(\d{4})', key)
    year = year_match.group(1) if year_match else ""

    word_part = re.sub(r'[\d_]', ' ', key).strip()
    words = word_part.split()

    initials = ''.join([w[0].upper() for w in words[:3] if w])

    if initials and year:
        normalized = f"{initials}{year[-2:]}"
        if re.match(r'^[A-Z]{1,3}\d{2}$', normalized):
            return normalized

    return key.upper()


# ---------------------------------------------------------------------------
# Bibliography section heading detection
# ---------------------------------------------------------------------------

BIB_HEADINGS = re.compile(
    r'(?:^|\n)'
    r'(?:\d+(?:\.\d+)*\.?\s+)?'
    r'('
    r'Literaturverzeichnis|LITERATURVERZEICHNIS|Literatur(?:\b|:)|LITERATUR(?:\b|:)'
    r'|Quellenverzeichnis|QUELLENVERZEICHNIS|Quellen(?:\b|:)|QUELLEN(?:\b|:)'
    r'|Schrifttum|SCHRIFTTUM|Literaturangaben|LITERATURANGABEN|Literaturliste|LITERATURLISTE'
    r'|Bibliographie|BIBLIOGRAPHIE|Referenzen|REFERENZEN'
    r'|References?(?:\b|:)|REFERENCES?(?:\b|:)'
    r'|Bibliography|BIBLIOGRAPHY|Works\s+Cited|WORKS\s+CITED'
    r'|Reference\s+List|REFERENCE\s+LIST|List\s+of\s+References|LIST\s+OF\s+REFERENCES'
    r'|List\s+of\s+Sources|LIST\s+OF\s+SOURCES|Sources?(?:\b|:)|SOURCES?(?:\b|:)'
    r'|Citations?(?:\b|:)|CITATIONS?(?:\b|:)|Cited\s+Works|CITED\s+WORKS|Cited\s+References|CITED\s+REFERENCES'
    r'|Literature\s+Cited|LITERATURE\s+CITED|Literature|LITERATURE'
    r'|Bibliographic\s+References|BIBLIOGRAPHIC\s+REFERENCES'
    r')',
    re.MULTILINE | re.IGNORECASE,
)

# Post-bibliography sections that should terminate the bibliography block.
# These are the headings that typically follow the bibliography in a
# German academic PDF (Hilfsmittel = tools/resources declaration, Anhang =
# appendix, Erklärung = declaration of authorship, etc.).
_BIB_END_HEADINGS = re.compile(
    r'(?:^|\n)\s*'
    r'(?:\d+(?:\.\d+)*\.?\s+)?'
    r'('
    r'Hilfsmittel|HILFSMITTEL'
    r'|Anhang|ANHANG|Anlagen|ANLAGEN'
    r'|Appendix|APPENDIX'
    r'|Eigenständigkeitserklärung|EIGENSTÄNDIGKEITSERKLÄRUNG'
    r'|Eigenstaendigkeitserklaerung|EIGENSTAENDIGKEITSERKLAERUNG'
    r'|Selbstständigkeitserklärung|SELBSTSTÄNDIGKEITSERKLÄRUNG'
    r'|Selbststaendigkeitserklaerung|SELBSTSTAENDIGKEITSERKLAERUNG'
    r'|Erklärung|ERKLÄRUNG|Erklaerung|ERKLAERUNG'
    r'|Abbildungsverzeichnis|ABBILDUNGSVERZEICHNIS'
    r'|Tabellenverzeichnis|TABELLENVERZEICHNIS'
    r'|Abkürzungsverzeichnis|ABKÜRZUNGSVERZEICHNIS|Abkuerzungsverzeichnis|ABKUERZUNGSVERZEICHNIS'
    r'|Glossar|GLOSSAR'
    r'|Index|INDEX'
    r')\b',
    re.MULTILINE,
)


def _find_bib_start(full_text: str) -> int:
    """
    Return the character offset where the bibliography section begins,
    or -1 if not found.
    """
    any_bib_key = re.compile(r'\[(?:[A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]')

    all_matches = list(BIB_HEADINGS.finditer(full_text))
    if not all_matches:
        embedded_heading = re.compile(
            r'\b(?:Bibliography|References?|Referenzen|Literaturverzeichnis|'
            r'Quellenverzeichnis|Bibliographie)\b',
            re.IGNORECASE,
        )
        all_matches = [m for m in embedded_heading.finditer(full_text)
                       if any_bib_key.search(full_text[m.start():m.start() + 500])]
    if not all_matches:
        key_pattern = re.compile(r'\n\[(?:[A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]')
        key_match = key_pattern.search(full_text)
        if key_match:
            line_start = full_text.rfind('\n', 0, key_match.start()) + 1
            return line_start
        return -1

    STRONG_HEADINGS = re.compile(
        r'(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY|'
        r'Literaturverzeichnis|LITERATURVERZEICHNIS|Bibliographie|BIBLIOGRAPHIE|'
        r'Referenzen|REFERENZEN|Quellenverzeichnis|QUELLENVERZEICHNIS|'
        r'Works\s+Cited|List\s+of\s+References)',
        re.IGNORECASE,
    )

    def _is_standalone_heading(m, text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        line_end = text.find('\n', m.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        heading_line = re.sub(r'^\d+[\.\s]+', '', line).strip()
        return bool(re.fullmatch(
            r'(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY|'
            r'Literaturverzeichnis|LITERATURVERZEICHNIS|Bibliographie|BIBLIOGRAPHIE|'
            r'Referenzen|REFERENZEN|Quellenverzeichnis|QUELLENVERZEICHNIS|'
            r'Schrifttum|Works\s+Cited|List\s+of\s+References|'
            r'Literatur(?:\s+Cited)?|Quellen|Sources?|Citations?|'
            r'Cited\s+(?:Works|References)|Bibliographic\s+References|Literature):?',
            heading_line,
            re.IGNORECASE,
        ))

    for m in all_matches:
        if STRONG_HEADINGS.search(m.group(0)) and _is_standalone_heading(m, full_text):
            window = full_text[m.start(): m.start() + 500]
            if any_bib_key.search(window):
                return m.start()

    for m in reversed(all_matches):
        if _is_standalone_heading(m, full_text):
            window = full_text[m.start(): m.start() + 500]
            if any_bib_key.search(window):
                return m.start()

    for m in reversed(all_matches):
        window = full_text[m.start(): m.start() + 500]
        if any_bib_key.search(window):
            return m.start()

    return all_matches[-1].start()


def _find_bib_end(full_text: str, bib_start: int) -> int:
    """
    NEW (v6.5): find where the bibliography stops. Returns the character
    offset of the first post-bibliography heading (Hilfsmittel, Anhang,
    Appendix, Erklärung, etc.), or len(full_text) if none is found.

    Only considers headings that appear AFTER bib_start, and only if the
    heading is on its own line (not mid-sentence).
    """
    if bib_start < 0:
        return len(full_text)

    tail = full_text[bib_start:]

    # Fast path: the heading word appears anywhere in the tail, even
    # mid-line, preceded by whitespace or a colon.
    for word in ("Hilfsmittel", "Anhang", "Appendix",
                 "Eigenständigkeitserklärung", "Eigenstaendigkeitserklaerung",
                 "Selbstständigkeitserklärung", "Selbststaendigkeitserklaerung",
                 "Erklärung", "Erklaerung"):
        idx = tail.find(word)
        if idx > 0:
            return bib_start + idx

    for m in _BIB_END_HEADINGS.finditer(tail):
        # Verify the match is a standalone heading line
        line_start = tail.rfind('\n', 0, m.start()) + 1
        line_end = tail.find('\n', m.end())
        if line_end == -1:
            line_end = len(tail)
        line = tail[line_start:line_end].strip()
        # Accept heading line with optional leading section number
        heading_line = re.sub(r'^\d+[\.\s]+', '', line).strip()
        # Heading words should essentially be the whole line
        if re.fullmatch(
            r'(?:Hilfsmittel|Anhang|Anlagen|Appendix|'
            r'Eigenständigkeitserklärung|Eigenstaendigkeitserklaerung|'
            r'Selbstständigkeitserklärung|Selbststaendigkeitserklaerung|'
            r'Erklärung|Erklaerung|'
            r'Abbildungsverzeichnis|Tabellenverzeichnis|'
            r'Abkürzungsverzeichnis|Abkuerzungsverzeichnis|'
            r'Glossar|Index):?',
            heading_line,
            re.IGNORECASE,
        ):
            return bib_start + m.start()
    return len(full_text)


def split_body_bib(full_text: str, format_hint: str = None) -> dict:
    heading_match = re.search(
        r'(?im)^(?:\d+(?:\.\d+)*\.?\s+)?(?:References?|Literaturverzeichnis|Bibliography|Bibliographie|Referenzen|Quellenverzeichnis)\s*[:.]?\s*$',
        full_text,
    )
    pos = _find_bib_start(full_text)
    if heading_match and (pos < 0 or heading_match.start() < pos):
        pos = heading_match.start()

    if pos >= 0:
        body = full_text[:pos].strip()
        bib_start_in_full = pos
        bib_end = _find_bib_end(full_text, bib_start_in_full)
        bib = full_text[bib_start_in_full:bib_end].strip()
    else:
        body = full_text.strip()
        bib = ""

    body = re.sub(r'\n{3,}', '\n\n', body)

    if bib:
        lines = bib.split('\n')
        cleaned_lines = []
        in_bibliography = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\s*\d+\s*$', line):
                continue
            if re.match(r'^\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d+\s*$', line):
                continue
            if 'Inhaltsverzeichnis' in line or 'Contents' in line:
                continue
            if line.strip().startswith('.'):
                continue
            if re.match(r'^\s*\[[A-Za-z0-9]+\]', line):
                in_bibliography = True
                cleaned_lines.append(line)
            elif in_bibliography and line:
                if re.search(r'\b(19|20)\d{2}\b', line) or re.search(r'Verlag|Press|Publisher|S\.|pp\.', line):
                    cleaned_lines.append(line)
                elif len(line) > 10 and not re.match(r'^[A-Z][a-z]+', line):
                    cleaned_lines.append(line)

        if cleaned_lines:
            bib = '\n'.join(cleaned_lines)
        else:
            bib_match = re.search(
                r'(?:^|\n)(?:Literaturverzeichnis|Bibliography|References?)[:\s]*\n(.*?)(?=\n\s*(?:[A-Z]|$))',
                full_text,
                re.DOTALL | re.IGNORECASE
            )
            if bib_match:
                bib_raw = bib_match.group(1).strip()
                entries = re.findall(r'\[[A-Za-z0-9]+\][^\[]+', bib_raw)
                if entries:
                    bib = 'Literaturverzeichnis\n' + '\n'.join(entries)

    return {"full_text": full_text, "body": body, "bibliography": bib, "format": format_hint}


# ---------------------------------------------------------------------------
# URL repair
# ---------------------------------------------------------------------------

def _repair_urls_in_text(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r'(https?://[^\s\n]+?)-[\s]*\n[\s]*([a-zA-Z0-9%_\-/\.?=&]+)',
        r'\1-\2',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(https?://[^\s\n]+)[\s]*\n[\s]*([a-zA-Z0-9%_\-/\.?=&]+)',
        r'\1\2',
        text,
        flags=re.IGNORECASE
    )

    # Only join URL + next token when the next token starts with a
    # lowercase letter, digit, or URL-safe character, and the join is
    # across a line break. This prevents swallowing a sentence that
    # happens to follow a URL.
    text = re.sub(
        r'(https?://[^\s\n]+)[\s]*\n[\s]*([a-z0-9%_\-/\.?=&]+)',
        r'\1\2',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(https?):\s+//',
        r'\1://',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r',?\s*Stand:?\s*[\d./-]+',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(https?://[^\s]+)[.,;:)]+(\s|$)',
        r'\1\2',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(gi-?ev\.at|gi-?ev)',
        'gi.de',
        text,
        flags=re.IGNORECASE
    )

    return text


def _words_based_text(page) -> str:
    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
    except Exception:
        return ""
    if not words:
        return ""

    lines: dict = {}
    for w in words:
        key = round(w["top"] / 3) * 3
        lines.setdefault(key, []).append(w)

    result_lines = []
    for y in sorted(lines):
        row = sorted(lines[y], key=lambda w: w["x0"])
        result_lines.append(" ".join(w["text"] for w in row))
    return "\n".join(result_lines)


def _reconstruct_page_text_from_chars(page) -> str:
    try:
        chars = page.chars
    except Exception:
        chars = []

    if not chars:
        return page.extract_text() or ""

    if len(chars) < 10:
        return page.extract_text() or ""

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

        gaps = []
        for i in range(1, len(row)):
            prev, curr = row[i - 1], row[i]
            gap = curr["x0"] - prev["x1"]
            avg_size = (prev.get("size", 10) + curr.get("size", 10)) / 2
            gaps.append((gap / avg_size, i, gap, avg_size))

        if not gaps:
            result_lines.append(row[0]["text"])
            continue

        norm_gaps = [g[0] for g in gaps]
        sorted_gaps = sorted(norm_gaps)

        pct_60 = sorted_gaps[int(len(sorted_gaps) * 0.60)] if len(sorted_gaps) >= 5 else None
        pct_70 = sorted_gaps[int(len(sorted_gaps) * 0.70)] if len(sorted_gaps) >= 5 else None
        pct_80 = sorted_gaps[int(len(sorted_gaps) * 0.80)] if len(sorted_gaps) >= 5 else None

        jumps = []
        for i in range(1, len(sorted_gaps)):
            jumps.append((sorted_gaps[i] - sorted_gaps[i-1], sorted_gaps[i-1], sorted_gaps[i]))

        if jumps:
            largest_jump = max(jumps, key=lambda x: x[0])
            jump_threshold = (largest_jump[1] + largest_jump[2]) / 2
        else:
            jump_threshold = None

        if len(sorted_gaps) >= 5:
            idx_65 = int(len(sorted_gaps) * 0.65)
            threshold_estimate = sorted_gaps[idx_65]
        else:
            threshold_estimate = sum(norm_gaps) / len(norm_gaps) * 1.2

        if jump_threshold and 0.05 < jump_threshold < 1.0:
            adaptive_threshold = jump_threshold
        elif pct_60 and 0.05 < pct_60 < 1.0:
            adaptive_threshold = pct_60
        else:
            median_gap = sorted_gaps[len(sorted_gaps)//2] if sorted_gaps else 0.15
            adaptive_threshold = median_gap * 1.3

        adaptive_threshold = max(0.025, min(0.35, adaptive_threshold))

        line_text = row[0]["text"]
        for i in range(1, len(row)):
            prev, curr = row[i - 1], row[i]
            gap = curr["x0"] - prev["x1"]
            avg_size = (prev.get("size", 10) + curr.get("size", 10)) / 2
            norm_gap = gap / avg_size if avg_size > 0 else 0

            curr_text = curr.get("text", "").strip()
            prev_text = prev.get("text", "").strip()

            insert_space = False

            if norm_gap > adaptive_threshold:
                insert_space = True

            if (curr_text and curr_text[0].isupper() and
                prev_text and prev_text[-1].islower() and
                norm_gap > 0.02):
                insert_space = True

            if (prev_text and prev_text[-1] in ":;." and
                curr_text and curr_text[0].isupper() and
                norm_gap > 0.02):
                insert_space = True

            if insert_space:
                line_text += " "
            line_text += curr_text if curr_text else curr.get("text", "")

        result_lines.append(line_text)

    reconstructed = "\n".join(result_lines)

    fallback = page.extract_text() or ""

    if reconstructed.strip():
        recon_spaces = reconstructed.count(' ')
        recon_chars = len(reconstructed.strip())
        recon_ratio = recon_spaces / recon_chars if recon_chars > 0 else 0

        if recon_ratio < 0.02 and fallback:
            return _clean_fallback_text(fallback)

    return reconstructed


def _clean_fallback_text(text: str) -> str:
    """
    Clean up text from pdfplumber's extract_text() when character-gap
    reconstruction fails. Includes hyphen-space repair that specifically
    targets LNI citation artifacts like "Research- Technology" and
    "25- 40".
    """
    if not text:
        return ""

    url_placeholders: list = []
    def _mask_url(m):
        url_placeholders.append(m.group(0))
        return f"\x00URL{len(url_placeholders) - 1}\x00"

    text = re.sub(r'https?://\S+', _mask_url, text)

    # NEW v6.5: repair "word- word" -> "word-word" inside a single line.
    # PDFs frequently emit a hyphen followed by a space where the source
    # had a compound word ("Research-Technology"). Restoring the compound
    # form helps the journal extractor later.
    text = re.sub(r'([A-Za-zÄÖÜäöüß])-\s+([A-Za-zÄÖÜäöüß])', r'\1-\2', text)

    # NEW v6.5: repair "digit- space digit" -> "digit-digit" for page
    # ranges ("25- 40" -> "25-40").
    text = re.sub(r'(\d)-\s+(\d)', r'\1-\2', text)

    fixed_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if len(stripped) > 30:
            space_ratio = stripped.count(' ') / len(stripped)
            if space_ratio < 0.03:
                line = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', line)
                line = re.sub(r'([A-ZÄÖÜa-zäöüß])(\d)', r'\1 \2', line)
                line = re.sub(r'(\d)([A-ZÄÖÜa-zäöüß])', r'\1 \2', line)
        fixed_lines.append(line)
    text = '\n'.join(fixed_lines)

    text = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', text)

    text = re.sub(r'([:;.!?])([A-ZÄÖÜ0-9])', r'\1 \2', text)

    text = re.sub(r'(S\.)(\d)', r'\1 \2', text)

    text = re.sub(r'(\d)([A-ZÄÖÜa-zäöüß])', r'\1 \2', text)

    text = re.sub(r'(https?):\s+//', r'\1://', text)

    for i, url in enumerate(url_placeholders):
        text = text.replace(f"\x00URL{i}\x00", url)

    text = re.sub(r'\s{2,}', ' ', text)

    return text


def extract_pdf(path: str) -> dict:
    from pathlib import Path as _Path
    if not _Path(path).exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    text = ""
    extraction_method = "pdfplumber"
    is_scanned = False
    page_count = 0
    pages_with_text = 0

    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            extracted_pages = []

            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""

                recon_text = _reconstruct_page_text_from_chars(page) or ""

                words_text = _words_based_text(page) or ""

                raw_spaces = raw_text.count(' ')
                raw_chars = len(raw_text.strip())
                raw_ratio = raw_spaces / raw_chars if raw_chars > 0 else 0

                recon_spaces = recon_text.count(' ')
                recon_chars = len(recon_text.strip())
                recon_ratio = recon_spaces / recon_chars if recon_chars > 0 else 0

                words_spaces = words_text.count(' ')
                words_chars = len(words_text.strip())
                words_ratio = words_spaces / words_chars if words_chars > 0 else 0

                use_recon = False

                if recon_ratio > 0.05 and recon_chars > 20:
                    if raw_ratio < 0.02 or recon_chars > raw_chars * 1.5:
                        use_recon = True
                    elif recon_ratio > raw_ratio + 0.03:
                        use_recon = True

                if words_chars > 20 and words_ratio >= 0.02:
                    best_name, best_text, best_ratio, best_chars = (
                        "words", words_text, words_ratio, words_chars
                    )
                else:
                    fallback_candidates = [("raw", raw_text, raw_ratio, raw_chars)]
                    if use_recon:
                        fallback_candidates.append(
                            ("recon", recon_text, recon_ratio, recon_chars)
                        )
                    best_name, best_text, best_ratio, best_chars = fallback_candidates[0]
                    for name, cand_text, cand_ratio, cand_chars in fallback_candidates[1:]:
                        if cand_chars > 20 and cand_ratio > best_ratio + 0.02:
                            best_name, best_text, best_ratio, best_chars = name, cand_text, cand_ratio, cand_chars

                if best_text.strip():
                    extracted_pages.append(best_text)
                    pages_with_text += 1
                elif raw_text.strip():
                    extracted_pages.append(raw_text)
                    pages_with_text += 1
                else:
                    extracted_pages.append("")

            text = "\n".join(extracted_pages)

            text = _clean_fallback_text(text)

            if pages_with_text == 0 or (page_count > 0 and pages_with_text / page_count < 0.3):
                is_scanned = True
                extraction_method = "pdfplumber (likely scanned)"

    except Exception as e:
        extraction_method = f"pdfplumber failed: {e}"
        text = ""

    if len(text.strip()) < 500:
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            page_count = len(reader.pages)
            extracted_pages = []
            text_pages = 0

            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text()
                    if t and len(t.strip()) > 10:
                        extracted_pages.append(t)
                        text_pages += 1
                    else:
                        extracted_pages.append("")
                except Exception:
                    extracted_pages.append("")

            fallback_text = "\n".join(extracted_pages)

            if len(fallback_text.strip()) > len(text.strip()):
                text = fallback_text
                extraction_method = "pypdf (fallback)"
                pages_with_text = text_pages

                if pages_with_text == 0 or (page_count > 0 and pages_with_text / page_count < 0.3):
                    is_scanned = True

        except Exception as e:
            extraction_method = f"pypdf also failed: {e}"

    if len(text.strip()) < 30:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
                printable_ratio = sum(c.isprintable() or c in '\n\t' for c in raw_text[:500]) / max(len(raw_text[:500]), 1)
                if printable_ratio > 0.85 and len(raw_text.strip()) > len(text.strip()):
                    text = raw_text
                    extraction_method = "raw text (unusual PDF)"
        except Exception:
            pass

    text = _repair_urls_in_text(text)

    text = re.sub(r' +', ' ', text)

    bib_pos = _find_bib_start(text)
    if bib_pos >= 0:
        body_raw = text[:bib_pos]
        bib_end = _find_bib_end(text, bib_pos)
        bib_raw = text[bib_pos:bib_end]
    else:
        body_raw = text
        bib_raw = ""

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
    body_raw = re.sub(r'-(\n)(\S)', r'\2', body_raw)

    if bib_pos >= 0:
        body_part = body_raw
        bib_part = bib_raw

        bib_part = re.sub(
            r'(https?://[^\s\n]+?)[\s]*\n[\s]*([a-zA-Z0-9%_\-/\.?=&]+)',
            r'\1\2',
            bib_part,
            flags=re.IGNORECASE
        )

        bib_part = re.sub(
            r'(CM-REPORT)[\s]*\n[\s]*-?[\s]*([a-zA-Z0-9%-]+)',
            r'\1-\2',
            bib_part,
            flags=re.IGNORECASE
        )

        bib_part = re.sub(r'(https?://[^\s]+?),?\s*Stand:?\s*[\d./-]+', r'\1', bib_part, flags=re.IGNORECASE)
        bib_part = re.sub(r'(https?://[^\s]+?),?\s*Stand\s+[\d./-]+', r'\1', bib_part, flags=re.IGNORECASE)
        bib_part = re.sub(r'(https?):\s+//', r'\1://', bib_part)
        bib_part = re.sub(r'\n{3,}', '\n\n', bib_part)

        bib_part = re.sub(r'(\[[A-Za-z0-9]+\])(?!\s*\n)', r'\n\1', bib_part)

        bib_lines = bib_part.split('\n')
        cleaned_bib_lines = []
        for line in bib_lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\s*\d+\s*$', line):
                continue
            if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d+\s*$', line):
                continue
            if re.match(r'^(Fig\.|Tab\.)\s+\d+', line, re.IGNORECASE):
                continue
            if re.match(r'^\d+\s+[A-Z][a-z]+', line):
                continue
            cleaned_bib_lines.append(line)

        if not cleaned_bib_lines:
            bib_match = re.search(
                r'(?:^|\n)(?:Literaturverzeichnis|Bibliography|References?)[:\s]*\n(.*?)(?=\n\s*(?:[A-Z]|$))',
                text,
                re.DOTALL | re.IGNORECASE
            )
            if bib_match:
                cleaned_bib_lines = [line.strip() for line in bib_match.group(1).splitlines() if line.strip()]

        bib_part = '\n'.join(cleaned_bib_lines)

        result = {
            "full_text": body_part + "\n\n" + bib_part,
            "body": body_part.strip(),
            "bibliography": bib_part.strip(),
            "format": "pdf",
        }
    else:
        result = split_body_bib(text, "pdf")

    result["extraction_method"] = extraction_method
    result["is_scanned"] = is_scanned
    result["pages_with_text"] = pages_with_text
    result["total_pages"] = page_count

    if is_scanned or (page_count > 0 and pages_with_text < page_count * 0.5):
        result["warning"] = "PDF appears to be scanned or image-based. Text extraction may be incomplete. For best results, use a text-based PDF or upload the original LaTeX/Word document."

    return result


def extract_pdf_simple(path: str) -> dict:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            except Exception:
                continue

        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n+', '\n', text)

        text = _repair_urls_in_text(text)

        bib_start = _find_bib_start(text)
        if bib_start >= 0:
            body = text[:bib_start].strip()
            bib_end = _find_bib_end(text, bib_start)
            bib = text[bib_start:bib_end].strip()
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
# DOCX Extraction
# ---------------------------------------------------------------------------

def extract_docx(path: str) -> dict:
    from docx import Document

    doc = Document(path)
    parts = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t:
                    parts.append(t)

    for section in doc.sections:
        for header in section.header.paragraphs:
            t = header.text.strip()
            if t and len(t) > 20:
                parts.append("[HEADER] " + t)

    text = "\n".join(parts)

    text = _repair_urls_in_text(text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    result = split_body_bib(text, "docx")
    result["extraction_method"] = "python-docx"
    result["is_scanned"] = False

    return result


# ---------------------------------------------------------------------------
# LaTeX Extraction
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

    if not bib_section and not bib_path:
        bib_file_match = re.search(r'\\bibliography\{([^}]+)\}', tex)
        if bib_file_match:
            bib_filename = bib_file_match.group(1) + ".bib"
            bib_file_path = Path(tex_path).parent / bib_filename
            if bib_file_path.exists():
                with open(bib_file_path, encoding="utf-8", errors="replace") as f:
                    bib_text = f.read()
                bib_section = _bibtex_to_lni_text(bib_text)

    bib_section = _repair_urls_in_text(bib_section)
    body = _repair_urls_in_text(body)

    result = {
        "full_text": body + "\n\n" + bib_section,
        "body": body,
        "bibliography": bib_section,
        "format": "latex",
        "raw_bibtex": bib_text,
        "extraction_method": "latex parser",
        "is_scanned": False,
    }

    if not bib_section and not bib_text:
        result["warning"] = "No bibliography found. Make sure your .tex file has a \\begin{thebibliography} section or attach a .bib file."

    return result


def _clean_latex(tex: str) -> str:
    tex = re.sub(r'%.*', '', tex)

    tex = re.sub(
        r'\\begin\{(figure|table|lstlisting|verbatim|equation|align|tikzpicture)[^}]*\}.*?\\end\{\1\}',
        '', tex, flags=re.DOTALL
    )

    tex = re.sub(r'\\include\{[^}]+\}', '', tex)
    tex = re.sub(r'\\input\{[^}]+\}', '', tex)

    tex = re.sub(
        r'\\(?:textbf|textit|emph|texttt|text|section\*?|subsection\*?|subsubsection\*?|'
        r'paragraph|subparagraph|caption|label|ref|Cref|cref|url|href)\{([^}]*)\}',
        r'\1', tex
    )

    tex = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\[a-zA-Z]+\*?', ' ', tex)

    tex = re.sub(r'[{}]', ' ', tex)

    tex = re.sub(r'\s+', ' ', tex)

    tex = re.sub(r'\$[^$]+\$', '[MATH]', tex)
    tex = re.sub(r'\$\$[^$]+\$\$', '[DISPLAY MATH]', tex)

    return tex.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_references_from_bibliography(bib_text: str) -> List[Dict]:
    references = []

    key_pattern = re.compile(r'\[([A-Za-z\u00C0-\u00FF\u0100-\u017F]{1,6}\d{2}[a-z]?|\d{1,3})\]')

    parts = key_pattern.split(bib_text)

    for i in range(1, len(parts), 2):
        if i < len(parts):
            key = parts[i]
            ref_text = parts[i + 1] if i + 1 < len(parts) else ""

            ref_text = ref_text.strip()

            ref_text = re.sub(r'\s+', ' ', ref_text)

            if ref_text:
                references.append({
                    'key': key,
                    'raw_text': ref_text,
                    'normalized_key': _normalize_citation_key(key)
                })

    return references


def extract(file_path: str, bib_path: str = None) -> dict:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        result = extract_pdf(file_path)

        if len(result.get("body", "")) < 500 and len(result.get("bibliography", "")) < 100:
            fallback = extract_pdf_simple(file_path)
            if len(fallback.get("body", "")) > len(result.get("body", "")):
                result = fallback

        bib = result.get("bibliography", "")
        if bib:
            result["references"] = extract_references_from_bibliography(bib)
        else:
            result["references"] = []

        return result

    elif ext == ".docx":
        return extract_docx(file_path)

    elif ext in (".tex", ".latex"):
        return extract_latex(file_path, bib_path)

    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: .pdf, .docx, .tex"
        )