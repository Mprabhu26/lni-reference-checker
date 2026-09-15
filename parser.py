"""
STEP 2: Bibliography Parser
----------------------------
Parses the bibliography section of an LNI-formatted document.
Extracts citation keys like [AB00], [Ez10], [GI19] and their metadata.

v7.8 fixes:
  - Removed duplicate _extract_lni_venue_fallback definition. The
    second (old) definition was silently overriding the new one.
  - _extract_venue_structural now refuses to return a fragment that
    starts with a preposition ("of", "for", "in", "on", "and", "the").
    If the initial match starts with such a word, back up to include
    the preceding capitalized words. Fixes "of Production Economics"
    -> "International Journal of Production Economics".
  - Article journal extraction tries _extract_venue_structural before
    _extract_journal_smart, and rejects structurally suspicious results.
  - Author extraction: prefer the FIRST colon as the author/title
    boundary, not the last, so "Diener, F.; Špaček, M.: Title" yields
    authors "Diener, F.; Špaček, M." and title "Title".

v7.7 (preserved):
  - _normalize_author_list: "et al." moved to end.
  - Publisher extraction: word-boundary aware.

v7.4 (preserved):
  - Fixed duplicate entry merging, URL extraction, garbage filtering.
"""

import re
import html
import datetime
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BibEntry:
    key: str
    raw_text: str
    entry_type: Optional[str] = None
    authors: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    booktitle: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    urldate: Optional[str] = None
    editor: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    completeness_issues: list = field(default_factory=list)
    needs_ai_parsing: bool = False
    key_consistent: Optional[bool] = None
    key_mismatch_detail: Optional[str] = None
    original_key: Optional[str] = None


REQUIRED_FIELDS = {
    "book":          ["authors", "title", "publisher", "year"],
    "article":       ["authors", "title", "journal", "year", "pages"],
    "proceedings":   ["authors", "title", "booktitle", "year", "pages"],
    "inproceedings": ["authors", "title", "booktitle", "year", "pages"],
    "website":       ["title", "url"],
    "misc":          ["title", "year"],
    "unknown":       ["authors", "title", "year"],
}

_PUBLISHER_WORDS = re.compile(
    r'\b(?:Verlag|Press|Publishers?|Sons|GmbH|Books?|'
    r'Springer|Wiley|Elsevier|ACM|IEEE|MIT|O\'Reilly|'
    r'Prentice\s*Hall|Addison[- ]Wesley|Cambridge|Oxford|'
    r'Hanser|dpunkt|McGraw|Macmillan|Routledge|Sage|Taylor|Francis|CRC|'
    r'De Gruyter|Nomos|Beck|Mohr|Kohlhammer|Juventa|UTB)\b',
    re.IGNORECASE,
)

_PROCEEDINGS_WORDS = re.compile(
    r'\bIn:\s*|\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b|'
    r'\bTagung\b|\bKonferenz\b|\bHrsg\b',
    re.IGNORECASE,
)

_INCOLLECTION_PATTERN = re.compile(
    r'\bIn\s*\([^)]*\bed\.?s?\.?\s*\)\s*:', re.IGNORECASE,
)

_JOURNAL_WORDS = re.compile(
    r'(?:Jg\.|Vol\.|Volume|Band|Heft|Nr\.|Issue|No\.)\s*[\d]+',
    re.IGNORECASE,
)

_CONFERENCE_NAMES = re.compile(
    r'\b(?:NeurIPS|CVPR|ICCV|ECCV|ICML|ICLR|ACL|EMNLP|NAACL|AAAI|IJCAI|'
    r'NIPS|COLT|UAI|AISTATS|ICRA|IROS|CHI|UIST|SIGGRAPH|SIGIR|SIGMOD|'
    r'VLDB|SOSP|OSDI|USENIX|NDSS|IEEE|SPIE|LNI|GI|INFORMATIK)\b',
    re.IGNORECASE,
)

_JOURNAL_NAME_HINTS = re.compile(
    r'\b(?:Journal|Zeitschrift|Magazin|Review|Transactions|Letters|'
    r'Bulletin|Annals|Communications|Informatik|Computing|'
    r'Quarterly|Professional|Magazine|Publication|'
    r'Systems|Research|Reports?|Science|Technology|Management)\b',
    re.IGNORECASE | re.UNICODE,
)

# Words that must not START a journal name — if a candidate begins
# with one of these, prepend the preceding words.
_LEADING_PREPOSITIONS = re.compile(
    r'^(?:of|for|in|on|and|the|to|at|by|from|with|der|die|das|und|fur|von|mit|im|an|zu)\b',
    re.IGNORECASE,
)


def _normalize_unicode(text: str) -> str:
    if not text:
        return text
    return unicodedata.normalize('NFC', text)


def _normalize_metadata_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = html.unescape(str(value)).replace('\u00a0', ' ')
    value = re.sub(r'\s+', ' ', value).strip()
    value = re.sub(r'\b([A-Z])\s+and\s+([A-Z])\b', r'\1&\2', value, flags=re.IGNORECASE)
    return value or None


def _normalize_key_semantically(key: str) -> str:
    if not key:
        return key
    key = key.strip().strip('[]').strip()
    key = _normalize_unicode(key)
    if re.match(r'^[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?$', key, re.UNICODE):
        return key
    match = re.search(r'([A-Za-zÀ-ÿ]+)(?:\s*,\s*[A-Za-zÀ-ÿ]+)?\s+(\d{4}|\d{2})\b', key, re.UNICODE)
    if match:
        surname = match.group(1); year = match.group(2)
        year_2digit = year[2:] if len(year) == 4 else year
        return f"{surname[:2].upper()}{year_2digit}"
    match = re.search(r'([A-Za-zÀ-ÿ]+)\s+([A-Za-zÀ-ÿ]+)\s+(\d{4}|\d{2})\b', key, re.UNICODE)
    if match:
        surname = match.group(2); year = match.group(3)
        year_2digit = year[2:] if len(year) == 4 else year
        return f"{surname[:2].upper()}{year_2digit}"
    return key[:6].upper()


def _extract_title_smart(raw: str) -> Optional[str]:
    if not raw:
        return None
    for quote_char in ['"', "'"]:
        pattern = f'{re.escape(quote_char)}([^{quote_char}]{{10,200}}){re.escape(quote_char)}'
        match = re.search(pattern, raw)
        if match:
            title = match.group(1).strip()
            if len(title) > 8:
                return title
    venue_pattern = r'\b(?:In|Journal|Zeitschrift|Verlag|Publisher|Press)\b\s*:?'
    venue_match = re.search(venue_pattern, raw, re.IGNORECASE)
    if venue_match:
        pre_venue = raw[:venue_match.start()].strip()
        if ':' in pre_venue:
            title = pre_venue.split(':')[-1].strip()
        elif ';' in pre_venue:
            title = pre_venue.split(';')[-1].strip()
        else:
            sentences = re.split(r'(?<=[.!?])\s+', pre_venue)
            title = sentences[-1].strip() if sentences else None
        if title and len(title) > 8:
            return title
    publisher_pattern = r'(?:Verlag|Publisher|Press|Springer|Wiley|Elsevier|ISBN|DOI|pp\.?)'
    pub_match = re.search(publisher_pattern, raw, re.IGNORECASE)
    if pub_match:
        pre_pub = raw[:pub_match.start()].strip()
        segments = re.split(r'(?:by\b|von\b|und\b|;|,)', pre_pub, flags=re.IGNORECASE)
        for segment in reversed(segments):
            segment = segment.strip()
            if len(segment) > 8 and segment[0].isupper():
                return segment
    segments = re.split(r'[.!?:;,]', raw)
    longest = None
    for segment in segments:
        segment = segment.strip()
        if (len(segment) > 15 and segment and segment[0].isupper() and
            not re.match(r'^[A-Z]{1,3}\s+\d{4}$', segment) and
            not re.match(r'^\d+\s*$', segment)):
            if not longest or len(segment) > len(longest):
                longest = segment
    return longest


def _find_publisher_start(rest: str, match_start: int) -> int:
    if match_start <= 0:
        return 0
    lo = max(0, match_start - 60)
    window = rest[lo:match_start]
    boundary_chars = ['. ', ', ', '; ', ': ', ' - ', ' – ']
    best = 0
    for bc in boundary_chars:
        idx = window.rfind(bc)
        if idx > best:
            best = idx + len(bc)
    if best == 0:
        last_space = window.rfind(' ')
        if last_space > 0:
            prev_space = window.rfind(' ', 0, last_space)
            best = (prev_space + 1) if prev_space >= 0 else 0
    return lo + best


def _normalize_author_list(authors: str) -> str:
    if not authors:
        return authors
    parts = [p.strip() for p in authors.split(';') if p.strip()]
    has_et_al = False
    cleaned = []
    for p in parts:
        if re.search(r'\bet\s+al\.?', p, re.IGNORECASE):
            has_et_al = True
            p = re.sub(r'\s*[,;]?\s*\bet\s+al\.?\s*', '', p, flags=re.IGNORECASE).strip()
            p = p.rstrip(',;').strip()
        if p:
            cleaned.append(p)
    if not cleaned:
        return authors
    result = '; '.join(cleaned)
    if has_et_al:
        result = result + ' et al.'
    return result


def _extract_venue_structural(rest: str) -> Optional[str]:
    """
    Extract a journal name from ". <Journal> <volume>..." pattern.

    If the initial match starts with a preposition like "of", back up
    to include the preceding capitalized words so that
    "International Journal of Production Economics" is not truncated
    to "of Production Economics".
    """
    if not rest:
        return None

    candidates = []
    for m in re.finditer(r'\.\s+', rest):
        after = rest[m.end():m.end() + 250]
        if not re.match(r'[A-ZÄÖÜ]', after):
            continue
        if not re.search(r'\d', after[:180]):
            continue
        if m.start() >= 1 and rest[m.start() - 1].isupper():
            prev = rest[max(0, m.start() - 3):m.start()]
            if re.match(r'\b[A-Z]$', prev.strip()):
                continue
        candidates.append(m.end())

    if not candidates:
        return None

    start = candidates[-1]
    tail = rest[start:]

    m = re.match(
        r'([A-ZÄÖÜ][A-Za-zäöüÄÖÜ&\-\.\s,\d]*?)'
        r'(?=\s+\d+\s*[\(\,]|\s+\d+\s*$|\s+S\.|\s+pp?\.|\s+Vol\.|\s+No\.|\s+Nr\.|,\s*\d{4})',
        tail,
    )
    if not m:
        return None

    candidate = m.group(1).strip().rstrip('.,;:')

    # If the candidate begins with a preposition, back up.
    # Find the preceding capitalized words in `rest[:start]`.
    if _LEADING_PREPOSITIONS.match(candidate):
        before = rest[:start].rstrip()
        # Match trailing capitalized words ending at start.
        prev_m = re.search(
            r'([A-ZÄÖÜ][A-Za-zäöüÄÖÜ&\-\.\s]+?)\.\s*$',
            before,
        )
        if prev_m:
            prefix = prev_m.group(1).strip()
            candidate = prefix + ' ' + candidate

    candidate = re.sub(r'\s+(?:and|of|the|in|on|for)$', '', candidate, flags=re.IGNORECASE)
    if len(candidate) < 4 or len(candidate) > 200:
        return None
    if not candidate[0].isupper():
        return None
    return candidate


def _extract_lni_venue_fallback(entry: BibEntry, rest: str) -> None:
    """Extract journal/booktitle/publisher from 'Title. Venue Volume, ...'."""
    if not rest:
        return
    title_end = re.search(
        r'(?<!\w)\.\s+([A-Z][^.]{0,200}?)'
        r'(?:,\s*(?:S\.|pp?\.|Vol\.|Nr\.|Band|Heft)|\s+\d+\s*,)',
        rest,
    )
    if not title_end:
        return
    venue_text = title_end.group(1).strip()
    if not venue_text or len(venue_text) < 3:
        return
    venue_match = re.match(
        r'^(.+?)\s+(\d+[a-z]?|\([^)]+\)|[A-Z]\d)', venue_text, re.UNICODE,
    )
    if not venue_match:
        venue_match = re.match(r'^([^,]+)', venue_text)
        if not venue_match:
            return
        venue_name = venue_match.group(1).strip()
    else:
        venue_name = venue_match.group(1).strip()
    if len(venue_name) < 3 or len(venue_name) > 150:
        return
    venue_lower = venue_name.lower()
    if any(kw in venue_lower for kw in ['journal', 'zeitschrift', 'transactions',
                                          'letters', 'review', 'quarterly', 'bulletin',
                                          'annals', 'research', 'studies', 'science',
                                          'computing', 'systems', 'technology']):
        if not entry.journal:
            entry.journal = venue_name
    elif any(kw in venue_lower for kw in ['proceedings', 'conference', 'workshop',
                                            'symposium', 'tagung', 'konferenz',
                                            'arbeitstagung', 'lecture notes']):
        if not entry.booktitle:
            entry.booktitle = venue_name
    elif any(kw in venue_lower for kw in ['springer', 'wiley', 'elsevier', 'press',
                                            'verlag', 'publisher', 'academic',
                                            'university', 'mit', 'oxford', 'cambridge',
                                            'routledge', 'sage', 'hanser', 'dpunkt']):
        if not entry.publisher:
            entry.publisher = venue_name
    else:
        if not entry.journal:
            entry.journal = venue_name


def _extract_journal_smart(raw: str) -> Optional[str]:
    """Extract journal name with multiple fallback strategies."""
    if not raw:
        return None

    journal_marker = r'(?:Journal|Zeitschrift|Magazine|Review|Transactions)\s*:?\s*([^,;.]+)'
    match = re.search(journal_marker, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and not re.match(r'^\d+', journal):
            return journal

    vol_pattern = r'(?:Vol\.|Volume|Band|Jg\.)\s*[\d]+\s*,\s*([^,;.]+)'
    match = re.search(vol_pattern, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and not re.fullmatch(r'\d+.*', journal):
            return journal

    journal_keywords = r'(?:Journal|Zeitschrift|Magazine|Review|Transactions|Letters|Computing|Informatik)\s+(?:of|fur|for|de|di)?\s*([^,;.]+)'
    match = re.search(journal_keywords, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and journal.count(' ') <= 5:
            return journal

    in_pattern = r'In\s*:?\s*([^,;.]+?)(?:\s*(?:pp?\.?|S\.|Vol|volume|pages?)\s*[\d]|,\s*\d{4}|$)'
    match = re.search(in_pattern, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and _JOURNAL_NAME_HINTS.search(journal):
            return journal

    comma_segments = raw.split(',')
    for segment in comma_segments:
        segment = segment.strip()
        if not (_JOURNAL_NAME_HINTS.search(segment)
                and len(segment) > 5 and len(segment) < 100):
            continue
        period_positions = [m.start() for m in re.finditer(r'\.\s', segment)]
        if period_positions:
            after_first = segment[period_positions[0] + 2:].strip()
            if len(after_first) > 20:
                last_period = period_positions[-1]
                tail = segment[last_period + 2:].strip()
                if tail and len(tail) > 3 and not re.search(r'\b(?:19|20)\d{2}\b', tail):
                    return tail
                continue
        if not re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:and|&|\b\w+\b))', segment):
            return segment

    return None


def _normalize_extracted_key(raw_key: str) -> str:
    if raw_key is None:
        return ""
    raw_key = _normalize_unicode(raw_key)
    key = raw_key.strip().strip('[]')
    key = re.sub(r'\s+', '', key)
    if re.fullmatch(r'\d{1,3}', key):
        return key
    if re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', key, re.UNICODE):
        return key
    if re.fullmatch(r'([A-Za-zÀ-ÿ]{1,6}\d{2})([a-z])', key, re.UNICODE):
        return key
    return key


def parse_bibliography(bib_text: str) -> list:
    if not bib_text or not bib_text.strip():
        return []
    bib_text = _normalize_unicode(bib_text)
    lines = bib_text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\s*\d+\s*$', line):
            continue
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d+\s*$', line):
            continue
        if re.match(r'^(Fig\.|Tab\.)\s+\d+', line, re.IGNORECASE):
            continue
        if 'Inhaltsverzeichnis' in line or 'Contents' in line:
            continue
        cleaned_lines.append(line)
    bib_text = '\n'.join(cleaned_lines)

    key_positions = []
    for m in re.finditer(r'\[([^\]\n]{1,40})\]', bib_text):
        candidate = m.group(1).strip()
        original_key = candidate
        normalized = _normalize_extracted_key(candidate)
        if not normalized:
            normalized = original_key if original_key else candidate
        if re.fullmatch(r'\d{1,3}', normalized):
            valid = True
        elif re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', normalized, re.UNICODE):
            valid = True
        else:
            semantically_normalized = _normalize_key_semantically(normalized)
            if semantically_normalized and re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', semantically_normalized, re.UNICODE):
                valid = True
                normalized = semantically_normalized
            else:
                valid = True
        if valid:
            key_positions.append((m.start(), normalized, m.end(), original_key))

    if not key_positions:
        for m in re.finditer(r'\b([A-Za-zÀ-ÿ]{1,6}\s*\d{2}\s*[a-z]?)\b', bib_text, re.UNICODE):
            normalized = _normalize_extracted_key(m.group(1))
            if normalized and re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', normalized, re.UNICODE):
                key_positions.append((m.start(), normalized, m.end(), normalized))

    if not key_positions:
        return []

    entries = []
    for i, pos_tuple in enumerate(key_positions):
        if len(pos_tuple) == 4:
            start, key, end, original_key = pos_tuple
        else:
            start, key, end = pos_tuple
            original_key = key
        next_start = key_positions[i + 1][0] if i + 1 < len(key_positions) else len(bib_text)
        raw = bib_text[end:next_start].strip()
        if not raw:
            continue
        raw = re.sub(r'\s+', ' ', raw).strip()
        if len(raw) < 15 and not re.search(r'[A-Z]', raw):
            continue
        entry = BibEntry(key=key, raw_text=raw)
        _classify_and_parse(entry, raw)
        for field_name in ('authors', 'title', 'journal', 'booktitle', 'publisher'):
            setattr(entry, field_name, _normalize_metadata_value(getattr(entry, field_name)))
        _check_completeness(entry)
        _validate_key_vs_metadata(entry)
        if original_key and original_key != key:
            if not re.fullmatch(r'\d{1,3}', key) and not re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', key, re.UNICODE):
                entry.completeness_issues.insert(0, f"Malformed citation key: '[{original_key}]' does not follow LNI format (expected e.g. [Sm20], [ABC15], or [1-999]).")
        entries.append(entry)
    return entries


def _classify_and_parse(entry: BibEntry, raw: str) -> None:
    if len(raw) > 500 and entry.key:
        key_pattern = re.compile(r'\[{}\]\s*([^\[]+?)(?=\s*\[[A-Za-z0-9]+\]|\Z)'.format(re.escape(entry.key)), re.DOTALL)
        match = key_pattern.search(raw)
        if match:
            extracted = match.group(1).strip()
            if 20 < len(extracted) < 800:
                raw = extracted
                entry.raw_text = raw

    raw = re.sub(r'\s+([.,;:)])', r'\1', raw)
    raw = re.sub(r'\bm\s+Ay\b', 'may', raw, flags=re.IGNORECASE)
    # FIXED: normalize PDF-extracted spaces inside initials: "A . ;" -> "A.;"
    raw = re.sub(r'([A-Z])\s+\.\s*', r'\1. ', raw)
    # FIXED: normalize "Name ,Initial" -> "Name, Initial"  
    raw = re.sub(r'([A-Za-zÀ-ÿ])\s+,\s+', r'\1, ', raw)

    doi_match = re.search(
        r'(?:doi:\s*|https?://doi\.org/|DOI:\s*)([^\s,;\]]+)',
        raw, re.IGNORECASE,
    )
    if doi_match:
        entry.doi = doi_match.group(1).rstrip('.')

    isbn_match = re.search(
        r'(?:ISBN[:\s-]*)([\d][\d -]{8,16}[\dXx])',
        raw, re.IGNORECASE,
    )
    if isbn_match:
        entry.isbn = re.sub(r'[\s-]', '', isbn_match.group(1))

    url_match = re.search(r'(https?://\S+|https?:\s+//\S+|www\.\S+|www\.[^\s,;]+)', raw)
    if url_match:
        raw_url = re.sub(r'^(https?):\s+//', r'\1://', url_match.group(1))
        raw_url = raw_url.strip().rstrip('.,;:)]}')
        url_start = url_match.start()
        if 'doi.org' in raw_url.lower() or 'dx.doi.org' in raw_url.lower():
            entry.doi = re.sub(r'https?://(dx\.)?doi\.org/', '', raw_url).strip()
            raw = re.sub(r'https?://(dx\.)?doi\.org/[^\s,;]+', '', raw).strip()
        elif 'arxiv.org' in raw_url.lower():
            arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', raw_url, re.IGNORECASE)
            if arxiv_match:
                entry.doi = f"arXiv:{arxiv_match.group(1)}"
            raw = re.sub(r'https?://arxiv\.org/[^\s,;]+', '', raw).strip()
        else:
            entry.entry_type = "website"
            if raw_url.startswith('www.'):
                raw_url = 'https://' + raw_url
            entry.url = raw_url
            date_match = re.search(
                r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am|Accessed:)\s*([\d./-]+)',
                raw, re.IGNORECASE,
            )
            if date_match:
                entry.urldate = date_match.group(1)
            pre_url = raw[:url_start].strip().rstrip(',')
            org_match = re.match(r'^([A-ZÄÖÜ][A-Za-z0-9äöüÄÖÜ\s\.\-]{1,80}?),\s*', pre_url)
            if org_match:
                org_cand = org_match.group(1).strip().rstrip(',.')
                if 2 < len(org_cand) < 80 and not re.search(r'\b(19|20)\d{2}\b', org_cand):
                    entry.authors = org_cand
                    rest_after_org = pre_url[org_match.end():].strip()
                    if rest_after_org and not rest_after_org.startswith('www.'):
                        entry.title = rest_after_org.rstrip(',.')
                    else:
                        entry.title = org_cand
                else:
                    entry.title = pre_url.rstrip(',.')
            else:
                entry.title = pre_url.rstrip(',.')
            if not entry.title and entry.authors:
                entry.title = entry.authors
            year_m = re.search(r'\b(19|20)\d{2}\b', raw[:url_start])
            if year_m:
                entry.year = year_m.group(0)
            elif entry.urldate:
                date_year = re.search(r'(19|20)\d{2}', entry.urldate)
                if date_year:
                    entry.year = date_year.group(0)
            if entry.url and entry.title:
                return
            else:
                if entry.url and not entry.title:
                    entry.title = entry.url
                return

    if (re.search(r'\b(?:Accessed|Abruf|Stand|abgerufen am|besucht am)\b', raw, re.IGNORECASE)
            and re.search(r'\b(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov)\b', raw, re.IGNORECASE)):
        entry.entry_type = "website"
        entry.url = None
        domain_match = re.search(
            r'(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu)',
            raw, re.IGNORECASE)
        if domain_match:
            start_pos = domain_match.start()
            delimiter_match = re.search(
                r'\.\s+(?!/)|\s+–\s|(?:\s+)(?:Accessed|Abruf|Stand|accessed|besucht am)',
                raw[domain_match.end():], re.IGNORECASE)
            if delimiter_match:
                end_pos = domain_match.end() + delimiter_match.start()
            else:
                end_pos = len(raw)
            url_text = raw[start_pos:end_pos].strip().rstrip('.,;:)]}– ')
            if url_text:
                entry.url = url_text
                if not entry.url.lower().startswith(('http://', 'https://')):
                    entry.url = 'https://' + entry.url
        if not entry.url:
            domain_match = re.search(
                r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)\S*',
                raw, re.IGNORECASE)
            if domain_match:
                entry.url = domain_match.group(0).rstrip('.,;:)]}')
                if not entry.url.lower().startswith(('http://', 'https://')):
                    entry.url = 'https://' + entry.url
        date_match = re.search(
            r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am|Accessed:)\s*'
            r'(\d{4}-\d{2}-\d{2}|[\d./-]+)', raw, re.IGNORECASE)
        if date_match:
            entry.urldate = date_match.group(1)
        title_raw = raw
        if entry.url:
            title_raw = re.sub(r'\s*(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)\S*', '', title_raw, flags=re.IGNORECASE)
        if entry.urldate:
            title_raw = re.sub(r'\s*(?:Stand:|Abruf:|abgerufen am|accessed|besucht am|Accessed:)\s*[\d./-]+', '', title_raw, flags=re.IGNORECASE)
        title_raw = title_raw.strip(' .,;:')
        org_match = re.match(r'^([A-ZÄÖÜ][A-Za-z0-9äöüÄÖÜ\s\.\-]{1,80}?),\s*', title_raw)
        if org_match:
            org_cand = org_match.group(1).strip().rstrip(',.')
            if 2 < len(org_cand) < 80 and not re.search(r'\b(19|20)\d{2}\b', org_cand):
                entry.authors = org_cand
                entry.title = title_raw[org_match.end():].strip().rstrip(',.')
            else:
                entry.title = title_raw
        else:
            entry.title = title_raw
        if not entry.title:
            entry.title = "Website"
        return

    structural_venue = _extract_venue_structural(raw)

    _has_volume = bool(re.search(r'(?:Jg\.|Vol\.|Nr\.|Band)\s*\d', raw, re.IGNORECASE))
    _has_lni_volume = bool(re.search(
        r',\s*\d{1,4}\s*(?:\([\d\-]+\))?\s*:\s*\d', raw
    ))
    if _has_lni_volume:
        _has_volume = True
    _has_explicit_conf = bool(re.search(
        r'\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b'
        r'|\bTagung\b|\bKonferenz\b|\bHrsg\b',
        raw, re.IGNORECASE))
    _has_journal_name = bool(re.search(
        r'(?:Nature|Science|Cell|PLOS|PNAS|JMLR|IEEE Trans|ACM Trans|'
        r'Journal of|Transactions on|Letters|Annals of|Reviews? in|'
        r'Zeitschrift für|Informatik Spektrum)',
        raw,
        re.IGNORECASE,
    ))

    if _has_lni_volume:
        entry.entry_type = "article"
    elif _has_explicit_conf:
        entry.entry_type = "proceedings"
    elif _CONFERENCE_NAMES.search(raw) and not (_PUBLISHER_WORDS.search(raw) and _has_volume):
        entry.entry_type = "proceedings"
    elif _PUBLISHER_WORDS.search(raw) and not _has_volume and not structural_venue:
        entry.entry_type = "book"
    elif _has_volume or _has_journal_name:
        entry.entry_type = "article"
    elif _CONFERENCE_NAMES.search(raw):
        entry.entry_type = "proceedings"
    elif _INCOLLECTION_PATTERN.search(raw):
        entry.entry_type = "misc"
    elif _PROCEEDINGS_WORDS.search(raw) and (_has_explicit_conf or not _has_volume):
        entry.entry_type = "proceedings"
    elif structural_venue:
        entry.entry_type = "article"
    elif _JOURNAL_WORDS.search(raw) or _JOURNAL_NAME_HINTS.search(raw):
        entry.entry_type = "article"
    elif _PUBLISHER_WORDS.search(raw):
        entry.entry_type = "book"
    else:
        if re.search(r'MIT Press|Springer|Elsevier|Wiley|O\'Reilly|Pearson|Cambridge|Oxford|McGraw|Macmillan|Prentice\s*Hall', raw, re.IGNORECASE):
            entry.entry_type = "book"
        elif re.search(r'Journal|Transactions|Letters|Magazine|Review|IEEE|ACM', raw, re.IGNORECASE):
            entry.entry_type = "article"
        elif re.search(r'In:\s*[A-Z][a-zA-Z0-9\s]+', raw, re.IGNORECASE):
            entry.entry_type = "proceedings"
        elif re.search(r'ar\s*[Xx]iv', raw, re.IGNORECASE):
            entry.entry_type = "misc"
        elif re.search(r'\b(?:Ph\.?\s*D\.?|PhD|Master|Diploma|Dissertation)\s+[Tt]hes', raw, re.IGNORECASE):
            entry.entry_type = "misc"
        elif re.search(r'\b(?:Bundesamt|Bundesministerium|Ministerium|Umweltbundesamt|Bundesbeh|agency|Authority|Institute|Institut|Commission)\b', raw, re.IGNORECASE):
            entry.entry_type = "misc"
        elif re.search(r'\bpp\.\s*\d', raw, re.IGNORECASE):
            entry.entry_type = "misc"
        elif re.search(r'\b(?:UN|SDG|UNESCO|WHO|FAO|IPCC|ISO)\b', raw):
            entry.entry_type = "misc"
        else:
            entry.entry_type = "unknown"
            entry.needs_ai_parsing = True

    quote_m = re.search(r'["\u201c]([^"\u201d]{3,300})["\u201d]', raw)
    if quote_m:
        before = raw[:quote_m.start()].strip()
        after = raw[quote_m.end():].strip()
        entry.title = quote_m.group(1).strip().rstrip('.,;: ')
        year_before_m = re.search(r'\((19|20)\d{2}\)\.?\s*$', before)
        if year_before_m:
            entry.year = re.sub(r'[().\s]', '', year_before_m.group(0))
            before = before[:year_before_m.start()].strip()
        before = before.rstrip('.,;: ')
        if before and len(before) < 180 and re.match(r'^[A-ZÄÖÜ]', before):
            entry.authors = before
        rest = after
        if not entry.year:
            year_after_m = re.search(r'\(((?:19|20)\d{2})\)|\b((?:19|20)\d{2})\b', rest)
            if year_after_m:
                entry.year = year_after_m.group(1) or year_after_m.group(2)
                rest = (rest[:year_after_m.start()] + rest[year_after_m.end():]).strip()
        rest = rest.strip('(). ').strip()
    else:
        rest = raw

    # Author extraction — prefer the FIRST colon.
    if not quote_m:
        eds_sep = re.search(r',\s*eds?\.\s+(?=[A-Z])', raw, re.IGNORECASE)
        if eds_sep:
            authors_cand = raw[:eds_sep.start()].strip()
            title_cand = raw[eds_sep.end():].strip()
            if re.match(r'^[A-Z\u00c0-\u00de]', authors_cand) and not re.search(r'\b(19|20)\d{2}\b', authors_cand):
                entry.authors = authors_cand
                rest = title_cand

    author_pattern = None if (quote_m or entry.authors) else re.match(
        r'^((?:[A-ZÄÖÜ\u00c0-\u00d6\u00d8-\u00de][a-zA-Z\u00c0-\u00ff\-]+'
        r'(?:,\s*[A-Za-z\u00c0-\u00ff\.\s\-]+)?'
        r'(?:;\s*)?)+):\s*(.*)',
        raw,
    )
    if quote_m:
        pass
    elif author_pattern:
        candidate = author_pattern.group(1).strip()
        candidate = re.sub(r'[\s,;]+eds?\.?\s*$', '', candidate, flags=re.IGNORECASE).strip()
        if len(candidate) < 300 and ':' not in candidate:
            entry.authors = candidate
            rest = author_pattern.group(2).strip()

    # If no authors yet, try the FIRST colon in the raw string.
    # This is the fix for "Diener, F.; Špaček, M.: Title".
    if not quote_m and not entry.authors:
        first_colon = raw.find(':')
        if 0 < first_colon < 200:
            before = raw[:first_colon].strip()
            after = raw[first_colon + 1:].strip()
            if (after and after[0].isupper()
                    and not re.search(r'\b(19|20)\d{2}\b', before)
                    and not re.search(r'\b(?:In:|Vol\.|Jg\.|doi:)\b', before, re.IGNORECASE)
                    and len(before) < 300
                    and ':' not in before):
                # Require the "before" to look like an author list:
                # ends with an initial or a name
                if re.search(r'[A-Z\u00c0-\u00de](?:\.|\b)(?:\s*,\s*[A-Z]\.?)?\s*$', before) \
                   or re.search(r'[a-z\u00e0-\u00ff]{2,}\s*$', before):
                    candidate = re.sub(r'[\s,;]+eds?\.?\s*$', '', before, flags=re.IGNORECASE).strip()
                    # FIXED: also normalise PDF space-before-dot artifacts in author list
                    candidate = re.sub(r'([A-Z])\s+\.\s*', r'\1. ', candidate)
                    entry.authors = candidate
                    rest = after

    if not quote_m and not entry.authors:
        colon_positions = [m.start() for m in re.finditer(r':', raw)]
        author_colon = None
        for pos in colon_positions:
            before = raw[:pos].strip()
            after = raw[pos + 1:].strip()
            if not after or not after[0].isupper():
                continue
            if re.search(r'\b(19|20)\d{2}\b', before):
                continue
            if re.search(r'\b(?:In:|Vol\.|Jg\.|doi:)\b', before, re.IGNORECASE):
                continue
            if re.search(r'[A-Z\u00c0-\u00de][a-z\u00e0-\u00ff]{1,}\s*$', before):
                author_colon = pos
                break
        if author_colon is not None and 0 < author_colon:
            candidate = raw[:author_colon].strip()
            candidate = re.sub(r'[\s,;]+eds?\.?\s*$', '', candidate, flags=re.IGNORECASE).strip()
            entry.authors = candidate
            rest = raw[author_colon + 1:].strip()
        else:
            colon_idx = raw.find(':')
            if 0 < colon_idx < 120:
                candidate = raw[:colon_idx].strip()
                if re.match(r'^[A-Z\u00c0-\u00de]', candidate):
                    candidate = re.sub(r'[\s,;]+eds?\.?\s*$', '', candidate, flags=re.IGNORECASE).strip()
                    entry.authors = candidate
                    rest = raw[colon_idx + 1:].strip()
                else:
                    entry.needs_ai_parsing = True
            else:
                entry.needs_ai_parsing = True

    if entry.authors:
        entry.authors = _normalize_author_list(entry.authors)

    if not entry.year:
        year_matches = re.findall(r'\b(?:19|20)\d{2}\b', rest)
        if year_matches:
            entry.year = year_matches[-1]
        else:
            broken1 = re.search(r'\b(19|20)\s+(\d{2})\b', rest)
            if broken1:
                entry.year = broken1.group(1) + broken1.group(2)
            else:
                broken2 = re.search(r'\b(\d)\s+(\d{3})\b', rest)
                if broken2 and broken2.group(1) in '12':
                    entry.year = broken2.group(1) + broken2.group(2)

    _roman = r'[ivxlcdmIVXLCDM]+'
    pages_match = re.search(
        r'(?:\bS\.|\bpp?\.)\s*'
        r'(\d+\s*[-–—]+\s*\d+|\d+|'
        rf'{_roman}\s*[-–—]+\s*{_roman}|{_roman})',
        rest, re.IGNORECASE,
    )
    if pages_match:
        entry.pages = pages_match.group(1).replace(' ', '')

    pub_match = _PUBLISHER_WORDS.search(rest)
    if pub_match:
        boundary = _find_publisher_start(rest, pub_match.start())
        candidate = rest[boundary:pub_match.end()].strip().lstrip(',. ')
        words = candidate.split()
        if len(words) > 5:
            candidate = ' '.join(words[-5:])
        entry.publisher = candidate[:80]

    vol_match = re.search(
        r'(?:Jg\.|Vol\.|Volume|Band)\s*(\d+)', rest, re.IGNORECASE
    )
    if vol_match:
        entry.volume = vol_match.group(1)

    nr_match = re.search(
        r'(?:Nr\.|No\.|Issue|Heft)\s*(\d+)', rest, re.IGNORECASE
    )
    if nr_match:
        entry.number = nr_match.group(1)

    if not entry.volume or not entry.number:
        lni_vol_match = re.search(
            r',\s*(\d{1,4})\s*\(([\d\-]+)\)\s*:\s*(\d+\s*[–—\-]+\s*\d+|\d+)',
            rest,
        )
        if lni_vol_match:
            if not entry.volume:
                entry.volume = lni_vol_match.group(1)
            if not entry.number:
                entry.number = lni_vol_match.group(2)
            if not entry.pages:
                entry.pages = lni_vol_match.group(3).replace(' ', '')
        else:
            lni_vol_only = re.search(
                r',\s*(\d{1,4})\s*:\s*(\d+\s*[–—\-]+\s*\d+|\d+)',
                rest,
            )
            if lni_vol_only:
                if not entry.volume:
                    entry.volume = lni_vol_only.group(1)
                if not entry.pages:
                    entry.pages = lni_vol_only.group(2).replace(' ', '')

    if quote_m:
        pass
    elif rest:
        rest_clean = re.sub(r',?\s*https?://\S+', '', rest)
        rest_clean = re.sub(r',?\s*(19|20)\d{2}\s*$', '', rest_clean)

        candidates = []
        stop_patterns = [
            r'\.\s+In\s+[\(\[]',
            r'\.\s+In:\s+',
            r'[?!]\s+In:\s+',
            r',\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            r'\.\s+(?:19|20)\d{2}[,\.]',
            r'\.\s+[A-ZÄÖÜ][^\s].*?(?:Verlag|Press|Publishers?|Springer|Wiley|Elsevier)',
            r'\.\s+[A-ZÄÖÜ][^,\.]{2,40},\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.|Issue)',
        ]
        for pat in stop_patterns:
            m = re.search(pat, rest_clean, re.IGNORECASE)
            if m and m.start() > 5:
                c = rest_clean[:m.start()].strip().rstrip('.')
                candidates.append(c)

        m = re.search(r'(?<![A-ZÄÖÜ])\.\s+[A-ZÄÖÜ]', rest_clean)
        if m and m.start() > 5:
            candidates.append(rest_clean[:m.start()].strip().rstrip('.'))

        first_period_parts = rest_clean.split('.')
        if len(first_period_parts) > 1:
            first_period = first_period_parts[0].strip()
            remainder = rest_clean[len(first_period_parts[0]):].lstrip('.')
            if remainder and remainder[:1] == ' ' and len(remainder) > 1 and remainder[1].isupper():
                if first_period:
                    candidates.append(first_period)

        if candidates:
            max_len = int(len(rest_clean) * 0.85)
            valid = [c for c in candidates if 5 < len(c) <= max_len]
            if valid:
                title_text = max(valid, key=len)
            else:
                title_text = max(candidates, key=len)
        else:
            title_text = rest_clean[:120]

        entry.title = title_text.strip().strip('.,;:') or None
        if not entry.title:
            entry.needs_ai_parsing = True

    if not entry.title:
        smart_title = _extract_title_smart(rest)
        if smart_title:
            entry.title = smart_title

    if entry.entry_type == "proceedings":
        bt_match = re.search(
            r'In\s*[\(\[]([^\)\]]+)[\)\]]'
            r'|In:\s*(.+?)(?=,\s*(?:pp?\.|S\.)|\s+pp?\.)',
            rest, re.IGNORECASE,
        )
        if bt_match:
            entry.booktitle = (
                bt_match.group(1) or bt_match.group(2) or ''
            ).strip()
        if not entry.booktitle:
            raw_bt = re.search(
                r'\bIn:\s*(.+?)(?=\.\s|,\s*\d{4}\b|$)',
                raw, re.IGNORECASE,
            )
            if raw_bt:
                candidate = raw_bt.group(1).strip(' ,.')
                if not re.fullmatch(r'(?:19|20)\d{2}', candidate):
                    entry.booktitle = candidate

    # Journal for articles.
    if entry.entry_type == "article":
        if not entry.journal:
            lni_j_match = re.search(
                r'([A-Za-z][A-Za-z0-9 &\-]+?),\s*\d{1,4}\s*(?:\([\d\-]+\))?\s*:',
                rest,
            )
            if lni_j_match:
                candidate = lni_j_match.group(1).strip().rstrip('.,;: ')
                last_period_pos = rest.rfind('.', 0, lni_j_match.start())
                if last_period_pos >= 0:
                    candidate = rest[last_period_pos + 1:lni_j_match.end() - 1]
                    candidate = candidate.strip().rstrip(',').strip()
                    comma_vol = re.search(r',\s*\d{1,4}\s*(?:\([\d\-]+\))?\s*:', candidate)
                    if comma_vol:
                        candidate = candidate[:comma_vol.start()].strip()
                if candidate and len(candidate) >= 3 and not re.match(r'^\d', candidate):
                    entry.journal = candidate

        if not entry.journal:
            j_match = re.search(
                r'(?:\.\s+In:\s+|\.\s+)([A-Za-zäöüÄÖÜ][^,\.]{2,80}?),\s*(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
                rest, re.IGNORECASE,
            )
            if j_match:
                entry.journal = j_match.group(1).strip()

        if not entry.journal:
            in_match = re.search(
                r'\bIn:\s+(.+?)(?=,\s*(?:Nr\.|Vol\.|pp?\.|S\.)|\s+\d+\s*(?:,|\.|\s+Nr\.)|\s+vol\.?)',
                rest, re.IGNORECASE,
            )
            if in_match:
                candidate = in_match.group(1).strip(' ,.')
                if candidate and not re.search(r'Proceedings|Conference|Lecture Notes', candidate, re.IGNORECASE):
                    entry.journal = candidate

        if not entry.journal:
            j_match2 = re.search(
                r'[.:]\s+([A-Za-zäöüÄÖÜ][A-Za-zäöüÄÖÜ\s]{4,60})\s+\d+\s*[\s(,]',
                rest,
            )
            if j_match2:
                candidate = j_match2.group(1).strip().rstrip(',.')
                if (not re.match(r'^(19|20)\d{2}$', candidate)
                        and len(candidate) > 4
                        and not candidate.lower().startswith('s.')):
                    entry.journal = candidate

        if not entry.journal:
            j_match3 = re.search(
                r'\.\s+([A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc][A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-]{2,60}?),\s*S\.',
                rest,
            )
            if j_match3:
                candidate = j_match3.group(1).strip().rstrip(',.')
                if (not re.match(r'^(19|20)\d{2}$', candidate)
                        and len(candidate) > 2):
                    entry.journal = candidate

        if not entry.journal:
            structural = _extract_venue_structural(rest)
            if structural and not _LEADING_PREPOSITIONS.match(structural):
                entry.journal = structural

        if not entry.journal:
            smart_journal = _extract_journal_smart(rest)
            if smart_journal:
                entry.journal = smart_journal

    if not entry.journal and not entry.booktitle and not entry.publisher:
            # If journal and publisher are the same string, it's a publisher, not a journal.
        if entry.journal and entry.publisher and entry.journal.strip().lower() == entry.publisher.strip().lower():
            entry.journal = None
        _extract_lni_venue_fallback(entry, rest)


def validate_lni_key(key: str) -> list:
    errors = []
    if key.isdigit():
        return errors
    match = re.match(r'^([A-ZÀ-ÿ][A-Za-zÀ-ÿ]*)(\d{2})([a-z])?$', key, re.UNICODE)
    if not match:
        errors.append(
            f"Key '{key}' does not follow LNI format (e.g. Ez10, ABC01, Mü18). "
            f"Initials must start with an uppercase letter."
        )
    else:
        letters = match.group(1)
        if len(letters) < 2 or len(letters) > 6:
            errors.append(
                f"Author initials in '{key}' should be 2–6 characters, "
                f"got {len(letters)}."
            )
    return errors


def _extract_surnames(authors_str: str) -> list:
    surnames = []
    if not authors_str:
        return surnames
    for author in re.split(r';\s*| and\s+', authors_str):
        author = author.strip()
        if not author:
            continue
        if re.search(r'et\s+al\.?', author, re.IGNORECASE):
            continue
        if ',' in author:
            surname = author.split(',')[0].strip()
        else:
            parts = author.split()
            if parts:
                surname = parts[-1].strip()
            else:
                continue
        surname = re.sub(r'^(?:van|von|de|der|den|del|della|di|du|la|le)\s+', '', surname, flags=re.IGNORECASE)
        surname = surname.lower()
        for source, replacement in [('ä', 'a'), ('ö', 'o'), ('ü', 'u'), ('ß', 'ss')]:
            surname = surname.replace(source, replacement)
        if surname:
            surnames.append(surname)
    return surnames


def _validate_key_vs_metadata(entry: BibEntry) -> None:
    if entry.entry_type == "website":
        entry.key_consistent = None
        entry.key_mismatch_detail = None
        return
    if entry.key.isdigit():
        entry.key_consistent = True
        entry.key_mismatch_detail = None
        return
    match = re.match(r'^([A-Z][A-Za-z]*)(\d{2})([a-z])?$', entry.key)
    if not match:
        entry.key_consistent = None
        return
    key_initials = match.group(1).lower()
    key_year_2d = match.group(2)

    year_ok: Optional[bool] = None
    if entry.year:
        try:
            bib_year_int = int(entry.year)
            year_ok = str(bib_year_int)[-2:] == key_year_2d
        except ValueError:
            expected_2d = entry.year[-2:]
            year_ok = (expected_2d == key_year_2d)

    initials_ok: Optional[bool] = None
    if entry.authors:
        surnames = _extract_surnames(entry.authors)
        if surnames:
            n = len(surnames)
            def _norm_surname(s: str) -> str:
                s = s.lower()
                for bad, good in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
                        ('é', 'e'), ('è', 'e'), ('ê', 'e'), ('à', 'a'),
                        ('â', 'a'), ('î', 'i'), ('ô', 'o'), ('û', 'u'),
                        ('š', 's'), ('ž', 'z'), ('č', 'c'), ('ř', 'r'),
                        ('ł', 'l'), ('ą', 'a'), ('ę', 'e'), ('ś', 's'),
                        ('ź', 'z'), ('ż', 'z'), ('ő', 'o'), ('ű', 'u')]:
                    s = s.replace(bad, good)
                s = re.sub(r'[^a-z]', '', s)
                return s
            
            normed = [_norm_surname(s) for s in surnames]
            normed = [s for s in normed if s]
            if not normed:
                entry.key_consistent = None
                return
            n = len(normed)
            def _compound_initials(authors_str: str) -> set:
                initials = set()
                for author in re.split(r';\s*', authors_str):
                    author = author.strip()
                    if not author:
                        continue
                    if ',' in author:
                        surname = author.split(',')[0].strip()
                    else:
                        parts = author.split()
                        surname = parts[-1] if parts else author
                    if surname and surname[0].isalpha():
                        initials.add(surname[0].lower())
                    for part in re.split(r'[-\s]+', surname):
                        if part and part[0].isalpha():
                            initials.add(part[0].lower())
                return initials
            compound_initial_pool = _compound_initials(entry.authors)
            valid_forms = set()
            valid_forms.add(normed[0][:2])
            if n >= 2:
                valid_forms.add(''.join(s[0] for s in normed[:min(n, 3)]))
                if n >= 3:
                    valid_forms.add(normed[0][0] + normed[-1][0])
            if n == 1:
                valid_forms.add(normed[0][0])
            if n == 2:
                valid_forms.add(normed[0][0] + normed[1][0])
                if len(normed[0]) >= 2:
                    valid_forms.add(normed[0][:2] + normed[1][0])
            if n == 3:
                valid_forms.add(normed[0][0] + normed[1][0] + normed[2][0])
            if n >= 4:
                valid_forms.add(normed[0][:2])
            for i in range(n):
                for j in range(n):
                    if i != j and normed[i] and normed[j]:
                        valid_forms.add(normed[i][0] + normed[j][0])
            if n >= 3:
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            if len({i, j, k}) == 3 and normed[i] and normed[j] and normed[k]:
                                valid_forms.add(normed[i][0] + normed[j][0] + normed[k][0])
            initials_ok = any(
                key_initials == form or key_initials.startswith(form) or form.startswith(key_initials)
                for form in valid_forms
                if form and len(form) >= 1
            )
            missing_initials_for_3plus = False
            if initials_ok and n >= 3 and len(key_initials) < n and len(key_initials) < 3:
                all_chars_are_initials = all(
                    any(s[0] == c for s in normed) for c in key_initials
                )
                is_first_surname_prefix = normed[0].startswith(key_initials)
                if all_chars_are_initials and not is_first_surname_prefix:
                    initials_ok = False
                    missing_initials_for_3plus = True
                    if len(key_initials) >= 2:
                        author_initials_pool = {s[0] for s in normed}
                        if set(key_initials) <= author_initials_pool:
                            missing_initials_for_3plus = False
            if initials_ok is False and n >= 2 and not missing_initials_for_3plus:
                author_initials = {s[0] for s in normed}
                if set(key_initials) <= author_initials:
                    if len(key_initials) >= n or len(key_initials) <= 2:
                        initials_ok = None
            if initials_ok is False and set(key_initials) <= compound_initial_pool:
                initials_ok = None

    checks = [c for c in [year_ok, initials_ok] if c is not None]
    if not checks:
        entry.key_consistent = None
        return
    entry.key_consistent = all(checks)
    if not entry.key_consistent:
        details = []
        if year_ok is False:
            details.append(f"key year '{key_year_2d}' ≠ parsed year '{entry.year}'")
        if initials_ok is False:
            if n >= 2 and key_initials == (normed[0][0] + normed[1][0]):
                entry.key_consistent = True
                entry.key_mismatch_detail = None
                return
            if n == 1 and key_initials == normed[0][:2]:
                entry.key_consistent = True
                entry.key_mismatch_detail = None
                return
            if len(key_initials) == 2 and not missing_initials_for_3plus:
                author_initials_pool = {s[0] for s in normed}
                if set(key_initials) <= author_initials_pool:
                    entry.key_consistent = None
                    entry.key_mismatch_detail = None
                    return
            details.append(
                f"key initials '{key_initials}' don't match authors '{entry.authors[:40]}'"
            )
        if details:
            entry.key_mismatch_detail = "; ".join(details)


def _check_completeness(entry: BibEntry) -> None:
    _validate_key_vs_metadata(entry)
    errors = validate_lni_key(entry.key)
    for err in errors:
        entry.completeness_issues.append(f"Invalid key format: {err}")
    if entry.original_key and entry.original_key != entry.key:
        if not re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', entry.original_key, re.UNICODE):
            entry.completeness_issues.append(
                f"Citation key '[{entry.original_key}]' violates LNI format "
                f"(expected [Ab00] format, got [{entry.original_key}]). "
                f"Auto-normalized to [{entry.key}]."
            )
    if entry.key_consistent is False and entry.key_mismatch_detail:
        entry.completeness_issues.append(
            f"Key inconsistency: {entry.key_mismatch_detail}"
        )
    entry_type = entry.entry_type or "unknown"
    lookup_type = "proceedings" if entry_type == "inproceedings" else entry_type
    required = REQUIRED_FIELDS.get(lookup_type, REQUIRED_FIELDS["unknown"])
    
    # FIX: Before marking as unknown, check if entry looks like real paper
    if entry_type == "unknown":
        # Check for keywords indicating real venue
        venue_keywords = {"journal", "proceedings", "conference", "workshop", "springer",
                         "wiley", "elsevier", "acm", "ieee", "international", "review"}
        raw_lower = entry.raw_text.lower() if entry.raw_text else ""
        venue_str = (entry.venue or "").lower() if entry.venue else ""
        
        has_venue_keyword = any(kw in raw_lower or kw in venue_str for kw in venue_keywords)
        
        if has_venue_keyword and entry.authors and entry.year:
            # Likely real paper with real venue - don't mark as unknown
            entry_type = "article"  # Default to article if not classifiable
            lookup_type = "article"
            required = REQUIRED_FIELDS.get("article", [])
        else:
            # Only mark incomplete if truly no venue indicators found
            entry.completeness_issues.append(
                "Entry type could not be determined (no journal, booktitle, "
                "or publisher field found) — LNI requires a classifiable "
                "venue for every entry."
            )
    has_explicit_venue = bool(re.search(
        r'\bIn\s*:?\s*|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b|\bTagung\b|\bKonferenz\b',
        entry.raw_text,
        flags=re.IGNORECASE,
    ))
    for field_name in required:
        if field_name == "pages" and (
            entry_type in ("proceedings", "inproceedings")
                or (entry_type == "article" and entry.raw_text.find("In:") >= 0
                    and (entry.volume or entry.number))):
            continue
        if field_name == "booktitle" and entry_type in ("proceedings", "inproceedings") and has_explicit_venue:
            continue
        if not getattr(entry, field_name, None):
            entry.completeness_issues.append(
                f"Missing required field: '{field_name}'"
            )
    if entry.authors and entry.entry_type not in ("website", "misc", "online"):
        for name in entry.authors.split(';'):
            name = name.strip()
            if re.match(
                r'^[A-ZÄÖÜ][a-zäöüß]{2,}\s+[A-ZÄÖÜ][a-zäöüß]{2,}$', name
            ) and ',' not in name:
                entry.completeness_issues.append(
                    f"Author '{name}' appears to be 'Firstname Lastname' — "
                    "LNI requires 'Lastname, Firstname'."
                )
                break
    if entry.year:
        try:
            if int(entry.year) > datetime.date.today().year + 1:
                entry.completeness_issues.append(
                    f"Year '{entry.year}' is in the future — likely an error."
                )
        except ValueError:
            pass
    if entry.pages:
        m = re.search(r'(\d+)\s*[-–—]+\s*(\d+)', entry.pages)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            span = hi - lo
            if span > 100:
                entry.completeness_issues.append(
                    f"Page range {lo}–{hi} spans {span} pages — "
                    "unusually large for a single article."
                )
    if entry.volume:
        vol_str = str(entry.volume).strip()
        suspicious_volumes = {'666', '777', '888', '999', '111', '222', '333', '444', '555'}
        if vol_str in suspicious_volumes:
            entry.completeness_issues.append(
                f"Volume number '{vol_str}' is suspiciously repetitive — "
                "likely fabricated or non-standard."
            )


def entries_to_dict(entries: list) -> dict:
    return {e.key: e for e in entries}


def parse_raw_references(raw_refs: list) -> list:
    entries = []
    for ref in raw_refs:
        key = ref.get('key', '')
        raw = ref.get('raw_text', '')
        entry = BibEntry(key=key, raw_text=raw)
        year_m = re.search(r'\b(19|20)\d{2}\b', raw)
        if year_m:
            entry.year = year_m.group(0)
        colon_idx = raw.find(':')
        if colon_idx > 0:
            author_part = raw[:colon_idx].strip()
            if re.search(r'[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,|\s+and\s+|;)', author_part):
                entry.authors = author_part
        title_m = re.search(r'["\']([^"\']{10,})["\']', raw)
        if title_m:
            entry.title = title_m.group(1)
        pages_m = re.search(r'(?:pp?\.?|S\.)\s*([0-9\-]+)', raw)
        if pages_m:
            entry.pages = pages_m.group(1)
        url_m = re.search(r'(https?://[^\s,\.]+)', raw)
        if url_m:
            entry.url = url_m.group(1).rstrip('.,;:')
        if 'In' in raw or 'in' in raw:
            in_m = re.search(r'In\s+\(([^)]+)\)', raw)
            if in_m:
                entry.booktitle = in_m.group(1)
        pub_m = re.search(r'(?:Verlag|Publisher|Press)\s+([^,\.]+)', raw)
        if pub_m:
            entry.publisher = pub_m.group(1).strip()
        raw_lower = raw.lower()
        if 'journal' in raw_lower:
            entry.entry_type = 'article'
        elif 'proceedings' in raw_lower or 'conference' in raw_lower:
            entry.entry_type = 'inproceedings'
        elif 'http' in raw_lower or 'www' in raw_lower:
            entry.entry_type = 'online'
        elif 'verlag' in raw_lower or 'publisher' in raw_lower:
            entry.entry_type = 'book'
        else:
            entry.entry_type = 'misc'
        entries.append(entry)
    return entries