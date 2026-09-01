"""
STEP 2: Bibliography Parser
----------------------------
Parses the bibliography section of an LNI-formatted document.
Extracts citation keys like [AB00], [Ez10], [GI19] and their metadata.

FIXED v7.4:
  - Fixed duplicate entry merging (Ez99 vs Ez10, GI09 vs GI14)
  - Better URL extraction for website entries
  - Fixed garbage filtering for proceedings PDFs
"""

import re
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


# Required fields per entry type (LNI standard)
REQUIRED_FIELDS = {
    "book":          ["authors", "title", "publisher", "year"],
    "article":       ["authors", "title", "journal", "year", "pages"],
    "proceedings":   ["authors", "title", "booktitle", "year", "pages"],
    "inproceedings": ["authors", "title", "booktitle", "year", "pages"],
    "website":       ["title", "url"],
    "misc":          ["title", "year"],
    "unknown":       ["authors", "title", "year"],
}

# Known publishers / venues for quick type sniffing
_PUBLISHER_WORDS = re.compile(
    r'(?:Verlag|Press|Publishers?|Sons|GmbH|Books?|'
    r'Springer|Wiley|Elsevier|ACM|IEEE|MIT|O\'Reilly|'
    r'Prentice\s*Hall|Addison[- ]Wesley|Cambridge|Oxford|'
    r'Hanser|dpunkt|McGraw|Macmillan|Routledge|Sage|Taylor|Francis|CRC|'
    r'De Gruyter|Nomos|Beck|Mohr|Kohlhammer|Juventa|UTB)',
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


# ============================================================================
# UNICODE SUPPORT FIX (BUG 4)
# ============================================================================

def _normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    if not text:
        return text
    return unicodedata.normalize('NFC', text)


# ============================================================================
# KEY NORMALIZATION FIX (BUG 3)
# ============================================================================

def _normalize_key_semantically(key: str) -> str:
    """Convert various key formats to canonical LNI format.
    Examples: Smith2020 -> Sm20, Knuth1984 -> Kn84"""
    if not key:
        return key
    
    key = key.strip().strip('[]').strip()
    key = _normalize_unicode(key)
    
    if re.match(r'^[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?$', key, re.UNICODE):
        return key
    
    match = re.search(r'([A-Za-zÀ-ÿ]+)(?:\s*,\s*[A-Za-zÀ-ÿ]+)?\s+(\d{4}|\d{2})\b', key, re.UNICODE)
    if match:
        surname = match.group(1)
        year = match.group(2)
        year_2digit = year[2:] if len(year) == 4 else year
        initials = surname[:2].upper()
        return f"{initials}{year_2digit}"
    
    match = re.search(r'([A-Za-zÀ-ÿ]+)\s+([A-Za-zÀ-ÿ]+)\s+(\d{4}|\d{2})\b', key, re.UNICODE)
    if match:
        surname = match.group(2)
        year = match.group(3)
        year_2digit = year[2:] if len(year) == 4 else year
        initials = surname[:2].upper()
        return f"{initials}{year_2digit}"
    
    return key[:6].upper()


# ============================================================================
# TITLE EXTRACTION FIX (BUG 1)
# ============================================================================

def _extract_title_smart(raw: str) -> Optional[str]:
    """Extract title with context-aware punctuation handling."""
    if not raw:
        return None
    
    # Strategy 1: Quoted title
    for quote_char in ['"', "'"]:
        pattern = f'{re.escape(quote_char)}([^{quote_char}]{{10,200}}){re.escape(quote_char)}'
        match = re.search(pattern, raw)
        if match:
            title = match.group(1).strip()
            if len(title) > 8:
                return title
    
    # Strategy 2: Venue-marker based
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
    
    # Strategy 3: Before publisher keywords
    publisher_pattern = r'(?:Verlag|Publisher|Press|Springer|Wiley|Elsevier|ISBN|DOI|pp\.?)'
    pub_match = re.search(publisher_pattern, raw, re.IGNORECASE)
    
    if pub_match:
        pre_pub = raw[:pub_match.start()].strip()
        segments = re.split(r'(?:by\b|von\b|und\b|;|,)', pre_pub, flags=re.IGNORECASE)
        for segment in reversed(segments):
            segment = segment.strip()
            if len(segment) > 8 and segment[0].isupper():
                return segment
    
    # Strategy 4: Longest capitalized segment
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


# ============================================================================
# JOURNAL EXTRACTION FIX (BUG 2)
# ============================================================================

def _extract_journal_smart(raw: str) -> Optional[str]:
    """Extract journal name with multiple fallback strategies."""
    if not raw:
        return None
    
    # Strategy 1: Explicit journal marker
    journal_marker = r'(?:Journal|Zeitschrift|Magazine|Review|Transactions)\s*:?\s*([^,;.]+)'
    match = re.search(journal_marker, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and not re.match(r'^\d+', journal):
            return journal
    
    # Strategy 2: Vol. NN, [Journal Name]
    vol_pattern = r'(?:Vol\.|Volume|Band|Jg\.)\s*[\d]+\s*,\s*([^,;.]+)'
    match = re.search(vol_pattern, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and not re.fullmatch(r'\d+.*', journal):
            return journal
    
    # Strategy 3: Journal keywords
    journal_keywords = r'(?:Journal|Zeitschrift|Magazine|Review|Transactions|Letters|Computing|Informatik)\s+(?:of|fur|for|de|di)?\s*([^,;.]+)'
    match = re.search(journal_keywords, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and journal.count(' ') <= 5:
            return journal
    
    # Strategy 4: Between "In:" and year/pages
    in_pattern = r'In\s*:?\s*([^,;.]+?)(?:\s*(?:pp?\.?|S\.|Vol|volume|pages?)\s*[\d]|,\s*\d{4}|$)'
    match = re.search(in_pattern, raw, re.IGNORECASE)
    if match:
        journal = match.group(1).strip()
        if len(journal) > 3 and _JOURNAL_NAME_HINTS.search(journal):
            return journal
    
    # Strategy 5: Comma-separated segments
    comma_segments = raw.split(',')
    for segment in comma_segments:
        segment = segment.strip()
        if (_JOURNAL_NAME_HINTS.search(segment) and
            len(segment) > 5 and len(segment) < 100 and
            not re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:and|&|\b\w+\b))', segment)):
            return segment
    
    return None


def _normalize_extracted_key(raw_key: str) -> str:
    """Normalize PDF-parsed citation keys such as 'Wa14 b' -> 'Wa14b'. Handles Unicode."""
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
    """Parse bibliography section into BibEntry objects."""
    if not bib_text or not bib_text.strip():
        return []

    # Normalize Unicode at start (BUG 4 FIX)
    bib_text = _normalize_unicode(bib_text)
    
    # Keep only entry-like lines and strip obvious PDF junk, but do not rely on
    # keys being at the start of a line; Proceedings PDFs often wrap references so
    # the citation key is embedded in the middle of a line or separated by spaces.
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
        original_key = candidate  # Keep original before normalization
        normalized = _normalize_extracted_key(candidate)
        
        # BUG 1 FIX: Do NOT silently drop malformed keys.
        # Always add valid key format OR use original raw key and flag as format error.
        if not normalized:
            # Normalization completely failed — use raw key but mark for format checking
            normalized = original_key if original_key else candidate
        
        if re.fullmatch(r'\d{1,3}', normalized):
            valid = True
        elif re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', normalized, re.UNICODE):
            valid = True
        else:
            # Try semantic normalization before rejecting (e.g., Smith2020 -> Sm20)
            semantically_normalized = _normalize_key_semantically(normalized)
            if semantically_normalized and re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', semantically_normalized, re.UNICODE):
                valid = True
                normalized = semantically_normalized
            else:
                # Key is malformed, but we KEEP it anyway for the user to see
                valid = True  # Allow it through; completeness check will flag it
        
        if valid:
            key_positions.append((m.start(), normalized, m.end(), original_key))

    # Fallback: if the PDF broke the key as 'Wa14 b', the above regex still catches
    # it via the bracketed text; if not, accept direct key-like strings with spaces.
    if not key_positions:
        for m in re.finditer(r'\b([A-Za-zÀ-ÿ]{1,6}\s*\d{2}\s*[a-z]?)\b', bib_text, re.UNICODE):
            normalized = _normalize_extracted_key(m.group(1))
            if normalized and re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', normalized, re.UNICODE):
                key_positions.append((m.start(), normalized, m.end(), normalized))

    if not key_positions:
        return []

    entries = []
    for i, pos_tuple in enumerate(key_positions):
        # BUG 1 FIX: Handle both 3-tuple (fallback) and 4-tuple (with original_key)
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
        _check_completeness(entry)
        _validate_key_vs_metadata(entry)
        
        # BUG 1 FIX: Flag malformed keys as completeness issues so they appear in UI
        if original_key and original_key != key:
            if not re.fullmatch(r'\d{1,3}', key) and not re.fullmatch(r'[A-Za-zÀ-ÿ]{1,6}\d{2}[a-z]?', key, re.UNICODE):
                entry.completeness_issues.insert(0, f"Malformed citation key: '[{original_key}]' does not follow LNI format (expected e.g. [Sm20], [ABC15], or [1-999]).")
        
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Internal classification + field extraction
# ---------------------------------------------------------------------------

def _classify_and_parse(entry: BibEntry, raw: str) -> None:
    # ── FIX: Handle proceedings PDF garbage extraction ──────────────────────
    # If the raw text is too long and looks like it contains garbage from the PDF,
    # try to extract just the bibliography entry part
    
    # Check if this entry has a valid LNI key and the raw text is extremely long
    if len(raw) > 500 and entry.key:
        # Try to find the entry in the raw text starting from the key
        key_pattern = re.compile(r'\[{}\]\s*([^\[]+?)(?=\s*\[[A-Za-z0-9]+\]|\Z)'.format(re.escape(entry.key)), re.DOTALL)
        match = key_pattern.search(raw)
        if match:
            extracted = match.group(1).strip()
            # If the extracted text is reasonable length, use it
            if 20 < len(extracted) < 800:
                raw = extracted
                entry.raw_text = raw
    
    # PDF extraction can split punctuation, e.g. "Vol . 521".
    raw = re.sub(r'\s+([.,;:)])', r'\1', raw)
    raw = re.sub(r'\bm\s+Ay\b', 'may', raw, flags=re.IGNORECASE)

    # ── DOI ──────────────────────────────────────────────────────────────────
    doi_match = re.search(
        r'(?:doi:\s*|https?://doi\.org/|DOI:\s*)([^\s,;\]]+)',
        raw, re.IGNORECASE,
    )
    if doi_match:
        entry.doi = doi_match.group(1).rstrip('.')

    # ── ISBN ─────────────────────────────────────────────────────────────────
    isbn_match = re.search(
        r'(?:ISBN[:\s-]*)([\d][\d -]{8,16}[\dXx])',
        raw, re.IGNORECASE,
    )
    if isbn_match:
        entry.isbn = re.sub(r'[\s-]', '', isbn_match.group(1))

    # ── URL ──────────────────────────────────────────────────────────────────
    # ── FIX: Better URL extraction for website entries ──────────────────────
    # Look for URLs with or without http:// prefix, including www. domains
    url_match = re.search(r'(https?://\S+|https?:\s+//\S+|www\.\S+|www\.[^\s,;]+)', raw)
    if url_match:
        raw_url = re.sub(r'^(https?):\s+//', r'\1://', url_match.group(1))
        raw_url = raw_url.strip().rstrip('.,;:)]}')
        url_start = url_match.start()

        # Check if this is a DOI URL
        if 'doi.org' in raw_url.lower() or 'dx.doi.org' in raw_url.lower():
            entry.doi = re.sub(r'https?://(dx\.)?doi\.org/', '', raw_url).strip()
            raw = re.sub(r'https?://(dx\.)?doi\.org/[^\s,;]+', '', raw).strip()
        elif 'arxiv.org' in raw_url.lower():
            arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', raw_url, re.IGNORECASE)
            if arxiv_match:
                entry.doi = f"arXiv:{arxiv_match.group(1)}"
            raw = re.sub(r'https?://arxiv\.org/[^\s,;]+', '', raw).strip()
        else:
            # This is a regular URL - classify as website
            entry.entry_type = "website"
            
            # Ensure URL is properly formatted
            if raw_url.startswith('www.'):
                raw_url = 'https://' + raw_url
            
            entry.url = raw_url
            
            # Extract date from "Stand:" or "accessed"
            date_match = re.search(
                r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am|Accessed:)\s*([\d./-]+)',
                raw, re.IGNORECASE,
            )
            if date_match:
                entry.urldate = date_match.group(1)

            # Extract title and author for website entries
            pre_url = raw[:url_start].strip().rstrip(',')
            
            # Check if there's an organization name before the URL
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
            
            # Try to extract year from the raw text
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

    # ── FIX: Also handle website entries that were already classified ──────
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

    # ── Entry type classification ────────────────────────────────────────────
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
        raw, re.IGNORECASE,
    ))
    
    if _has_lni_volume:
        entry.entry_type = "article"
    elif _has_explicit_conf:
        entry.entry_type = "proceedings"
    elif _CONFERENCE_NAMES.search(raw) and not (_PUBLISHER_WORDS.search(raw) and _has_volume):
        entry.entry_type = "proceedings"
    elif _PUBLISHER_WORDS.search(raw) and not _has_volume:
        entry.entry_type = "book"
    elif _has_volume or _has_journal_name:
        entry.entry_type = "article"
    elif _CONFERENCE_NAMES.search(raw):
        entry.entry_type = "proceedings"
    elif _INCOLLECTION_PATTERN.search(raw):
        entry.entry_type = "misc"
    elif _PROCEEDINGS_WORDS.search(raw) and (_has_explicit_conf or not _has_volume):
        entry.entry_type = "proceedings"
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

    # ── Quoted-title format ─────────────────────────────────────────────────
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

    # ── Author extraction ────────────────────────────────────────────────────
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

    # ── Year ──────────────────────────────────────────────────────────────────
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

    # ── Pages ─────────────────────────────────────────────────────────────────
    _roman = r'[ivxlcdmIVXLCDM]+'
    pages_match = re.search(
        r'(?:\bS\.|\bpp?\.)\s*'
        r'(\d+\s*[-–—]+\s*\d+|\d+|'
        rf'{_roman}\s*[-–—]+\s*{_roman}|{_roman})',
        rest, re.IGNORECASE,
    )
    if pages_match:
        entry.pages = pages_match.group(1).replace(' ', '')

    # ── Publisher ─────────────────────────────────────────────────────────────
    pub_match = _PUBLISHER_WORDS.search(rest)
    if pub_match:
        start = max(0, pub_match.start() - 30)
        candidate = rest[start:pub_match.end()].strip().lstrip(',. ')
        entry.publisher = candidate[:80]

    # ── Volume / Number ──────────────────────────────────────────────────────
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

    # ── LNI parenthetical format ────────────────────────────────────────────
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

    # ── Title extraction ─────────────────────────────────────────────────────
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

        first_period = rest_clean.split('.')[0].strip()
        if first_period:
            candidates.append(first_period)

        if candidates:
            max_len = int(len(rest_clean) * 0.85)
            valid = [c for c in candidates if 5 < len(c) <= max_len]
            if valid:
                title_text = min(valid, key=len)
            else:
                title_text = min(candidates, key=len)
        else:
            title_text = rest_clean[:120]

        if entry.entry_type == "article" and title_text:
            title_text = re.sub(
                r'\.\s+[A-ZÄÖÜ][^.]{1,50}$', '', title_text.strip()
            )

        entry.title = title_text.strip().strip('.,;:') or None
        if not entry.title:
            entry.needs_ai_parsing = True
    
    # BUG 1 FIX: Use smart title extraction if regular extraction failed
    if not entry.title:
        smart_title = _extract_title_smart(rest)
        if smart_title:
            entry.title = smart_title

    # ── Booktitle for proceedings ────────────────────────────────────────────
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

    # ── Journal name for articles ────────────────────────────────────────────
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

        j_match = re.search(
            r'(?:\.\s+In:\s+|\.\s+)([A-Za-zäöüÄÖÜ][^,\.]{2,80}?),\s*(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            rest, re.IGNORECASE,
        )
        if j_match and not entry.journal:
            entry.journal = j_match.group(1).strip()
        else:
            in_match = re.search(
                r'\bIn:\s+(.+?)(?=,\s*(?:Nr\.|Vol\.|pp?\.|S\.)|\s+\d+\s*(?:,|\.|\s+Nr\.)|\s+vol\.?)',
                rest, re.IGNORECASE,
            )
            if in_match:
                candidate = in_match.group(1).strip(' ,.')
                if candidate and not re.search(r'Proceedings|Conference|Lecture Notes', candidate, re.IGNORECASE):
                    entry.journal = candidate
            if entry.journal:
                return
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
    
    # BUG 2 FIX: Use smart journal extraction if regular extraction failed
    if not entry.journal and entry.entry_type == "article":
        smart_journal = _extract_journal_smart(rest)
        if smart_journal:
            entry.journal = smart_journal


# ---------------------------------------------------------------------------
# LNI key format validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Key-vs-metadata consistency check
# ---------------------------------------------------------------------------

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
                                   ('â', 'a'), ('î', 'i'), ('ô', 'o'), ('û', 'u')]:
                    s = s.replace(bad, good)
                s = re.sub(r'[^a-z]', '', s)
                return s

            normed = [_norm_surname(s) for s in surnames]

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
                    if surname:
                        initials.add(surname[0].lower())
                    for part in re.split(r'[-\s]+', surname):
                        if part:
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
            if len(surnames) == 2 and key_initials == (normed[0][0] + normed[1][0]):
                entry.key_consistent = True
                entry.key_mismatch_detail = None
                return
            
            if len(surnames) == 1 and key_initials == normed[0][:2]:
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


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _check_completeness(entry: BibEntry) -> None:
    _validate_key_vs_metadata(entry)

    # BUG 5 FIX: Check if key was originally malformed but auto-normalized
    # This ensures the UI is aware of LNI format violations
    errors = validate_lni_key(entry.key)
    for err in errors:
        entry.completeness_issues.append(f"Invalid key format: {err}")
    
    # If key was auto-normalized from a long form, flag it as a format violation
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

    if entry_type == "unknown":
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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def entries_to_dict(entries: list) -> dict:
    return {e.key: e for e in entries}


def parse_raw_references(raw_refs: list) -> list:
    """
    Parse raw extracted references into BibEntry objects.
    Input: list of dicts with 'key' and 'raw_text'
    Output: list of BibEntry objects with fields extracted
    """
    entries = []
    
    for ref in raw_refs:
        key = ref.get('key', '')
        raw = ref.get('raw_text', '')
        
        entry = BibEntry(key=key, raw_text=raw)
        
        # Extract year
        year_m = re.search(r'\b(19|20)\d{2}\b', raw)
        if year_m:
            entry.year = year_m.group(0)
        
        # Extract authors (before first colon typically)
        colon_idx = raw.find(':')
        if colon_idx > 0:
            author_part = raw[:colon_idx].strip()
            if re.search(r'[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,|\s+and\s+|;)', author_part):
                entry.authors = author_part
        
        # Extract title (in quotes)
        title_m = re.search(r'["\']([^"\']{10,})["\']', raw)
        if title_m:
            entry.title = title_m.group(1)
        
        # Extract pages
        pages_m = re.search(r'(?:pp?\.?|S\.)\s*([0-9\-]+)', raw)
        if pages_m:
            entry.pages = pages_m.group(1)
        
        # Extract URL
        url_m = re.search(r'(https?://[^\s,\.]+)', raw)
        if url_m:
            entry.url = url_m.group(1).rstrip('.,;:')
        
        # Extract publisher/venue
        if 'In' in raw or 'in' in raw:
            in_m = re.search(r'In\s+\(([^)]+)\)', raw)
            if in_m:
                entry.booktitle = in_m.group(1)
        
        pub_m = re.search(r'(?:Verlag|Publisher|Press)\s+([^,\.]+)', raw)
        if pub_m:
            entry.publisher = pub_m.group(1).strip()
        
        # Detect type
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