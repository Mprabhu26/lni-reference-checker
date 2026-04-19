"""
STEP 2: Bibliography Parser
----------------------------
Parses the Literaturverzeichnis section of an LNI-formatted document.
Extracts citation keys like [AB00], [Ez10], [GI19] and their metadata.

LNI key format:
  - 1 author:       First 2 letters of surname + 2-digit year  → [Ez10]
  - 2-3 authors:    First letter of each surname + year         → [ABC01]
  - 3+ authors:     First 2 letters of first author + year      → [Az09]
  - No author:      First 2 letters of title + year             → [Di02]
  - Multiple works same year: append a, b, c...                 → [Wa14a]
"""

import re
from dataclasses import dataclass, field
from typing import Optional


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
    completeness_issues: list = field(default_factory=list)


REQUIRED_FIELDS = {
    "book":          ["authors", "title", "publisher", "year"],
    "article":       ["authors", "title", "journal", "year", "pages"],
    "proceedings":   ["authors", "title", "booktitle", "year", "pages"],
    "inproceedings": ["authors", "title", "booktitle", "year", "pages"],
    "website":       ["title", "url", "urldate"],
    "misc":          ["title", "year"],
    "unknown":       ["authors", "title", "year"],
}


def parse_bibliography(bib_text: str) -> list:
    entries = []
    entry_pattern = re.compile(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]\s+(.*?)(?=\n\[|\Z)',
        re.DOTALL
    )
    for match in entry_pattern.finditer(bib_text):
        key = match.group(1)
        raw = re.sub(r'\s+', ' ', match.group(2).strip().replace('\n', ' '))
        entry = BibEntry(key=key, raw_text=raw)
        _classify_and_parse(entry, raw)
        _check_completeness(entry)
        entries.append(entry)
    return entries


def _classify_and_parse(entry: BibEntry, raw: str):
    # DOI extraction first — most reliable identifier
    doi_match = re.search(
        r'(?:doi:\s*|https?://doi\.org/|DOI:\s*)([^\s,;\]]+)',
        raw, re.IGNORECASE
    )
    if doi_match:
        entry.doi = doi_match.group(1).rstrip('.')

    # Website detection
    url_match = re.search(r'(https?://\S+|www\.\S+)', raw)
    if url_match:
        entry.entry_type = "website"
        entry.url = url_match.group(1).rstrip('.,;)')
        date_match = re.search(
            r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am)[:\s]*([\d./-]+)',
            raw, re.IGNORECASE
        )
        if date_match:
            entry.urldate = date_match.group(1)
        title_part = raw[:url_match.start()].strip().rstrip(',.')
        entry.title = title_part if title_part else None
        return

    # Entry type classification
    if re.search(r'(?:Jg\.|Vol\.|Band|Heft|Nr\.|Issue|No\.)\s*[\d]+', raw, re.IGNORECASE):
        entry.entry_type = "article"
    elif re.search(r'\bIn\s*[\(\[]|Proc\.|Proceedings|Conference|Workshop|Symposium|Hrsg\b|eds?\b', raw, re.IGNORECASE):
        entry.entry_type = "proceedings"
    else:
        entry.entry_type = "book"

    # Robust author extraction using LNI name pattern
    # LNI: "Lastname, Firstname [; Lastname2, Firstname2]:"
    author_pattern = re.match(
        r'^((?:[A-ZÄÖÜ][a-zäöüß\-]+(?:,\s*[A-Za-zÄÖÜäöüß\.\s\-]+)?'
        r'(?:;\s*)?)+):\s*(.*)',
        raw
    )
    rest = raw
    if author_pattern:
        candidate = author_pattern.group(1).strip()
        if len(candidate) < 150:
            entry.authors = candidate
            rest = author_pattern.group(2).strip()
    else:
        # Fallback: split on first colon that appears within first 100 chars
        colon_idx = raw.find(':')
        if 0 < colon_idx < 100:
            entry.authors = raw[:colon_idx].strip()
            rest = raw[colon_idx + 1:].strip()

    # Year
    year_match = re.search(r'\b(19|20)\d{2}\b', rest)
    if year_match:
        entry.year = year_match.group(0)

    # Pages
    pages_match = re.search(r'(?:S\.|pp?\.)\s*(\d+\s*[-–—]{1,2}\s*\d+)', rest, re.IGNORECASE)
    if pages_match:
        entry.pages = pages_match.group(1).replace(' ', '')

    # Publisher
    pub_match = re.search(
        r'([A-ZÄÖÜ][^\.,]+(?:Verlag|Press|Sons|Publishers?|Springer|Wiley|Elsevier|ACM|IEEE|MIT))',
        rest
    )
    if pub_match:
        entry.publisher = pub_match.group(1).strip()

    # Title — carefully handle colons within titles
    if rest:
        # Stop at markers that signal the title has ended
        stop_patterns = [
            r'\.\s+In\s+[\(\[]',
            r',\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            r'\.\s+(?:19|20)\d{2}[,\.]',
            r'\.\s+[A-ZÄÖÜ][^\s].*?(?:Verlag|Press|Sons|Publishers?|Springer|Wiley|Elsevier)',
            # For articles: ". JournalName, Vol." — stop at the period before journal
            r'\.\s+[A-ZÄÖÜ][^,\.]{2,40},\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.|Issue)',
        ]
        title_text = rest
        for pat in stop_patterns:
            m = re.search(pat, rest, re.IGNORECASE)
            if m:
                title_text = rest[:m.start()].strip().rstrip('.')
                break
        else:
            # Take up to first period that isn't an abbreviation
            m = re.search(r'(?<![A-ZÄÖÜ])\.\s+[A-ZÄÖÜ]', rest)
            if m:
                title_text = rest[:m.start()].strip().rstrip('.')
            else:
                title_text = rest.split('.')[0].strip()
        # For articles: strip trailing ". JournalName" that leaked before ", Vol."
        if entry.entry_type == "article" and title_text:
            title_text = re.sub(r'\.\s+[A-ZÄÖÜ][^.]{1,40}$', '', title_text.strip())
        entry.title = title_text.strip().strip('.,;:') or None

    # Booktitle for proceedings
    if entry.entry_type == "proceedings":
        bt_match = re.search(r'In\s*[\(\[]([^\)\]]+)[\)\]]|In:\s*([^,\.]+)', rest, re.IGNORECASE)
        if bt_match:
            entry.booktitle = (bt_match.group(1) or bt_match.group(2) or '').strip()

    # Journal name for articles
    if entry.entry_type == "article":
        j_match = re.search(r'\.\s+([A-ZÄÖÜ][^,\.]+?),\s*(?:Jg|Vol|Nr|Band|No)', rest, re.IGNORECASE)
        if j_match:
            entry.journal = j_match.group(1).strip()


def validate_lni_key(key: str) -> list:
    errors = []
    match = re.match(r'^([A-Za-z]+)(\d{2})([a-z])?$', key)
    if not match:
        errors.append(f"Key '{key}' does not follow LNI format (e.g. Ez10, ABC01).")
    else:
        letters = match.group(1)
        if len(letters) < 2 or len(letters) > 6:
            errors.append(f"Author initials in '{key}' should be 2–6 characters, got {len(letters)}.")
    return errors


def _check_completeness(entry: BibEntry):
    for err in validate_lni_key(entry.key):
        entry.completeness_issues.append(f"Invalid key format: {err}")

    required = REQUIRED_FIELDS.get(entry.entry_type, REQUIRED_FIELDS["unknown"])
    for field_name in required:
        if not getattr(entry, field_name, None):
            entry.completeness_issues.append(f"Missing required field: '{field_name}'")

    # LNI formatting rules
    if entry.pages and re.search(r'\d-\d', entry.pages) and '--' not in (entry.pages or ''):
        entry.completeness_issues.append(
            "Page range uses single dash '-' — LNI requires double dash '--' (e.g. S. 12--34)."
        )
    if entry.authors:
        for name in entry.authors.split(';'):
            name = name.strip()
            # Detect "Firstname Lastname" (wrong order)
            if re.match(r'^[A-ZÄÖÜ][a-z]+\s+[A-ZÄÖÜ][a-z]+$', name):
                entry.completeness_issues.append(
                    f"Author '{name}' may be in 'Firstname Lastname' order — "
                    "LNI requires 'Lastname, Firstname'."
                )
                break


def entries_to_dict(entries: list) -> dict:
    return {e.key: e for e in entries}