"""
STEP 2: Bibliography Parser
----------------------------
Parses the Literaturverzeichnis section of an LNI-formatted PDF.
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
    entry_type: Optional[str] = None      # book, article, proceedings, website, misc
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
    completeness_issues: list = field(default_factory=list)


# Required fields per entry type (LNI rules)
REQUIRED_FIELDS = {
    "book":        ["authors", "title", "publisher", "year"],
    "article":     ["authors", "title", "journal", "year", "pages"],
    "proceedings": ["authors", "title", "booktitle", "year", "pages"],
    "inproceedings": ["authors", "title", "booktitle", "year", "pages"],
    "website":     ["title", "url", "urldate"],
    "misc":        ["title", "year"],
    "unknown":     ["authors", "title", "year"],
}


def parse_bibliography(bib_text: str) -> list[BibEntry]:
    """
    Parse bibliography section and return a list of BibEntry objects.
    Handles both LNI Word-style [Key] and detects entry boundaries.
    """
    entries = []

    # Match entries starting with [KEY] pattern
    # LNI format: [Ab99] Author, ...: Title. ...
    entry_pattern = re.compile(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]\s+(.*?)(?=\n\[|\Z)',
        re.DOTALL
    )

    matches = list(entry_pattern.finditer(bib_text))

    for match in matches:
        key = match.group(1)
        raw = match.group(2).strip().replace('\n', ' ')
        raw = re.sub(r'\s+', ' ', raw)

        entry = BibEntry(key=key, raw_text=raw)
        _classify_and_parse(entry, raw)
        _check_completeness(entry)
        entries.append(entry)

    return entries


def _classify_and_parse(entry: BibEntry, raw: str):
    """Heuristically detect entry type and extract fields."""

    # Detect URLs → website
    url_match = re.search(r'(https?://\S+|www\.\S+)', raw)
    if url_match:
        entry.entry_type = "website"
        entry.url = url_match.group(1)
        # Try to get urldate (Stand: DD.MM.YYYY or accessed YYYY-MM-DD)
        date_match = re.search(
            r'(?:Stand:|Abruf:|abgerufen am|accessed)[:\s]*([\d./-]+)', raw, re.IGNORECASE
        )
        if date_match:
            entry.urldate = date_match.group(1)
        # Title is everything before the URL
        title_part = raw[:url_match.start()].strip().rstrip(',.')
        entry.title = title_part if title_part else None
        return

    # Detect journal articles → has "Jg." or volume/issue pattern
    journal_match = re.search(
        r'(?:Jg\.|Vol\.|Band|Heft|Nr\.|Issue)\s*[\d]+', raw, re.IGNORECASE
    )
    if journal_match:
        entry.entry_type = "article"

    # Detect proceedings → "In (" or "Proc." or "Proceedings"
    elif re.search(r'\bIn\s*\(|Proc\.|Proceedings|Conference|Workshop|Symposium', raw, re.IGNORECASE):
        entry.entry_type = "proceedings"

    # Default to book
    else:
        entry.entry_type = "book"

    # Extract authors: text before first ":"
    colon_idx = raw.find(':')
    if colon_idx > 0:
        entry.authors = raw[:colon_idx].strip()
        rest = raw[colon_idx + 1:].strip()
    else:
        rest = raw

    # Extract year: 4-digit number
    year_match = re.search(r'\b(19|20)\d{2}\b', rest)
    if year_match:
        entry.year = year_match.group(0)

    # Extract pages: S. 12-34 or pp. 12-34 or just 12-34
    pages_match = re.search(r'(?:S\.|pp?\.)\s*(\d+[-–]\d+)', rest, re.IGNORECASE)
    if pages_match:
        entry.pages = pages_match.group(1)

    # Extract publisher (word before city/year pattern, or after last comma before year)
    # Heuristic: publisher is often the last "Verlag" or proper noun before year
    publisher_match = re.search(r'([A-ZÄÖÜ][^\.,]+(?:Verlag|Press|Sons|Publishers?))', rest)
    if publisher_match:
        entry.publisher = publisher_match.group(1).strip()

    # Extract title: first sentence/phrase after authors
    if entry.authors:
        # Title is typically the first complete phrase ending with "."
        title_match = re.match(r'\s*([^.]+?)\.', rest)
        if title_match:
            entry.title = title_match.group(1).strip()
    else:
        title_match = re.match(r'([^.]+?)\.', rest)
        if title_match:
            entry.title = title_match.group(1).strip()

    # For proceedings: extract booktitle from "In (...): Title"
    if entry.entry_type == "proceedings":
        bt_match = re.search(r'In\s*\([^)]+\):\s*([^,\.]+)', rest, re.IGNORECASE)
        if bt_match:
            entry.booktitle = bt_match.group(1).strip()

    # For articles: journal name
    if entry.entry_type == "article":
        # Journal is usually after title, before volume info
        j_match = re.search(r'\.\s+([A-ZÄÖÜ][^,]+?),\s*(?:Jg|Vol|Nr|Band)', rest, re.IGNORECASE)
        if j_match:
            entry.journal = j_match.group(1).strip()


def validate_lni_key(key: str) -> list[str]:
    """
    Validates if a key like [Ab00] or [ABC01] follows LNI rules.
    Rules: 2-6 letters + 2 digits + optional lowercase suffix (a, b, c...).
    """
    errors = []
    match = re.match(r'^([A-Za-z]+)(\d{2})([a-z])?$', key)
    if not match:
        errors.append(f"Key '{key}' does not follow LNI format [AuthorYear] (e.g. Ez10, ABC01).")
    else:
        letters = match.group(1)
        if len(letters) < 2 or len(letters) > 6:
            errors.append(f"Author initials in key '{key}' should be 2-6 characters, got {len(letters)}.")
    return errors


def _check_completeness(entry: BibEntry):
    """Check if all required fields are present for this entry type, and validate the key format."""
    # Validate LNI citation key format
    key_errors = validate_lni_key(entry.key)
    for err in key_errors:
        entry.completeness_issues.append(f"Invalid key format: {err}")

    # Validate required fields per entry type
    required = REQUIRED_FIELDS.get(entry.entry_type, REQUIRED_FIELDS["unknown"])
    for field_name in required:
        val = getattr(entry, field_name, None)
        if not val:
            entry.completeness_issues.append(
                f"Missing required field: '{field_name}'"
            )


def entries_to_dict(entries: list[BibEntry]) -> dict:
    """Convert list of BibEntry to dict keyed by citation key."""
    return {e.key: e for e in entries}


if __name__ == "__main__":
    # Quick test with sample LNI bibliography text
    sample = """
Literaturverzeichnis

[AB00] Abel, K.; Bibel, U.: Formatierungsrichtlinien für Tagungsbände. Format-Verlag, Bonn, 2000.

[ABC01] Abraham, N.; Bibel, U.; Corleone, P.: Formatting Contributions for Proceedings. In (Glück, H.I., Hrsg.): Proc. 7th Int. Conf. on Formatting, San Francisco. Noah & Sons, S. 46-53, 2001.

[Ez10] Ezgarani, O.: The Magic Format -- Your Way to Pretty Books. Noah & Sons, 2010.

[Gl06] Glück, H. I.: Formatierung leicht gemacht. Formatierungsjournal, Jg. 11, Nr. 09, S. 23-27, 2009.

[GI19] GI, Gesellschaft für Informatik e.V., www.gi-ev.de, Stand: 21.03.2019.

[XX14] Anteil an Frauen in der Informatik. Statistics Worldwide, 2014.
    """

    entries = parse_bibliography(sample)
    for e in entries:
        print(f"\n[{e.key}] type={e.entry_type}")
        print(f"  authors:   {e.authors}")
        print(f"  title:     {e.title}")
        print(f"  year:      {e.year}")
        print(f"  publisher: {e.publisher}")
        print(f"  url:       {e.url}")
        print(f"  issues:    {e.completeness_issues}")