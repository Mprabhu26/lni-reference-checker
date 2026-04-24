"""
STEP 2: Bibliography Parser
----------------------------
Parses the bibliography section of an LNI-formatted document.
Extracts citation keys like [AB00], [Ez10], [GI19] and their metadata.

LNI key format:
  - 1 author:       First 2 letters of surname + 2-digit year  → [Ez10]
  - 2–3 authors:    First letter of each surname + year         → [ABC01]
  - 3+ authors:     First 2 letters of first author + year      → [Az09]
  - No author:      First 2 letters of title + year             → [Di02]
  - Multiple works same year: append a, b, c...                 → [Wa14a]

FIXES vs v3:
  - validate_key_vs_metadata(): new deterministic check — verifies that the
    author initials and year encoded in the key actually match the parsed
    metadata. Mismatch is a strong fake/typo signal passed to AI.
  - Entry type classification extended with more German/English journal markers.
  - Title extraction: new heuristic keeps the longest plausible candidate across
    multiple stop-pattern attempts, reducing truncated titles.
  - needs_ai_parsing flag added to BibEntry: set True when the regex cannot
    confidently extract title or authors so ai_checker can re-parse those entries.
  - Booktitle extraction improved for "In: Proceedings of ..." patterns.
  - Author order check made more precise (fewer false positives on two-word names
    that happen to be title-case common nouns).
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
    isbn: Optional[str] = None
    completeness_issues: list = field(default_factory=list)
    # Set True when regex extraction is uncertain — triggers AI re-parsing
    needs_ai_parsing: bool = False
    # Result of validate_key_vs_metadata (passed to AI as a signal)
    key_consistent: Optional[bool] = None


# Required fields per entry type (LNI standard)
REQUIRED_FIELDS = {
    "book":          ["authors", "title", "publisher", "year"],
    "article":       ["authors", "title", "journal", "year", "pages"],
    "proceedings":   ["authors", "title", "booktitle", "year", "pages"],
    "inproceedings": ["authors", "title", "booktitle", "year", "pages"],
    "website":       ["title", "url", "urldate"],
    "misc":          ["title", "year"],
    "unknown":       ["authors", "title", "year"],
}

# Known publishers / venues for quick type sniffing
_PUBLISHER_WORDS = re.compile(
    r'(?:Verlag|Press|Publishers?|Sons|GmbH|Books?|'
    r'Springer|Wiley|Elsevier|ACM|IEEE|MIT|O\'Reilly|Pearson|'
    r'Hanser|dpunkt|Addison[- ]Wesley|Cambridge|Oxford)',
    re.IGNORECASE,
)

_PROCEEDINGS_WORDS = re.compile(
    r'\bIn\s*[\(\[]|Proc\.|Proceedings|Conference|Workshop|Symposium|'
    r'Tagung|Konferenz|Hrsg\b|[Ee]ds?\b|editors?\b',
    re.IGNORECASE,
)

_JOURNAL_WORDS = re.compile(
    r'(?:Jg\.|Vol\.|Volume|Band|Heft|Nr\.|Issue|No\.)\s*[\d]+',
    re.IGNORECASE,
)

# German/English journal title fragments — used to fall back to "article" type
_JOURNAL_NAME_HINTS = re.compile(
    r'\b(?:Journal|Zeitschrift|Magazin|Review|Transactions|Letters|'
    r'Bulletin|Annals|Communications|Informatik|Computing)\b',
    re.IGNORECASE,
)


def parse_bibliography(bib_text: str) -> list:
    entries = []
    entry_pattern = re.compile(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]\s+(.*?)(?=\n\[|\Z)',
        re.DOTALL,
    )
    for match in entry_pattern.finditer(bib_text):
        key = match.group(1)
        raw = re.sub(r'\s+', ' ', match.group(2).strip().replace('\n', ' '))
        entry = BibEntry(key=key, raw_text=raw)
        _classify_and_parse(entry, raw)
        _check_completeness(entry)
        _validate_key_vs_metadata(entry)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Internal classification + field extraction
# ---------------------------------------------------------------------------

def _classify_and_parse(entry: BibEntry, raw: str) -> None:
    # ── DOI (most reliable identifier — extract first) ────────────────────────
    doi_match = re.search(
        r'(?:doi:\s*|https?://doi\.org/|DOI:\s*)([^\s,;\]]+)',
        raw, re.IGNORECASE,
    )
    if doi_match:
        entry.doi = doi_match.group(1).rstrip('.')

    # ── ISBN ──────────────────────────────────────────────────────────────────
    isbn_match = re.search(
        r'(?:ISBN[:\s-]*)([\d][\d -]{8,16}[\dXx])',
        raw, re.IGNORECASE,
    )
    if isbn_match:
        entry.isbn = re.sub(r'[\s-]', '', isbn_match.group(1))

    # ── Website detection ─────────────────────────────────────────────────────
    url_match = re.search(r'(https?://\S+|www\.\S+)', raw)
    if url_match:
        entry.entry_type = "website"
        entry.url = url_match.group(1).rstrip('.,;)')
        date_match = re.search(
            r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am)[:\s]*([\d./-]+)',
            raw, re.IGNORECASE,
        )
        if date_match:
            entry.urldate = date_match.group(1)
        title_part = raw[:url_match.start()].strip().rstrip(',.')
        entry.title = title_part if title_part else None
        if not entry.title:
            entry.needs_ai_parsing = True
        return

    # ── Entry type classification ─────────────────────────────────────────────
    if _JOURNAL_WORDS.search(raw) or _JOURNAL_NAME_HINTS.search(raw):
        entry.entry_type = "article"
    elif _PROCEEDINGS_WORDS.search(raw):
        entry.entry_type = "proceedings"
    elif _PUBLISHER_WORDS.search(raw):
        entry.entry_type = "book"
    else:
        # Cannot determine confidently — will be validated for completeness as
        # "unknown" and flagged for optional AI re-parsing
        entry.entry_type = "unknown"
        entry.needs_ai_parsing = True

    # ── Author extraction ─────────────────────────────────────────────────────
    # LNI format: "Lastname, Firstname [; Lastname2, Firstname2]: Title."
    author_pattern = re.match(
        r'^((?:[A-ZÄÖÜ][a-zäöüß\-]+(?:,\s*[A-Za-zÄÖÜäöüß\.\s\-]+)?'
        r'(?:;\s*)?)+):\s*(.*)',
        raw,
    )
    rest = raw
    if author_pattern:
        candidate = author_pattern.group(1).strip()
        # Guard against capturing the entire entry as "author"
        if len(candidate) < 180 and ':' not in candidate:
            entry.authors = candidate
            rest = author_pattern.group(2).strip()
    if not entry.authors:
        # Fallback: split on first colon that appears within first 120 chars
        colon_idx = raw.find(':')
        if 0 < colon_idx < 120:
            candidate = raw[:colon_idx].strip()
            # Sanity-check: looks like "Surname, First" or "Surname"
            if re.match(r'^[A-ZÄÖÜ][a-zäöüß\-]+', candidate):
                entry.authors = candidate
                rest = raw[colon_idx + 1:].strip()
            else:
                entry.needs_ai_parsing = True
        else:
            entry.needs_ai_parsing = True

    # ── Year ──────────────────────────────────────────────────────────────────
    year_match = re.search(r'\b(19|20)\d{2}\b', rest)
    if year_match:
        entry.year = year_match.group(0)

    # ── Pages ─────────────────────────────────────────────────────────────────
    pages_match = re.search(
        r'(?:S\.|pp?\.)\s*(\d+\s*[-–—]{1,2}\s*\d+)',
        rest, re.IGNORECASE,
    )
    if pages_match:
        entry.pages = pages_match.group(1).replace(' ', '')

    # ── Publisher ─────────────────────────────────────────────────────────────
    pub_match = _PUBLISHER_WORDS.search(rest)
    if pub_match:
        # Grab up to 60 chars before and including the publisher word
        start = max(0, pub_match.start() - 30)
        candidate = rest[start:pub_match.end()].strip().lstrip(',. ')
        entry.publisher = candidate[:80]

    # ── Volume / Number ───────────────────────────────────────────────────────
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

    # ── Title extraction ──────────────────────────────────────────────────────
    # We try multiple stop-patterns and keep the LONGEST result that is still
    # shorter than the entire rest string, since a too-long "title" means the
    # stop pattern missed and the rest leaked in.
    if rest:
        candidates = []

        stop_patterns = [
            r'\.\s+In\s+[\(\[]',
            r'\.\s+In:\s+',
            r',\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            r'\.\s+(?:19|20)\d{2}[,\.]',
            r'\.\s+[A-ZÄÖÜ][^\s].*?(?:Verlag|Press|Publishers?|Springer|Wiley|Elsevier)',
            r'\.\s+[A-ZÄÖÜ][^,\.]{2,40},\s+(?:Jg\.|Vol\.|Nr\.|Band|No\.|Issue)',
        ]
        for pat in stop_patterns:
            m = re.search(pat, rest, re.IGNORECASE)
            if m and m.start() > 5:
                c = rest[:m.start()].strip().rstrip('.')
                candidates.append(c)

        # Fallback: stop at the first sentence-ending period not followed by
        # a known abbreviation
        m = re.search(r'(?<![A-ZÄÖÜ])\.\s+[A-ZÄÖÜ]', rest)
        if m and m.start() > 5:
            candidates.append(rest[:m.start()].strip().rstrip('.'))

        # Last resort: first period
        first_period = rest.split('.')[0].strip()
        if first_period:
            candidates.append(first_period)

        if candidates:
            # Keep the longest candidate that doesn't exceed 85% of rest length
            max_len = int(len(rest) * 0.85)
            valid = [c for c in candidates if 5 < len(c) <= max_len]
            if valid:
                title_text = max(valid, key=len)
            else:
                title_text = min(candidates, key=len)  # very short rest
        else:
            title_text = rest[:120]

        # For articles: strip trailing ". JournalName" that leaked in
        if entry.entry_type == "article" and title_text:
            title_text = re.sub(
                r'\.\s+[A-ZÄÖÜ][^.]{1,50}$', '', title_text.strip()
            )

        entry.title = title_text.strip().strip('.,;:') or None
        if not entry.title:
            entry.needs_ai_parsing = True

    # ── Booktitle for proceedings ─────────────────────────────────────────────
    if entry.entry_type == "proceedings":
        bt_match = re.search(
            r'In\s*[\(\[]([^\)\]]+)[\)\]]'
            r'|In:\s*([^,\.]{5,80})',
            rest, re.IGNORECASE,
        )
        if bt_match:
            entry.booktitle = (
                bt_match.group(1) or bt_match.group(2) or ''
            ).strip()

    # ── Journal name for articles ─────────────────────────────────────────────
    if entry.entry_type == "article":
        j_match = re.search(
            r'\.\s+([A-ZÄÖÜ][^,\.]+?),\s*(?:Jg|Vol|Nr|Band|No)',
            rest, re.IGNORECASE,
        )
        if j_match:
            entry.journal = j_match.group(1).strip()


# ---------------------------------------------------------------------------
# LNI key format validation (deterministic)
# ---------------------------------------------------------------------------

def validate_lni_key(key: str) -> list:
    """Return a list of format-error strings (empty = valid)."""
    errors = []
    match = re.match(r'^([A-Za-z]+)(\d{2})([a-z])?$', key)
    if not match:
        errors.append(
            f"Key '{key}' does not follow LNI format (e.g. Ez10, ABC01)."
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
# Key-vs-metadata consistency check (deterministic, new in v4)
# ---------------------------------------------------------------------------

def _validate_key_vs_metadata(entry: BibEntry) -> None:
    """
    Verify that the initials and year encoded in the LNI citation key are
    consistent with the parsed author(s) and year.

    Sets entry.key_consistent = True / False / None (None = cannot check).
    Appends a completeness_issues warning if inconsistent.

    This gives the AI a deterministic, high-confidence signal that something
    is wrong with a reference even when API lookups return no results.
    """
    match = re.match(r'^([A-Za-z]+)(\d{2})([a-z])?$', entry.key)
    if not match:
        entry.key_consistent = None
        return

    key_initials = match.group(1).lower()
    key_year_2d  = match.group(2)          # e.g. "10" for year "2010"

    # ── Year check ────────────────────────────────────────────────────────────
    year_ok: Optional[bool] = None
    if entry.year:
        expected_2d = entry.year[-2:]      # last two digits
        year_ok = (expected_2d == key_year_2d)

    # ── Author initials check ─────────────────────────────────────────────────
    initials_ok: Optional[bool] = None
    if entry.authors:
        # Split on semicolons to get individual authors
        raw_authors = [a.strip() for a in re.split(r';', entry.authors) if a.strip()]
        # Extract first surname from each "Surname, Firstname" or "Surname" token
        surnames = []
        for auth in raw_authors:
            # LNI format is "Surname, Firstname" — take part before comma
            surname = auth.split(',')[0].strip()
            if surname:
                surnames.append(surname)

        if surnames:
            n = len(surnames)
            # Build the set of ALL valid expected initials for this author list.
            # LNI allows some variation in practice, so we accept any valid form:
            #   1 author:   first 2 letters of surname         → "ez" for Ezkiri
            #   2–3 authors: first letter of each surname      → "ms" for Mueller+Schmidt
            #   4+ authors:  first 2 letters of first surname  → "mu" for Mueller et al.
            # Additionally, students sometimes apply the 4+-author rule to 2–3 author
            # entries by mistake, so we also accept the 2-letter prefix as a fallback
            # for any count to avoid false positives.
            valid_forms = set()
            # Always accept first-2-letters of first surname (covers 1-author + common mistake)
            valid_forms.add(surnames[0][:2].lower())
            if n >= 2:
                # Accept per-surname initials form (strict LNI for 2–3 authors)
                valid_forms.add(''.join(s[0].lower() for s in surnames[:min(n, 3)]))

            initials_ok = any(
                key_initials.startswith(form) or form.startswith(key_initials)
                for form in valid_forms
            )

    # ── Combine ───────────────────────────────────────────────────────────────
    checks = [c for c in [year_ok, initials_ok] if c is not None]
    if not checks:
        entry.key_consistent = None
        return

    entry.key_consistent = all(checks)

    if not entry.key_consistent:
        details = []
        if year_ok is False:
            details.append(
                f"key year '{key_year_2d}' ≠ parsed year '{entry.year}'"
            )
        if initials_ok is False:
            details.append(
                f"key initials '{key_initials}' don't match authors '{entry.authors[:40]}'"
            )
        entry.completeness_issues.append(
            "LNI key inconsistency: " + "; ".join(details) +
            " — possible wrong key, renamed author, or fabricated entry."
        )


# ---------------------------------------------------------------------------
# Completeness check (deterministic)
# ---------------------------------------------------------------------------

def _check_completeness(entry: BibEntry) -> None:
    for err in validate_lni_key(entry.key):
        entry.completeness_issues.append(f"Invalid key format: {err}")

    entry_type = entry.entry_type or "unknown"
    # Treat proceedings/inproceedings identically
    lookup_type = "proceedings" if entry_type == "inproceedings" else entry_type
    required = REQUIRED_FIELDS.get(lookup_type, REQUIRED_FIELDS["unknown"])

    for field_name in required:
        if not getattr(entry, field_name, None):
            entry.completeness_issues.append(
                f"Missing required field: '{field_name}'"
            )

    # LNI page-range dash: must be double dash "--"
    if entry.pages:
        if re.search(r'\d-\d', entry.pages) and '--' not in (entry.pages or ''):
            entry.completeness_issues.append(
                "Page range uses single dash '-' — LNI requires '--' "
                "(e.g. S. 12--34)."
            )

    # LNI author order: must be "Lastname, Firstname"
    if entry.authors:
        for name in entry.authors.split(';'):
            name = name.strip()
            # Pattern: TitleCaseWord SPACE TitleCaseWord with no comma
            # Require that neither word is a short particle (von, de, van…)
            if re.match(
                r'^[A-ZÄÖÜ][a-zäöüß]{2,}\s+[A-ZÄÖÜ][a-zäöüß]{2,}$', name
            ) and ',' not in name:
                entry.completeness_issues.append(
                    f"Author '{name}' appears to be 'Firstname Lastname' — "
                    "LNI requires 'Lastname, Firstname'."
                )
                break

    # Future-year check
    if entry.year:
        try:
            import datetime
            if int(entry.year) > datetime.date.today().year + 1:
                entry.completeness_issues.append(
                    f"Year '{entry.year}' is in the future — likely an error."
                )
        except ValueError:
            pass

    # Implausible page range (e.g. pp. 1–500 for a single article)
    if entry.pages:
        m = re.search(r'(\d+)\s*[-–—]+\s*(\d+)', entry.pages)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            span = hi - lo
            if span > 200:
                entry.completeness_issues.append(
                    f"Page range {lo}–{hi} spans {span} pages — "
                    "unusually large for a single article."
                )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def entries_to_dict(entries: list) -> dict:
    return {e.key: e for e in entries}