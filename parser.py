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
import datetime
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
    "website":       ["title", "url"],   # urldate optional
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
    r'\bIn:\s*|\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b|'
    r'\bTagung\b|\bKonferenz\b|\bHrsg\b|\bEds?\.\B|\beditors?\b',
    re.IGNORECASE,
)

_JOURNAL_WORDS = re.compile(
    r'(?:Jg\.|Vol\.|Volume|Band|Heft|Nr\.|Issue|No\.)\s*[\d]+',
    re.IGNORECASE,
)

# German/English journal title fragments — used to fall back to "article" type
_JOURNAL_NAME_HINTS = re.compile(
    r'\b(?:Journal|Zeitschrift|Magazin|Review|Transactions|Letters|'
    r'Bulletin|Annals|Communications|Informatik|Computing|'
    r'Quarterly|Professional|Magazine|Publication|Proceedings|'
    r'Systems|Research|Reports?|Science|Technology|Management)\b',
    re.IGNORECASE,
)


def parse_bibliography(bib_text: str) -> list:
    """Parse bibliography section into BibEntry objects.

    Two-pass approach so that invalid keys like [vaswani2017] or [X] are
    captured AND can receive completeness issues, and mid-entry newlines
    (which appear in real student PDFs) are handled correctly.
    """
    if not bib_text or not bib_text.strip():
        return []

    # Broad key: any [something] at start of a logical line (valid OR invalid).
    ANY_KEY = re.compile(r'(?:^|\n)\s*\[([^\]\n]{1,30})\]', re.MULTILINE)
    positions = [(m.start(), m.group(1), m.end()) for m in ANY_KEY.finditer(bib_text)]
    if not positions:
        return []

    entries = []
    for i, (chunk_start, key, body_start) in enumerate(positions):
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(bib_text)
        raw_body = bib_text[body_start:body_end]
        # Collapse ALL whitespace (including mid-entry newlines) to single spaces
        raw = re.sub(r'\s+', ' ', raw_body).strip()
        if not raw:
            continue

        # Handle compound numeric keys like "1,2" or "1, 2" — split into
        # individual entries that each share the same raw body text.
        sub_keys = [k.strip() for k in key.split(',') if k.strip()] \
            if ',' in key else [key]

        for sub_key in sub_keys:
            entry = BibEntry(key=sub_key, raw_text=raw)
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

    # ── URL / DOI detection ──────────────────────────────────────────────────
    # Extract URL as-is, no cleaning except basic whitespace
    url_match = re.search(r'(https?://\S+|https?:\s+//\S+|www\.\S+)', raw)
    if url_match:
        raw_url = re.sub(r'^(https?):\s+//', r'\1://', url_match.group(1))
        raw_url = raw_url.strip()
        url_start = url_match.start()
        
        # ── Check if this is a DOI URL (doi.org) ──────────────────────────────
        # If it's a DOI URL, extract DOI and continue processing as normal
        # (don't classify as "website" - it's an academic paper with a DOI)
        if 'doi.org' in raw_url.lower() or 'dx.doi.org' in raw_url.lower():
            entry.doi = re.sub(r'https?://(dx\.)?doi\.org/', '', raw_url).strip()
            # Remove the DOI from raw text so it doesn't interfere with title extraction
            raw = re.sub(r'https?://(dx\.)?doi\.org/[^\s,;]+', '', raw).strip()
            # Continue with normal classification (not website)
            # The entry will be classified as article/proceedings/book below
        
        # ── Check if this is a arXiv URL ──────────────────────────────────────
        elif 'arxiv.org' in raw_url.lower():
            # Extract arXiv ID
            arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', raw_url, re.IGNORECASE)
            if arxiv_match:
                entry.doi = f"arXiv:{arxiv_match.group(1)}"
            # Remove arXiv URL from raw text
            raw = re.sub(r'https?://arxiv\.org/[^\s,;]+', '', raw).strip()
            # Continue with normal classification
        
        # ── Otherwise, it's a real website URL ──────────────────────────────
        else:
            entry.entry_type = "website"
            entry.url = raw_url

            # Extract urldate separately if present
            date_match = re.search(
                r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am)[:\s]*([\d./-]+)',
                raw, re.IGNORECASE,
            )
            if date_match:
                entry.urldate = date_match.group(1)

            # Extract authors and clean title from text before the URL
            pre_url = raw[:url_start].strip().rstrip(',')

            # Pattern 1: "Lastname[ Lastname2], Firstname[ and ...]: Title"
            author_m = re.match(
                r'^((?:[A-ZÄÖÜ][a-zäöüß\-]+(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)*'
                r'(?:,\s*[A-Za-zÄÖÜäöüß.\s\-]+)?'
                r'(?:\s+and\s+|\s*;\s*)?)+):\s*(.*)',
                pre_url,
            )
            if author_m:
                cand = author_m.group(1).strip()
                if len(cand) < 150 and not re.search(r'\b(19|20)\d{2}\b', cand):
                    entry.authors = cand
                    pre_url = author_m.group(2).strip()

            # Pattern 2: Organisation name
            if not entry.authors:
                org_m = re.match(r'^([A-ZÄÖÜ][A-Za-z0-9äöüÄÖÜ\s\.\-]{1,50}?):\s*(.*)', pre_url)
                if org_m:
                    org_cand = org_m.group(1).strip().rstrip(',.')
                    rest_after = org_m.group(2).strip()
                    if 2 < len(org_cand) < 80 and not re.search(r'\b(19|20)\d{2}\b', org_cand) and rest_after:
                        entry.authors = org_cand
                        pre_url = rest_after

            # Extract year from pre-URL region
            year_m = re.search(r'\b(19|20)\d{2}\b', raw[:url_start])
            if year_m:
                entry.year = year_m.group(0)

            # Clean title: strip leading/trailing year
            title_text = re.sub(r'^\s*\b(19|20)\d{2}\b\s*', '', pre_url).strip()
            title_text = re.sub(r',?\s*\b(19|20)\d{2}\b\s*$', '', title_text).strip().rstrip(',.')
            entry.title = title_text if title_text else None
            if not entry.title:
                entry.needs_ai_parsing = True
            return

    # ── Entry type classification ─────────────────────────────────────────────
    # IMPORTANT: check proceedings FIRST — a proceedings entry often contains
    # journal-like words (e.g. "Informatik", "Jg.") that would falsely trigger
    # the article branch. "In: Proceedings" is an unambiguous proceedings signal
    # and must take priority over any journal name hints.
    # A volume marker (Vol./Jg./Nr.) is a strong article signal — it overrides
    # the bare 'In:' proceedings heuristic (which also fires for 'In: Nature, Vol.').
    # Exception: explicit conference/proceedings words always win.
    _has_volume = bool(re.search(r'(?:Jg\.|Vol\.|Nr\.|Band)\s*\d', raw, re.IGNORECASE))
    _has_explicit_conf = bool(re.search(
        r'\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b'
        r'|\bTagung\b|\bKonferenz\b|\bHrsg\b|\bEds?\.\B',
        raw, re.IGNORECASE))

    if _PROCEEDINGS_WORDS.search(raw) and (_has_explicit_conf or not _has_volume):
        entry.entry_type = "proceedings"
    elif _JOURNAL_WORDS.search(raw) or _JOURNAL_NAME_HINTS.search(raw) or _has_volume:
        entry.entry_type = "article"
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
    # Matches: digit ranges (S. 12--34), roman numeral ranges (S. xiii–xxiii),
    # and single page numbers (S. 292) — all valid LNI page field values.
    _roman = r'[ivxlcdmIVXLCDM]+'
    pages_match = re.search(
        r'(?:S\.|pp?\.)\s*'
        r'((?:\d+\s*[-–—]{{1,2}}\s*\d+|'   # digit range: 12--34
        r'\d+|'                             # single digit page: 292
        r'{_roman}\s*[-–—]{{1,2}}\s*{_roman}|'  # roman range: xiii–xxiii
        r'{_roman}))'                       # single roman: xiv
        .format(_roman=_roman),
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
            # Keep the SHORTEST valid candidate — earlier stop = cleaner title boundary.
            # A longer candidate almost always means a stop pattern was missed and
            # journal/venue text leaked into the title.
            max_len = int(len(rest) * 0.85)
            valid = [c for c in candidates if 5 < len(c) <= max_len]
            if valid:
                title_text = min(valid, key=len)
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
            r'(?:\.\s+In:\s+|\.\s+)([A-Za-zäöüÄÖÜ][^,\.]{2,60}?),\s*(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            rest, re.IGNORECASE,
        )
        if j_match:
            entry.journal = j_match.group(1).strip()
        else:
            # Fallback: "Title. Journal Name 53(4)" — volume number after journal name
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
                # Fallback 3: "Title. Journal Name, S. <pages>" — no volume at all
                # Covers entries like "MIS quarterly, S. xiii\u2013xxiii, 2002"
                j_match3 = re.search(
                    r'\.\s+([A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc][A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-]{2,60}?),\s*S\.',
                    rest,
                )
                if j_match3:
                    candidate = j_match3.group(1).strip().rstrip(',.')
                    if (not re.match(r'^(19|20)\d{2}$', candidate)
                            and len(candidate) > 2):
                        entry.journal = candidate


# ---------------------------------------------------------------------------
# LNI key format validation (deterministic)
# ---------------------------------------------------------------------------

def validate_lni_key(key: str) -> list:
    """Return a list of format-error strings (empty = valid)."""
    errors = []
    
    # Skip validation for numeric keys
    if key.isdigit():
        return errors
    
    # LNI keys: initials must be UPPERCASE (first letter of surname)
    match = re.match(r'^([A-Z][A-Za-z]*)(\d{2})([a-z])?$', key)
    if not match:
        errors.append(
            f"Key '{key}' does not follow LNI format (e.g. Ez10, ABC01). "
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
    if entry.key.isdigit():
        entry.key_consistent = True
        return
    match = re.match(r'^([A-Z][A-Za-z]*)(\d{2})([a-z])?$', entry.key)
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

            def _norm_surname(s: str) -> str:
                """Normalise a surname for initial comparison: lowercase + map umlauts."""
                s = s.lower()
                # Map German umlauts and common transliterations
                for bad, good in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss'),
                                   ('é','e'),('è','e'),('ê','e'),('à','a'),
                                   ('â','a'),('î','i'),('ô','o'),('û','u')]:
                    s = s.replace(bad, good)
                return s

            normed = [_norm_surname(s) for s in surnames]

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
            valid_forms.add(normed[0][:2])
            # Also accept 1-letter prefix alone (e.g. key "M" vs surname "Mueller")
            valid_forms.add(normed[0][:1])
            if n >= 2:
                # Accept per-surname initials form (strict LNI for 2–3 authors)
                valid_forms.add(''.join(s[0] for s in normed[:min(n, 3)]))
                # Accept 4+-author style (2-letter prefix of first surname)
                valid_forms.add(normed[0][:2])
            if n >= 4:
                # Accept first letter of each of first 4 surnames
                valid_forms.add(''.join(s[0] for s in normed[:4]))

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
    # Skip for website/misc — author may be an organisation name
    if entry.authors and entry.entry_type not in ("website", "misc", "online"):
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