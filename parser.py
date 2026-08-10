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

FIXES v7.0:
  - Entry type classification: better detection of books, articles, proceedings
  - Key mismatch detection: fixed surname extraction (handles "Kingma, Diederik")
  - No longer flags correctly formatted entries as "issues"
  - Better handling of "et al." in author lists
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
    needs_ai_parsing: bool = False
    key_consistent: Optional[bool] = None
    key_mismatch_detail: Optional[str] = None


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
    r'Springer|Wiley|Elsevier|ACM|IEEE|MIT|O\'Reilly|Pearson|'
    r'Hanser|dpunkt|Addison[- ]Wesley|Cambridge|Oxford|'
    r'McGraw|Macmillan|Routledge|Sage|Taylor|Francis|CRC|'
    r'De Gruyter|Nomos|Beck|Mohr|Kohlhammer|Juventa|UTB)',
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

# Conference/venue names that should be classified as proceedings
_CONFERENCE_NAMES = re.compile(
    r'\b(?:NeurIPS|CVPR|ICCV|ECCV|ICML|ICLR|ACL|EMNLP|NAACL|AAAI|IJCAI|'
    r'NIPS|COLT|UAI|AISTATS|ICRA|IROS|CHI|UIST|SIGGRAPH|SIGIR|SIGMOD|'
    r'VLDB|SOSP|OSDI|USENIX|NDSS|IEEE|SPIE|LNI|GI|INFORMATIK)\b',
    re.IGNORECASE,
)

# Journal name hints
_JOURNAL_NAME_HINTS = re.compile(
    r'\b(?:Journal|Zeitschrift|Magazin|Review|Transactions|Letters|'
    r'Bulletin|Annals|Communications|Informatik|Computing|'
    r'Quarterly|Professional|Magazine|Publication|'
    r'Systems|Research|Reports?|Science|Technology|Management)\b',
    re.IGNORECASE,
)


def parse_bibliography(bib_text: str) -> list:
    """Parse bibliography section into BibEntry objects."""
    if not bib_text or not bib_text.strip():
        return []

    ANY_KEY = re.compile(r'(?:^|\n)\s*\[([^\]\n]{1,30})\]', re.MULTILINE)
    bracket_positions = [(m.start(), m.group(1), m.end(), True)
                          for m in ANY_KEY.finditer(bib_text)]

    # FIX v8.3: Malformed entries (missing LNI brackets entirely) were being
    # silently dropped or merged into the previous/next bracketed entry,
    # which caused real references to disappear from the bibliography and
    # falsely show up as "cited but missing". Detect unbracketed entry
    # starts too: a line/segment beginning with a bare key-like token
    # followed by " - ", ":", or "(" and author-like text (a capitalized
    # surname), e.g. "LBH15 - LeCun, Yann and ...", "adam_optimizer_2014:
    # Kingma, D.P...", or a line starting directly with "Surname, Initial."
    _LNI_MARKER_WORDS = {
        'in', 'vol', 'volume', 'no', 's', 'pp', 'p', 'hrsg', 'eds', 'ed',
        'jg', 'nr', 'band', 'heft', 'issue', 'stand', 'doi', 'isbn', 'and',
    }
    UNBRACKETED_KEY = re.compile(
        r'(?:^|\n|;\s|\.\s)\s*'
        r'([A-Za-z][A-Za-z0-9_]{2,30})\s*[-:]\s*'
        r'(?=[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?,\s)',
        re.MULTILINE,
    )
    # FIX v8.4: Catch line-start keys followed by ( or [ without : or -
    # Examples: "APA20 (Author...", "IEEE21 [...". These are valid but not
    # caught by the original UNBRACKETED_KEY pattern which requires : or -.
    LINESTART_KEY = re.compile(
        r'(?:^|\n)\s*([A-Z][A-Z0-9]+)\s+(?=[\(\[])',
        re.MULTILINE,
    )
    UNBRACKETED_AUTHOR_START = re.compile(
        r'(?:^|\n)\s*([A-Z][a-zA-Z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)?,?\s*(?:&|and|,)\s)',
        re.MULTILINE,
    )

    unbracketed_positions = []

    raw_unbracketed_key = [(m.start(1), m.group(1), m.end(), False)
                            for m in UNBRACKETED_KEY.finditer(bib_text)
                            if m.group(1).lower() not in _LNI_MARKER_WORDS]
    # Add line-start keys
    raw_linestart_key = [(m.start(1), m.group(1), m.end(), False)
                          for m in LINESTART_KEY.finditer(bib_text)]
    
    raw_unbracketed_author = []
    for m in UNBRACKETED_AUTHOR_START.finditer(bib_text):
        start = m.start(1)
        if any(abs(c[0] - start) < 5 for c in raw_unbracketed_key):
            continue
        window = bib_text[start:start + 200]
        surname_m = re.match(r'([A-Z][a-zA-Z]+)', window)
        year_m = re.search(r'\((\d{4})\)|\b(\d{4})\b', window)
        surname = surname_m.group(1) if surname_m else "Unknown"
        year = (year_m.group(1) or year_m.group(2))[-2:] if year_m else "??"
        raw_unbracketed_author.append((start, f"{surname[:2]}{year}", start, False))

    all_candidates = sorted(raw_unbracketed_key + raw_linestart_key + raw_unbracketed_author, key=lambda p: p[0])

    # A bracketed entry's own text ends at the next paragraph break (blank
    # line) after its key, or at the next bracket, whichever comes first.
    # Candidates found before that boundary are genuinely part of the
    # bracketed entry's running prose (e.g. "...Nature, Vol. 521...") and
    # must be excluded; candidates after it are separate, unrelated
    # (malformed) entries and must be kept.
    PARA_BREAK = re.compile(r'(?:\n\s*\n)|(?:[.\)]\s*\n(?=[A-Za-z]))')

    def _entry_own_text_end(bracket_key_end, next_bracket_start):
        m = PARA_BREAK.search(bib_text, bracket_key_end, next_bracket_start)
        return m.start() if m else next_bracket_start

    bracket_own_spans = []
    for i, bp in enumerate(bracket_positions):
        next_bracket_start = bracket_positions[i + 1][0] if i + 1 < len(bracket_positions) else len(bib_text)
        # An unbracketed candidate belongs to THIS bracketed entry's own
        # text only if there's no newline between the bracket's key and
        # the candidate (still mid-sentence / same wrapped line). Once a
        # newline is crossed, treat any candidate as a new, separate
        # (possibly malformed) entry rather than assuming it's still part
        # of this entry's prose.
        newline_m = re.search(r'\n', bib_text[bp[2]:next_bracket_start])
        own_end = bp[2] + newline_m.start() if newline_m else next_bracket_start
        bracket_own_spans.append((bp[0], own_end))

    for start, key, end, was_bracketed in all_candidates:
        if any(s <= start < e for s, e in bracket_own_spans):
            continue
        unbracketed_positions.append((start, key, end, was_bracketed))

    positions = sorted(bracket_positions + unbracketed_positions, key=lambda p: p[0])
    if not positions:
        return []

    entries = []
    for i, (chunk_start, key, body_start, was_bracketed) in enumerate(positions):
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(bib_text)
        raw_body = bib_text[body_start:body_end]
        raw = re.sub(r'\s+', ' ', raw_body).strip()
        if not raw:
            continue

        # FIXED v8.2: Normalize keys by removing ALL internal spaces
        # PDF extraction produces keys like "RH15 a" instead of "RH15a"
        # This ensures they match citations extracted from body text
        sub_keys = [re.sub(r'\s+', '', k.strip()) for k in key.split(',') if k.strip()] \
            if ',' in key else [re.sub(r'\s+', '', key.strip())]

        for sub_key in sub_keys:
            entry = BibEntry(key=sub_key, raw_text=raw)
            _classify_and_parse(entry, raw)
            _check_completeness(entry)
            _validate_key_vs_metadata(entry)
            if not was_bracketed:
                entry.needs_ai_parsing = True
                entry.completeness_issues.append(
                    "Entry is missing LNI-required [Key] brackets — "
                    "key was inferred, not stated in the source."
                )
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Internal classification + field extraction
# ---------------------------------------------------------------------------

def _classify_and_parse(entry: BibEntry, raw: str) -> None:
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
    url_match = re.search(r'(https?://\S+|https?:\s+//\S+|www\.\S+)', raw)
    if url_match:
        raw_url = re.sub(r'^(https?):\s+//', r'\1://', url_match.group(1))
        raw_url = raw_url.strip()
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
            entry.url = raw_url
            date_match = re.search(
                r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am)[:\s]*([\d./-]+)',
                raw, re.IGNORECASE,
            )
            if date_match:
                entry.urldate = date_match.group(1)

            pre_url = raw[:url_start].strip().rstrip(',')
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

            if not entry.authors:
                org_m = re.match(r'^([A-ZÄÖÜ][A-Za-z0-9äöüÄÖÜ\s\.\-]{1,50}?):\s*(.*)', pre_url)
                if org_m:
                    org_cand = org_m.group(1).strip().rstrip(',.')
                    rest_after = org_m.group(2).strip()
                    if 2 < len(org_cand) < 80 and not re.search(r'\b(19|20)\d{2}\b', org_cand) and rest_after:
                        entry.authors = org_cand
                        pre_url = rest_after

            year_m = re.search(r'\b(19|20)\d{2}\b', raw[:url_start])
            if year_m:
                entry.year = year_m.group(0)

            title_text = re.sub(r'^\s*\b(19|20)\d{2}\b\s*', '', pre_url).strip()
            title_text = re.sub(r',?\s*\b(19|20)\d{2}\b\s*$', '', title_text).strip().rstrip(',.')
            entry.title = title_text if title_text else None
            if not entry.title:
                entry.needs_ai_parsing = True
            return

    # ── Entry type classification ───────────────────────────────────────────
    _has_volume = bool(re.search(r'(?:Jg\.|Vol\.|Nr\.|Band)\s*\d', raw, re.IGNORECASE))
    _has_explicit_conf = bool(re.search(
        r'\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b'
        r'|\bTagung\b|\bKonferenz\b|\bHrsg\b|\bEds?\.\B',
        raw, re.IGNORECASE))

    # Check for conference names first (strongest signal)
    if _CONFERENCE_NAMES.search(raw):
        entry.entry_type = "proceedings"
    elif _PROCEEDINGS_WORDS.search(raw) and (_has_explicit_conf or not _has_volume):
        entry.entry_type = "proceedings"
    elif _JOURNAL_WORDS.search(raw) or _JOURNAL_NAME_HINTS.search(raw) or _has_volume:
        entry.entry_type = "article"
    elif _PUBLISHER_WORDS.search(raw):
        entry.entry_type = "book"
    else:
        # ── FIX: Better default detection ──────────────────────────────────
        # Check for book publisher keywords
        if re.search(r'MIT Press|Springer|Elsevier|Wiley|O\'Reilly|Pearson|Cambridge|Oxford|McGraw|Macmillan', raw, re.IGNORECASE):
            entry.entry_type = "book"
        # Check for journal/conference keywords
        elif re.search(r'Journal|Transactions|Letters|Magazine|Review|IEEE|ACM', raw, re.IGNORECASE):
            entry.entry_type = "article"
        # Check for "In: ConferenceName"
        elif re.search(r'In:\s*[A-Z][a-zA-Z0-9\s]+', raw, re.IGNORECASE):
            entry.entry_type = "proceedings"
        else:
            entry.entry_type = "unknown"
            entry.needs_ai_parsing = True

    # ── Author extraction ──────────────────────────────────────────────────
    author_pattern = re.match(
        r'^((?:[A-ZÄÖÜ][a-zäöüß\-]+(?:,\s*[A-Za-zÄÖÜäöüß\.\s\-]+)?'
        r'(?:;\s*)?)+):\s*(.*)',
        raw,
    )
    rest = raw
    if author_pattern:
        candidate = author_pattern.group(1).strip()
        if len(candidate) < 180 and ':' not in candidate:
            entry.authors = candidate
            rest = author_pattern.group(2).strip()
    if not entry.authors:
        colon_idx = raw.find(':')
        if 0 < colon_idx < 120:
            candidate = raw[:colon_idx].strip()
            if re.match(r'^[A-ZÄÖÜ][a-zäöüß\-]+', candidate):
                entry.authors = candidate
                rest = raw[colon_idx + 1:].strip()
            else:
                entry.needs_ai_parsing = True
        else:
            entry.needs_ai_parsing = True

    # ── Year ─────────────────────────────────────────────────────────────────
    # FIXED v8.2: Handle PDF extraction artifacts where years are broken
    # Example: "2016" → "2 016" (space after first digit)
    # Example: "2016" → "20 16" (space after century)
    year_match = re.search(r'\b(19|20)\d{2}\b', rest)
    if year_match:
        entry.year = year_match.group(0)
    else:
        # Fallback 1: "20 16" or "19 95" (space between century and year)
        broken1 = re.search(r'\b(19|20)\s+(\d{2})\b', rest)
        if broken1:
            entry.year = broken1.group(1) + broken1.group(2)
        else:
            # Fallback 2: "2 016" (space after first digit - PDF extraction artifact)
            broken2 = re.search(r'\b(\d)\s+(\d{3})\b', rest)
            if broken2 and broken2.group(1) in '12':
                entry.year = broken2.group(1) + broken2.group(2)

    # ── Pages ────────────────────────────────────────────────────────────────
    _roman = r'[ivxlcdmIVXLCDM]+'
    pages_match = re.search(
        r'(?:S\.|pp?\.)\s*'
        r'((?:\d+\s*[-–—]{{1,2}}\s*\d+|'
        r'\d+|'
        r'{_roman}\s*[-–—]{{1,2}}\s*{_roman}|'
        r'{_roman}))'
        .format(_roman=_roman),
        rest, re.IGNORECASE,
    )
    if pages_match:
        entry.pages = pages_match.group(1).replace(' ', '')

    # ── Publisher ────────────────────────────────────────────────────────────
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

    # ── Title extraction ─────────────────────────────────────────────────────
    if rest:
        # Clean up rest: remove URLs and years from the end
        rest_clean = re.sub(r',?\s*https?://\S+', '', rest)
        rest_clean = re.sub(r',?\s*(19|20)\d{2}\s*$', '', rest_clean)

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

    # ── Booktitle for proceedings ──────────────────────────────────────────
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

    # ── Journal name for articles ──────────────────────────────────────────
    if entry.entry_type == "article":
        j_match = re.search(
            r'(?:\.\s+In:\s+|\.\s+)([A-Za-zäöüÄÖÜ][^,\.]{2,60}?),\s*(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            rest, re.IGNORECASE,
        )
        if j_match:
            entry.journal = j_match.group(1).strip()
        else:
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


# ---------------------------------------------------------------------------
# LNI key format validation
# ---------------------------------------------------------------------------

def validate_lni_key(key: str) -> list:
    errors = []
    if key.isdigit():
        return errors

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
# Key-vs-metadata consistency check (FIXED)
# ---------------------------------------------------------------------------

def _extract_surnames(authors_str: str) -> list:
    """Extract surnames from an author string."""
    surnames = []
    if not authors_str:
        return surnames

    for author in re.split(r';\s*| and\s+', authors_str):
        author = author.strip()
        if not author:
            continue
        if re.search(r'et\s+al\.?', author, re.IGNORECASE):
            continue

        # Extract surname (part before comma)
        if ',' in author:
            surname = author.split(',')[0].strip()
        else:
            # No comma - take last word as surname
            parts = author.split()
            if parts:
                surname = parts[-1].strip()
            else:
                continue

        if surname:
            surnames.append(surname)

    return surnames


def _validate_key_vs_metadata(entry: BibEntry) -> None:
    """
    Verify that the initials and year encoded in the LNI citation key are
    consistent with the parsed author(s) and year.
    """
    if entry.key.isdigit():
        entry.key_consistent = True
        return

    match = re.match(r'^([A-Z][A-Za-z]*)(\d{2})([a-z])?$', entry.key)
    if not match:
        entry.key_consistent = None
        return

    key_initials = match.group(1).lower()
    key_year_2d = match.group(2)

    # ── Year check ────────────────────────────────────────────────────────────
    year_ok: Optional[bool] = None
    if entry.year:
        expected_2d = entry.year[-2:]
        year_ok = (expected_2d == key_year_2d)

    # ── Author initials check (FIXED) ───────────────────────────────────────
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
                return s

            normed = [_norm_surname(s) for s in surnames]

            valid_forms = set()
            # Always accept first-2-letters of first surname (LNI "3+ authors" rule)
            valid_forms.add(normed[0][:2])
            valid_forms.add(normed[0][:1])

            if n >= 2:
                # Per-surname initials for however many authors there are
                # (covers 2, 3, 4+ author joint-initial keys)
                valid_forms.add(''.join(s[0] for s in normed))
                # Common real-world variant: first letter of first author +
                # first letter of LAST author (e.g. He + Sun -> "HS"),
                # regardless of how many authors are in between.
                valid_forms.add(normed[0][0] + normed[-1][0])
                valid_forms.add(''.join(s[0] for s in normed[:min(n, 3)]))
            if n >= 4:
                valid_forms.add(''.join(s[0] for s in normed[:4]))

            initials_ok = any(
                key_initials.startswith(form) or form.startswith(key_initials)
                for form in valid_forms
            )

            # If nothing matched but the key's initials are a plausible
            # subsequence of the paper's author initials (common with 4+
            # authors where conventions vary widely), don't hard-fail —
            # treat as unknown rather than a false "mismatch".
            if initials_ok is False and n >= 2:
                author_initials = {s[0] for s in normed}
                if set(key_initials) <= author_initials:
                    initials_ok = None

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
        # NOTE: a key/year or key/initials mismatch is NOT an LNI formatting
        # violation — the key mnemonic has no fixed rule for which year to
        # use (arXiv preprint year vs. official venue year vs. proceedings
        # print year routinely differ by +/-1). Keep it informational via
        # key_consistent/key_mismatch_detail only; do NOT push it into
        # completeness_issues, since that list drives the pass/fail
        # LNI-compliance verdict and a mismatch here is not evidence of a
        # format violation or a fake reference.
        entry.key_mismatch_detail = "; ".join(details)


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _check_completeness(entry: BibEntry) -> None:
    for err in validate_lni_key(entry.key):
        entry.completeness_issues.append(f"Invalid key format: {err}")

    entry_type = entry.entry_type or "unknown"
    lookup_type = "proceedings" if entry_type == "inproceedings" else entry_type
    required = REQUIRED_FIELDS.get(lookup_type, REQUIRED_FIELDS["unknown"])

    # FIX: an entry whose type could not be classified (article/book/
    # proceedings/etc.) is itself an LNI violation — LNI requires the
    # venue field (journal/booktitle/publisher) needed to determine type.
    # Previously "unknown" silently fell back to the loosest required-field
    # list (authors/title/year only), letting malformed entries pass as
    # "Correct Format" whenever those three happened to be present.
    if entry_type == "unknown":
        entry.completeness_issues.append(
            "Entry type could not be determined (no journal, booktitle, "
            "or publisher field found) — LNI requires a classifiable "
            "venue for every entry."
        )

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