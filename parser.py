"""
STEP 2: Bibliography Parser
----------------------------
Parses the bibliography section of an LNI-formatted document.
Extracts citation keys like [AB00], [Ez10], [GI19] and their metadata.

LNI key format:
  - 1 author:       First 2 letters of surname + 2-digit year  → [Ez10]
  - 2 authors:      First letter of each author + year         → [KB14]
  - 3+ authors:     First 2 letters of first author + year     → [De18]
  - No author:      First 2 letters of title + year            → [Di02]
  - Multiple works same year: append a, b, c...                → [Wa14a]

FIXED v7.1:
  - Key validation now correctly handles 2-author keys (e.g., KB14 = Kingma + Ba)
  - Better support for all valid LNI key formats
  - No false "Key mismatch" warnings for correctly formatted keys
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
                            if m.group(1).lower() not in _LNI_MARKER_WORDS
                            # Reject candidates with no digit/underscore — a
                            # hyphenated publisher name (e.g. "Format-Verlag,
                            # Bonn, 1999") matches the same "WORD - Capitalized,"
                            # shape as a genuine unbracketed key (e.g.
                            # "smith2020: Jones, A...") but real keys almost
                            # always carry a year digit or underscore; plain
                            # dictionary words never do. This avoids splitting
                            # one bibliography entry into two at a publisher
                            # name that happens to look key-like.
                            and re.search(r'[0-9_]', m.group(1))]
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

    PARA_BREAK = re.compile(r'(?:\n\s*\n)|(?:[.\)]\s*\n(?=[A-Za-z]))')

    def _entry_own_text_end(bracket_key_end, next_bracket_start):
        m = PARA_BREAK.search(bib_text, bracket_key_end, next_bracket_start)
        return m.start() if m else next_bracket_start

    bracket_own_spans = []
    for i, bp in enumerate(bracket_positions):
        next_bracket_start = bracket_positions[i + 1][0] if i + 1 < len(bracket_positions) else len(bib_text)
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
    url_match = re.search(r'(https?://\S+|https?:\s+//\S+|www\.\S+)', raw)
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

    # PDF extraction can drop the URL scheme from web references. Access-date
    # wording plus a domain is still enough to classify these as websites.
    if (re.search(r'\b(?:Accessed|Abruf|Stand|abgerufen am|besucht am)\b', raw, re.IGNORECASE)
            and re.search(r'\b(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov)\b', raw, re.IGNORECASE)):
        entry.entry_type = "website"
        
        # ── IMPROVED URL EXTRACTION ─────────────────────────────────────────
        # PDFs often break URLs with spaces/newlines. Find the domain, then
        # capture everything after it until clear delimiters (period-space,
        # dash, "Accessed" keyword).
        
        entry.url = None
        
        # Find any domain-like pattern (including subdomains and special TLDs)
        domain_match = re.search(
            r'(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu)',
            raw, re.IGNORECASE)
        
        if domain_match:
            start_pos = domain_match.start()
            # Look for clear delimiters after the domain that signal end of URL
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
        
        # Fallback: if no URL found, do a simple domain search
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
        entry.title = re.sub(
            r'\s+(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov)\S*.*$',
            '', raw, flags=re.IGNORECASE).strip(' .,;:') or None
        return

    # ── Entry type classification ────────────────────────────────────────────
    _has_volume = bool(re.search(r'(?:Jg\.|Vol\.|Nr\.|Band)\s*\d', raw, re.IGNORECASE))
    _has_explicit_conf = bool(re.search(
        r'\bProc\.|\bProceedings\b|\bConference\b|\bWorkshop\b|\bSymposium\b'
        r'|\bTagung\b|\bKonferenz\b|\bHrsg\b|\bEds?\.\B',
        raw, re.IGNORECASE))

    # Volume/journal indicators take highest priority — "In: Nature, Vol. 521"
    # is a journal article, not a proceedings, even though it contains "In:".
    _has_journal_name = bool(re.search(
        r'(?:Nature|Science|Cell|PLOS|PNAS|JMLR|IEEE Trans|ACM Trans|'
        r'Journal of|Transactions on|Letters|Annals of|Reviews? in|'
        r'Zeitschrift für|Informatik Spektrum)',
        raw, re.IGNORECASE,
    ))
    if _PUBLISHER_WORDS.search(raw) and not _has_volume:
        entry.entry_type = "book"
    elif _has_explicit_conf or re.search(r'\(eds?\.\)', raw, re.IGNORECASE):
        entry.entry_type = "proceedings"
    elif _has_volume or _has_journal_name:
        entry.entry_type = "article"
    elif _CONFERENCE_NAMES.search(raw):
        entry.entry_type = "proceedings"
    elif _PROCEEDINGS_WORDS.search(raw) and (_has_explicit_conf or not _has_volume):
        entry.entry_type = "proceedings"
    elif _JOURNAL_WORDS.search(raw) or _JOURNAL_NAME_HINTS.search(raw):
        entry.entry_type = "article"
    elif _PUBLISHER_WORDS.search(raw):
        entry.entry_type = "book"
    else:
        if re.search(r'MIT Press|Springer|Elsevier|Wiley|O\'Reilly|Pearson|Cambridge|Oxford|McGraw|Macmillan', raw, re.IGNORECASE):
            entry.entry_type = "book"
        elif re.search(r'Journal|Transactions|Letters|Magazine|Review|IEEE|ACM', raw, re.IGNORECASE):
            entry.entry_type = "article"
        elif re.search(r'In:\s*[A-Z][a-zA-Z0-9\s]+', raw, re.IGNORECASE):
            entry.entry_type = "proceedings"
        else:
            entry.entry_type = "unknown"
            entry.needs_ai_parsing = True

    # ── Quoted-title format ─────────────────────────────────────────────────
    # Handles APA/MLA-style entries the colon-separated LNI extraction below
    # can't parse, e.g.:
    #   'LeCun, Yann and Bengio, Yoshua and Hinton, Geoffrey (2015). "Deep
    #    Learning" Nature 521 pp. 436-444.'
    #   'Kingma, D.P., Ba, J. "Adam: A Method for Stochastic Optimization"
    #    (2014)'
    # Without this, a colon *inside* the quoted title (like "Adam: A
    # Method...") gets misread as the author/title separator, mangling both.
    quote_m = re.search(r'["\u201c]([^"\u201d]{3,300})["\u201d]', raw)
    if quote_m:
        before = raw[:quote_m.start()].strip()
        after = raw[quote_m.end():].strip()
        entry.title = quote_m.group(1).strip().rstrip('.,;: ')

        # Authors precede the title; strip a trailing "(YYYY)." if the year
        # sits before the quote (e.g. "... Geoffrey (2015). \"Deep...").
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
    author_pattern = None if quote_m else re.match(
        r'^((?:[A-ZÄÖÜ][a-zäöüß\-]+(?:,\s*[A-Za-zÄÖÜäöüß\.\s\-]+)?'
        r'(?:;\s*)?)+):\s*(.*)',
        raw,
    )
    if quote_m:
        pass  # authors/rest already resolved above
    elif author_pattern:
        candidate = author_pattern.group(1).strip()
        if len(candidate) < 180 and ':' not in candidate:
            entry.authors = candidate
            rest = author_pattern.group(2).strip()
    if not quote_m and not entry.authors:
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

    # ── Title extraction ─────────────────────────────────────────────────────
    if quote_m:
        pass  # entry.title already set from the quoted-title branch above
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
        j_match = re.search(
            r'(?:\.\s+In:\s+|\.\s+)([A-Za-zäöüÄÖÜ][^,\.]{2,80}?),\s*(?:Jg\.|Vol\.|Nr\.|Band|No\.)',
            rest, re.IGNORECASE,
        )
        if j_match:
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
    """
    Verify that the initials and year encoded in the LNI citation key are
    consistent with the parsed author(s) and year.
    
    FIXED v7.1: Now correctly handles all LNI key formats:
    - 1 author:  First 2 letters of surname (e.g., Ez10 for Ezhov)
    - 2 authors: First letter of each author (e.g., KB14 for Kingma + Ba)
    - 3+ authors: First 2 letters of first author (e.g., De18 for Devlin)
    """
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

    # ── Year check (±1 tolerance for ArXiv/preprint vs published year) ─────
    year_ok: Optional[bool] = None
    if entry.year:
        try:
            bib_year_int = int(entry.year)
            year_ok = str(bib_year_int)[-2:] == key_year_2d
        except ValueError:
            expected_2d = entry.year[-2:]
            year_ok = (expected_2d == key_year_2d)

    # ── Author initials check (FIXED) ──────────────────────────────────────
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
                # Remove non-alphanumeric
                s = re.sub(r'[^a-z]', '', s)
                return s

            normed = [_norm_surname(s) for s in surnames]

            # ── Build ALL valid LNI key forms ──────────────────────────────
            valid_forms = set()

            # Form 1: First 2 letters of first author's surname
            # Used for: 1 author (Ez10) OR 3+ authors (De18)
            valid_forms.add(normed[0][:2])

            # Form 2: First letter of each author (for 2 authors: KB14)
            if n >= 2:
                valid_forms.add(''.join(s[0] for s in normed[:min(n, 3)]))
                # Also first + last for 3+ authors (common variant)
                if n >= 3:
                    valid_forms.add(normed[0][0] + normed[-1][0])

            # Form 3: First letter of first author only (for 1 author)
            if n == 1:
                valid_forms.add(normed[0][0])

            # Form 4: For 2 authors, first letter of each (this is the correct LNI form!)
            if n == 2:
                # e.g., Kingma + Ba = KB
                valid_forms.add(normed[0][0] + normed[1][0])
                # Also try first two letters of first author + first letter of last
                if len(normed[0]) >= 2:
                    valid_forms.add(normed[0][:2] + normed[1][0])

            # Form 5: For 3 authors, first letter of each (e.g., ABC)
            if n == 3:
                valid_forms.add(normed[0][0] + normed[1][0] + normed[2][0])

            # Form 6: For 4+ authors, first 2 of first author (LNI standard)
            if n >= 4:
                valid_forms.add(normed[0][:2])

            # Form 7: ANY pair of author initials (handles keys like RH15a = Ren+He)
            # LNI does not strictly mandate which authors contribute to the key
            for i in range(n):
                for j in range(n):
                    if i != j and normed[i] and normed[j]:
                        valid_forms.add(normed[i][0] + normed[j][0])

            # Form 8: ANY triple of author initials (3-initial keys)
            if n >= 3:
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            if len({i, j, k}) == 3 and normed[i] and normed[j] and normed[k]:
                                valid_forms.add(normed[i][0] + normed[j][0] + normed[k][0])

            # ── Check if key_initials matches any valid form ──────────────
            initials_ok = any(
                key_initials == form or key_initials.startswith(form) or form.startswith(key_initials)
                for form in valid_forms
                if form and len(form) >= 1
            )

            # ── Extra strictness: key is SHORTER than the number of authors ──
            # LNI rule: for 3 authors, key should include initials of all 3.
            # e.g. LeCun+Bengio+Hinton → LBH15, NOT LB15.
            # If key initials count < author count AND all key chars are valid
            # initials, flag as mismatch (missing author initials).
            missing_initials_for_3plus = False
            if initials_ok and n >= 3 and len(key_initials) < n and len(key_initials) < 3:
                # Only flag if every char in the key IS a valid initial
                # (so we don't accidentally flag 2-char first-author keys like De18)
                all_chars_are_initials = all(
                    any(s[0] == c for s in normed) for c in key_initials
                )
                # But De18 is valid for 3+ authors (first-2-chars of first surname)
                # Distinguish: if key == normed[0][:len(key_initials)], it's a
                # first-surname truncation, which IS valid. Only flag if the
                # key looks like individual initials (each char from diff author).
                is_first_surname_prefix = normed[0].startswith(key_initials)
                if all_chars_are_initials and not is_first_surname_prefix:
                    initials_ok = False  # Missing initials for some authors
                    # This is a DEFINITIVE mismatch (key omits a required
                    # author's initial for a 3+-author entry, e.g. LB15 for
                    # LeCun+Bengio+Hinton). It must NOT be downgraded to
                    # "ambiguous" by the generic 2-char pool check below.
                    missing_initials_for_3plus = True
                    # Rescue: if every key char IS a valid author initial
                    # (e.g. RH for He+Zhang+Ren+Sun), the key is ambiguous
                    # — not a definitive mismatch. Clear the flag so the
                    # pool check below can promote to None (ambiguous).
                    if len(key_initials) >= 2:
                        author_initials_pool = {s[0] for s in normed}
                        if set(key_initials) <= author_initials_pool:
                            missing_initials_for_3plus = False

            # If nothing matched but the key's initials are a plausible
            # subsequence of the paper's author initials, treat as unknown
            if initials_ok is False and n >= 2 and not missing_initials_for_3plus:
                author_initials = {s[0] for s in normed}
                if set(key_initials) <= author_initials:
                    # Only promote to None (ambiguous) if key length equals
                    # author count — otherwise keep as False (missing initials)
                    if len(key_initials) >= n or len(key_initials) <= 2:
                        initials_ok = None

    # ── Combine ──────────────────────────────────────────────────────────────
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
            # For 2-author papers with correct key like KB14, don't flag as mismatch
            if len(surnames) == 2 and key_initials == (normed[0][0] + normed[1][0]):
                entry.key_consistent = True
                entry.key_mismatch_detail = None
                return
            
            # For 1-author papers with correct key like Ez10
            if len(surnames) == 1 and key_initials == normed[0][:2]:
                entry.key_consistent = True
                entry.key_mismatch_detail = None
                return

            # For 2-char keys where both chars match author initials in the pool,
            # the bibliography may have truncated the author list (e.g. only showing
            # He+Zhang when the real paper is He+Zhang+Ren+Sun). Any pair of initials
            # from the known authors could be valid — mark as ambiguous, not wrong.
            if len(key_initials) == 2 and not missing_initials_for_3plus:
                author_initials_pool = {s[0] for s in normed}
                if set(key_initials) <= author_initials_pool:
                    # Ambiguous — could be valid with full author list
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

    for err in validate_lni_key(entry.key):
        entry.completeness_issues.append(f"Invalid key format: {err}")

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

    for field_name in required:
        if field_name == "pages" and (
            entry_type in ("proceedings", "inproceedings")
                or (entry_type == "article" and entry.raw_text.find("In:") >= 0
                    and (entry.volume or entry.number))):
            continue
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
            if span > 100:
                entry.completeness_issues.append(
                    f"Page range {lo}–{hi} spans {span} pages — "
                    "unusually large for a single article."
                )
    
    # Check for suspicious volume numbers (e.g., 666, 777, 888, 999, 111, 222, 333, 444, 555)
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