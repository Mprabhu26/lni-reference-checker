"""
STEP 3: Citation Cross-Checker + Reference Verifier
-----------------------------------------------------
Verification pipeline (in order of reliability):
  1. DOI direct lookup  → CrossRef (100% definitive if DOI resolves)
  2. CrossRef title+author search
  3. Semantic Scholar
  4. OpenAlex          (fully open, 100k/day free)
  5. arXiv             (critical for CS/AI papers)
  6. Open Library      (ISBN-based book lookup — new in v4)
  7. Google Scholar    (scrape fallback, rate-limited)
  8. Web fingerprint   (DuckDuckGo, last resort)

Title similarity uses rapidfuzz.token_set_ratio (falls back to difflib).
All API calls run concurrently via ThreadPoolExecutor.

FIXES vs v3:
  - _CACHE: normalised (title, first-surname) → VerificationResult cache so
    batch mode doesn't re-query the same popular references 30 times.
  - _query_open_library(): new ISBN/title lookup via Open Library API (free,
    no key). Books are the weakest verification area in v3.
  - detect_self_citations(): now scans the whole document header zone
    (first 2 000 chars) and normalises umlauts before matching so
    "Müller" in the paper == "Mueller" or "Müller" in the bib.
  - extract_citations_from_body(): also catches numeric-style refs [1], [12]
    and reports them separately so the professor knows a mixed format was used.
  - check_lni_macros(): extended to catch en-dash used as hyphen, missing
    non-breaking space before citations, and footnote citations.
  - compute_score(): now uses AI fake_count (passed in) instead of raw
    API not_found count, which was penalising legitimate niche papers.
"""

import re
import time
import hashlib
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from parser import BibEntry


# ---------------------------------------------------------------------------
# Thread-safe in-memory cache (keyed by normalised title + first surname)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, "VerificationResult"] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(entry: BibEntry) -> str:
    title = (entry.title or "").lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    first_surname = ""
    if entry.authors:
        first_surname = entry.authors.split(';')[0].split(',')[0].strip().lower()
    raw = f"{title}|{first_surname}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(entry: BibEntry) -> Optional["VerificationResult"]:
    if not entry.title:
        return None
    key = _cache_key(entry)
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _put_cache(entry: BibEntry, result: "VerificationResult") -> None:
    if not entry.title:
        return
    key = _cache_key(entry)
    with _CACHE_LOCK:
        _CACHE[key] = result


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------

def _title_similarity(title1: str, title2: str) -> float:
    if not title1 or not title2:
        return 0.0

    def _normalize(t: str) -> str:
        t = t.lower()
        # Normalise common umlaut/accent variants
        for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss')]:
            t = t.replace(a, b)
        t = re.sub(r'[^\w\s]', ' ', t)
        stopwords = {
            'the','a','an','in','of','for','on','and','to','with',
            'der','die','das','und','fur','von','mit','im','an','zu',
            'eine','ein','des','dem',
        }
        return ' '.join(w for w in t.split() if w not in stopwords)

    t1, t2 = _normalize(title1), _normalize(title2)
    if not t1 or not t2:
        return 0.0

    try:
        from rapidfuzz.fuzz import token_set_ratio
        return token_set_ratio(t1, t2) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


# ---------------------------------------------------------------------------
# In-text citation extraction
# ---------------------------------------------------------------------------

def extract_citations_from_body(body_text: str) -> set:
    """
    Extract LNI-style citation keys [Ez10], [ABC01], etc.
    Also detects numeric citations [1], [12] and records them separately
    (stored in the returned set with a '__numeric__' prefix so the caller
    can warn the professor about mixed citation styles).
    """
    raw_matches = re.findall(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]',
        body_text,
    )
    keys = set()
    for match in raw_matches:
        for key in re.split(r',\s*', match):
            key = key.strip()
            if re.match(r'^[A-Za-z]{2,6}\d{2}[a-z]?$', key):
                keys.add(key)

    # Detect numeric-only citations and flag them
    numeric = re.findall(r'\[(\d{1,3})\]', body_text)
    if numeric:
        keys.add('__numeric_citations__')

    return keys


def extract_citation_contexts(body_text: str) -> dict:
    """Return up to 2 short context snippets for each cited key."""
    contexts = {}
    for match in re.finditer(
        r'([^.]{0,80})\[([A-Za-z]{2,6}\d{2}[a-z]?'
        r'(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]([^.]{0,80})',
        body_text,
    ):
        snippet = (
            match.group(1) + '[' + match.group(2) + ']' + match.group(3)
        ).strip()
        for key in re.split(r',\s*', match.group(2)):
            key = key.strip()
            if key:
                contexts.setdefault(key, [])
                if len(contexts[key]) < 2:
                    contexts[key].append(snippet)
    return contexts


def _normalise_name(name: str) -> str:
    """Fold umlauts and lower-case for fuzzy author matching."""
    name = name.lower()
    for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss')]:
        name = name.replace(a, b)
    return name


def detect_self_citations(bib_entries: dict, body_text: str) -> list:
    """
    Flag bib entries whose author surname overlaps with the paper's own authors.

    Improvement over v3:
    - Scans first 2 000 chars of body (covers title + author line of most papers)
    - Normalises umlauts so "Müller" in paper == "Müller" in bib
    - Requires surname length > 3 to avoid short noise tokens
    """
    # Collect candidate surnames from top of document
    header = body_text[:2000]
    raw_candidates = re.findall(r'\b([A-ZÄÖÜ][a-zäöüß]{3,})\b', header)
    candidates_normalised = {_normalise_name(c) for c in raw_candidates}

    self_cites = []
    for key, entry in bib_entries.items():
        if not entry.authors:
            continue
        for name in re.split(r'[;,]', entry.authors):
            name = name.strip()
            if len(name) <= 3:
                continue
            if _normalise_name(name) in candidates_normalised:
                self_cites.append({
                    "key":           key,
                    "title":         entry.title or "",
                    "matched_author": name,
                })
                break
    return self_cites


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

@dataclass
class CrossCheckResult:
    cited_not_in_bib: list = field(default_factory=list)
    in_bib_not_cited: list = field(default_factory=list)
    correctly_used: list   = field(default_factory=list)


def cross_check(bib_entries: dict, cited_keys: set) -> CrossCheckResult:
    # Strip synthetic flags before comparing
    real_cited = {k for k in cited_keys if not k.startswith('__')}
    bib_keys   = set(bib_entries.keys())
    r = CrossCheckResult()
    r.cited_not_in_bib = sorted(real_cited - bib_keys)
    r.in_bib_not_cited  = sorted(bib_keys - real_cited)
    r.correctly_used    = sorted(real_cited & bib_keys)
    return r


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str              # verified | not_found | partial_match | error
    confidence: float
    matched_title: Optional[str] = None
    doi:           Optional[str] = None
    open_access_url: Optional[str] = None
    note:          Optional[str] = None
    sources_checked: list = field(default_factory=list)
    web_evidence:  Optional[str] = None


# ---------------------------------------------------------------------------
# Individual source queries
# ---------------------------------------------------------------------------

def _query_crossref(entry: BibEntry) -> VerificationResult:
    try:
        headers = {"User-Agent": "LNI-Checker/4.0 (mailto:lni@checker.de)"}

        # Direct DOI lookup — 100% definitive
        if entry.doi:
            resp = requests.get(
                f"https://api.crossref.org/works/{entry.doi}",
                timeout=10, headers=headers,
            )
            if resp.status_code == 200:
                work        = resp.json().get("message", {})
                found_title = (work.get("title") or [""])[0]
                similarity  = _title_similarity(entry.title or "", found_title)
                oa_url      = _check_unpaywall(entry.doi)
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=max(similarity, 0.95),
                    matched_title=found_title, doi=entry.doi,
                    open_access_url=oa_url,
                    note="DOI verified via CrossRef",
                    sources_checked=["CrossRef (DOI)"],
                )

        # Title + author search
        params: dict = {
            "query.title": entry.title,
            "rows": 3,
            "select": "DOI,title,author,published-print",
        }
        if entry.authors:
            params["query.author"] = (
                entry.authors.split(';')[0].split(',')[0].strip()
            )

        resp = requests.get(
            "https://api.crossref.org/works",
            params=params, timeout=10, headers=headers,
        )
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0,
                note="No results from CrossRef",
                sources_checked=["CrossRef"],
            )

        best        = items[0]
        found_title = (best.get("title") or [""])[0]
        doi         = best.get("DOI", "")
        similarity  = _title_similarity(entry.title or "", found_title)

        if similarity >= 0.75:
            oa_url = _check_unpaywall(doi) if doi else None
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=similarity,
                matched_title=found_title, doi=doi,
                open_access_url=oa_url,
                note="Found via CrossRef",
                sources_checked=["CrossRef"],
            )
        elif similarity >= 0.4:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="partial_match", confidence=similarity,
                matched_title=found_title, doi=doi,
                note="Partial match on CrossRef — verify manually",
                sources_checked=["CrossRef"],
            )
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="not_found", confidence=similarity,
            matched_title=found_title,
            note="Title did not match CrossRef results",
            sources_checked=["CrossRef"],
        )

    except requests.exceptions.Timeout:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note="CrossRef timeout", sources_checked=["CrossRef"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"CrossRef error: {e}", sources_checked=["CrossRef"],
        )


def _query_semantic_scholar(entry: BibEntry) -> VerificationResult:
    try:
        query = entry.title or ""
        if entry.authors:
            query += " " + entry.authors.split(';')[0].split(',')[0].strip()

        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query, "limit": 3,
                "fields": "title,authors,year,openAccessPdf,externalIds",
            },
            timeout=10,
            headers={"User-Agent": "LNI-Checker/4.0"},
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])
        if not papers:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0,
                note="Not found in Semantic Scholar",
                sources_checked=["Semantic Scholar"],
            )

        best        = papers[0]
        found_title = best.get("title", "")
        similarity  = _title_similarity(entry.title or "", found_title)
        oa          = best.get("openAccessPdf")
        oa_url      = oa.get("url") if oa else None
        ext_ids     = best.get("externalIds", {})
        doi         = ext_ids.get("DOI") or ext_ids.get("doi")

        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.4  else
            "not_found"
        )
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status=status, confidence=similarity,
            matched_title=found_title, doi=doi,
            open_access_url=oa_url,
            note="Found via Semantic Scholar",
            sources_checked=["Semantic Scholar"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"Semantic Scholar error: {e}",
            sources_checked=["Semantic Scholar"],
        )


def _query_openalex(entry: BibEntry) -> VerificationResult:
    try:
        search_str = entry.title or ""
        if entry.authors:
            first = entry.authors.split(';')[0].split(',')[0].strip()
            search_str = f"{search_str} {first}"

        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": search_str, "per_page": 3},
            timeout=10,
            headers={"User-Agent": "LNI-Checker/4.0 (mailto:lni@checker.de)"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0,
                note="Not found in OpenAlex",
                sources_checked=["OpenAlex"],
            )

        best        = results[0]
        found_title = best.get("title", "")
        similarity  = _title_similarity(entry.title or "", found_title)
        doi_raw     = best.get("doi", "") or ""
        doi         = doi_raw.replace("https://doi.org/", "") or None
        oa_url      = best.get("open_access", {}).get("oa_url")

        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.4  else
            "not_found"
        )
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status=status, confidence=similarity,
            matched_title=found_title, doi=doi,
            open_access_url=oa_url,
            note="Found via OpenAlex",
            sources_checked=["OpenAlex"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"OpenAlex error: {e}",
            sources_checked=["OpenAlex"],
        )


def _query_arxiv(entry: BibEntry) -> VerificationResult:
    """arXiv API — critical for CS/AI papers. Free, no key, 3 req/s polite."""
    try:
        import urllib.parse
        title = entry.title or ""
        query = f'ti:"{urllib.parse.quote(title)}"'
        if entry.authors:
            first_surname = entry.authors.split(';')[0].split(',')[0].strip()
            if first_surname:
                query += f' AND au:{urllib.parse.quote(first_surname)}'

        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": 3, "sortBy": "relevance"},
            timeout=10,
            headers={"User-Agent": "LNI-Checker/4.0"},
        )
        if resp.status_code != 200:
            return VerificationResult(
                key=entry.key, title=title,
                status="error", confidence=0.0,
                note=f"arXiv HTTP {resp.status_code}",
                sources_checked=["arXiv"],
            )

        entries_xml = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        if not entries_xml:
            return VerificationResult(
                key=entry.key, title=title,
                status="not_found", confidence=0.0,
                note="Not found on arXiv",
                sources_checked=["arXiv"],
            )

        title_match = re.search(r'<title>(.*?)</title>', entries_xml[0], re.DOTALL)
        found_title = re.sub(r'\s+', ' ', title_match.group(1)).strip() if title_match else ""

        pdf_match = re.search(
            r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', entries_xml[0]
        )
        oa_url = pdf_match.group(1) if pdf_match else None

        doi_match = re.search(
            r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', entries_xml[0]
        )
        doi = doi_match.group(1).strip() if doi_match else None

        similarity = _title_similarity(title, found_title)
        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.4  else
            "not_found"
        )
        return VerificationResult(
            key=entry.key, title=title,
            status=status, confidence=similarity,
            matched_title=found_title, doi=doi,
            open_access_url=oa_url,
            note="Found on arXiv",
            sources_checked=["arXiv"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"arXiv error: {e}",
            sources_checked=["arXiv"],
        )


def _query_open_library(entry: BibEntry) -> VerificationResult:
    """
    Open Library API (openlibrary.org) — free, no key.
    Particularly strong for book verification:
      - ISBN lookup if available (100% definitive)
      - Title search fallback
    This is new in v4 and fills the gap left by CrossRef/Semantic Scholar
    for books that are not academic papers.
    """
    try:
        # ── ISBN direct lookup ─────────────────────────────────────────────────
        if entry.isbn:
            isbn_clean = re.sub(r'[\s-]', '', entry.isbn)
            resp = requests.get(
                "https://openlibrary.org/api/books",
                params={
                    "bibkeys": f"ISBN:{isbn_clean}",
                    "format": "json",
                    "jscmd": "data",
                },
                timeout=10,
                headers={"User-Agent": "LNI-Checker/4.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                record = data.get(f"ISBN:{isbn_clean}")
                if record:
                    found_title = record.get("title", "")
                    similarity  = _title_similarity(entry.title or "", found_title)
                    oa_url      = record.get("url")
                    return VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status="verified",
                        confidence=max(similarity, 0.95),
                        matched_title=found_title,
                        open_access_url=oa_url,
                        note=f"ISBN {isbn_clean} verified via Open Library",
                        sources_checked=["Open Library (ISBN)"],
                    )

        # ── Title search fallback ─────────────────────────────────────────────
        if not entry.title:
            return VerificationResult(
                key=entry.key, title="",
                status="error", confidence=0.0,
                note="No title or ISBN for Open Library lookup",
                sources_checked=["Open Library"],
            )

        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"title": entry.title, "limit": 3},
            timeout=10,
            headers={"User-Agent": "LNI-Checker/4.0"},
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if not docs:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="not_found", confidence=0.0,
                note="Not found in Open Library",
                sources_checked=["Open Library"],
            )

        best        = docs[0]
        found_title = best.get("title", "")
        similarity  = _title_similarity(entry.title, found_title)
        ol_key      = best.get("key", "")
        oa_url      = f"https://openlibrary.org{ol_key}" if ol_key else None

        # Cross-check author if available
        if entry.authors and similarity >= 0.5:
            ol_authors = best.get("author_name", [])
            if ol_authors:
                first_surname = entry.authors.split(';')[0].split(',')[0].strip().lower()
                author_match = any(
                    first_surname in a.lower() for a in ol_authors
                )
                if not author_match:
                    similarity = max(0.0, similarity - 0.15)

        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.4  else
            "not_found"
        )
        return VerificationResult(
            key=entry.key, title=entry.title,
            status=status, confidence=similarity,
            matched_title=found_title,
            open_access_url=oa_url,
            note="Found via Open Library",
            sources_checked=["Open Library"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"Open Library error: {e}",
            sources_checked=["Open Library"],
        )


def _query_google_scholar(entry: BibEntry) -> VerificationResult:
    """Google Scholar scrape fallback. Returns quickly if rate-limited."""
    try:
        query = entry.title or ""
        if entry.authors:
            query += " " + entry.authors.split(';')[0].split(',')[0].strip()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(
            "https://scholar.google.com/scholar",
            params={"q": query, "hl": "en", "num": 3},
            headers=headers, timeout=12,
        )

        if resp.status_code == 429:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="error", confidence=0.0,
                note="Google Scholar rate-limited (429)",
                sources_checked=["Google Scholar"],
            )
        if resp.status_code != 200:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="error", confidence=0.0,
                note=f"Google Scholar HTTP {resp.status_code}",
                sources_checked=["Google Scholar"],
            )

        found_titles = re.findall(
            r'class="gs_rt"[^>]*>(?:<[^>]+>)*([^<]+)', resp.text
        )
        found_titles = [
            re.sub(r'<[^>]+>', '', t).strip()
            for t in found_titles if t.strip()
        ]

        if not found_titles:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0,
                note="No results on Google Scholar",
                sources_checked=["Google Scholar"],
            )

        best_title = found_titles[0]
        similarity = _title_similarity(entry.title or "", best_title)
        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.40 else
            "not_found"
        )
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status=status, confidence=similarity,
            matched_title=best_title,
            note="Found via Google Scholar",
            sources_checked=["Google Scholar"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"Google Scholar error: {e}",
            sources_checked=["Google Scholar"],
        )


def _query_duckduckgo_web(entry: BibEntry) -> VerificationResult:
    """Last-resort web fingerprint via DuckDuckGo HTML (no API key needed)."""
    try:
        title = entry.title or ""
        author_hint = ""
        if entry.authors:
            author_hint = entry.authors.split(';')[0].split(',')[0].strip()

        query = f'"{title}"'
        if author_hint:
            query += f" {author_hint}"
        if entry.year:
            query += f" {entry.year}"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LNI-Checker/4.0)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers, timeout=12,
        )

        if resp.status_code != 200:
            return VerificationResult(
                key=entry.key, title=title,
                status="error", confidence=0.0,
                note=f"DuckDuckGo HTTP {resp.status_code}",
                sources_checked=["Web (DDG)"],
            )

        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL
        )
        snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:3]]

        title_lower   = title.lower()
        combined_text = " ".join(snippets).lower()
        title_words   = {w for w in title_lower.split() if len(w) > 3}
        if not title_words:
            return VerificationResult(
                key=entry.key, title=title,
                status="not_found", confidence=0.0,
                note="No web results",
                sources_checked=["Web (DDG)"],
            )

        matching = sum(1 for w in title_words if w in combined_text)
        coverage = matching / len(title_words)

        if coverage >= 0.7:
            evidence = snippets[0][:150] if snippets else ""
            return VerificationResult(
                key=entry.key, title=title,
                status="verified" if coverage >= 0.85 else "partial_match",
                confidence=round(coverage, 2),
                note="Found via web search (DuckDuckGo)",
                web_evidence=evidence,
                sources_checked=["Web (DDG)"],
            )

        return VerificationResult(
            key=entry.key, title=title,
            status="not_found", confidence=round(coverage, 2),
            note=f"Limited web evidence ({int(coverage*100)}% keyword match)",
            sources_checked=["Web (DDG)"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note=f"Web search error: {e}",
            sources_checked=["Web (DDG)"],
        )


def _verify_website(entry: BibEntry) -> VerificationResult:
    url = entry.url
    if not url:
        return VerificationResult(
            key=entry.key, title=entry.title or "(website)",
            status="error", confidence=0.0,
            note="No URL found", sources_checked=["URL check"],
        )
    try:
        if not url.startswith("http"):
            url = "https://" + url
        resp = requests.head(url, timeout=8, allow_redirects=True)
        if resp.status_code < 400:
            return VerificationResult(
                key=entry.key, title=entry.title or url,
                status="verified", confidence=1.0,
                open_access_url=url,
                note=f"URL reachable (HTTP {resp.status_code})",
                sources_checked=["URL check"],
            )
        return VerificationResult(
            key=entry.key, title=entry.title or url,
            status="not_found", confidence=0.0,
            note=f"URL returned HTTP {resp.status_code}",
            sources_checked=["URL check"],
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or url,
            status="error", confidence=0.0,
            note=f"URL check failed: {e}",
            sources_checked=["URL check"],
        )


def _check_unpaywall(doi: str) -> Optional[str]:
    import os
    email = os.environ.get("UNPAYWALL_EMAIL", "lni-checker@uni-project.de")
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={email}",
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("is_oa"):
                loc = data.get("best_oa_location")
                if loc:
                    return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main verifier — parallel execution with caching
# ---------------------------------------------------------------------------

def verify_reference(entry: BibEntry) -> VerificationResult:
    """
    Try all verification sources in parallel.
    Results are cached by (normalised_title, first_surname) to avoid
    redundant API calls when the same paper is cited in multiple submissions.
    """
    if not entry.title and not entry.doi and not entry.isbn:
        return VerificationResult(
            key=entry.key, title="(no title parsed)",
            status="error", confidence=0.0,
            note="Could not extract title — check parsing",
            sources_checked=[],
        )

    if entry.entry_type == "website":
        return _verify_website(entry)

    # Check cache first
    cached = _get_cached(entry)
    if cached:
        # Return a shallow copy with the correct key (different entries may
        # share the same paper)
        import copy
        result = copy.copy(cached)
        result.key = entry.key
        result.note = (result.note or "") + " [cached]"
        return result

    # Choose source functions: for books prefer Open Library over DDG
    is_book = entry.entry_type in ("book", "unknown") and (
        entry.isbn or entry.publisher
    )
    source_fns = [
        _query_crossref,
        _query_semantic_scholar,
        _query_openalex,
        _query_arxiv,
        _query_open_library,   # new in v4
        _query_google_scholar,
        _query_duckduckgo_web,
    ]

    results: List[VerificationResult] = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(fn, entry): fn.__name__ for fn in source_fns}
        for future in as_completed(futures, timeout=25):
            try:
                r = future.result()
                results.append(r)
            except Exception:
                pass

    if not results:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="error", confidence=0.0,
            note="All sources failed",
            sources_checked=[],
        )

    # Aggregate: prefer verified > partial_match > not_found > error
    priority = {"verified": 3, "partial_match": 2, "not_found": 1, "error": 0}
    results.sort(
        key=lambda r: (priority.get(r.status, 0), r.confidence), reverse=True
    )
    best = results[0]

    # Merge sources_checked from all results
    all_sources = []
    for r in results:
        all_sources.extend(r.sources_checked)
    best.sources_checked = all_sources

    # Harvest web_evidence from DDG if best doesn't have it
    for r in results:
        if r.web_evidence and not best.web_evidence:
            best.web_evidence = r.web_evidence

    # Confidence boost: multiple independent sources agree
    verified_count = sum(1 for r in results if r.status == "verified")
    if verified_count >= 2 and best.status == "verified":
        best.confidence = min(best.confidence + 0.05 * (verified_count - 1), 1.0)
        best.note = f"Confirmed by {verified_count} independent sources"

    _put_cache(entry, best)
    return best


def verify_all_references(bib_entries: dict, delay: float = 0.0) -> list:
    """Verify all entries concurrently. delay param kept for API compatibility."""
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_key = {
            executor.submit(verify_reference, entry): key
            for key, entry in bib_entries.items()
        }
        for future in as_completed(future_to_key, timeout=90):
            try:
                results.append(future.result())
            except Exception as e:
                key = future_to_key[future]
                results.append(VerificationResult(
                    key=key, title="",
                    status="error", confidence=0.0,
                    note=f"Verification crashed: {e}",
                    sources_checked=[],
                ))

    key_order = list(bib_entries.keys())
    results.sort(
        key=lambda r: key_order.index(r.key) if r.key in key_order else 999
    )
    return results


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(bib_entries: dict, threshold: float = 0.85) -> list:
    entries = list(bib_entries.values())
    dupes   = []
    seen    = set()
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if not a.title or not b.title:
                continue
            pair = tuple(sorted([a.key, b.key]))
            if pair in seen:
                continue
            seen.add(pair)
            score = _title_similarity(a.title, b.title)
            if score >= threshold:
                dupes.append({
                    "key_a":      a.key,
                    "key_b":      b.key,
                    "title_a":    a.title,
                    "title_b":    b.title,
                    "similarity": round(score, 2),
                    "type":       "exact" if score >= 0.97 else "near-duplicate",
                })
    return dupes


# ---------------------------------------------------------------------------
# LNI style checks
# ---------------------------------------------------------------------------

def check_lni_macros(body_text: str) -> list:
    suggestions = []

    macro_rules = [
        (r'\be\.g\.', r'\eg',   r"Use LNI macro '\eg' instead of 'e.g.'"),
        (r'\bi\.e\.', r'\ie',   r"Use LNI macro '\ie' instead of 'i.e.'"),
        (r'\bcf\.',   r'\cf',   r"Use LNI macro '\cf' instead of 'cf.'"),
        (r'\bet al\.', r'\etal', r"Use LNI macro '\etal' instead of 'et al.'"),
    ]
    for pattern, macro, message in macro_rules:
        matches = re.findall(pattern, body_text, re.IGNORECASE)
        if matches:
            suggestions.append({
                "type":    "Macro",
                "message": message,
                "count":   len(matches),
            })

    # ALL-CAPS headings (LNI uses sentence / title case)
    all_caps = re.findall(r'\n([A-Z]{4,}(?:\s+[A-Z]{2,})*)\n', body_text)
    if all_caps:
        suggestions.append({
            "type":    "Heading",
            "message": (
                f"Found {len(all_caps)} ALL-CAPS heading(s) — "
                "LNI uses sentence case for headings."
            ),
            "count":   len(all_caps),
        })

    # Manual \textbf{} where LNI semantic macros should be used
    textbf_count = len(re.findall(r'\\textbf\{[^}]{1,20}\}', body_text))
    if textbf_count:
        suggestions.append({
            "type":    "Emphasis",
            "message": r"Manual \textbf{} detected — prefer LNI semantic macros.",
            "count":   textbf_count,
        })

    # Straight quotes instead of LaTeX curly quotes
    straight_count = len(re.findall(r'(?<!`)"[^"]{1,60}"', body_text))
    if straight_count:
        suggestions.append({
            "type":    "Quotes",
            "message": r'Straight quotes (") detected — use ``...'"''"' or \\enquote{}.',
            "count":   straight_count,
        })

    # En-dash used as word-hyphen (e.g. "state–of–the–art")
    endash_count = len(re.findall(r'\w–\w', body_text))
    if endash_count:
        suggestions.append({
            "type":    "Dash",
            "message": (
                "En-dash (–) used as word hyphen — use a regular hyphen (-) "
                "for compound words; reserve -- for page ranges."
            ),
            "count":   endash_count,
        })

    # Missing non-breaking space before citation: "text[ABC01]" vs "text [ABC01]"
    nbs_count = len(re.findall(r'\w\[(?:[A-Za-z]{2,6}\d{2})', body_text))
    if nbs_count:
        suggestions.append({
            "type":    "Spacing",
            "message": (
                "Citation directly follows a word without a space — "
                "LNI style requires 'text~[Key]' (non-breaking space)."
            ),
            "count":   nbs_count,
        })

    # Numeric-style citations mixed in ([1], [2] alongside [Ez10])
    has_lni     = bool(re.search(r'\[[A-Za-z]{2,6}\d{2}\]', body_text))
    has_numeric = bool(re.search(r'\[\d{1,3}\]', body_text))
    if has_lni and has_numeric:
        suggestions.append({
            "type":    "CitationStyle",
            "message": (
                "Mixed citation styles detected — both LNI-style [Ez10] "
                "and numeric [1] citations found. LNI requires [Author+Year] only."
            ),
            "count":   len(re.findall(r'\[\d{1,3}\]', body_text)),
        })

    return suggestions


# ---------------------------------------------------------------------------
# Scoring — uses AI fake_count, not raw API not_found count
# ---------------------------------------------------------------------------

def compute_score(
    bib_list: list,
    xcheck: CrossCheckResult,
    verification_results: list,
    style_suggestions: list,
    duplicates: list,
    ai_fake_count: int = 0,
) -> dict:
    """
    Compute a deterministic score.

    ai_fake_count: the number of entries the AI judged FAKE (passed in from
    ai_checker). We use this instead of counting raw API 'not_found' results,
    which penalised legitimate niche/old papers that simply aren't indexed.
    """
    score    = 100
    penalties = []

    missing = len(xcheck.cited_not_in_bib)
    p = min(missing * 10, 30)
    if p:
        penalties.append({"category": "Missing from bibliography",
                           "count": missing, "deduction": p})
    score -= p

    orphaned = len(xcheck.in_bib_not_cited)
    p = min(orphaned * 5, 20)
    if p:
        penalties.append({"category": "Cited nowhere in text",
                           "count": orphaned, "deduction": p})
    score -= p

    incomplete = sum(1 for e in bib_list if e.completeness_issues)
    p = min(incomplete * 5, 20)
    if p:
        penalties.append({"category": "Incomplete entries",
                           "count": incomplete, "deduction": p})
    score -= p

    # Use AI fake count (more accurate than raw API status)
    p = min(ai_fake_count * 10, 30)
    if p:
        penalties.append({"category": "Likely fabricated references (AI verdict)",
                           "count": ai_fake_count, "deduction": p})
    score -= p

    p = min(len(duplicates) * 5, 10)
    if p:
        penalties.append({"category": "Duplicate entries",
                           "count": len(duplicates), "deduction": p})
    score -= p

    p = min(len(style_suggestions) * 2, 6)
    if p:
        penalties.append({"category": "LNI style violations",
                           "count": len(style_suggestions), "deduction": p})
    score -= p

    score = max(score, 0)
    grade = (
        "A" if score >= 90 else
        "B" if score >= 75 else
        "C" if score >= 60 else
        "D" if score >= 45 else
        "F"
    )
    return {"score": score, "grade": grade, "penalties": penalties, "max_score": 100}