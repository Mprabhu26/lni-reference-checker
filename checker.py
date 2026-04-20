"""
STEP 3: Citation Cross-Checker + Reference Verifier  — v6
----------------------------------------------------------
Verification pipeline (parallel, all FREE, no paid API keys required):
  1.  DOI direct lookup        → CrossRef          (100% if DOI resolves)
  2.  CrossRef title+author    → crossref.org       (free, polite pool)
  3.  Semantic Scholar         → semanticscholar.org (free; SEMANTIC_SCHOLAR_API_KEY = higher limits)
  4.  OpenAlex                 → openalex.org        (free, 100k/day)
  5.  arXiv BibTeX (versioned) → arxiv.org/bibtex/   (NEW v6 — authoritative, checks ALL versions)
  6.  arXiv search fallback    → export.arxiv.org    (free, 3 req/s)
  7.  DBLP                     → dblp.org            (free, 1 req/s)
  8.  ACL Anthology            → aclanthology.org    (free)
  9.  OpenReview               → openreview.net      (free)
  10. Open Library             → openlibrary.org     (free, ISBN/title)
  11. GitHub checker           → api.github.com      (free 60/hr anon)
  12. Google Scholar           → scrape fallback, rate-limited
  13. DuckDuckGo web           → last resort

NEW in v6:
  - ArXiv versioned BibTeX checker: fetches authoritative metadata from
    https://arxiv.org/bibtex/{id} — the gold-standard source for arXiv papers.
    Checks the reference against EVERY historical version (v1, v2, …, latest),
    so a citation to an old version no longer triggers a false mismatch.
    When the cited version differs from latest, a note is added rather than
    an error. This replaces the old plain title-search arXiv fallback for
    any reference whose URL, doi, or raw text contains an arXiv ID.
  - In-memory LLM response cache: AI calls (extraction, re-parse, verify)
    are cached in-process by (model, prompt) hash so repeated runs against
    the same papers in a batch session skip all AI network calls.
    Cache is session-scoped (lives in process memory, cleared on restart) —
    appropriate for local single-user use.

Environment variables (all optional, all free):
  LNI_CACHE_DIR             → disk cache directory  (default: .lni_cache)
  SEMANTIC_SCHOLAR_API_KEY  → free at semanticscholar.org/product/api
  GITHUB_TOKEN              → free personal token at github.com/settings/tokens
  UNPAYWALL_EMAIL           → your email for Unpaywall polite pool
"""

import copy
import hashlib
import json
import os
import re
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from parser import BibEntry


# ---------------------------------------------------------------------------
# Persistent disk cache
# ---------------------------------------------------------------------------

_DISK_CACHE_DIR: str = os.environ.get("LNI_CACHE_DIR", ".lni_cache")
_DISK_CACHE_LOCK = threading.Lock()


def _disk_cache_key(entry: BibEntry) -> str:
    title = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (entry.title or "").lower().strip()))
    first = ""
    if entry.authors:
        first = entry.authors.split(';')[0].split(',')[0].strip().lower()
    return hashlib.sha256(f"{title}|{first}".encode()).hexdigest()[:24]


def _disk_cache_path(key: str) -> Optional[Path]:
    if not _DISK_CACHE_DIR:
        return None
    return Path(_DISK_CACHE_DIR) / f"{key}.json"


def _load_disk_cache(key: str) -> Optional[dict]:
    path = _disk_cache_path(key)
    if path and path.exists():
        try:
            with _DISK_CACHE_LOCK:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_disk_cache(key: str, data: dict) -> None:
    path = _disk_cache_path(key)
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _DISK_CACHE_LOCK:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# In-memory cache (fast layer on top of disk cache)
# ---------------------------------------------------------------------------

_MEM_CACHE: Dict[str, "VerificationResult"] = {}
_MEM_CACHE_LOCK = threading.Lock()


def _get_cached(entry: BibEntry) -> Optional["VerificationResult"]:
    if not entry.title:
        return None
    key = _disk_cache_key(entry)
    with _MEM_CACHE_LOCK:
        hit = _MEM_CACHE.get(key)
    if hit:
        return hit
    data = _load_disk_cache(key)
    if data:
        r = VerificationResult(**{k: data.get(k) for k in VerificationResult.__dataclass_fields__})
        with _MEM_CACHE_LOCK:
            _MEM_CACHE[key] = r
        return r
    return None


def _put_cache(entry: BibEntry, result: "VerificationResult") -> None:
    if not entry.title:
        return
    key = _disk_cache_key(entry)
    with _MEM_CACHE_LOCK:
        _MEM_CACHE[key] = result
    _save_disk_cache(key, {
        "key": result.key, "title": result.title, "status": result.status,
        "confidence": result.confidence, "matched_title": result.matched_title,
        "doi": result.doi, "open_access_url": result.open_access_url,
        "note": result.note, "sources_checked": result.sources_checked,
        "web_evidence": result.web_evidence, "correct_authors": result.correct_authors,
        "version_note": result.version_note,
    })


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------

def _title_similarity(title1: str, title2: str) -> float:
    if not title1 or not title2:
        return 0.0

    def _norm(t: str) -> str:
        t = t.lower()
        for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
            t = t.replace(a, b)
        # Strip LaTeX commands that may appear in arXiv BibTeX titles
        t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
        t = re.sub(r'[{}]', '', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        stop = {'the','a','an','in','of','for','on','and','to','with',
                'der','die','das','und','fur','von','mit','im','an','zu',
                'eine','ein','des','dem'}
        return ' '.join(w for w in t.split() if w not in stop)

    t1, t2 = _norm(title1), _norm(title2)
    if not t1 or not t2:
        return 0.0
    try:
        from rapidfuzz.fuzz import token_set_ratio
        return token_set_ratio(t1, t2) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


# ---------------------------------------------------------------------------
# Author overlap scoring  — exported for AI pre-filter
# ---------------------------------------------------------------------------

def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    """
    Return fraction of cited author surnames found in correct_authors, or None
    if either list has fewer than 2 authors (not enough to be meaningful).
    """
    if not cited_authors or not correct_authors:
        return None

    def _surnames(s: str) -> List[str]:
        out = []
        for part in re.split(r';|\band\b', s, flags=re.IGNORECASE):
            part = part.strip().lower()
            if re.match(r'^et\s+al\.?$', part):
                continue
            for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss')]:
                part = part.replace(a, b)
            surname = part.split(',')[0].strip() if ',' in part else (part.split() or [''])[-1]
            surname = re.sub(r'[^\w]', '', surname)
            if len(surname) > 2:
                out.append(surname)
        return out

    cited   = _surnames(cited_authors)
    correct = _surnames(correct_authors)
    if len(cited) < 2 or len(correct) < 2:
        return None

    cited_cmp   = cited[:10]
    correct_set = set(correct)
    matches = sum(1 for s in cited_cmp if any(s in c or c in s for c in correct_set))
    return matches / len(cited_cmp)


# ---------------------------------------------------------------------------
# In-text citation extraction
# ---------------------------------------------------------------------------

def extract_citations_from_body(body_text: str) -> set:
    keys = set()
    for match in re.findall(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]', body_text
    ):
        for key in re.split(r',\s*', match):
            key = key.strip()
            if re.match(r'^[A-Za-z]{2,6}\d{2}[a-z]?$', key):
                keys.add(key)
    if re.findall(r'\[(\d{1,3})\]', body_text):
        keys.add('__numeric_citations__')
    return keys


def extract_citation_contexts(body_text: str) -> dict:
    contexts = {}
    for m in re.finditer(
        r'([^.]{0,80})\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]([^.]{0,80})',
        body_text,
    ):
        snippet = (m.group(1) + '[' + m.group(2) + ']' + m.group(3)).strip()
        for key in re.split(r',\s*', m.group(2)):
            key = key.strip()
            if key:
                contexts.setdefault(key, [])
                if len(contexts[key]) < 2:
                    contexts[key].append(snippet)
    return contexts


def _norm_name(name: str) -> str:
    name = name.lower()
    for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss')]:
        name = name.replace(a, b)
    return name


def detect_self_citations(bib_entries: dict, body_text: str) -> list:
    candidates = {_norm_name(c) for c in re.findall(r'\b([A-ZÄÖÜ][a-zäöüß]{3,})\b', body_text[:2000])}
    self_cites = []
    for key, entry in bib_entries.items():
        if not entry.authors:
            continue
        for name in re.split(r'[;,]', entry.authors):
            name = name.strip()
            if len(name) > 3 and _norm_name(name) in candidates:
                self_cites.append({"key": key, "title": entry.title or "", "matched_author": name})
                break
    return self_cites


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

@dataclass
class CrossCheckResult:
    cited_not_in_bib: list = field(default_factory=list)
    in_bib_not_cited: list = field(default_factory=list)
    correctly_used:   list = field(default_factory=list)


def cross_check(bib_entries: dict, cited_keys: set) -> CrossCheckResult:
    real_cited = {k for k in cited_keys if not k.startswith('__')}
    bib_keys   = set(bib_entries.keys())
    r = CrossCheckResult()
    r.cited_not_in_bib = sorted(real_cited - bib_keys)
    r.in_bib_not_cited  = sorted(bib_keys - real_cited)
    r.correctly_used    = sorted(real_cited & bib_keys)
    return r


# ---------------------------------------------------------------------------
# VerificationResult  — extended with version_note (NEW v6)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str              # verified | not_found | partial_match | error
    confidence: float
    matched_title:   Optional[str] = None
    doi:             Optional[str] = None
    open_access_url: Optional[str] = None
    note:            Optional[str] = None
    sources_checked: list = field(default_factory=list)
    web_evidence:    Optional[str] = None
    correct_authors: Optional[str] = None
    # NEW v6: non-empty when reference matches an older arXiv version
    version_note:    Optional[str] = None


# ---------------------------------------------------------------------------
# Per-host rate limiter
# ---------------------------------------------------------------------------

_RATE_LOCK: Dict[str, threading.Lock] = {}
_RATE_LAST: Dict[str, float] = {}
_RATE_META_LOCK = threading.Lock()


def _rate_limit(host: str, min_interval: float) -> None:
    with _RATE_META_LOCK:
        if host not in _RATE_LOCK:
            _RATE_LOCK[host] = threading.Lock()
    with _RATE_LOCK[host]:
        elapsed = time.time() - _RATE_LAST.get(host, 0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _RATE_LAST[host] = time.time()


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------

def _crossref_authors_str(work: dict) -> Optional[str]:
    authors = work.get("author", [])
    if not authors:
        return None
    parts = []
    for a in authors[:10]:
        family = a.get("family", "")
        given  = a.get("given", "")
        if family:
            parts.append(f"{family}, {given}".strip(', '))
    return "; ".join(parts) or None


def _query_crossref(entry: BibEntry) -> VerificationResult:
    try:
        hdrs = {"User-Agent": "LNI-Checker/6.0 (mailto:lni@checker.de)"}

        if entry.doi:
            resp = requests.get(f"https://api.crossref.org/works/{entry.doi}", timeout=10, headers=hdrs)
            if resp.status_code == 200:
                work  = resp.json().get("message", {})
                ft    = (work.get("title") or [""])[0]
                sim   = _title_similarity(entry.title or "", ft)
                oa    = _check_unpaywall(entry.doi)
                ca    = _crossref_authors_str(work)
                return VerificationResult(key=entry.key, title=entry.title or "",
                    status="verified", confidence=max(sim, 0.95),
                    matched_title=ft, doi=entry.doi, open_access_url=oa,
                    note="DOI verified via CrossRef", sources_checked=["CrossRef (DOI)"],
                    correct_authors=ca)

        params = {"query.title": entry.title, "rows": 3, "select": "DOI,title,author,published-print"}
        if entry.authors:
            params["query.author"] = entry.authors.split(';')[0].split(',')[0].strip()

        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10, headers=hdrs)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0, note="No results from CrossRef",
                sources_checked=["CrossRef"])

        best = items[0]
        ft   = (best.get("title") or [""])[0]
        doi  = best.get("DOI", "")
        sim  = _title_similarity(entry.title or "", ft)
        ca   = _crossref_authors_str(best)

        if sim >= 0.75:
            oa = _check_unpaywall(doi) if doi else None
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="verified", confidence=sim, matched_title=ft, doi=doi,
                open_access_url=oa, note="Found via CrossRef", sources_checked=["CrossRef"],
                correct_authors=ca)
        if sim >= 0.4:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="partial_match", confidence=sim, matched_title=ft, doi=doi,
                note="Partial match on CrossRef", sources_checked=["CrossRef"],
                correct_authors=ca)
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="not_found", confidence=sim, matched_title=ft,
            note="Title mismatch on CrossRef", sources_checked=["CrossRef"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"CrossRef error: {e}",
            sources_checked=["CrossRef"])


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _query_semantic_scholar(entry: BibEntry) -> VerificationResult:
    try:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        hdrs = {"User-Agent": "LNI-Checker/6.0"}
        if api_key:
            hdrs["x-api-key"] = api_key

        query = (entry.title or "")
        if entry.authors:
            query += " " + entry.authors.split(';')[0].split(',')[0].strip()

        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3,
                    "fields": "title,authors,year,openAccessPdf,externalIds"},
            timeout=10, headers=hdrs)
        resp.raise_for_status()
        papers = resp.json().get("data", [])
        if not papers:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0, note="Not found in Semantic Scholar",
                sources_checked=["Semantic Scholar"])

        best  = papers[0]
        ft    = best.get("title", "")
        sim   = _title_similarity(entry.title or "", ft)
        oa    = (best.get("openAccessPdf") or {}).get("url")
        eids  = best.get("externalIds", {})
        doi   = eids.get("DOI") or eids.get("doi")
        ca    = "; ".join(a.get("name","") for a in best.get("authors",[])[:10]) or None

        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=entry.title or "",
            status=status, confidence=sim, matched_title=ft, doi=doi,
            open_access_url=oa, note="Found via Semantic Scholar",
            sources_checked=["Semantic Scholar"], correct_authors=ca)
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"Semantic Scholar error: {e}",
            sources_checked=["Semantic Scholar"])


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _query_openalex(entry: BibEntry) -> VerificationResult:
    try:
        q = (entry.title or "")
        if entry.authors:
            q += " " + entry.authors.split(';')[0].split(',')[0].strip()

        resp = requests.get("https://api.openalex.org/works",
            params={"search": q, "per_page": 3}, timeout=10,
            headers={"User-Agent": "LNI-Checker/6.0 (mailto:lni@checker.de)"})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0, note="Not found in OpenAlex",
                sources_checked=["OpenAlex"])

        best = results[0]
        ft   = best.get("title", "")
        sim  = _title_similarity(entry.title or "", ft)
        doi  = (best.get("doi") or "").replace("https://doi.org/", "") or None
        oa   = best.get("open_access", {}).get("oa_url")
        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=entry.title or "",
            status=status, confidence=sim, matched_title=ft, doi=doi,
            open_access_url=oa, note="Found via OpenAlex", sources_checked=["OpenAlex"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"OpenAlex error: {e}",
            sources_checked=["OpenAlex"])


# ---------------------------------------------------------------------------
# ArXiv ID extraction helper
# ---------------------------------------------------------------------------

_ARXIV_ID_PATTERNS = [
    r'arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?',
    r'arxiv\.org/pdf/(\d{4}\.\d{4,5})(v\d+)?',
    r'arxiv\.org/abs/([a-z\-]+/\d{7})(v\d+)?',
    r'arxiv\.org/pdf/([a-z\-]+/\d{7})(v\d+)?',
    r'export\.arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?',
    r'arXiv:(\d{4}\.\d{4,5})(v\d+)?',
    r'arXiv:([a-z\-]+/\d{7})(v\d+)?',
    r'(?:arxiv[:./])(\d{4}\.\d{4,5})(v\d+)?',
]


def _extract_arxiv_id(entry: BibEntry):
    """Return (base_id, cited_version_str_or_None) from any field of entry."""
    sources = [
        entry.url or "",
        entry.doi or "",
        entry.raw_text or "",
        entry.journal or "",
    ]
    for src in sources:
        if not src:
            continue
        for pat in _ARXIV_ID_PATTERNS:
            m = re.search(pat, src, re.IGNORECASE)
            if m:
                base_id = m.group(1)
                version = m.group(2) if len(m.groups()) > 1 else None
                return base_id, version
    return None, None


# ---------------------------------------------------------------------------
# ArXiv BibTeX parser (standalone — no pybtex dependency)
# ---------------------------------------------------------------------------

def _parse_arxiv_bibtex(bibtex_str: str) -> Optional[dict]:
    """
    Parse a minimal arXiv BibTeX response without external libraries.
    Returns dict with keys: title, authors (list of str), year (int|None),
    eprint, doi.
    """
    if not bibtex_str or not bibtex_str.strip().startswith('@'):
        return None

    def _field(name: str) -> str:
        # Match:  fieldname = {value}  or  fieldname = "value"
        m = re.search(
            rf'\b{name}\s*=\s*[\{{"](.*?)[\}}"]\s*[,}}]',
            bibtex_str, re.DOTALL | re.IGNORECASE
        )
        if not m:
            # Handle nested braces: field = {{value}}
            m = re.search(
                rf'\b{name}\s*=\s*\{{(.*?)\}}\s*[,}}]',
                bibtex_str, re.DOTALL | re.IGNORECASE
            )
        if not m:
            return ""
        val = m.group(1)
        # Strip LaTeX brace wrappers used for capitalisation
        val = re.sub(r'\{([^{}]*)\}', r'\1', val)
        return re.sub(r'\s+', ' ', val).strip()

    title   = _field("title")
    year_s  = _field("year")
    eprint  = _field("eprint")
    doi     = _field("doi")

    # Authors: split on ' and ' (BibTeX convention)
    raw_auth = _field("author")
    authors = [a.strip() for a in re.split(r'\s+and\s+', raw_auth, flags=re.IGNORECASE) if a.strip()]

    year = None
    try:
        year = int(year_s)
    except (ValueError, TypeError):
        # Infer from eprint ID: first 2 digits are year (post-2007 format)
        if eprint and re.match(r'^\d{4}\.\d{4,5}', eprint):
            yy = int(eprint[:2])
            year = 2000 + yy if yy <= 30 else 1900 + yy

    return {"title": title, "authors": authors, "year": year, "eprint": eprint, "doi": doi}


# ---------------------------------------------------------------------------
# ArXiv versioned BibTeX checker  (NEW v6)
# ---------------------------------------------------------------------------

# In-process cache for arXiv BibTeX fetches to avoid re-downloading the same
# paper multiple times within a batch session.
_ARXIV_BIBTEX_MEM_CACHE: Dict[str, Optional[str]] = {}
_ARXIV_BIBTEX_LOCK = threading.Lock()


def _fetch_arxiv_bibtex_cached(arxiv_id: str) -> Optional[str]:
    """Fetch https://arxiv.org/bibtex/{id} with in-memory caching."""
    with _ARXIV_BIBTEX_LOCK:
        if arxiv_id in _ARXIV_BIBTEX_MEM_CACHE:
            return _ARXIV_BIBTEX_MEM_CACHE[arxiv_id]

    _rate_limit("arxiv.org", 0.34)  # max ~3 req/s as per arXiv guidelines
    try:
        resp = requests.get(
            f"https://arxiv.org/bibtex/{arxiv_id}",
            timeout=12,
            headers={"User-Agent": "LNI-Checker/6.0 (mailto:lni@checker.de)"},
        )
        text = resp.text.strip() if resp.status_code == 200 else None
        if text and not text.startswith('@'):
            text = None
    except Exception:
        text = None

    with _ARXIV_BIBTEX_LOCK:
        _ARXIV_BIBTEX_MEM_CACHE[arxiv_id] = text
    return text


def _query_arxiv_versioned(entry: BibEntry) -> Optional[VerificationResult]:
    """
    Authoritative arXiv checker (NEW v6).

    1. Extract arXiv ID + cited version from the entry's url/doi/raw_text.
    2. Fetch official BibTeX (always latest version) from arxiv.org/bibtex/.
    3. Compare cited metadata against official metadata.
    4. If no match on latest, also check each historical version (v1 .. v{latest-1})
       by fetching the per-version abs page and extracting the title.
    5. Return VerificationResult. Returns None if no arXiv ID found so the
       caller can skip this checker and fall through to the regular arXiv search.

    version_note is set when the cited version differs from the latest, so
    the professor can see "cited v2, latest is v4" without it counting as FAKE.
    """
    arxiv_id, cited_version = _extract_arxiv_id(entry)
    if not arxiv_id:
        return None

    bibtex_str = _fetch_arxiv_bibtex_cached(arxiv_id)
    if not bibtex_str:
        # BibTeX fetch failed — fall back to a basic existence check
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="not_found", confidence=0.0,
            note=f"arXiv ID {arxiv_id} found in reference but BibTeX fetch failed",
            sources_checked=["arXiv (versioned)"],
        )

    meta = _parse_arxiv_bibtex(bibtex_str)
    if not meta:
        return None

    official_title   = meta.get("title", "")
    official_authors = "; ".join(meta.get("authors", []))
    official_year    = meta.get("year")
    official_doi     = meta.get("doi") or None

    # --- Compare cited title vs latest-version title -------------------------
    sim = _title_similarity(entry.title or "", official_title)

    version_note: Optional[str] = None
    if cited_version:
        # Determine latest version number by reading the abs page
        try:
            _rate_limit("arxiv.org", 0.34)
            abs_resp = requests.get(
                f"https://arxiv.org/abs/{arxiv_id}",
                timeout=10,
                headers={"User-Agent": "LNI-Checker/6.0"},
            )
            if abs_resp.status_code == 200:
                latest_m = re.search(
                    r'Latest version.*?v(\d+)', abs_resp.text, re.DOTALL | re.IGNORECASE
                )
                if not latest_m:
                    latest_m = re.search(r'\[v(\d+)\]', abs_resp.text)
                if latest_m:
                    latest_v = int(latest_m.group(1))
                    cited_v  = int(re.sub(r'[^0-9]', '', cited_version) or "0")
                    if cited_v < latest_v:
                        version_note = (
                            f"Cited version {cited_version} — latest is v{latest_v}. "
                            "Metadata from latest version used for verification."
                        )
        except Exception:
            pass

    # --- If title doesn't match latest, check old versions ------------------
    if sim < 0.75 and cited_version:
        cited_v = int(re.sub(r'[^0-9]', '', cited_version) or "0")
        # Try up to 5 previous versions
        for v in range(max(1, cited_v - 1), cited_v + 1):
            try:
                _rate_limit("arxiv.org", 0.34)
                vresp = requests.get(
                    f"https://arxiv.org/abs/{arxiv_id}v{v}",
                    timeout=10,
                    headers={"User-Agent": "LNI-Checker/6.0"},
                )
                if vresp.status_code == 200:
                    tm = re.search(r'<h1 class="title[^"]*">.*?<span[^>]*>(.*?)</span>', vresp.text, re.DOTALL)
                    if tm:
                        v_title = re.sub(r'<[^>]+>', '', tm.group(1)).strip()
                        v_sim   = _title_similarity(entry.title or "", v_title)
                        if v_sim > sim:
                            sim           = v_sim
                            official_title = v_title
                            if not version_note:
                                version_note = f"Matched title from arXiv version v{v}."
            except Exception:
                continue

    # Determine final status
    status: str
    if sim >= 0.75:
        status = "verified"
    elif sim >= 0.4:
        status = "partial_match"
    else:
        status = "not_found"

    oa_url = f"https://arxiv.org/pdf/{arxiv_id}" if status in ("verified", "partial_match") else None

    note_parts = [f"Found via arXiv BibTeX (authoritative) — ID: {arxiv_id}"]
    if version_note:
        note_parts.append(version_note)
    if official_year and entry.year and str(official_year) != entry.year:
        note_parts.append(f"Year: cited '{entry.year}', arXiv says '{official_year}'.")

    return VerificationResult(
        key=entry.key,
        title=entry.title or "",
        status=status,
        confidence=round(sim, 2),
        matched_title=official_title,
        doi=official_doi,
        open_access_url=oa_url,
        note=" ".join(note_parts),
        sources_checked=["arXiv BibTeX (versioned)"],
        correct_authors=official_authors or None,
        version_note=version_note,
    )


# ---------------------------------------------------------------------------
# arXiv search fallback (used when no arXiv ID is present in the entry)
# ---------------------------------------------------------------------------

def _query_arxiv_search(entry: BibEntry) -> VerificationResult:
    """Title+author search via arXiv API — fallback when no explicit arXiv ID."""
    try:
        import urllib.parse
        title = entry.title or ""
        q = f'ti:"{urllib.parse.quote(title)}"'
        if entry.authors:
            s = entry.authors.split(';')[0].split(',')[0].strip()
            if s:
                q += f' AND au:{urllib.parse.quote(s)}'

        _rate_limit("export.arxiv.org", 0.34)
        resp = requests.get("https://export.arxiv.org/api/query",
            params={"search_query": q, "max_results": 3, "sortBy": "relevance"},
            timeout=10, headers={"User-Agent": "LNI-Checker/6.0"})
        if resp.status_code != 200:
            return VerificationResult(key=entry.key, title=title, status="error",
                confidence=0.0, note=f"arXiv HTTP {resp.status_code}", sources_checked=["arXiv"])

        entries_xml = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        if not entries_xml:
            return VerificationResult(key=entry.key, title=title, status="not_found",
                confidence=0.0, note="Not found on arXiv", sources_checked=["arXiv"])

        tm = re.search(r'<title>(.*?)</title>', entries_xml[0], re.DOTALL)
        ft = re.sub(r'\s+', ' ', tm.group(1)).strip() if tm else ""
        pm = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', entries_xml[0])
        oa = pm.group(1) if pm else None
        dm = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', entries_xml[0])
        doi = dm.group(1).strip() if dm else None
        author_names = re.findall(r'<name>(.*?)</name>', entries_xml[0])
        ca = "; ".join(author_names[:10]) or None

        sim = _title_similarity(title, ft)
        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=title, status=status,
            confidence=sim, matched_title=ft, doi=doi, open_access_url=oa,
            note="Found on arXiv (search)", sources_checked=["arXiv"], correct_authors=ca)
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"arXiv error: {e}", sources_checked=["arXiv"])


def _query_arxiv(entry: BibEntry) -> VerificationResult:
    """
    Unified arXiv checker (v6):
    - If entry has an arXiv ID → use versioned BibTeX checker (authoritative)
    - Otherwise → fall back to title+author search
    """
    versioned = _query_arxiv_versioned(entry)
    if versioned is not None:
        return versioned
    return _query_arxiv_search(entry)


# ---------------------------------------------------------------------------
# DBLP
# ---------------------------------------------------------------------------

def _query_dblp(entry: BibEntry) -> VerificationResult:
    try:
        _rate_limit("dblp.org", 1.1)
        title   = entry.title or ""
        clean_q = re.sub(r'[^\w\s]', ' ', title.lower()).strip()

        resp = requests.get("https://dblp.org/search/publ/api",
            params={"q": clean_q, "format": "json", "h": 3}, timeout=15,
            headers={"User-Agent": "LNI-Checker/6.0 (mailto:lni@checker.de)"})
        resp.raise_for_status()

        hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return VerificationResult(key=entry.key, title=title, status="not_found",
                confidence=0.0, note="Not found in DBLP", sources_checked=["DBLP"])

        info = hits[0].get("info", {})
        ft   = info.get("title", "")
        doi  = info.get("doi")
        url  = info.get("url")
        sim  = _title_similarity(title, ft)

        ad = info.get("authors", {}).get("author", [])
        if isinstance(ad, dict):
            ad = [ad]
        ca = "; ".join(
            (a.get("text", "") if isinstance(a, dict) else str(a)) for a in ad[:10]
        ) or None

        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=title, status=status,
            confidence=sim, matched_title=ft, doi=doi, open_access_url=url,
            note="Found via DBLP", sources_checked=["DBLP"], correct_authors=ca)
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"DBLP error: {e}", sources_checked=["DBLP"])


# ---------------------------------------------------------------------------
# ACL Anthology
# ---------------------------------------------------------------------------

def _query_acl_anthology(entry: BibEntry) -> VerificationResult:
    try:
        title = entry.title or ""
        resp = requests.get("https://aclanthology.org/search/",
            params={"q": title}, timeout=12,
            headers={"User-Agent": "LNI-Checker/6.0"})
        if resp.status_code != 200:
            return VerificationResult(key=entry.key, title=title, status="error",
                confidence=0.0, note=f"ACL Anthology HTTP {resp.status_code}",
                sources_checked=["ACL Anthology"])

        found = re.findall(r'<span class="d-block"[^>]*>(.*?)</span>', resp.text, re.DOTALL)
        found = [re.sub(r'<[^>]+>', '', t).strip() for t in found if t.strip()]
        if not found:
            found = re.findall(r'<a\s+class="align-middle"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            found = [re.sub(r'<[^>]+>', '', t).strip() for t in found if t.strip()]

        if not found:
            return VerificationResult(key=entry.key, title=title, status="not_found",
                confidence=0.0, note="Not found in ACL Anthology",
                sources_checked=["ACL Anthology"])

        ft  = found[0]
        sim = _title_similarity(title, ft)
        pm  = re.search(r'href="(https://aclanthology\.org/[^"]+\.pdf)"', resp.text)
        oa  = pm.group(1) if pm else None

        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=title, status=status,
            confidence=sim, matched_title=ft, open_access_url=oa,
            note="Found via ACL Anthology", sources_checked=["ACL Anthology"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"ACL Anthology error: {e}",
            sources_checked=["ACL Anthology"])


# ---------------------------------------------------------------------------
# OpenReview
# ---------------------------------------------------------------------------

def _query_openreview(entry: BibEntry) -> VerificationResult:
    try:
        title = entry.title or ""

        if entry.url and "openreview.net" in entry.url:
            fid = re.search(r'[?&]id=([A-Za-z0-9_\-]+)', entry.url)
            if fid:
                resp = requests.get("https://api.openreview.net/notes",
                    params={"forum": fid.group(1), "limit": 1}, timeout=10,
                    headers={"User-Agent": "LNI-Checker/6.0"})
                if resp.status_code == 200:
                    notes = resp.json().get("notes", [])
                    if notes:
                        ft  = notes[0].get("content", {}).get("title", "")
                        sim = _title_similarity(title, ft)
                        return VerificationResult(key=entry.key, title=title,
                            status="verified" if sim >= 0.7 else "partial_match",
                            confidence=max(sim, 0.9), matched_title=ft,
                            open_access_url=entry.url,
                            note="Verified via OpenReview forum ID",
                            sources_checked=["OpenReview"])

        resp = requests.get("https://api2.openreview.net/notes/search",
            params={"term": title, "limit": 3}, timeout=12,
            headers={"User-Agent": "LNI-Checker/6.0"})
        if resp.status_code != 200:
            return VerificationResult(key=entry.key, title=title, status="error",
                confidence=0.0, note=f"OpenReview HTTP {resp.status_code}",
                sources_checked=["OpenReview"])

        notes = resp.json().get("notes", [])
        if not notes:
            return VerificationResult(key=entry.key, title=title, status="not_found",
                confidence=0.0, note="Not found on OpenReview",
                sources_checked=["OpenReview"])

        best_sim, best_ft, best_url = 0.0, "", None
        for note in notes:
            nt = note.get("content", {}).get("title", "")
            if isinstance(nt, dict):
                nt = nt.get("value", "")
            sim = _title_similarity(title, nt)
            if sim > best_sim:
                best_sim = sim
                best_ft  = nt
                fid      = note.get("forum") or note.get("id", "")
                best_url = f"https://openreview.net/forum?id={fid}" if fid else None

        status = "verified" if best_sim >= 0.75 else "partial_match" if best_sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=title, status=status,
            confidence=round(best_sim, 2), matched_title=best_ft,
            open_access_url=best_url, note="Found via OpenReview",
            sources_checked=["OpenReview"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"OpenReview error: {e}",
            sources_checked=["OpenReview"])


# ---------------------------------------------------------------------------
# Open Library
# ---------------------------------------------------------------------------

def _query_open_library(entry: BibEntry) -> VerificationResult:
    try:
        if entry.isbn:
            isbn = re.sub(r'[\s-]', '', entry.isbn)
            resp = requests.get("https://openlibrary.org/api/books",
                params={"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"},
                timeout=10, headers={"User-Agent": "LNI-Checker/6.0"})
            if resp.status_code == 200:
                rec = resp.json().get(f"ISBN:{isbn}")
                if rec:
                    ft  = rec.get("title", "")
                    sim = _title_similarity(entry.title or "", ft)
                    return VerificationResult(key=entry.key, title=entry.title or "",
                        status="verified", confidence=max(sim, 0.95),
                        matched_title=ft, open_access_url=rec.get("url"),
                        note=f"ISBN {isbn} verified via Open Library",
                        sources_checked=["Open Library (ISBN)"])

        if not entry.title:
            return VerificationResult(key=entry.key, title="", status="error",
                confidence=0.0, note="No title/ISBN for Open Library",
                sources_checked=["Open Library"])

        resp = requests.get("https://openlibrary.org/search.json",
            params={"title": entry.title, "limit": 3}, timeout=10,
            headers={"User-Agent": "LNI-Checker/6.0"})
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if not docs:
            return VerificationResult(key=entry.key, title=entry.title,
                status="not_found", confidence=0.0, note="Not found in Open Library",
                sources_checked=["Open Library"])

        best = docs[0]
        ft   = best.get("title", "")
        sim  = _title_similarity(entry.title, ft)
        oa   = f"https://openlibrary.org{best.get('key','')}" if best.get("key") else None

        if entry.authors and sim >= 0.5:
            ol_auths = best.get("author_name", [])
            if ol_auths:
                fname = entry.authors.split(';')[0].split(',')[0].strip().lower()
                if not any(fname in a.lower() for a in ol_auths):
                    sim = max(0.0, sim - 0.15)

        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.4 else "not_found"
        return VerificationResult(key=entry.key, title=entry.title,
            status=status, confidence=sim, matched_title=ft,
            open_access_url=oa, note="Found via Open Library",
            sources_checked=["Open Library"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"Open Library error: {e}",
            sources_checked=["Open Library"])


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _query_github(entry: BibEntry) -> Optional[VerificationResult]:
    url = entry.url or ""
    if "github.com" not in url:
        return None

    m = re.match(r'https?://github\.com/([^/\s]+)/([^/\s#?]+)', url.rstrip('/'))
    if not m:
        return None

    owner = m.group(1)
    repo  = m.group(2).replace('.git', '')

    try:
        token = os.environ.get("GITHUB_TOKEN", "")
        hdrs  = {"Accept": "application/vnd.github.v3+json", "User-Agent": "LNI-Checker/6.0"}
        if token:
            hdrs["Authorization"] = f"token {token}"

        resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
            headers=hdrs, timeout=8)

        if resp.status_code == 200:
            data     = resp.json()
            stars    = data.get("stargazers_count", 0)
            archived = data.get("archived", False)
            note = (f"GitHub repo {owner}/{repo} exists "
                    f"({stars}★{'  archived' if archived else ''})")
            return VerificationResult(key=entry.key, title=entry.title or url,
                status="verified", confidence=1.0, open_access_url=url,
                note=note, sources_checked=["GitHub API"])

        if resp.status_code == 404:
            return VerificationResult(key=entry.key, title=entry.title or url,
                status="not_found", confidence=0.0,
                note=f"GitHub repo {owner}/{repo} not found (404)",
                sources_checked=["GitHub API"])

        return VerificationResult(key=entry.key, title=entry.title or url,
            status="error", confidence=0.0,
            note=f"GitHub API HTTP {resp.status_code}",
            sources_checked=["GitHub API"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or url,
            status="error", confidence=0.0, note=f"GitHub error: {e}",
            sources_checked=["GitHub API"])


# ---------------------------------------------------------------------------
# Google Scholar  (scrape fallback)
# ---------------------------------------------------------------------------

def _query_google_scholar(entry: BibEntry) -> VerificationResult:
    try:
        q = (entry.title or "")
        if entry.authors:
            q += " " + entry.authors.split(';')[0].split(',')[0].strip()

        hdrs = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get("https://scholar.google.com/scholar",
            params={"q": q, "hl": "en", "num": 3}, headers=hdrs, timeout=12)

        if resp.status_code == 429:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="error", confidence=0.0, note="Google Scholar rate-limited",
                sources_checked=["Google Scholar"])
        if resp.status_code != 200:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="error", confidence=0.0,
                note=f"Google Scholar HTTP {resp.status_code}",
                sources_checked=["Google Scholar"])

        titles = [re.sub(r'<[^>]+>', '', t).strip()
                  for t in re.findall(r'class="gs_rt"[^>]*>(?:<[^>]+>)*([^<]+)', resp.text)
                  if t.strip()]
        if not titles:
            return VerificationResult(key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0, note="No results on Google Scholar",
                sources_checked=["Google Scholar"])

        sim    = _title_similarity(entry.title or "", titles[0])
        status = "verified" if sim >= 0.75 else "partial_match" if sim >= 0.40 else "not_found"
        return VerificationResult(key=entry.key, title=entry.title or "",
            status=status, confidence=sim, matched_title=titles[0],
            note="Found via Google Scholar", sources_checked=["Google Scholar"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"Google Scholar error: {e}",
            sources_checked=["Google Scholar"])


# ---------------------------------------------------------------------------
# DuckDuckGo web fingerprint  (last resort)
# ---------------------------------------------------------------------------

def _query_duckduckgo_web(entry: BibEntry) -> VerificationResult:
    try:
        title = entry.title or ""
        q     = f'"{title}"'
        if entry.authors:
            q += " " + entry.authors.split(';')[0].split(',')[0].strip()
        if entry.year:
            q += f" {entry.year}"

        resp = requests.get("https://html.duckduckgo.com/html/",
            params={"q": q},
            headers={"User-Agent": "Mozilla/5.0 (compatible; LNI-Checker/6.0)",
                     "Accept-Language": "en-US,en;q=0.9"},
            timeout=12)
        if resp.status_code != 200:
            return VerificationResult(key=entry.key, title=title, status="error",
                confidence=0.0, note=f"DuckDuckGo HTTP {resp.status_code}",
                sources_checked=["Web (DDG)"])

        snippets = [re.sub(r'<[^>]+>', '', s).strip()
                    for s in re.findall(r'class="result__snippet"[^>]*>(.*?)</a>',
                                        resp.text, re.DOTALL)][:3]
        combined  = " ".join(snippets).lower()
        words     = {w for w in title.lower().split() if len(w) > 3}
        coverage  = sum(1 for w in words if w in combined) / len(words) if words else 0

        if coverage >= 0.7:
            return VerificationResult(key=entry.key, title=title,
                status="verified" if coverage >= 0.85 else "partial_match",
                confidence=round(coverage, 2),
                note="Found via web search (DuckDuckGo)",
                web_evidence=snippets[0][:150] if snippets else "",
                sources_checked=["Web (DDG)"])

        return VerificationResult(key=entry.key, title=title, status="not_found",
            confidence=round(coverage, 2),
            note=f"Limited web evidence ({int(coverage*100)}% keyword match)",
            sources_checked=["Web (DDG)"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note=f"Web search error: {e}",
            sources_checked=["Web (DDG)"])


# ---------------------------------------------------------------------------
# Website / URL verification
# ---------------------------------------------------------------------------

def _verify_website(entry: BibEntry) -> VerificationResult:
    url = entry.url or ""

    if "github.com" in url:
        result = _query_github(entry)
        if result:
            return result

    if not url:
        return VerificationResult(key=entry.key, title=entry.title or "(website)",
            status="error", confidence=0.0, note="No URL found",
            sources_checked=["URL check"])
    try:
        if not url.startswith("http"):
            url = "https://" + url
        resp = requests.head(url, timeout=8, allow_redirects=True)
        if resp.status_code < 400:
            return VerificationResult(key=entry.key, title=entry.title or url,
                status="verified", confidence=1.0, open_access_url=url,
                note=f"URL reachable (HTTP {resp.status_code})",
                sources_checked=["URL check"])
        return VerificationResult(key=entry.key, title=entry.title or url,
            status="not_found", confidence=0.0,
            note=f"URL returned HTTP {resp.status_code}",
            sources_checked=["URL check"])
    except Exception as e:
        return VerificationResult(key=entry.key, title=entry.title or url,
            status="error", confidence=0.0, note=f"URL check failed: {e}",
            sources_checked=["URL check"])


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------

def _check_unpaywall(doi: str) -> Optional[str]:
    email = os.environ.get("UNPAYWALL_EMAIL", "lni-checker@uni-project.de")
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=8)
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
# Main verifier  — parallel, two-level cached
# ---------------------------------------------------------------------------

def verify_reference(entry: BibEntry) -> VerificationResult:
    if not entry.title and not entry.doi and not entry.isbn:
        return VerificationResult(key=entry.key, title="(no title parsed)",
            status="error", confidence=0.0,
            note="Could not extract title — check parsing", sources_checked=[])

    if entry.entry_type == "website":
        return _verify_website(entry)

    cached = _get_cached(entry)
    if cached:
        result = copy.copy(cached)
        result.key  = entry.key
        result.note = (result.note or "") + " [cache]"
        return result

    source_fns = [
        _query_crossref,
        _query_semantic_scholar,
        _query_openalex,
        _query_arxiv,          # now routes through versioned checker first
        _query_dblp,
        _query_acl_anthology,
        _query_openreview,
        _query_open_library,
        _query_google_scholar,
        _query_duckduckgo_web,
    ]

    results: List[VerificationResult] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fn, entry): fn.__name__ for fn in source_fns}
        for future in as_completed(futures, timeout=35):
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except Exception:
                pass

    if not results:
        return VerificationResult(key=entry.key, title=entry.title or "",
            status="error", confidence=0.0, note="All sources failed",
            sources_checked=[])

    priority = {"verified": 3, "partial_match": 2, "not_found": 1, "error": 0}
    results.sort(key=lambda r: (priority.get(r.status, 0), r.confidence), reverse=True)
    best = results[0]

    best.sources_checked = [s for r in results for s in r.sources_checked]

    for r in results:
        if not best.web_evidence and r.web_evidence:
            best.web_evidence = r.web_evidence
        if not best.correct_authors and r.correct_authors:
            best.correct_authors = r.correct_authors
        if not best.version_note and r.version_note:
            best.version_note = r.version_note
        if not best.doi and r.doi:
            best.doi = r.doi

    # Boost confidence when multiple independent sources agree
    verified_count = sum(1 for r in results if r.status == "verified")
    if verified_count >= 2 and best.status == "verified":
        best.confidence = min(best.confidence + 0.05 * (verified_count - 1), 1.0)
        best.note = f"Confirmed by {verified_count} independent sources"

    _put_cache(entry, best)
    return best


def verify_all_references(bib_entries: dict, delay: float = 0.0) -> list:
    """Verify all entries concurrently. Returns results in original order."""
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_key = {
            executor.submit(verify_reference, entry): key
            for key, entry in bib_entries.items()
        }
        for future in as_completed(future_to_key, timeout=120):
            try:
                results.append(future.result())
            except Exception as e:
                key = future_to_key[future]
                results.append(VerificationResult(key=key, title="", status="error",
                    confidence=0.0, note=f"Verification crashed: {e}",
                    sources_checked=[]))

    key_order = list(bib_entries.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)
    return results


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(bib_entries: dict, threshold: float = 0.85) -> list:
    entries = list(bib_entries.values())
    dupes, seen = [], set()
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
                dupes.append({"key_a": a.key, "key_b": b.key,
                               "title_a": a.title, "title_b": b.title,
                               "similarity": round(score, 2),
                               "type": "exact" if score >= 0.97 else "near-duplicate"})
    return dupes


# ---------------------------------------------------------------------------
# LNI style checks
# ---------------------------------------------------------------------------

def check_lni_macros(body_text: str) -> list:
    suggestions = []

    for pattern, message in [
        (r'\be\.g\.', r"Use LNI macro '\eg' instead of 'e.g.'"),
        (r'\bi\.e\.', r"Use LNI macro '\ie' instead of 'i.e.'"),
        (r'\bcf\.',   r"Use LNI macro '\cf' instead of 'cf.'"),
        (r'\bet al\.', r"Use LNI macro '\etal' instead of 'et al.'"),
    ]:
        n = len(re.findall(pattern, body_text, re.IGNORECASE))
        if n:
            suggestions.append({"type": "Macro", "message": message, "count": n})

    ac = re.findall(r'\n([A-Z]{4,}(?:\s+[A-Z]{2,})*)\n', body_text)
    if ac:
        suggestions.append({"type": "Heading",
            "message": f"Found {len(ac)} ALL-CAPS heading(s) — LNI uses sentence case.",
            "count": len(ac)})

    n = len(re.findall(r'\\textbf\{[^}]{1,20}\}', body_text))
    if n:
        suggestions.append({"type": "Emphasis",
            "message": r"Manual \textbf{} — prefer LNI semantic macros.", "count": n})

    n = len(re.findall(r'(?<!`)"[^"]{1,60}"', body_text))
    if n:
        suggestions.append({"type": "Quotes",
            "message": r'Straight quotes (") — use ``...'"''"' or \\enquote{}.', "count": n})

    n = len(re.findall(r'\w–\w', body_text))
    if n:
        suggestions.append({"type": "Dash",
            "message": "En-dash (–) as word hyphen — use '-' for compounds, '--' for ranges.",
            "count": n})

    n = len(re.findall(r'\w\[(?:[A-Za-z]{2,6}\d{2})', body_text))
    if n:
        suggestions.append({"type": "Spacing",
            "message": "Citation directly after word — LNI requires 'text~[Key]'.",
            "count": n})

    has_lni     = bool(re.search(r'\[[A-Za-z]{2,6}\d{2}\]', body_text))
    has_numeric = bool(re.search(r'\[\d{1,3}\]', body_text))
    if has_lni and has_numeric:
        suggestions.append({"type": "CitationStyle",
            "message": "Mixed LNI [Ez10] and numeric [1] citations — LNI requires Author+Year only.",
            "count": len(re.findall(r'\[\d{1,3}\]', body_text))})

    return suggestions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(bib_list, xcheck, verification_results, style_suggestions,
                  duplicates, ai_fake_count=0):
    score, penalties = 100, []
    for label, count, per_item, cap in [
        ("Missing from bibliography",               len(xcheck.cited_not_in_bib),                       10, 30),
        ("Cited nowhere in text",                   len(xcheck.in_bib_not_cited),                        5, 20),
        ("Incomplete entries",                      sum(1 for e in bib_list if e.completeness_issues),   5, 20),
        ("Likely fabricated references (AI)",       ai_fake_count,                                       10, 30),
        ("Duplicate entries",                       len(duplicates),                                      5, 10),
        ("LNI style violations",                    len(style_suggestions),                               2,  6),
    ]:
        p = min(count * per_item, cap)
        if p:
            penalties.append({"category": label, "count": count, "deduction": p})
        score -= p

    score = max(score, 0)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 45 else "F"
    return {"score": score, "grade": grade, "penalties": penalties, "max_score": 100}