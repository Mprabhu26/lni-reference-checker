"""
STEP 3: Citation Cross-Checker + Reference Verifier — v6.2
----------------------------------------------------------
COMPLETELY REWRITTEN for thorough online verification.

Verification pipeline (parallel, all FREE, no paid API keys required):

PHASE 1: FAST LOOKUPS (all run in parallel, 3 second timeout)
  1. DOI direct lookup        → CrossRef (100% if DOI resolves)
  2. arXiv ID lookup          → arXiv API (if ID present)
  3. ISBN lookup              → Open Library

PHASE 2: TITLE/AUTHOR SEARCH (run in parallel, 5 second timeout)
  4. CrossRef title+author    → crossref.org
  5. Semantic Scholar         → semanticscholar.org
  6. OpenAlex                 → openalex.org
  7. DBLP                     → dblp.org
  8. ACL Anthology            → aclanthology.org

PHASE 3: DEEP SEARCH (run only if needed, 8 second timeout)
  9. arXiv search fallback    → export.arxiv.org
  10. OpenReview              → openreview.net
  11. Google Scholar scrape   → scholar.google.com
  12. DuckDuckGo web search   → html.duckduckgo.com

For each reference, we aggregate results from ALL sources and determine:
- REAL: Found in 2+ sources with title match >0.7 OR 1 source with DOI/arXiv ID
- SUSPICIOUS: Found in 1 source with title match 0.4-0.7 OR no sources but plausible
- FAKE: No sources match AND multiple strong fake indicators

All results are cached in disk + memory for batch processing efficiency.
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
from typing import Optional, List, Dict, Tuple
from parser import BibEntry

from local_db import search_cache, save_to_cache, get_cache_stats, init_cache_db

# Import web search verifier (new in v6.3)
from web_search_verifier import verify_with_web_search
from review_queue import is_venue_whitelisted, get_review_decision, get_false_positive

# ---------------------------------------------------------------------------
# Persistent disk cache
# ---------------------------------------------------------------------------

_DISK_CACHE_DIR: str = os.environ.get("LNI_CACHE_DIR", ".lni_cache")
_DISK_CACHE_LOCK = threading.Lock()
_MEM_CACHE: Dict[str, "VerificationResult"] = {}
_MEM_CACHE_LOCK = threading.Lock()

# ADDED: Missing cache dictionaries for arXiv and general results
_ARXIV_BIBTEX_MEM_CACHE: Dict[str, str] = {}
_ARXIV_CACHE_LOCK = threading.Lock()
_VERIFICATION_RESULT_CACHE: Dict[str, "VerificationResult"] = {}
_VERIFICATION_CACHE_LOCK = threading.Lock()

_RATE_LOCK: Dict[str, threading.Lock] = {}
_RATE_LAST: Dict[str, float] = {}
_RATE_META_LOCK = threading.Lock()


def _rate_limit(host: str, min_interval: float) -> None:
    """Simple rate limiter per host."""
    with _RATE_META_LOCK:
        if host not in _RATE_LOCK:
            _RATE_LOCK[host] = threading.Lock()
    with _RATE_LOCK[host]:
        elapsed = time.time() - _RATE_LAST.get(host, 0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _RATE_LAST[host] = time.time()


def _disk_cache_key(entry: BibEntry) -> str:
    title = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (entry.title or "").lower().strip()))
    first = ""
    if entry.authors:
        first = entry.authors.split(';')[0].split(',')[0].strip().lower()
    return hashlib.sha256(f"{title}|{first}".encode()).hexdigest()[:24]


def _get_cached(entry: BibEntry) -> Optional["VerificationResult"]:
    if not entry.title:
        return None
    key = _disk_cache_key(entry)
    with _MEM_CACHE_LOCK:
        hit = _MEM_CACHE.get(key)
    if hit:
        return hit
    
    path = Path(_DISK_CACHE_DIR) / f"{key}.json"
    if path.exists():
        try:
            with _DISK_CACHE_LOCK:
                data = json.loads(path.read_text(encoding="utf-8"))
            r = VerificationResult(**{k: data.get(k) for k in VerificationResult.__dataclass_fields__})
            with _MEM_CACHE_LOCK:
                _MEM_CACHE[key] = r
            return r
        except Exception:
            pass
    return None


def _put_cache(entry: BibEntry, result: "VerificationResult") -> None:
    if not entry.title:
        return
    key = _disk_cache_key(entry)
    with _MEM_CACHE_LOCK:
        _MEM_CACHE[key] = result
    
    path = Path(_DISK_CACHE_DIR) / f"{key}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fixed: Write with lock held for atomicity
        with _DISK_CACHE_LOCK:
            path.write_text(json.dumps({
                "key": result.key, "title": result.title, "status": result.status,
                "confidence": result.confidence, "matched_title": result.matched_title,
                "doi": result.doi, "open_access_url": result.open_access_url,
                "note": result.note, "sources_checked": result.sources_checked,
                "web_evidence": result.web_evidence, "correct_authors": result.correct_authors,
                "version_note": result.version_note,
            }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Title similarity (improved)
# ---------------------------------------------------------------------------

def _title_similarity(title1: str, title2: str) -> float:
    """Calculate title similarity with normalization."""
    if not title1 or not title2:
        return 0.0

    def _norm(t: str) -> str:
        t = t.lower()
        # German umlauts
        for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
            t = t.replace(a, b)
        # Strip LaTeX commands
        t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
        t = re.sub(r'[{}]', '', t)
        # Remove punctuation
        t = re.sub(r'[^\w\s]', ' ', t)
        # Remove common stopwords
        stop = {'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with',
                'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu',
                'eine', 'ein', 'des', 'dem', 'is', 'are', 'was', 'were', 'be', 'by'}
        return ' '.join(w for w in t.split() if w not in stop and len(w) > 2)

    t1, t2 = _norm(title1), _norm(title2)
    if not t1 or not t2:
        return 0.0
    
    try:
        from rapidfuzz.fuzz import token_sort_ratio
        return token_sort_ratio(t1, t2) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    """Return fraction of cited author surnames found in correct_authors."""
    if not cited_authors or not correct_authors:
        return None

    def _surnames(s: str) -> List[str]:
        out = []
        for part in re.split(r';|\band\b', s, flags=re.IGNORECASE):
            part = part.strip().lower()
            if re.match(r'^et\s+al\.?$', part):
                continue
            for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
                part = part.replace(a, b)
            surname = part.split(',')[0].strip() if ',' in part else (part.split() or [''])[-1]
            surname = re.sub(r'[^\w]', '', surname)
            if len(surname) > 2:
                out.append(surname)
        return out

    cited = _surnames(cited_authors)
    correct = _surnames(correct_authors)
    if not cited or not correct:
        return None

    correct_set = set(correct)
    matches = sum(1 for s in cited[:5] if any(s in c or c in s for c in correct_set))
    return matches / min(len(cited[:5]), 5)


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str              # verified | partial_match | not_found | error
    confidence: float
    matched_title: Optional[str] = None
    doi: Optional[str] = None
    open_access_url: Optional[str] = None
    note: Optional[str] = None
    sources_checked: list = field(default_factory=list)
    web_evidence: Optional[str] = None
    correct_authors: Optional[str] = None
    version_note: Optional[str] = None
    aggregated_sources: list = field(default_factory=list)
    is_retracted: bool = False
    retraction_doi: Optional[str] = None
    retraction_note: Optional[str] = None
    # Corrected metadata from databases (for BibTeX export)
    corrected_title: Optional[str] = None
    corrected_authors: Optional[str] = None
    corrected_year: Optional[str] = None
    corrected_publisher: Optional[str] = None
    corrected_journal: Optional[str] = None
    corrected_volume: Optional[str] = None
    corrected_pages: Optional[str] = None


# ---------------------------------------------------------------------------
# PHASE 1: Fast Lookups by Identifier
# ---------------------------------------------------------------------------

def _check_retraction(doi: str) -> tuple:
    """Check CrossRef for retraction notice on a DOI.

    CrossRef marks retracted papers with update-to[type=retraction].
    Returns (is_retracted: bool, retraction_doi: str, note: str).
    """
    if not doi:
        return False, None, None
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/6.3 (mailto:{mailto})" if mailto else "LNI-Checker/6.3"
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=5,
            headers={"User-Agent": ua},
        )
        if resp.status_code != 200:
            return False, None, None
        work = resp.json().get("message", {})
        updates = work.get("update-to", [])
        for u in updates:
            if u.get("type", "").lower() == "retraction":
                ret_doi = u.get("DOI", "")
                ret_date = u.get("updated", {}).get("date-parts", [[""]])[0]
                date_str = "-".join(str(p) for p in ret_date if p) if ret_date else "unknown date"
                return True, ret_doi, f"Retracted {date_str}. Retraction notice DOI: {ret_doi}"
    except Exception:
        pass
    return False, None, None


def _extract_corrected_metadata(work: dict) -> dict:
    """Pull clean corrected metadata from a CrossRef work record."""
    authors = work.get("author", [])
    author_str = "; ".join(
        f"{a.get('family', '')}, {a.get('given', '')}" for a in authors[:5]
    ) if authors else None

    issued = work.get("issued", {}).get("date-parts", [[None]])[0]
    year = str(issued[0]) if issued and issued[0] else None

    container = (work.get("container-title") or [""])[0]
    volume = work.get("volume", "")
    pages = work.get("page", "")
    publisher = work.get("publisher", "")

    return {
        "corrected_authors": author_str,
        "corrected_year": year,
        "corrected_journal": container or None,
        "corrected_publisher": publisher or None,
        "corrected_volume": str(volume) if volume else None,
        "corrected_pages": str(pages) if pages else None,
    }


def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    """Direct DOI lookup via CrossRef — also checks for retraction."""
    if not entry.doi:
        return None

    _rate_limit("crossref.org", 0.2)
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/6.3 (mailto:{mailto})" if mailto else "LNI-Checker/6.3"
        resp = requests.get(
            f"https://api.crossref.org/works/{entry.doi}",
            timeout=5,
            headers={"User-Agent": ua},
        )
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            meta = _extract_corrected_metadata(work)

            # Retraction check
            is_retracted, ret_doi, ret_note = _check_retraction(entry.doi)

            status = "verified" if not is_retracted else "retracted"
            confidence = 0.95 if sim > 0.7 else 0.7
            note_str = f"DOI verified via CrossRef (match: {int(sim*100)}%)"
            if is_retracted:
                note_str = f"⚠ RETRACTED. {ret_note}"
                confidence = 1.0   # 100% confident it's retracted

            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status=status,
                confidence=confidence,
                matched_title=title,
                doi=entry.doi,
                open_access_url=_check_unpaywall(entry.doi),
                note=note_str,
                sources_checked=["CrossRef (DOI)"],
                correct_authors=meta["corrected_authors"],
                is_retracted=is_retracted,
                retraction_doi=ret_doi,
                retraction_note=ret_note,
                corrected_title=title if title else None,
                corrected_authors=meta["corrected_authors"],
                corrected_year=meta["corrected_year"],
                corrected_journal=meta["corrected_journal"],
                corrected_publisher=meta["corrected_publisher"],
                corrected_volume=meta["corrected_volume"],
                corrected_pages=meta["corrected_pages"],
            )
    except Exception:
        pass
    return None


def _lookup_by_arxiv_id(entry: BibEntry) -> Optional[VerificationResult]:
    """Direct arXiv ID lookup."""
    # Extract arXiv ID from any field
    arxiv_patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
        r'arXiv:([a-z\-]+/\d{7})',
    ]
    
    arxiv_id = None
    for field in [entry.url or "", entry.doi or "", entry.raw_text or ""]:
        for pat in arxiv_patterns:
            m = re.search(pat, field, re.IGNORECASE)
            if m:
                arxiv_id = m.group(1)
                break
        if arxiv_id:
            break
    
    if not arxiv_id:
        return None
    
    _rate_limit("arxiv.org", 0.34)
    try:
        # Fetch BibTeX from arXiv
        resp = requests.get(
            f"https://arxiv.org/bibtex/{arxiv_id}",
            timeout=5,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            bibtex = resp.text
            # Parse title from BibTeX
            title_match = re.search(r'title\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
            title = title_match.group(1) if title_match else None
            if title:
                sim = _title_similarity(entry.title or "", title)
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="verified",
                    confidence=0.92,
                    matched_title=title,
                    open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    note=f"arXiv ID {arxiv_id} verified (match: {int(sim*100)}%)",
                    sources_checked=["arXiv (ID)"],
                )
    except Exception:
        pass
    return None


def _lookup_by_isbn(entry: BibEntry) -> Optional[VerificationResult]:
    """Direct ISBN lookup via Open Library."""
    if not entry.isbn:
        return None
    
    isbn_clean = re.sub(r'[\s-]', '', entry.isbn)
    _rate_limit("openlibrary.org", 0.5)
    try:
        resp = requests.get(
            f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn_clean}&format=json&jscmd=data",
            timeout=5,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            data = resp.json()
            key = f"ISBN:{isbn_clean}"
            if key in data:
                book = data[key]
                title = book.get("title", "")
                sim = _title_similarity(entry.title or "", title)
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="verified" if sim > 0.6 else "partial_match",
                    confidence=0.88 if sim > 0.6 else 0.5,
                    matched_title=title,
                    open_access_url=book.get("url"),
                    note=f"ISBN {isbn_clean} verified via Open Library",
                    sources_checked=["Open Library (ISBN)"],
                )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# PHASE 2: Title/Author Search
# ---------------------------------------------------------------------------

def _search_crossref(entry: BibEntry) -> Optional[VerificationResult]:
    """Search CrossRef by title and author."""
    if not entry.title:
        return None
    
    _rate_limit("crossref.org", 0.2)
    try:
        params = {"query.title": entry.title, "rows": 5}
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            params["query.author"] = first_author
        
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/6.2 (mailto:{mailto})" if mailto else "LNI-Checker/6.2"
        resp = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=6,
            headers={"User-Agent": ua}
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            for item in items[:3]:
                title = (item.get("title") or [""])[0]
                sim = _title_similarity(entry.title, title)
                if sim >= 0.5:
                    doi = item.get("DOI", "")
                    authors = item.get("author", [])
                    author_str = "; ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors[:3]]) if authors else None
                    
                    meta = _extract_corrected_metadata(item)
                    is_ret, ret_doi, ret_note = _check_retraction(doi) if doi else (False, None, None)
                    ret_note_full = f"⚠ RETRACTED. {ret_note}" if is_ret else f"Found on CrossRef (match: {int(sim*100)}%)"
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="retracted" if is_ret else ("verified" if sim >= 0.75 else "partial_match"),
                        confidence=1.0 if is_ret else sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=_check_unpaywall(doi) if doi else None,
                        note=ret_note_full,
                        sources_checked=["CrossRef"],
                        correct_authors=author_str,
                        is_retracted=is_ret,
                        retraction_doi=ret_doi,
                        retraction_note=ret_note,
                        corrected_title=title if title else None,
                        corrected_authors=meta["corrected_authors"],
                        corrected_year=meta["corrected_year"],
                        corrected_journal=meta["corrected_journal"],
                        corrected_publisher=meta["corrected_publisher"],
                        corrected_volume=meta["corrected_volume"],
                        corrected_pages=meta["corrected_pages"],
                    )
    except Exception:
        pass
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    """Search Semantic Scholar by title."""
    if not entry.title:
        return None
    
    _rate_limit("api.semanticscholar.org", 0.2)
    try:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"User-Agent": "LNI-Checker/6.2"}
        if api_key:
            headers["x-api-key"] = api_key
        
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": entry.title, "limit": 5, "fields": "title,authors,year,openAccessPdf,externalIds"},
            timeout=6,
            headers=headers
        )
        if resp.status_code == 200:
            papers = resp.json().get("data", [])
            for paper in papers[:3]:
                title = paper.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.5:
                    authors = paper.get("authors", [])
                    author_str = "; ".join([a.get("name", "") for a in authors[:3]]) if authors else None
                    oa = (paper.get("openAccessPdf") or {}).get("url")
                    doi = paper.get("externalIds", {}).get("DOI")
                    
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified" if sim >= 0.75 else "partial_match",
                        confidence=sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=oa,
                        note=f"Found on Semantic Scholar (match: {int(sim*100)}%)",
                        sources_checked=["Semantic Scholar"],
                        correct_authors=author_str,
                    )
    except Exception:
        pass
    return None


def _search_openalex(entry: BibEntry) -> Optional[VerificationResult]:
    """Search OpenAlex by title."""
    if not entry.title:
        return None
    
    _rate_limit("api.openalex.org", 0.2)
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": entry.title, "per-page": 5},
            timeout=6,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for work in results[:3]:
                title = work.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.5:
                    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
                    oa = work.get("open_access", {}).get("oa_url")
                    
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified" if sim >= 0.75 else "partial_match",
                        confidence=sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=oa,
                        note=f"Found on OpenAlex (match: {int(sim*100)}%)",
                        sources_checked=["OpenAlex"],
                    )
    except Exception:
        pass
    return None


def _search_dblp(entry: BibEntry) -> Optional[VerificationResult]:
    """Search DBLP by title."""
    if not entry.title:
        return None
    
    _rate_limit("dblp.org", 1.0)
    try:
        clean_title = re.sub(r'[^\w\s]', ' ', entry.title.lower()).strip()
        resp = requests.get(
            "https://dblp.org/search/publ/api",
            params={"q": clean_title, "format": "json", "h": 5},
            timeout=6,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            for hit in hits[:3]:
                info = hit.get("info", {})
                title = info.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.5:
                    doi = info.get("doi")
                    url = info.get("url")
                    
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified" if sim >= 0.75 else "partial_match",
                        confidence=sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=url,
                        note=f"Found on DBLP (match: {int(sim*100)}%)",
                        sources_checked=["DBLP"],
                    )
    except Exception:
        pass
    return None


def _search_acl(entry: BibEntry) -> Optional[VerificationResult]:
    """Search ACL Anthology (for NLP papers)."""
    if not entry.title:
        return None
    
    _rate_limit("aclanthology.org", 0.5)
    try:
        resp = requests.get(
            "https://aclanthology.org/search/",
            params={"q": entry.title},
            timeout=6,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            # Parse results from HTML
            titles = re.findall(r'<span class="d-block"[^>]*>(.*?)</span>', resp.text, re.DOTALL)
            for t in titles[:3]:
                title = re.sub(r'<[^>]+>', '', t).strip()
                sim = _title_similarity(entry.title, title)
                if sim >= 0.6:
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified" if sim >= 0.8 else "partial_match",
                        confidence=sim,
                        matched_title=title,
                        note=f"Found on ACL Anthology (match: {int(sim*100)}%)",
                        sources_checked=["ACL Anthology"],
                    )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# PHASE 3: Deep Search (only when needed)
# ---------------------------------------------------------------------------

def _search_arxiv_fallback(entry: BibEntry) -> Optional[VerificationResult]:
    """Fallback arXiv search by title."""
    if not entry.title:
        return None
    
    _rate_limit("export.arxiv.org", 0.34)
    try:
        import urllib.parse
        query = f'ti:"{urllib.parse.quote(entry.title)}"'
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            query += f' AND au:{urllib.parse.quote(first_author)}'
        
        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": 3},
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.2"}
        )
        if resp.status_code == 200:
            entries_xml = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
            for entry_xml in entries_xml[:2]:
                title_match = re.search(r'<title>(.*?)</title>', entry_xml, re.DOTALL)
                if title_match:
                    title = re.sub(r'\s+', ' ', title_match.group(1)).strip()
                    sim = _title_similarity(entry.title, title)
                    if sim >= 0.6:
                        pdf_match = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', entry_xml)
                        oa = pdf_match.group(1) if pdf_match else None
                        
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title,
                            status="verified" if sim >= 0.8 else "partial_match",
                            confidence=sim,
                            matched_title=title,
                            open_access_url=oa,
                            note=f"Found on arXiv (match: {int(sim*100)}%)",
                            sources_checked=["arXiv (search)"],
                        )
    except Exception:
        pass
    return None



def _search_openreview(entry: BibEntry) -> Optional[VerificationResult]:
    """Search OpenReview for ML/AI conference papers (NeurIPS, ICLR, ICML, etc.)."""
    if not entry.title:
        return None

    _rate_limit("openreview.net", 0.5)
    try:
        resp = requests.get(
            "https://api2.openreview.net/notes/search",
            params={"term": entry.title, "limit": 5, "offset": 0},
            timeout=7,
            headers={"User-Agent": "LNI-Checker/6.3"},
        )
        if resp.status_code != 200:
            return None
        notes = resp.json().get("notes", [])
        for note in notes[:5]:
            content = note.get("content", {})
            # content values may be dicts with a "value" key (OpenReview v2 schema)
            def _val(field):
                v = content.get(field, "")
                return v.get("value", "") if isinstance(v, dict) else (v or "")

            found_title = _val("title")
            if not found_title:
                continue
            sim = _title_similarity(entry.title, found_title)
            if sim >= 0.6:
                authors_list = _val("authors")
                if isinstance(authors_list, list):
                    author_str = "; ".join(str(a) for a in authors_list[:4])
                else:
                    author_str = str(authors_list)
                forum_id = note.get("forum", note.get("id", ""))
                oa_url = f"https://openreview.net/forum?id={forum_id}" if forum_id else None
                return VerificationResult(
                    key=entry.key,
                    title=entry.title,
                    status="verified" if sim >= 0.78 else "partial_match",
                    confidence=sim,
                    matched_title=found_title,
                    open_access_url=oa_url,
                    note=f"Found on OpenReview (match: {int(sim * 100)}%)",
                    sources_checked=["OpenReview"],
                    correct_authors=author_str,
                )
    except Exception:
        pass
    return None

def _search_google_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    """Google Scholar scrape fallback."""
    if not entry.title:
        return None
    
    _rate_limit("scholar.google.com", 2.0)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(
            "https://scholar.google.com/scholar",
            params={"q": entry.title, "hl": "en", "num": 3},
            timeout=8,
            headers=headers
        )
        if resp.status_code == 200:
            titles = re.findall(r'class="gs_rt"[^>]*>(?:<[^>]+>)*([^<]+)', resp.text)
            for title in titles[:2]:
                title_clean = re.sub(r'\[[^\]]+\]', '', title).strip()
                sim = _title_similarity(entry.title, title_clean)
                if sim >= 0.6:
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="partial_match",
                        confidence=sim,
                        matched_title=title_clean,
                        note=f"Found on Google Scholar (match: {int(sim*100)}%)",
                        sources_checked=["Google Scholar"],
                    )
    except Exception:
        pass
    return None


def _search_duckduckgo(entry: BibEntry) -> Optional[VerificationResult]:
    """DuckDuckGo web search as last resort."""
    if not entry.title:
        return None
    
    _rate_limit("duckduckgo.com", 1.0)
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f'"{entry.title}"'},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if resp.status_code == 200:
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            if snippets:
                combined = " ".join(snippets[:2]).lower()
                title_words = set(w for w in entry.title.lower().split() if len(w) > 3)
                matches = sum(1 for w in title_words if w in combined)
                coverage = matches / len(title_words) if title_words else 0
                
                if coverage >= 0.5:
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="partial_match" if coverage >= 0.6 else "not_found",
                        confidence=coverage,
                        web_evidence=snippets[0][:200],
                        note=f"Web search found evidence (coverage: {int(coverage*100)}%)",
                        sources_checked=["Web (DDG)"],
                    )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Website verification
# ---------------------------------------------------------------------------

def _verify_website(entry: BibEntry) -> VerificationResult:
    """Verify website/URL references."""
    url = entry.url or ""
    if not url:
        return VerificationResult(
            key=entry.key,
            title=entry.title or "(website)",
            status="error",
            confidence=0.0,
            note="No URL provided for website",
            sources_checked=[],
        )
    
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        _rate_limit(url.split("/")[2], 0.5)
        resp = requests.head(url, timeout=5, allow_redirects=True)
        
        if resp.status_code < 400:
            return VerificationResult(
                key=entry.key,
                title=entry.title or url,
                status="verified",
                confidence=0.95,
                open_access_url=url,
                note=f"URL reachable (HTTP {resp.status_code})",
                sources_checked=["URL check"],
            )
        else:
            return VerificationResult(
                key=entry.key,
                title=entry.title or url,
                status="not_found",
                confidence=0.0,
                note=f"URL returned HTTP {resp.status_code}",
                sources_checked=["URL check"],
            )
    except Exception as e:
        return VerificationResult(
            key=entry.key,
            title=entry.title or url,
            status="error",
            confidence=0.0,
            note=f"URL check failed: {str(e)[:100]}",
            sources_checked=["URL check"],
        )


# ---------------------------------------------------------------------------
# Unpaywall helper
# ---------------------------------------------------------------------------

def _check_unpaywall(doi: str) -> Optional[str]:
    """Check Unpaywall for open access version.

    Requires UNPAYWALL_EMAIL environment variable to be set to a real
    institutional or personal email address, as required by Unpaywall's
    polite pool policy (https://unpaywall.org/products/api).
    Returns None silently if the variable is not set — no fake email fallback.
    """
    email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
    if not email or email == "lni-checker@uni-project.de":
        # No valid email configured — skip rather than violating Unpaywall ToS
        return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=5)
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
# Main verification function - AGGREGATES ALL SOURCES
# ---------------------------------------------------------------------------

def verify_reference(entry: BibEntry) -> VerificationResult:
    """
    Verify a single reference by checking ALL available online sources.
    Returns aggregated result with highest confidence.
    """
    # Websites are handled separately
    if entry.entry_type == "website":
        return _verify_website(entry)
    
    # Check cache first
    cached = _get_cached(entry)
    if cached:
        result = copy.copy(cached)
        result.note = (result.note or "") + " [cached]"
        return result
    cached_paper = search_cache(entry.title or "", entry.authors or "")
    if cached_paper:
        sim = _title_similarity(entry.title or "", cached_paper.title)
        if sim >= 0.7:
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="verified",
                confidence=cached_paper.confidence,
                matched_title=cached_paper.title,
                doi=cached_paper.doi,
                open_access_url=cached_paper.url,
                note=f"Found in local cache (from {cached_paper.source})",
                sources_checked=["local_cache"],
                correct_authors=cached_paper.authors,
            )
        
        # Check professor review decisions (manual override)
    review = get_review_decision(entry.title or "", entry.authors or "")
    if review:
        if review.get("decision") == "verified":
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="verified",
                confidence=0.99,
                matched_title=entry.title,
                doi=review.get("verified_doi"),
                open_access_url=review.get("verified_url"),
                note=f"Professor verified: {review.get('professor_note', 'Manually approved')}",
                sources_checked=["professor_review"],
                correct_authors=entry.authors,
            )
        elif review.get("decision") == "rejected":
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="not_found",
                confidence=0.0,
                note=f"Professor marked as rejected: {review.get('professor_note', '')}",
                sources_checked=["professor_review"],
            )
    
    # Check professor false-positive corrections (paper was flagged but professor said REAL)
    fp_record = get_false_positive(entry.title or "", entry.authors or "")
    if fp_record:
        return VerificationResult(
            key=entry.key,
            title=entry.title or "",
            status="verified",
            confidence=0.95,
            matched_title=entry.title,
            note=f"Professor previously corrected this as REAL: {fp_record.get('notes', '')}",
            sources_checked=["professor_false_positive_correction"],
            correct_authors=entry.authors,
        )

    # Check if venue is whitelisted (German conference = don't penalize)
    venue = entry.journal or entry.booktitle or entry.publisher or ""
    whitelist_check = is_venue_whitelisted(venue)
    if whitelist_check.get("whitelisted"):
        # Still verify, but don't mark as FAKE
        pass  # This just flags to AI that venue is trusted
        
    
    all_results: List[VerificationResult] = []
    
    # PHASE 1: Fast identifier lookups (run in parallel)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        futures.append(executor.submit(_lookup_by_doi, entry))
        futures.append(executor.submit(_lookup_by_arxiv_id, entry))
        futures.append(executor.submit(_lookup_by_isbn, entry))
        
        for future in as_completed(futures, timeout=5):
            try:
                r = future.result()
                if r:
                    all_results.append(r)
            except Exception:
                pass
    
    # PHASE 2: Title/author search (run in parallel)
    if not any(r.status == "verified" for r in all_results):
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            futures.append(executor.submit(_search_crossref, entry))
            futures.append(executor.submit(_search_semantic_scholar, entry))
            futures.append(executor.submit(_search_openalex, entry))
            futures.append(executor.submit(_search_dblp, entry))
            futures.append(executor.submit(_search_acl, entry))
            
            for future in as_completed(futures, timeout=7):
                try:
                    r = future.result()
                    if r:
                        all_results.append(r)
                except Exception:
                    pass
    
    # PHASE 3: Deep search (only if no good match found yet)
    best_so_far = max(all_results, key=lambda r: r.confidence, default=None) if all_results else None
    if not best_so_far or best_so_far.confidence < 0.6:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.append(executor.submit(_search_arxiv_fallback, entry))
            futures.append(executor.submit(_search_openreview, entry))   # NeurIPS/ICLR/ICML
            futures.append(executor.submit(_search_google_scholar, entry))
            futures.append(executor.submit(_search_duckduckgo, entry))

            for future in as_completed(futures, timeout=12):
                try:
                    r = future.result()
                    if r:
                        all_results.append(r)
                except Exception:
                    pass
    
    # =========================================================
    # Web search + LLM fallback — only fires when composite signals
    # suggest genuinely ambiguous / SUSPICIOUS entries, not every miss.
    # This avoids burning Groq/Gemini quota on clearly-absent papers.
    # =========================================================
    best_after_apis = max(all_results, key=lambda r: r.confidence, default=None) if all_results else None
    _run_web_search = not all_results  # definitely run if all APIs returned nothing

    if not _run_web_search and best_after_apis and best_after_apis.confidence < 0.5:
        # APIs found something but low-confidence — worth a web cross-check
        _run_web_search = True

    if _run_web_search:
        try:
            web_result = verify_with_web_search(
                {"title": entry.title, "authors": entry.authors, "year": entry.year},
                best_after_apis.status if best_after_apis else "not_found"
            )
            
            if web_result.get("status") == "verified":
                # Save to local cache BEFORE returning
                save_to_cache(
                    title=entry.title or "",
                    authors=entry.authors or "",
                    year=entry.year or "",
                    doi=web_result.get("doi", ""),
                    url=web_result.get("open_access_url", ""),
                    source="web_search",
                    confidence=web_result.get("confidence", 0.8)
                )
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="verified",
                    confidence=web_result.get("confidence", 0.8),
                    matched_title=web_result.get("matched_title"),
                    open_access_url=web_result.get("open_access_url"),
                    note=web_result.get("note", "Verified via web search"),
                    sources_checked=["web_search", "llm_verification"]
                )
            else:
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="not_found",
                    confidence=web_result.get("confidence", 0.0),
                    note=web_result.get("note", "No results found in any academic database or web search"),
                    sources_checked=["api_phases_1_2_3", "web_search_attempted"]
                )
        except Exception as e:
            # Fallback to original behavior if web search fails
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="not_found",
                confidence=0.0,
                note=f"No results found in any academic database (web search error: {str(e)[:50]})",
                sources_checked=[],
            )
    
    # Aggregate results - find the best one
    priority = {"verified": 3, "partial_match": 2, "not_found": 1, "error": 0}
    all_results.sort(key=lambda r: (priority.get(r.status, 0), r.confidence), reverse=True)
    best = all_results[0]
    
    # Aggregate all sources checked
    all_sources = []
    for r in all_results:
        for src in r.sources_checked:
            if src not in all_sources:
                all_sources.append(src)
    best.sources_checked = all_sources
    
    # Collect additional evidence
    for r in all_results:
        if not best.web_evidence and r.web_evidence:
            best.web_evidence = r.web_evidence
        if not best.correct_authors and r.correct_authors:
            best.correct_authors = r.correct_authors
        if not best.doi and r.doi:
            best.doi = r.doi
        if not best.open_access_url and r.open_access_url:
            best.open_access_url = r.open_access_url
        # Propagate retraction flag from any source
        if r.is_retracted:
            best.is_retracted = True
            best.retraction_doi = best.retraction_doi or r.retraction_doi
            best.retraction_note = best.retraction_note or r.retraction_note
            best.status = "retracted"
        # Propagate corrected metadata (prefer CrossRef over others)
        if not best.corrected_title and r.corrected_title:
            best.corrected_title = r.corrected_title
        if not best.corrected_authors and r.corrected_authors:
            best.corrected_authors = r.corrected_authors
        if not best.corrected_year and r.corrected_year:
            best.corrected_year = r.corrected_year
        if not best.corrected_journal and r.corrected_journal:
            best.corrected_journal = r.corrected_journal
        if not best.corrected_publisher and r.corrected_publisher:
            best.corrected_publisher = r.corrected_publisher
        if not best.corrected_volume and r.corrected_volume:
            best.corrected_volume = r.corrected_volume
        if not best.corrected_pages and r.corrected_pages:
            best.corrected_pages = r.corrected_pages
    
    # Boost confidence if multiple sources agree
       
    verified_count = sum(1 for r in all_results if r.status == "verified")
    if verified_count >= 2 and best.status == "verified":
        best.confidence = min(best.confidence + 0.08, 0.98)
        best.note = f"Confirmed by {verified_count} independent sources. {best.note}"
    
    # Save to local cache if verified
    if best.status == "verified":
        save_to_cache(
            title=entry.title or "",
            authors=entry.authors or "",
            year=entry.year or "",
            doi=best.doi or "",
            url=best.open_access_url or "",
            source="api",
            confidence=best.confidence
        )
    
    # Cache the result
    _put_cache(entry, best)
    return best


def verify_all_references(bib_entries: dict) -> list:
    """Verify all entries concurrently. Returns results in original order."""
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_key = {executor.submit(verify_reference, entry): key for key, entry in bib_entries.items()}
        for future in as_completed(future_to_key, timeout=120):
            try:
                results.append(future.result())
            except Exception as e:
                key = future_to_key[future]
                results.append(VerificationResult(
                    key=key, title="", status="error", confidence=0.0,
                    note=f"Verification crashed: {str(e)[:100]}", sources_checked=[]
                ))
    
    # Return in original order
    key_order = list(bib_entries.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)
    return results


# ---------------------------------------------------------------------------
# In-text citations extraction
# ---------------------------------------------------------------------------

def extract_citations_from_body(body_text: str) -> set:
    keys = set()
    
    # 1. LNI format: [ABC01], [Vas17], [Dev19], etc.
    lni_matches = re.findall(
        r'\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]', 
        body_text
    )
    for match in lni_matches:
        for key in re.split(r',\s*', match):
            key = key.strip()
            if re.match(r'^[A-Za-z]{2,6}\d{2}[a-z]?$', key):
                keys.add(key)
    
    # 2. Numeric format: [1], [2], [3], etc.
    numeric_matches = re.findall(r'\[(\d{1,3})\]', body_text)
    if numeric_matches:
        # Add a special marker so we know numeric citations exist
        keys.add('__numeric_citations__')
        # Also add each number as a string for potential matching
        for num in numeric_matches:
            keys.add(f'__NUM_{num}__')
    
    return keys


def extract_citation_contexts(body_text: str) -> dict:
    contexts = {}
    # Match BOTH LNI and numeric citations
    for m in re.finditer(
        r'([^.]{0,80})\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*|\d+(?:,\s*\d+)*)\]([^.]{0,80})',
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
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        name = name.replace(a, b)
    return name


def detect_self_citations(bib_entries: dict, body_text: str) -> list:
    candidates = {_norm_name(c) for c in re.findall(r'\b([A-ZÄÖÜ][a-zäöüß]{3,})\b', body_text)}
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


@dataclass
class CrossCheckResult:
    cited_not_in_bib: list = field(default_factory=list)
    in_bib_not_cited: list = field(default_factory=list)
    correctly_used: list = field(default_factory=list)


def cross_check(bib_entries: dict, cited_keys: set) -> CrossCheckResult:
    """Cross-check in-text citations against bibliography entries.

    Handles two citation styles independently:
      - LNI author-year keys  [ABC01]  — exact string match
      - Numeric keys          [1],[2]  — count-based match only
        (numeric keys from the body text will NOT be compared as strings against
        author-year bib keys, which would cause false "cited but missing" alarms)
    """
    bib_keys = set(str(k) for k in bib_entries.keys())

    # Separate LNI keys from numeric markers
    lni_cited: set = set()
    numeric_cited_numbers: set = set()
    has_numeric = False

    for k in cited_keys:
        k_str = str(k)
        if k_str == '__numeric_citations__':
            has_numeric = True
        elif k_str.startswith('__NUM_'):
            num = k_str.replace('__NUM_', '').replace('__', '')
            numeric_cited_numbers.add(num)
        else:
            lni_cited.add(k_str)

    # Determine if bib keys are numeric style too
    bib_numeric_keys = {k for k in bib_keys if re.match(r'^[0-9]+$', k)}
    bib_lni_keys = bib_keys - bib_numeric_keys

    r = CrossCheckResult()

    if has_numeric and bib_numeric_keys:
        # Both sides are numeric — do exact numeric matching
        r.cited_not_in_bib = sorted(numeric_cited_numbers - bib_numeric_keys)
        r.in_bib_not_cited = sorted(bib_numeric_keys - numeric_cited_numbers)
        r.correctly_used = sorted(numeric_cited_numbers & bib_numeric_keys)

    elif has_numeric and not bib_numeric_keys:
        # Body uses numeric [1][2] but bib uses LNI keys — count-only validation.
        # We cannot do per-key matching, so we only flag count mismatches.
        n_cited = len(numeric_cited_numbers)
        n_bib = len(bib_lni_keys)
        r.cited_not_in_bib = []   # cannot determine which specific key is missing
        r.in_bib_not_cited = []
        r.correctly_used = list(bib_lni_keys)  # treat all bib entries as matched
        if n_cited > n_bib:
            # More citations than bib entries — surface as a single synthetic warning
            r.cited_not_in_bib = [f"__NUMERIC_OVERFLOW__ ({n_cited} citations > {n_bib} bib entries)"]
        elif n_bib > n_cited + 2:
            # Noticeably more bib entries than citations
            r.in_bib_not_cited = [f"__NUMERIC_UNDERREF__ ({n_bib} bib entries, only {n_cited} numeric citations)"]

    else:
        # Standard LNI author-year matching
        r.cited_not_in_bib = sorted(lni_cited - bib_lni_keys)
        r.in_bib_not_cited = sorted(bib_lni_keys - lni_cited)
        r.correctly_used = sorted(lni_cited & bib_lni_keys)

    return r


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
                dupes.append({
                    "key_a": a.key, "key_b": b.key,
                    "title_a": a.title, "title_b": b.title,
                    "similarity": round(score, 2),
                    "type": "exact" if score >= 0.97 else "near-duplicate"
                })
    return dupes


def check_lni_macros(body_text: str) -> list:
    suggestions = []
    for pattern, message in [
        (r'\be\.g\.', r"Use LNI macro '\eg' instead of 'e.g.'"),
        (r'\bi\.e\.', r"Use LNI macro '\ie' instead of 'i.e.'"),
        (r'\bcf\.', r"Use LNI macro '\cf' instead of 'cf.'"),
        (r'\bet al\.', r"Use LNI macro '\etal' instead of 'et al.'"),
    ]:
        n = len(re.findall(pattern, body_text, re.IGNORECASE))
        if n:
            suggestions.append({"type": "Macro", "message": message, "count": n})
    
    ac = re.findall(r'\n([A-Z]{4,}(?:\s+[A-Z]{2,})*)\n', body_text)
    if ac:
        suggestions.append({"type": "Heading", "message": f"Found {len(ac)} ALL-CAPS heading(s) — LNI uses sentence case.", "count": len(ac)})
    
    n = len(re.findall(r'\\textbf\{[^}]{1,20}\}', body_text))
    if n:
        suggestions.append({"type": "Emphasis", "message": r"Manual \textbf{} — prefer LNI semantic macros.", "count": n})
    
    n = len(re.findall(r'(?<!`)"[^"]{1,60}"', body_text))
    if n:
        suggestions.append({"type": "Quotes", "message": r'Straight quotes (") — use ``...'"''"' or \\enquote{}.', "count": n})
    
    return suggestions


def compute_score(bib_list, xcheck, verification_results, style_suggestions, duplicates,
                  ai_fake_count=0, retracted_count=0, year_mismatches=0, author_mismatches=0):
    """Compute a 0-100 score with detailed per-category penalty breakdown.

    Penalty table (per-item deduction, max cap):
      Retracted citation          -25  (cap 50)  — citing retracted work, very serious
      Missing from bibliography   -10  (cap 30)  — in-text key has no bib entry
      Likely fabricated ref       -15  (cap 45)  — AI flagged as hallucinated
      Year mismatch               - 8  (cap 24)  — cited year ≠ CrossRef year by >1
      Author mismatch             - 5  (cap 20)  — cited authors don't match DB
      Cited nowhere in text       - 5  (cap 20)  — orphan bib entry
      Incomplete LNI entry        - 5  (cap 20)  — missing required fields
      Duplicate entries           - 5  (cap 10)  — same paper listed twice
      LNI style violations        - 2  (cap  6)  — macro / formatting issues
    """
    score, penalties = 100, []

    rows = [
        ("Retracted papers cited",      retracted_count,      25, 50),
        ("Missing from bibliography",   len(xcheck.cited_not_in_bib), 10, 30),
        ("Likely fabricated references",ai_fake_count,        15, 45),
        ("Year mismatch (>1 yr off)",   year_mismatches,       8, 24),
        ("Author mismatch",             author_mismatches,     5, 20),
        ("Cited nowhere in text",       len(xcheck.in_bib_not_cited), 5, 20),
        ("Incomplete LNI entries",      sum(1 for e in bib_list if e.completeness_issues), 5, 20),
        ("Duplicate entries",           len(duplicates),       5, 10),
        ("LNI style violations",        len(style_suggestions),2,  6),
    ]
    for label, count, per_item, cap in rows:
        p = min(count * per_item, cap)
        if p:
            penalties.append({"category": label, "count": count,
                              "deduction": p, "per_item": per_item, "cap": cap})
        score -= p

    score = max(score, 0)
    if score >= 95:   grade = "A+"
    elif score >= 90: grade = "A"
    elif score >= 85: grade = "A-"
    elif score >= 80: grade = "B+"
    elif score >= 75: grade = "B"
    elif score >= 70: grade = "B-"
    elif score >= 65: grade = "C+"
    elif score >= 60: grade = "C"
    elif score >= 55: grade = "C-"
    elif score >= 50: grade = "D"
    else:             grade = "F"

    return {"score": score, "grade": grade, "penalties": penalties, "max_score": 100}