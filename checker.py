"""
STEP 3: Citation Cross-Checker + Reference Verifier — v8.7
----------------------------------------------------------
FIXED v8.7:
  - Grey literature now properly calls AI verification instead of immediately returning SUSPICIOUS
  - Fixed URL repair for broken PDF extraction
  - Better fallback for web search
  - Proper AI integration for ALL entries including grey literature
"""

import hashlib
import json
import os
import re
import sys
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ai_checker import _ai_available
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from parser import BibEntry, _extract_surnames
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from local_db import search_cache, save_to_cache, get_cache_stats, init_cache_db
from web_search_verifier import verify_with_web_search
from review_queue import is_venue_whitelisted, get_review_decision, get_false_positive
from ai_checker import _is_grey_literature, _is_fabricated_title

# ---------------------------------------------------------------------------
# Configurable verification thresholds (can be overridden via environment)
# ---------------------------------------------------------------------------
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("LNI_TITLE_SIM_THRESHOLD", "0.80"))
AUTHOR_OVERLAP_THRESHOLD = float(os.getenv("LNI_AUTHOR_OVERLAP_THRESHOLD", "0.70"))
AUTHOR_MISMATCH_THRESHOLD = float(os.getenv("LNI_AUTHOR_MISMATCH_THRESHOLD", "0.50"))
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("LNI_CONFIDENCE_HIGH", "0.85"))

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

_MEM_CACHE: Dict[str, "VerificationResult"] = {}
_MEM_CACHE_LOCK = threading.Lock()
_ARXIV_BIBTEX_MEM_CACHE: Dict[str, str] = {}
_ARXIV_CACHE_LOCK = threading.Lock()

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
# Title normalisation + similarity
# ---------------------------------------------------------------------------

def _normalize_title(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    for src, dst in [
        ('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss'),
        ('à','a'),('á','a'),('â','a'),('ã','a'),
        ('è','e'),('é','e'),('ê','e'),('ë','e'),
        ('ì','i'),('í','i'),('î','i'),('ï','i'),
        ('ò','o'),('ó','o'),('ô','o'),('õ','o'),
        ('ù','u'),('ú','u'),('û','u'),
        ('ý','y'),('ÿ','y'),('ñ','n'),('ç','c'),
        ('ø','o'),('å','aa'),('æ','ae'),('œ','oe'),
    ]:
        t = t.replace(src, dst)
    t = re.sub(r'&[a-z]+;', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'[{}]', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    stop = {
        'the','a','an','in','of','for','on','and','to','with','its',
        'using','based','towards','toward','via','approach',
        'der','die','das','und','fur','fuer','von','mit','im','an',
        'zu','zur','zum','eine','ein','des','dem','den',
        'is','are','was','were','be','by','at','or','not',
    }
    return ' '.join(w for w in t.split() if w not in stop and len(w) > 2)


def _title_similarity(title1: str, title2: str) -> float:
    if not title1 or not title2:
        return 0.0
    t1, t2 = _normalize_title(title1), _normalize_title(title2)
    if not t1 or not t2:
        return 0.0
    try:
        from rapidfuzz.fuzz import token_sort_ratio, partial_ratio
        score_a = token_sort_ratio(t1, t2) / 100.0
        score_b = partial_ratio(t1[:120], t2[:120]) / 100.0
        fuzzy = max(score_a, score_b)
    except ImportError:
        from difflib import SequenceMatcher
        fuzzy = SequenceMatcher(None, t1, t2).ratio()
    words1, words2 = set(t1.split()), set(t2.split())
    sig1 = {w for w in words1 if len(w) >= 5}
    sig2 = {w for w in words2 if len(w) >= 5}
    overlap = len(sig1 & sig2) / max(len(sig1), len(sig2)) if sig1 and sig2 else 0.0
    return round(min(0.75 * fuzzy + 0.25 * overlap, 1.0), 4)


# ---------------------------------------------------------------------------
# Author overlap with umlaut tolerance and prefix matching
# ---------------------------------------------------------------------------

def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    """
    Calculate author surname overlap with:
    - Umlaut tolerance (Müller matches Muller, Mueller)
    - Prefix matching (Schmidt matches Schmid)
    - Single-author leniency
    """
    if not cited_authors or not correct_authors:
        return None
    
    def _normalize_surname(s: str) -> tuple:
        """Normalize surname for matching with umlaut tolerance.
        Returns (full_normalized, collapsed_umlaut) for multiple matching strategies."""
        s = s.lower()
        # Convert umlauts to 'ue'/'ae'/'oe' forms
        for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
                     ('à', 'a'), ('á', 'a'), ('â', 'a'), ('ã', 'a'),
                     ('è', 'e'), ('é', 'e'), ('ê', 'e'), ('ë', 'e'),
                     ('ì', 'i'), ('í', 'i'), ('î', 'i'), ('ï', 'i'),
                     ('ò', 'o'), ('ó', 'o'), ('ô', 'o'), ('õ', 'o'),
                     ('ù', 'u'), ('ú', 'u'), ('û', 'u'),
                     ('ý', 'y'), ('ÿ', 'y'), ('ñ', 'n'), ('ç', 'c')]:
            s = s.replace(a, b)
        
        # Create collapsed version that drops the 'e' from umlaut forms
        # so 'mueller' matches 'muller' and 'mueller' matches 'muller'
        collapsed = s
        for u_umlaut, u_dropped in [('ue', 'u'), ('ae', 'a'), ('oe', 'o')]:
            collapsed = collapsed.replace(u_umlaut, u_dropped)
        
        # Remove non-alphanumeric
        clean = re.sub(r'[^a-z0-9]', '', s)
        clean_collapsed = re.sub(r'[^a-z0-9]', '', collapsed)
        
        return clean, clean_collapsed
    
    def _extract_surnames_with_normalization(authors_str: str) -> set:
        """Extract surnames with both normal forms."""
        surnames = set()
        for part in re.split(r';|\band\b|\bund\b', authors_str, flags=re.IGNORECASE):
            part = part.strip()
            if not part:
                continue
            if re.match(r'^et\s+al\.?$', part.lower()):
                continue
            if ',' in part:
                surname_part = part.split(',')[0].strip()
            else:
                tokens = part.lower().split()
                particles = {'von', 'van', 'de', 'del', 'della', 'der', 'la', 'le', 'du', 'des', 'di'}
                non_particle = [t for t in tokens if t not in particles and not re.match(r'^[a-z]\.?$', t)]
                surname_part = non_particle[-1] if non_particle else (tokens[-1] if tokens else '')
            clean, collapsed = _normalize_surname(surname_part)
            if len(clean) > 2:
                surnames.add(clean)
                surnames.add(collapsed)
        return surnames
    
    cited = _extract_surnames_with_normalization(cited_authors)
    correct = _extract_surnames_with_normalization(correct_authors)
    
    if not cited or not correct:
        return None
    
    # Calculate match: full match OR prefix match (first 3-5 chars)
    matches = 0.0
    cited_list = list(cited)[:6]
    
    for s in cited_list:
        matched = False
        match_weight = 1.0
        
        for c in correct:
            # Exact match
            if s == c:
                matched = True
                break
            
            # Prefix match on first 5 chars
            if len(s) >= 5 and len(c) >= 5 and s[:5] == c[:5]:
                matched = True
                break
            
            # Prefix match on first 4 chars
            if len(s) >= 4 and len(c) >= 4 and s[:4] == c[:4]:
                matched = True
                match_weight = 0.9
                break
            
            # Prefix match on first 3 chars (for short surnames like 'Mül' vs 'Mul')
            if len(s) >= 3 and len(c) >= 3 and s[:3] == c[:3]:
                matched = True
                match_weight = 0.8
                break
        
        if matched:
            matches += match_weight
        else:
            # Partial match: first character only (for umlaut cases)
            if s and any(c and s[0] == c[0] for c in correct):
                matches += 0.5
    
    # For single-author entries, be more lenient
    if len(cited_list) == 1:
        # If we have any first-char match, boost to 0.7
        first_char_match = any(s and any(c and s[0] == c[0] for c in correct) for s in cited_list)
        if first_char_match and matches < 0.7:
            matches = 0.7
    
    return round(matches / min(len(cited_list), 6), 3)


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str          # "verified" | "suspicious" | "not_checked" | "error"
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
    corrected_title: Optional[str] = None
    corrected_authors: Optional[str] = None
    corrected_year: Optional[str] = None
    corrected_publisher: Optional[str] = None
    corrected_journal: Optional[str] = None
    corrected_volume: Optional[str] = None
    corrected_pages: Optional[str] = None
    title_match_score: Optional[float] = None
    author_match_score: Optional[float] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None


# ---------------------------------------------------------------------------
# Retraction check
# ---------------------------------------------------------------------------

def _check_retraction(doi: str) -> tuple:
    if not doi:
        return False, None, None
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/8.1 (mailto:{mailto})" if mailto else "LNI-Checker/8.1"
        resp = requests.get(f"https://api.crossref.org/works/{doi}",
                            timeout=5, headers={"User-Agent": ua})
        if resp.status_code != 200:
            return False, None, None
        work = resp.json().get("message", {})
        for u in work.get("update-to", []):
            if u.get("type", "").lower() == "retraction":
                ret_doi = u.get("DOI", "")
                parts = u.get("updated", {}).get("date-parts", [[""]])[0]
                date_str = "-".join(str(p) for p in parts if p) if parts else "unknown"
                return True, ret_doi, f"Retracted {date_str}. Retraction DOI: {ret_doi}"
    except Exception:
        pass
    return False, None, None


def _extract_corrected_metadata(work: dict) -> dict:
    authors = work.get("author", [])
    author_str = "; ".join(
        f"{a.get('family','')}, {a.get('given','')}" for a in authors[:5]
    ) if authors else None
    issued = work.get("issued", {}).get("date-parts", [[None]])[0]
    year = str(issued[0]) if issued and issued[0] else None
    container = (work.get("container-title") or [""])[0]
    return {
        "corrected_authors": author_str,
        "corrected_year": year,
        "corrected_journal": container or None,
        "corrected_publisher": work.get("publisher") or None,
        "corrected_volume": str(work.get("volume","")) or None,
        "corrected_pages": str(work.get("page","")) or None,
    }


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------

def _check_unpaywall(doi: str) -> Optional[str]:
    if not doi:
        return None
    mailto = os.environ.get("UNPAYWALL_EMAIL",
                            os.environ.get("CROSSREF_MAILTO","")).strip()
    if not mailto:
        return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                            params={"email": mailto}, timeout=5,
                            headers={"User-Agent": "LNI-Checker/8.1"})
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Academic API lookups
# ---------------------------------------------------------------------------

def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.doi:
        return None
    _rate_limit("crossref.org", 0.2)
    try:
        mailto = os.environ.get("CROSSREF_MAILTO","").strip()
        ua = f"LNI-Checker/8.1 (mailto:{mailto})" if mailto else "LNI-Checker/8.1"
        resp = requests.get(f"https://api.crossref.org/works/{entry.doi}",
                            timeout=5, headers={"User-Agent": ua})
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            if sim >= 0.75:
                meta = _extract_corrected_metadata(work)
                is_ret, ret_doi, ret_note = _check_retraction(entry.doi)
                author_sim = author_overlap_score(entry.authors or "", meta["corrected_authors"] or "") if meta["corrected_authors"] else None
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=sim,
                    matched_title=title, doi=entry.doi,
                    open_access_url=_check_unpaywall(entry.doi),
                    note=f"DOI verified via CrossRef ({int(sim*100)}%)",
                    sources_checked=["CrossRef (DOI)"],
                    correct_authors=meta["corrected_authors"],
                    is_retracted=is_ret, retraction_doi=ret_doi, retraction_note=ret_note,
                    corrected_title=title, corrected_authors=meta["corrected_authors"],
                    corrected_year=meta["corrected_year"],
                    corrected_journal=meta["corrected_journal"],
                    title_match_score=round(sim, 4),
                    author_match_score=round(author_sim, 4) if author_sim is not None else None,
                )
    except Exception:
        pass
    return None


def _lookup_by_arxiv_id(entry: BibEntry) -> Optional[VerificationResult]:
    arxiv_patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})', r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})', r'arXiv:([a-z\-]+/\d{7})',
    ]
    arxiv_id = None
    for field_val in [entry.url or "", entry.doi or "", entry.raw_text or ""]:
        for pat in arxiv_patterns:
            m = re.search(pat, field_val, re.IGNORECASE)
            if m:
                arxiv_id = m.group(1)
                break
        if arxiv_id:
            break
    if not arxiv_id:
        return None
    _rate_limit("arxiv.org", 0.34)
    try:
        resp = requests.get(f"https://arxiv.org/bibtex/{arxiv_id}",
                            timeout=5, headers={"User-Agent": "LNI-Checker/8.1"})
        if resp.status_code == 200:
            m = re.search(r'title\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
            title = m.group(1) if m else None
            if title and _title_similarity(entry.title or "", title) >= 0.75:
                sim = _title_similarity(entry.title or "", title)
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=sim,
                    matched_title=title,
                    open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    note=f"arXiv ID {arxiv_id} verified",
                    sources_checked=["arXiv (ID)"],
                )
    except Exception:
        pass
    return None


def _search_crossref(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None

    _rate_limit("crossref.org", 0.2)
    params = {"query.title": entry.title, "rows": 5}
    if entry.authors:
        first_author = entry.authors.split(';')[0].split(',')[0].strip()
        first_author = re.sub(r'\s+et\s+al\.?$', '', first_author, flags=re.IGNORECASE)
        if first_author and len(first_author) > 2:
            params["query.author"] = first_author

    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    ua = f"LNI-Checker/8.1 (mailto:{mailto})" if mailto else "LNI-Checker/8.1"

    for attempt in range(2):
        try:
            resp = requests.get("https://api.crossref.org/works",
                                params=params, timeout=12,
                                headers={"User-Agent": ua})
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                for item in items[:5]:
                    title = (item.get("title") or [""])[0]
                    if not title:
                        continue
                    sim = _title_similarity(entry.title, title)
                    if sim >= 0.75:
                        doi = item.get("DOI", "")
                        meta = _extract_corrected_metadata(item)
                        authors = item.get("author", [])
                        author_str = "; ".join(
                            f"{a.get('family','')}, {a.get('given','')}"
                            for a in authors[:3]
                        ) if authors else None
                        author_sim = author_overlap_score(entry.authors or "", meta["corrected_authors"] or "") if meta["corrected_authors"] else None
                        return VerificationResult(
                            key=entry.key, title=entry.title,
                            status="verified", confidence=sim,
                            matched_title=title, doi=doi,
                            open_access_url=_check_unpaywall(doi) if doi else None,
                            note=f"CrossRef match ({int(sim*100)}%)",
                            sources_checked=["CrossRef"],
                            correct_authors=author_str,
                            corrected_title=title,
                            corrected_authors=meta["corrected_authors"],
                            corrected_year=meta["corrected_year"],
                            corrected_journal=meta["corrected_journal"],
                            title_match_score=round(sim, 4),
                            author_match_score=round(author_sim, 4) if author_sim is not None else None,
                        )
                return None
            elif resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"CrossRef {resp.status_code} for {entry.key}")
                return None
        except requests.exceptions.Timeout:
            print(f"CrossRef timeout for {entry.key}")
            return None
        except Exception as e:
            print(f"CrossRef error for {entry.key}: {e}")
            return None
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    
    clean_title = entry.title
    for pattern in [r'\.\s*In:\s*.*$', r'\.\s*doi:\s*.*$', r'\.\s*https?://\S+$']:
        clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.strip().strip('.,;:')
    
    if not clean_title:
        clean_title = entry.title
    
    _rate_limit("api.semanticscholar.org", 0.25)
    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "LNI-Checker/8.1"}
    if ss_key:
        headers["x-api-key"] = ss_key
    
    search_queries = [
        clean_title,
        f'"{clean_title}"',
        clean_title[:100] if len(clean_title) > 100 else clean_title,
    ]
    
    if entry.authors:
        first_author = entry.authors.split(';')[0].split(',')[0].strip()
        first_author = re.sub(r'\s+et\s+al\.?$', '', first_author, flags=re.IGNORECASE)
        if first_author and len(first_author) > 2:
            search_queries.append(f'{clean_title} {first_author}')
    
    for attempt in range(2):
        for query in search_queries[:2]:
            try:
                resp = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": query[:200], "limit": 5,
                            "fields": "title,authors,year,openAccessPdf,externalIds"},
                    timeout=10, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for paper in data[:5]:
                        title = paper.get("title", "")
                        if not title:
                            continue
                        sim = _title_similarity(entry.title, title)
                        if sim >= 0.75:
                            authors = paper.get("authors", [])
                            author_str = "; ".join(
                                a.get("name", "") for a in authors[:3]
                            ) if authors else None
                            oa = (paper.get("openAccessPdf") or {}).get("url")
                            doi = paper.get("externalIds", {}).get("DOI")
                            author_sim = author_overlap_score(entry.authors or "", author_str or "") if author_str else None
                            return VerificationResult(
                                key=entry.key, title=entry.title,
                                status="verified", confidence=sim,
                                matched_title=title, doi=doi,
                                open_access_url=oa,
                                note=f"Semantic Scholar match ({int(sim*100)}%)",
                                sources_checked=["Semantic Scholar"],
                                correct_authors=author_str,
                                title_match_score=round(sim, 4),
                                author_match_score=round(author_sim, 4) if author_sim is not None else None,
                            )
                    for paper in data[:3]:
                        title = paper.get("title", "")
                        if title:
                            sim = _title_similarity(entry.title, title)
                            if sim >= 0.50:
                                authors = paper.get("authors", [])
                                author_str = "; ".join(
                                    a.get("name", "") for a in authors[:3]
                                ) if authors else None
                                oa = (paper.get("openAccessPdf") or {}).get("url")
                                doi = paper.get("externalIds", {}).get("DOI")
                                return VerificationResult(
                                    key=entry.key, title=entry.title,
                                    status="partial_match",
                                    confidence=sim,
                                    matched_title=title,
                                    doi=doi,
                                    open_access_url=oa,
                                    note=f"Partial Semantic Scholar match ({int(sim*100)}%)",
                                    sources_checked=["Semantic Scholar"],
                                    correct_authors=author_str,
                                    title_match_score=round(sim, 4),
                                )
                elif resp.status_code in (429, 503):
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Semantic Scholar {resp.status_code} for {entry.key}")
            except requests.exceptions.Timeout:
                print(f"Semantic Scholar timeout for {entry.key}")
            except Exception as e:
                print(f"Semantic Scholar error for {entry.key}: {e}")
    
    return None


_OPENALEX_SESSION: Optional[requests.Session] = None
_OPENALEX_SESSION_LOCK = threading.Lock()


def _get_openalex_session() -> requests.Session:
    global _OPENALEX_SESSION
    if _OPENALEX_SESSION is not None:
        return _OPENALEX_SESSION
    with _OPENALEX_SESSION_LOCK:
        if _OPENALEX_SESSION is None:
            s = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                connect=3,
                read=3,
                allowed_methods=frozenset(["GET"]),
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            _OPENALEX_SESSION = s
    return _OPENALEX_SESSION


def _search_openalex(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    
    clean_title = entry.title
    for pattern in [r'\.\s*In:\s*.*$', r'\.\s*doi:\s*.*$']:
        clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.strip().strip('.,;:')
    
    if not clean_title:
        clean_title = entry.title
    
    _rate_limit("api.openalex.org", 0.1)
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    params = {"search": clean_title[:200], "per-page": 5}
    if mailto:
        params["mailto"] = mailto
    
    try:
        session = _get_openalex_session()
        resp = session.get(
            "https://api.openalex.org/works",
            params=params, timeout=15,
            headers={"User-Agent": "LNI-Checker/8.1"})
        if resp.status_code != 200:
            print(f"OpenAlex {resp.status_code} for {entry.key}")
            return None
        results = resp.json().get("results", [])
        for work in results[:5]:
            title = work.get("title") or ""
            if not title:
                continue
            sim = _title_similarity(entry.title, title)
            if sim >= 0.75:
                doi = (work.get("doi") or "").replace("https://doi.org/", "")
                oa_url = (work.get("open_access") or {}).get("oa_url")
                auth_list = work.get("authorships", [])
                author_str = "; ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in auth_list[:3]
                ) if auth_list else None
                pub_year = str(work.get("publication_year") or "")
                author_sim = author_overlap_score(entry.authors or "", author_str or "") if author_str else None
                return VerificationResult(
                    key=entry.key, title=entry.title,
                    status="verified", confidence=sim,
                    matched_title=title, doi=doi or None,
                    open_access_url=oa_url,
                    note=f"OpenAlex match ({int(sim*100)}%)",
                    sources_checked=["OpenAlex"],
                    correct_authors=author_str,
                    corrected_title=title,
                    corrected_year=pub_year or None,
                    title_match_score=round(sim, 4),
                    author_match_score=round(author_sim, 4) if author_sim is not None else None,
                )
        for work in results[:3]:
            title = work.get("title") or ""
            if title:
                sim = _title_similarity(entry.title, title)
                if sim >= 0.50:
                    auth_list = work.get("authorships", [])
                    author_str = "; ".join(
                        a.get("author", {}).get("display_name", "")
                        for a in auth_list[:3]
                    ) if auth_list else None
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="partial_match",
                        confidence=sim,
                        matched_title=title,
                        note=f"Partial OpenAlex match ({int(sim*100)}%)",
                        sources_checked=["OpenAlex"],
                        correct_authors=author_str,
                        title_match_score=round(sim, 4),
                    )
    except requests.exceptions.Timeout:
        print(f"OpenAlex timeout for {entry.key}")
    except Exception as e:
        print(f"OpenAlex error for {entry.key}: {e}")
    return None


# ---------------------------------------------------------------------------
# URL fetch
# ---------------------------------------------------------------------------

def _fetch_url_strict(entry: BibEntry) -> Optional[VerificationResult]:
    from bs4 import BeautifulSoup

    url = (getattr(entry, "url", "") or "").strip()

    if not url or not url.startswith("http"):
        return None

    url = re.sub(r'\s+', '', url)
    url = re.sub(r',?\s*Stand:.*$', '', url, flags=re.IGNORECASE)
    url = re.sub(r'Stand:.*$', '', url, flags=re.IGNORECASE)

    _profiles = [
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.4 Safari/605.1.15"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-GB;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
    ]

    last_status = None
    for profile in _profiles:
        session = requests.Session()
        session.headers.update(profile)
        try:
            try:
                session.head(url, timeout=8, allow_redirects=True)
            except Exception:
                pass

            resp = session.get(url, timeout=15, allow_redirects=True)
            last_status = resp.status_code

            if resp.status_code == 200:
                content_type = resp.headers.get('Content-Type', '').lower()

                # PDFs: try to extract a real title from the fetched bytes
                # before giving up and escalating to the weaker AI/web-search
                # path. We already have the full PDF in memory (resp.content)
                # from the fetch above — no extra request needed.
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    pdf_title = ""
                    try:
                        import io
                        import pdfplumber
                        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                            meta_title = (pdf.metadata or {}).get("Title", "") or ""
                            pdf_title = meta_title.strip()
                            if not pdf_title and pdf.pages:
                                first_page_text = pdf.pages[0].extract_text() or ""
                                # Heuristic: the title is usually one of the
                                # first few non-empty lines on the cover/first
                                # page, before body text kicks in.
                                for line in first_page_text.split("\n")[:6]:
                                    line = line.strip()
                                    if len(line) >= 8:
                                        pdf_title = line
                                        break
                    except Exception:
                        pdf_title = ""

                    if pdf_title and entry.title:
                        sim = _title_similarity(entry.title, pdf_title)
                        if sim >= 0.70:
                            return VerificationResult(
                                key=entry.key,
                                title=entry.title or "",
                                status="verified",
                                confidence=round(sim, 4),
                                matched_title=pdf_title,
                                open_access_url=resp.url,
                                note=f"PDF verified: title match {int(sim*100)}% (extracted from document: '{pdf_title[:60]}')",
                                sources_checked=["url_fetch", "pdf_extract"],
                            )
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title or "",
                            status="url_blocked",
                            confidence=0.0,
                            open_access_url=resp.url,
                            note=(
                                f"PDF reachable (HTTP 200) but extracted title mismatch "
                                f"(cited: '{(entry.title or '')[:50]}' | PDF: '{pdf_title[:50]}' | sim: {int(sim*100)}%). "
                                f"Escalating to AI."
                            ),
                            sources_checked=["url_fetch", "pdf_extract"],
                        )

                    # Couldn't extract any usable title from the PDF itself
                    # (scanned/image-only PDF, no metadata, etc.) — fall back
                    # to AI/web-search as before.
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=resp.url,
                        note="PDF reachable (HTTP 200) but no extractable title (scanned/image-only or missing metadata). Escalating to AI.",
                        sources_checked=["url_fetch"],
                    )

                # Check if this is a documentation/tutorial site
                is_docs_site = any(x in url.lower() for x in ['/docs/', '/tutorials/', '/guide/', '/documentation/', '/manual/'])
                
                # For docs/tutorial sites returning 200, treat that as sufficient proof
                # (we already verified the domain exists and is live)
                if is_docs_site:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    page_title = ""
                    if soup.find("title"):
                        page_title = soup.find("title").get_text().strip()
                    if not page_title:
                        page_title = entry.title or "Documentation"
                    
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="verified",
                        confidence=0.75,
                        matched_title=page_title,
                        open_access_url=resp.url,
                        note=f"Documentation/tutorial URL verified (HTTP 200): {url}",
                        sources_checked=["url_verify"],
                    )
                
                soup = BeautifulSoup(resp.text, "html.parser")

                # Collect every title-like candidate rather than committing
                # to whichever tag happens to be non-empty first. SEO <title>
                # tags routinely bolt on year/audience keywords ("State of
                # the Cloud 2026 | Insights von Cloud-Führungskräften") that
                # the actual on-page heading doesn't have ("State of the
                # Cloud Report") — comparing only against <title> can fail a
                # match that the visible H1 would have passed cleanly.
                candidates = []
                if soup.find("title"):
                    t = soup.find("title").get_text().strip()
                    if t:
                        candidates.append(t)
                meta = soup.find("meta", property="og:title")
                if meta and meta.get("content", "").strip():
                    candidates.append(meta.get("content", "").strip())
                h1 = soup.find("h1")
                if h1 and h1.get_text().strip():
                    candidates.append(h1.get_text().strip())

                page_title = candidates[0] if candidates else ""

                _challenge = {"just a moment", "access denied", "attention required",
                              "403 forbidden", "404 not found", "please wait"}
                if page_title.lower() in _challenge or "cloudflare" in resp.text[:500].lower():
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=url,
                        note=f"URL alive (HTTP 200) but bot-protected. Escalating to AI.",
                        sources_checked=["url_fetch"],
                    )

                if not page_title:
                    # No title extractable — cannot verify, escalate
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=url,
                        note="URL reachable (HTTP 200) but no page title found. Escalating to AI.",
                        sources_checked=["url_fetch"],
                    )

                if entry.title:
                    # Try every extracted candidate (title tag, og:title, h1)
                    # and keep whichever matches best — an SEO title tag can
                    # legitimately differ from the visible heading, and we
                    # shouldn't fail a real match just because the first
                    # candidate we happened to check wasn't the best one.
                    best_sim, best_candidate = 0.0, page_title
                    for cand in candidates:
                        s = _title_similarity(entry.title, cand)
                        if s > best_sim:
                            best_sim, best_candidate = s, cand
                    sim = best_sim
                    page_title = best_candidate
                    
                    # ── LENIENT MATCHING FOR DOCUMENTATION/TUTORIAL SITES ─────
                    # URLs like python.langchain.com/docs/tutorials often have
                    # generic page titles. A 200 status is strong evidence they're
                    # real. Use lower threshold for docs/tutorials/guides.
                    is_docs_site = any(x in url.lower() for x in ['/docs/', '/tutorials/', '/guide/', '/documentation/', '/manual/'])
                    title_threshold = 0.65 if is_docs_site else 0.80
                    
                    # Require ≥threshold similarity — title must actually match
                    if sim >= title_threshold:
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title or "",
                            status="verified",
                            confidence=round(sim, 4),
                            matched_title=page_title,
                            open_access_url=resp.url,
                            note=f"URL verified: title match {int(sim*100)}% (page: '{page_title[:60]}')",
                            sources_checked=["url_verify"],
                        )
                    # Title found but doesn't match well enough — escalate with evidence
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=url,
                        note=(
                            f"URL reachable (HTTP 200) but title mismatch "
                            f"(cited: '{(entry.title or '')[:50]}' | page: '{page_title[:50]}' | sim: {int(sim*100)}%). "
                            f"Escalating to AI."
                        ),
                        sources_checked=["url_fetch"],
                    )

                # No cited title to compare against — escalate
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="url_blocked",
                    confidence=0.0,
                    open_access_url=url,
                    note=f"URL reachable (HTTP 200) but no cited title to compare. Escalating to AI.",
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (301, 302, 303, 307, 308):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="url_blocked", confidence=0.0,
                    open_access_url=url,
                    note=f"URL redirects (HTTP {resp.status_code}). Escalating to AI.",
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (403, 429):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="url_blocked", confidence=0.0,
                    open_access_url=url,
                    note=f"URL reachable but bot-blocked (HTTP {resp.status_code}). Escalating to AI.",
                    sources_checked=["url_fetch"],
                )

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            print(f"URL fetch error for {entry.key}: {e}")
            continue

    # ── Last resort: recover hyphens lost during PDF text extraction ─────────
    # A URL that hard-wrapped across lines in the source PDF with no literal
    # hyphen character at the break (a soft wrap, not a hard hyphen) leaves
    # extractor.py nothing to detect/repair — e.g. a real URL segment
    # "...240703-Bitkom-Charts-CloudReport-2024-final.pdf" comes out as
    # "...240703Bitkom-ChartsCloudReport-2024final.pdf". The signature is a
    # lowercase-letter-or-digit immediately followed by an uppercase letter,
    # exactly where a hyphen would normally separate slug segments. If the
    # URL we tried 404s and shows this pattern, retry once with hyphens
    # reinserted at those boundaries before giving up.
    if last_status in (404, None) and re.search(r'[a-z0-9][A-Z]', url):
        repaired_url = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', url)
        if repaired_url != url:
            for profile in _profiles:
                session = requests.Session()
                session.headers.update(profile)
                try:
                    resp = session.get(repaired_url, timeout=15, allow_redirects=True)
                    if resp.status_code == 200:
                        # Re-run the same PDF/HTML handling by recursing with
                        # a corrected entry URL — cheap since this only
                        # triggers on the rare 404+camelCase-boundary case.
                        repaired_entry = BibEntry(key=entry.key, raw_text=getattr(entry, "raw_text", ""))
                        repaired_entry.title = entry.title
                        repaired_entry.url = repaired_url
                        result = _fetch_url_strict(repaired_entry)
                        if result:
                            result.note = (
                                f"Original URL 404'd; recovered likely-correct URL "
                                f"(hyphens lost during PDF text extraction) and retried. "
                                + (result.note or "")
                            )
                            return result
                except Exception:
                    continue

    status_str = f"HTTP {last_status}" if last_status else "connection failed"
    return VerificationResult(
        key=entry.key, title=entry.title or "",
        status="url_blocked", confidence=0.0,
        open_access_url=url,
        note=f"URL unreachable after all attempts ({status_str}). Escalating to AI.",
        sources_checked=["url_fetch"],
    )


# ---------------------------------------------------------------------------
# LNI style checks
# ---------------------------------------------------------------------------

def check_lni_macros(body: str) -> List[dict]:
    suggestions = []
    if not body:
        return suggestions
    for i, line in enumerate(body.splitlines()):
        ctx = line.strip()[:120]
        if re.search(r'\[\d[\d,\s\-]*\]', line):
            suggestions.append({
                "type": "numeric_citation",
                "message": "Numeric citation detected. LNI requires author-year keys e.g. [AB20].",
                "context": ctx, "line": i + 1,
            })
        for m in re.finditer(r'\\cite\{([^}]+)\}', line):
            key = m.group(1).strip()
            if not re.match(r'^[A-Z]{1,4}\d{2}', key):
                suggestions.append({
                    "type": "cite_key_format",
                    "message": f"Key '{key}' may not follow LNI convention (e.g. AB20).",
                    "context": ctx, "line": i + 1,
                })
        if re.search(r'\bet\s+al\.', line, re.IGNORECASE):
            suggestions.append({
                "type": "et_al_in_text",
                "message": "'et al.' found — bibliography must list all authors.",
                "context": ctx, "line": i + 1,
            })
    return suggestions


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def extract_citations_from_body(body: str) -> set:
    r"""
    Extract citation keys from body text.
    
    Supports:
    - LNI format: [KEY], [KEY1], [KEY1a], [KEY1a, KEY2b], etc.
    - LaTeX: \cite{KEY}, \citet{KEY1, KEY2}
    - Numeric: [1], [1-3], [2, 5]
    """
    keys = set()
    if not body:
        return keys
    
    def normalize_brackets(m):
        content = m.group(1)
        normalized = re.sub(r'\s+', '', content)
        return '[' + normalized + ']'
    
    body_clean = re.sub(r'\[([A-Za-z0-9\s\n,;\-]+)\]', normalize_brackets, body)

    for m in re.finditer(r'\[([^\]]+)\]', body_clean):
        prefix = body_clean[max(0, m.start() - 12):m.start()].lower()
        if re.search(r'(?:e\.g\.?|z\.b\.?|cf\.?)\s*$', prefix):
            continue
        for k in re.split(r'\s*[,;]\s*', m.group(1)):
            k = k.strip()
            if re.fullmatch(r'\d+', k):
                keys.add(f'__NUM_{k}__')
                keys.add('__numeric_citations__')
            elif re.fullmatch(r'[A-Z][A-Za-z+]{0,5}\d{2}[a-z]?', k):
                keys.add(k)
    
    # LaTeX citations
    for m in re.finditer(r'\\(?:cite|citet|citep|Cite)\{([^}]+)\}', body_clean):
        for k in m.group(1).split(','):
            k = k.strip()
            if k:
                keys.add(k)
    
    return keys


def extract_citation_contexts(body: str) -> dict:
    contexts = {}
    if not body:
        return contexts
    
    def normalize_brackets(m):
        content = m.group(1)
        normalized = re.sub(r'\s+', '', content)
        return '[' + normalized + ']'
    
    body_clean = re.sub(r'\[([A-Za-z0-9\s\n,;\-]+)\]', normalize_brackets, body)
    
    for m in re.finditer(
        r'(.{0,80})(\[[A-Za-z][A-Za-z0-9+]{0,40}(?:[,;]\s*[A-Za-z][A-Za-z0-9+]{0,40})*\]|\[\d[\d,;\s\-]*\])(.{0,80})',
        body_clean,
    ):
        pre, cite, post = m.group(1), m.group(2), m.group(3)
        for k in re.split(r'\s*[,;]\s*', cite[1:-1]):
            k = k.strip()
            if k:
                contexts.setdefault(k, []).append(
                    f"...{pre}{cite}{post}...".strip()
                )
    return contexts


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc, field as _field

@_dc
class CrossCheckResult:
    correctly_used: List[str] = _field(default_factory=list)
    cited_not_in_bib: List[str] = _field(default_factory=list)
    in_bib_not_cited: List[str] = _field(default_factory=list)


def cross_check(bib_dict: dict, cited_keys: set) -> CrossCheckResult:
    result = CrossCheckResult()
    bib_keys = set(bib_dict.keys()) if bib_dict else set()
    bib_key_lookup = {key.lower(): key for key in bib_keys}
    real_cited = {k for k in cited_keys if k and not k.startswith('__')}
    numeric_cited = {
        k[6:-2] for k in cited_keys
        if k.startswith('__NUM_') and k.endswith('__')
    }
    real_cited.update(numeric_cited & bib_keys)
    real_cited = {bib_key_lookup.get(key.lower(), key) for key in real_cited}
    missing_numeric = numeric_cited - bib_keys
    result.correctly_used = sorted(real_cited & bib_keys)
    numeric_missing = missing_numeric if bib_keys and all(k.isdigit() for k in bib_keys) else set()
    result.cited_not_in_bib = sorted(
        key for key in (real_cited - bib_keys) | numeric_missing
        if key.lower() not in bib_key_lookup
    )
    result.in_bib_not_cited = sorted(bib_keys - real_cited)
    return result


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(bib_dict: dict, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> List[dict]:
    entries = list(bib_dict.values())
    duplicates = []
    seen_pairs = set()
    
    # ── STEP 1: Check for duplicate DOIs ──────────────────────────────────────
    doi_map = {}
    for entry in entries:
        if entry.doi:
            doi_map.setdefault(entry.doi, []).append(entry)
    
    for doi, matches in doi_map.items():
        if len(matches) > 1:
            for i in range(len(matches)):
                for j in range(i + 1, len(matches)):
                    a, b = matches[i], matches[j]
                    pair = tuple(sorted([a.key, b.key]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        duplicates.append({
                            "key_a": a.key,
                            "key_b": b.key,
                            "similarity": 1.0,
                            "title_a": a.title or "",
                            "title_b": b.title or "",
                            "reason": "Same DOI"
                        })
    
    # ── STEP 2: Check for duplicate titles ────────────────────────────────────
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if not a.title or not b.title:
                continue
            
            pair = tuple(sorted([a.key, b.key]))
            if pair in seen_pairs:
                continue
            
            def is_generic_title(t):
                if not t:
                    return True
                words = t.lower().split()
                if len(words) < 3:
                    return True
                if len(words) == 1 and t.isupper():
                    return True
                return False
            
            if is_generic_title(a.title) or is_generic_title(b.title):
                continue
            
            norm_a = _normalize_title(a.title)
            norm_b = _normalize_title(b.title)
            if norm_a and norm_b and norm_a == norm_b:
                if a.authors and b.authors:
                    auth_sim = author_overlap_score(a.authors, b.authors)
                    if auth_sim is None or auth_sim < AUTHOR_OVERLAP_THRESHOLD:
                        continue
                
                seen_pairs.add(pair)
                duplicates.append({
                    "key_a": a.key,
                    "key_b": b.key,
                    "similarity": 1.0,
                    "title_a": a.title,
                    "title_b": b.title,
                    "reason": "Exact title (after normalization)"
                })
                continue
            
            sim = _title_similarity(a.title, b.title)
            if sim >= threshold:
                if not (a.authors and b.authors):
                    continue
                
                auth_sim = author_overlap_score(a.authors, b.authors)
                if auth_sim is None or auth_sim < AUTHOR_OVERLAP_THRESHOLD:
                    continue
                
                seen_pairs.add(pair)
                duplicates.append({
                    "key_a": a.key,
                    "key_b": b.key,
                    "similarity": round(sim, 3),
                    "title_a": a.title,
                    "title_b": b.title,
                    "reason": "Title similarity + author overlap"
                })
                continue
            
            # Author-year match for flexible keys
            if (a.authors and b.authors and a.year and b.year
                    and a.year == b.year
                    and _title_similarity(a.title or "", b.title or "") >= threshold):
                def get_surnames(authors_str):
                    surnames = set()
                    for part in re.split(r';|\band\b|\bund\b', authors_str, flags=re.IGNORECASE):
                        part = part.strip()
                        if not part or re.match(r'^et\s+al\.?$', part.lower()):
                            continue
                        if ',' in part:
                            surname = part.split(',')[0].strip()
                        else:
                            tokens = part.lower().split()
                            particles = {'von', 'van', 'de', 'del', 'della', 'der', 'la', 'le', 'du', 'des', 'di'}
                            non_particle = [t for t in tokens if t not in particles]
                            surname = non_particle[-1] if non_particle else (tokens[-1] if tokens else '')
                        if len(surname) > 2:
                            surnames.add(surname[:3].lower())
                    return surnames
                
                surnames_a = get_surnames(a.authors)
                surnames_b = get_surnames(b.authors)
                
                if surnames_a and surnames_b:
                    overlap = len(surnames_a & surnames_b) / min(len(surnames_a), len(surnames_b))
                    if overlap >= 0.5:
                        seen_pairs.add(pair)
                        duplicates.append({
                            "key_a": a.key,
                            "key_b": b.key,
                            "similarity": round(overlap, 3),
                            "title_a": a.title,
                            "title_b": b.title,
                            "reason": f"Same year ({a.year}) + author overlap ({int(overlap*100)}%)"
                        })
    
    return duplicates


def get_duplicate_map(bib_dict: dict) -> Dict[str, str]:
    """Create a duplicate map that properly handles transitive relationships."""
    duplicates = find_duplicates(bib_dict)
    
    parent = {key: key for key in bib_dict.keys()}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            x_idx = list(bib_dict.keys()).index(px) if px in bib_dict else 999
            y_idx = list(bib_dict.keys()).index(py) if py in bib_dict else 999
            if x_idx < y_idx:
                parent[py] = px
            else:
                parent[px] = py
    
    for d in duplicates:
        union(d["key_a"], d["key_b"])
    
    dup_map = {}
    for key in bib_dict.keys():
        canonical = find(key)
        if key != canonical:
            dup_map[key] = canonical
    
    return dup_map


# ---------------------------------------------------------------------------
# Self-citation detection
# ---------------------------------------------------------------------------

def detect_self_citations(bib_dict: dict, body: str) -> List[dict]:
    self_cites = []
    body_lower = body.lower()
    self_signals = re.findall(
        r'(?:we|our|this paper|this work|the author|the authors|i )(?:.{0,60})'
        r'(\[[A-Z][A-Za-z+]{0,5}\d{2}[^\]]*\])',
        body_lower,
    )
    self_keys = set()
    for match in self_signals:
        for k in re.split(r',\s*', match[1:-1]):
            self_keys.add(k.strip().upper())
    for key, entry in bib_dict.items():
        if key.upper() in self_keys:
            self_cites.append({
                "key": key,
                "title": entry.title or "",
                "reason": "Citation appears near self-referential language",
                "matched_author": entry.authors or "",
            })
    return self_cites


# ---------------------------------------------------------------------------
# Field validation helpers
# ---------------------------------------------------------------------------

def _validate_entry_fields(entry: BibEntry) -> Optional[VerificationResult]:
    """Pre-verification validation. Returns error result if fields missing."""
    if entry.entry_type in ("website", "online"):
        if not entry.title:
            return VerificationResult(
                key=entry.key, title="", status="incomplete", confidence=0.0,
                note="Missing required fields: title. Cannot verify.",
                sources_checked=["structural_validation"],
            )
        return None

    missing_fields = []
    
    if not entry.title or entry.title.strip() == "":
        missing_fields.append("title")
    if not entry.authors or entry.authors.strip() == "":
        missing_fields.append("authors")
    if not entry.year or entry.year.strip() == "":
        missing_fields.append("year")
    
    if "title" in missing_fields or ("authors" in missing_fields and "year" in missing_fields):
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="incomplete",
            confidence=0.0,
            note=f"Missing required fields: {', '.join(missing_fields)}. Cannot verify.",
            sources_checked=["structural_validation"],
        )
    
    return None


# ---------------------------------------------------------------------------
# MAIN VERIFICATION FUNCTION — FIXED with grey literature AI support
# ---------------------------------------------------------------------------

def verify_reference(
    entry: BibEntry,
    dup_map: dict = None,
    allow_ai_fallback: bool = True,
) -> VerificationResult:
    """
    4-step pipeline with duplicate check FIRST.
    FIXED v8.7: Grey literature now properly uses AI fallback.
    """
    try:
        # Professor decisions are the strongest local evidence and should
        # prevent a known real paper from being reclassified by noisy APIs.
        review = get_review_decision(entry.title or "", entry.authors or "")
        if review:
            decision = (review.get("decision") or "").lower()
            if decision in ("verified", "real", "accepted"):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=1.0,
                    matched_title=entry.title or "",
                    correct_authors=entry.authors or "",
                    note="Confirmed by professor review.",
                    sources_checked=["professor_review"],
                )
            if decision in ("rejected", "fake"):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="fabricated", confidence=1.0,
                    matched_title=entry.title or "",
                    note="Marked as fake by professor review.",
                    sources_checked=["professor_review"],
                )

        # Check if this is a duplicate FIRST
        if dup_map and entry.key in dup_map:
            canonical_key = dup_map[entry.key]
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified",
                confidence=0.95,
                note=f"Duplicate of [{canonical_key}] — same paper",
                sources_checked=["duplicate_detection"],
                is_duplicate=True,
                duplicate_of=canonical_key,
            )
        
        # Validate fields BEFORE any API calls
        field_error = _validate_entry_fields(entry)
        if field_error:
            return field_error
        
        if not entry.title and not entry.doi:
            return VerificationResult(
                key=entry.key, title="",
                status="manual_review", confidence=0.0,
                note="No title or DOI — cannot verify. Manual review required.",
                sources_checked=[],
            )

        # ── STEP 1: Local SQLite DB ───────────────────────────────────────────────
        # search_cache() only ever returns rows with confirmed_real = 1 — i.e.
        # papers already verified (via API/AI) or explicitly confirmed by a
        # professor. Anything found here is authoritative and should short
        # circuit the rest of the pipeline; re-gating on confidence caused
        # already-cached/professor-confirmed entries to be silently re-checked
        # and sometimes re-flagged as SUSPICIOUS.
        cached = search_cache(entry.title or "", entry.authors or "")
        if cached:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=max(cached.confidence, 0.90),
                matched_title=cached.title, doi=cached.doi,
                open_access_url=cached.url,
                note=f"Found in local database (source: {cached.source})",
                sources_checked=["local_db"],
            )

        # ── STEP 1b: Grey literature detection ────────────────────────────────
        _entry_dict_for_grey = {
            "title":      entry.title or "",
            "authors":    getattr(entry, "authors", "") or "",
            "year":       entry.year or "",
            "url":        (getattr(entry, "url", "") or "").strip(),
            "publisher":  getattr(entry, "publisher", "") or "",
            "entry_type": getattr(entry, "entry_type", "") or "",
            "raw_text":   getattr(entry, "raw_text", "") or "",
        }
        _is_grey, _grey_reason = _is_grey_literature(_entry_dict_for_grey)

        if _is_grey:
            entry_url = _entry_dict_for_grey["url"]
            url_blocked = False
            url_note = ""

            # ── Try URL verification first ────────────────────────────────────────
            if entry_url and entry_url.startswith("http"):
                url_result = _fetch_url_strict(entry)
                if url_result and url_result.status == "verified":
                    save_to_cache(
                        title=url_result.matched_title or entry.title,
                        authors=entry.authors or "",
                        year=entry.year or "",
                        doi=entry.doi or "",
                        url=entry_url,
                        source="url_verify",
                        confidence=url_result.confidence,
                    )
                    return url_result
                elif url_result and url_result.status == "url_blocked":
                    url_blocked = True
                    url_note = url_result.note or "URL reachable but bot-blocked."
                else:
                    url_blocked = True
                    url_note = "URL fetch returned no result or non-200 status."
            else:
                url_note = "No URL in grey literature entry."

            # ── ✅ FIX: CALL AI FOR GREY LITERATURE ──────────────────────────────
            if allow_ai_fallback and _ai_available():
                grey_dict = {
                    **_entry_dict_for_grey,
                    "api_status": "url_blocked" if url_blocked else "not_found",
                    "api_matched_title": "",
                    "url_note": url_note,
                    "url_blocked": url_blocked,
                    "open_access_url": None,
                }
                try:
                    web_result = verify_with_web_search(grey_dict, "not_found")
                    
                    if web_result.get("status") == "verified" and web_result.get("confidence", 0) >= 0.55:
                        matched = web_result.get("matched_title") or entry.title
                        save_to_cache(
                            title=matched,
                            authors=entry.authors or "",
                            year=entry.year or "",
                            doi="",
                            url=web_result.get("open_access_url") or entry_url,
                            source="ai_grey_verified",
                            confidence=web_result["confidence"],
                        )
                        return VerificationResult(
                            key=entry.key, title=entry.title or "",
                            status="verified",
                            confidence=web_result["confidence"],
                            matched_title=matched,
                            open_access_url=web_result.get("open_access_url"),
                            note=web_result.get("note", f"Grey literature verified via AI ({_grey_reason})"),
                            sources_checked=["grey_lit", "ai_verified"],
                        )
                    else:
                        # AI couldn't verify it either
                        return VerificationResult(
                            key=entry.key, title=entry.title or "",
                            status="manual_review",
                            confidence=web_result.get("confidence", 0.4),
                            note=f"Grey literature ({_grey_reason}). {web_result.get('note', 'Could not be auto-verified.')}",
                            sources_checked=["grey_lit", "ai_attempted"],
                        )
                except Exception as e:
                    # AI call failed — log and return suspicious
                    print(f"[grey_lit] AI verification error for {entry.key}: {e}")
                    return VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status="manual_review",
                        confidence=0.4,
                        note=f"Grey literature ({_grey_reason}). AI verification failed: {str(e)[:100]}",
                        sources_checked=["grey_lit", "ai_error"],
                    )
            
            # ── No AI available ── return suspicious ──────────────────────────
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="manual_review",
                confidence=0.55,
                note=f"Grey literature ({_grey_reason}). Manual review required.",
                sources_checked=["grey_lit"],
            )

        # ── STEP 2: Academic APIs ─────────────────────────────────────────────────
        api_result: Optional[VerificationResult] = None

        if entry.doi:
            api_result = _lookup_by_doi(entry)

        if not api_result:
            api_result = _lookup_by_arxiv_id(entry)

        if not api_result:
            best: Optional[VerificationResult] = None
            ex = ThreadPoolExecutor(max_workers=3)
            futures = {ex.submit(fn, entry): fn for fn in
                       [_search_crossref, _search_semantic_scholar, _search_openalex]}

            max_wait = 50
            completed_count = 0
            total_futures = len(futures)
            try:
                for future in as_completed(futures, timeout=max_wait):
                    completed_count += 1
                    try:
                        r = future.result(timeout=5)
                    except (ConcurrentTimeoutError, Exception) as e:
                        fn_name = futures[future].__name__ if future in futures else "unknown"
                        # Silently skip individual API timeouts; if all timeout we'll fall back to AI
                        r = None
                    
                    if r is None:
                        continue
                    
                    if r.status == "verified" and r.confidence >= 0.75:
                        if best is None or r.confidence > best.confidence:
                            best = r
                        if best.confidence >= 0.95:
                            for f in futures:
                                f.cancel()
                            break
                    elif r.status == "partial_match" and (best is None or r.confidence > best.confidence):
                        best = r
            except ConcurrentTimeoutError:
                # Timeout waiting for futures to complete
                print(f"[API LOOKUP] Timeout: {completed_count}/{total_futures} APIs responded for '{entry.key}'", 
                      file=sys.stderr, flush=True)
                for f in futures:
                    f.cancel()
            except Exception as outer_ex:
                print(f"[API LOOKUP] Error for {entry.key}: {outer_ex}", file=sys.stderr, flush=True)
                for f in futures:
                    f.cancel()
            finally:
                for f in futures:
                    if not f.done():
                        f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
            
            api_result = best

        # Check if we have a good API match
        if api_result and api_result.status in ("verified", "partial_match"):
            title_sim = _title_similarity(entry.title or "", api_result.matched_title or "")

            def _looks_like_fake_author(authors_str: str) -> bool:
                if not authors_str:
                    return False
                first = authors_str.split(";")[0].strip()
                _fake_surnames = {
                    "ghost", "fake", "test", "example", "placeholder",
                    "unknown", "anonymous", "nobody", "someone", "author",
                    "dummy", "sample", "demo", "null", "none",
                }
                if "," in first:
                    surname = first.split(",")[0].strip().lower()
                else:
                    parts = first.split()
                    surname = parts[-1].lower() if parts else ""
                return surname in _fake_surnames

            entry_has_fake_author = _looks_like_fake_author(entry.authors or "")

            if api_result.status == "partial_match":
                author_overlap = author_overlap_score(
                    entry.authors or "", api_result.correct_authors or ""
                ) if entry.authors and api_result.correct_authors else None

                if (not entry_has_fake_author
                        and title_sim >= 0.75
                        and author_overlap is not None
                        and author_overlap >= 0.80):
                    api_result.status = "verified"
                    api_result.confidence = round(min(title_sim * 0.6 + author_overlap * 0.4, 0.92), 4)
                    api_result.note = (
                        f"Promoted from partial match: title {int(title_sim*100)}% + "
                        f"author {int(author_overlap*100)}% — likely real with format differences"
                    )
                else:
                    api_result = None

            if api_result and api_result.status == "verified" and title_sim >= 0.70:
                if entry_has_fake_author:
                    return VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status="manual_review",
                        confidence=0.30,
                        matched_title=api_result.matched_title,
                        note=(
                            "Suspicious author name detected. "
                            "API returned a match but the author string looks "
                            "like a placeholder — manual review required."
                        ),
                        sources_checked=api_result.sources_checked or [],
                    )

                author_ok = True
                if entry.authors and api_result.correct_authors:
                    overlap = author_overlap_score(entry.authors, api_result.correct_authors)
                    if overlap is not None and overlap < 0.40:
                        author_ok = False
                elif entry.authors and not api_result.correct_authors and api_result.confidence < 0.85:
                    author_ok = False
                
                if author_ok:
                    save_to_cache(
                        title=api_result.matched_title or entry.title,
                        authors=entry.authors or "",
                        year=entry.year or "",
                        doi=api_result.doi or entry.doi or "",
                        url=api_result.open_access_url or "",
                        source=(api_result.sources_checked[0]
                                if api_result.sources_checked else "api"),
                        confidence=api_result.confidence,
                    )
                    return api_result

        # ── STEP 3: URL fetch ─────────────────────────────────────────────────────
        entry_url = (getattr(entry, "url", "") or "").strip()

        if entry_url and entry_url.startswith("http"):
            url_result = _fetch_url_strict(entry)
            if url_result and url_result.status == "verified":
                save_to_cache(
                    title=url_result.matched_title or entry.title,
                    authors=entry.authors or "",
                    year=entry.year or "",
                    doi=entry.doi or "",
                    url=entry_url,
                    source="url_verify",
                    confidence=url_result.confidence,
                )
                return url_result

        # ── STEP 4: AI / web-search fallback ─────────────────────────────────────
        api_status = api_result.status if api_result else "not_found"
        api_matched_title = api_result.matched_title if api_result else ""
        
        entry_dict = {
            "title":      entry.title or "",
            "authors":    entry.authors or "",
            "year":       entry.year or "",
            "url":        entry_url,
            "publisher":  getattr(entry, "publisher", "") or "",
            "entry_type": getattr(entry, "entry_type", "") or "",
            "raw_text":   getattr(entry, "raw_text", "") or "",
            "api_status": api_status,
            "api_matched_title": api_matched_title,
            "url_note":   "",
            "open_access_url": api_result.open_access_url if api_result else None,
        }

        # Check if title is obviously fabricated
        is_fabricated, fab_confidence = _is_fabricated_title(entry.title or "")
        if is_fabricated and fab_confidence >= 0.80:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="fabricated",
                confidence=fab_confidence,
                note="Title matches known fabrication patterns.",
                sources_checked=["fabrication_detector"],
            )

        if not allow_ai_fallback:
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="manual_review",
                confidence=api_result.confidence if api_result else 0.0,
                matched_title=api_matched_title or None,
                doi=api_result.doi if api_result else None,
                open_access_url=api_result.open_access_url if api_result else None,
                note="No database or URL confirmation; queued for final AI review.",
                sources_checked=(api_result.sources_checked if api_result else ["none"]),
                correct_authors=api_result.correct_authors if api_result else None,
                title_match_score=api_result.title_match_score if api_result else None,
                author_match_score=api_result.author_match_score if api_result else None,
            )

        web_result = verify_with_web_search(entry_dict, api_status)

        if (web_result.get("status") == "verified"
                and web_result.get("confidence", 0.0) >= 0.55):
            matched = web_result.get("matched_title") or entry.title
            save_to_cache(
                title=matched,
                authors=entry.authors or "",
                year=entry.year or "",
                doi=entry.doi or "",
                url=web_result.get("open_access_url") or entry_url,
                source="ai_web_search",
                confidence=web_result["confidence"],
            )
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified",
                confidence=web_result["confidence"],
                matched_title=matched,
                open_access_url=web_result.get("open_access_url"),
                note=web_result.get("note", "Verified via AI web search"),
                sources_checked=web_result.get("sources_checked", ["web_search"]),
            )

        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="manual_review",
            confidence=web_result.get("confidence", 0.0),
            matched_title=api_result.matched_title if api_result else None,
            note=f"Not confirmed by any source. {web_result.get('note', 'Manual review required.')}",
            sources_checked=web_result.get("sources_checked", ["none"]),
        )
    
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="manual_review",
            confidence=0.0,
            note=f"Verification error: {str(e)[:100]}",
            sources_checked=["error"],
        )


def verify_all_references(bib_dict: dict) -> List[VerificationResult]:
    """
    Verify all references in the bibliography.
    FIXED: Duplicate entries inherit verification results.
    """
    entries = list(bib_dict.values())
    
    # ── STEP 1: Find duplicates BEFORE verification ──────────────────────────
    dup_map = get_duplicate_map(bib_dict)
    skip_keys = set(dup_map.keys())
    
    # ── STEP 2: Verify only unique entries ──────────────────────────────────
    results = []
    results_by_key = {}
    
    unique_entries = [e for e in entries if e.key not in skip_keys]
    
    worker_count = min(16, max(1, len(unique_entries)))
    ex = ThreadPoolExecutor(max_workers=worker_count)
    future_map = {
        ex.submit(verify_reference, e, dup_map): e
        for e in unique_entries
    }
    completed_keys = set()
    batch_timeout = 180
    try:
        for future in as_completed(future_map, timeout=batch_timeout):
            e = future_map[future]
            completed_keys.add(e.key)
            try:
                result = future.result()
            except Exception as exc:
                result = VerificationResult(
                    key=e.key, title=e.title or "",
                    status="manual_review", confidence=0.0,
                    note=f"Verification error: {exc}",
                )
            results.append(result)
            results_by_key[result.key] = result
    except ConcurrentTimeoutError:
        print(f"[VERIFY] Batch timeout after {batch_timeout}s", file=sys.stderr, flush=True)
    finally:
        for future in future_map:
            if not future.done():
                future.cancel()
        ex.shutdown(wait=False, cancel_futures=True)

    for e in unique_entries:
        if e.key not in completed_keys:
            result = VerificationResult(
                key=e.key, title=e.title or "",
                status="manual_review", confidence=0.0,
                note=f"Verification timed out after {batch_timeout} seconds.",
            )
            results.append(result)
            results_by_key[e.key] = result
    
    # ── STEP 3: Handle duplicates ─────────────────────────────────────────────
    for dup_key, canonical_key in dup_map.items():
        canonical_result = results_by_key.get(canonical_key)
        if canonical_result:
            dup_result = VerificationResult(
                key=dup_key,
                title=canonical_result.title,
                status=canonical_result.status,
                confidence=canonical_result.confidence,
                matched_title=canonical_result.matched_title,
                doi=canonical_result.doi,
                open_access_url=canonical_result.open_access_url,
                note=f"Duplicate of [{canonical_key}] — same paper",
                sources_checked=canonical_result.sources_checked,
                correct_authors=canonical_result.correct_authors,
                version_note=f"Duplicate entry: same as [{canonical_key}]",
                is_retracted=canonical_result.is_retracted,
                retraction_doi=canonical_result.retraction_doi,
                retraction_note=canonical_result.retraction_note,
                corrected_title=canonical_result.corrected_title,
                corrected_authors=canonical_result.corrected_authors,
                corrected_year=canonical_result.corrected_year,
                corrected_publisher=canonical_result.corrected_publisher,
                corrected_journal=canonical_result.corrected_journal,
                corrected_volume=canonical_result.corrected_volume,
                corrected_pages=canonical_result.corrected_pages,
                title_match_score=canonical_result.title_match_score,
                author_match_score=canonical_result.author_match_score,
                is_duplicate=True,
                duplicate_of=canonical_key,
            )
            results.append(dup_result)
        else:
            entry = bib_dict.get(dup_key)
            results.append(VerificationResult(
                key=dup_key,
                title=entry.title if entry else "",
                status="manual_review",
                confidence=0.0,
                note=f"Duplicate of [{canonical_key}] but canonical not found",
                is_duplicate=True,
                duplicate_of=canonical_key,
            ))
    
    # ── STEP 4: Sort results to maintain order ──────────────────────────────
    key_order = list(bib_dict.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)
    
    return results


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score(
    bib_list: list,
    xcheck: "CrossCheckResult",
    api_results: list,
    style_suggestions: list,
    duplicates: list,
    professor_confirmed_fakes: int = 0,
    retracted_count: int = 0,
    verification_results: list = None,
) -> dict:
    """
    Compute academic submission score with recalibrated penalties.
    """
    score = 100
    penalties = []

    # NO BIBLIOGRAPHY AT ALL — the paper cites sources in-text but has zero
    # bibliography entries. This is categorically worse than "a few missing
    # citations" (which assumes a bibliography exists and is just incomplete),
    # so it gets its own heavy, uncapped-by-the-missing-citations-cap penalty
    # instead of being silently absorbed into that 20pt-max bucket.
    if not bib_list and xcheck.cited_not_in_bib:
        deduct = 60
        score -= deduct
        penalties.append({
            "category": "No bibliography found",
            "count": len(xcheck.cited_not_in_bib),
            "deduction": deduct
        })

    # MISSING CITATIONS
    missing = len(xcheck.cited_not_in_bib)
    if missing and bib_list:
        deduct = min(missing * 5, 20)
        score -= deduct
        penalties.append({
            "category": "Missing citations",
            "count": missing,
            "deduction": deduct
        })

    # ORPHANED ENTRIES
    orphaned = len(xcheck.in_bib_not_cited)
    if orphaned:
        deduct = min(orphaned * 2, 10)
        score -= deduct
        penalties.append({
            "category": "Orphaned entries",
            "count": orphaned,
            "deduction": deduct
        })

    # STYLE/FORMAT ISSUES
    style_count = len(style_suggestions)
    if style_count:
        deduct = min(style_count * 2, 15)
        score -= deduct
        penalties.append({
            "category": "Style issues",
            "count": style_count,
            "deduction": deduct
        })

    # DUPLICATE ENTRIES
    dup_count = len(duplicates)
    if dup_count:
        deduct = min(dup_count * 3, 10)
        score -= deduct
        penalties.append({
            "category": "Duplicates",
            "count": dup_count,
            "deduction": deduct
        })

    # CONFIRMED FAKES
    if professor_confirmed_fakes:
        deduct = min(professor_confirmed_fakes * 10, 60)
        score -= deduct
        penalties.append({
            "category": "Confirmed fake references",
            "count": professor_confirmed_fakes,
            "deduction": deduct
        })

    # Automated verification findings are review signals, not confirmed fakes.
    # They must not affect the score until the professor confirms them manually.
    suspicious_count = sum(
        1 for v in (verification_results or [])
        if v.get("ai_verdict") == "SUSPICIOUS" or v.get("status") == "suspicious"
    )

    # KEY MISMATCHES (citation key doesn't match author/year metadata)
    key_mismatch_count = sum(1 for e in bib_list if getattr(e, "key_consistent", None) is False)
    if key_mismatch_count:
        deduct = min(key_mismatch_count * 3, 15)
        score -= deduct
        penalties.append({
            "category": "Key mismatches (author/year)",
            "count": key_mismatch_count,
            "deduction": deduct
        })

    # RETRACTED PAPERS
    if retracted_count:
        deduct = min(retracted_count * 8, 25)
        score -= deduct
        penalties.append({
            "category": "Retracted papers cited",
            "count": retracted_count,
            "deduction": deduct
        })

    # INCOMPLETE ENTRIES
    incomplete = sum(1 for e in bib_list if getattr(e, "completeness_issues", None))
    if incomplete:
        deduct = min(incomplete * 2, 10)
        score -= deduct
        penalties.append({
            "category": "Incomplete entries",
            "count": incomplete,
            "deduction": deduct
        })

    score = max(0, score)

    # ENTRY QUALITY SCORE
    entry_quality_score = None
    if verification_results and bib_list:
        real_count = sum(1 for v in verification_results if v.get("ai_verdict") == "REAL")
        total_entries = len(bib_list)
        entry_quality_score = int((real_count / total_entries * 100)) if total_entries else None

    grade = (
        "A" if score >= 90 else
        "B" if score >= 80 else
        "C" if score >= 60 else
        "D" if score >= 50 else
        "F"
    )

    summary_parts = []
    if not bib_list and xcheck.cited_not_in_bib:
        summary_parts.append("No bibliography section found at all")
    if professor_confirmed_fakes:
        summary_parts.append(f"{professor_confirmed_fakes} confirmed fake reference(s)")
    if suspicious_count:
        summary_parts.append(f"{suspicious_count} suspicious/unverified reference(s)")
    if retracted_count:
        summary_parts.append(f"{retracted_count} retracted paper(s)")
    if missing and bib_list:
        summary_parts.append(f"{missing} missing citation(s)")
    if orphaned:
        summary_parts.append(f"{orphaned} orphaned entry/entries")
    if style_count:
        summary_parts.append(f"{style_count} formatting issue(s)")

    summary = "; ".join(summary_parts) if summary_parts else "No issues detected."

    return {
        "score": score,
        "grade": grade,
        "penalties": penalties,
        "entry_quality_score": entry_quality_score,
        "summary": summary,
    }