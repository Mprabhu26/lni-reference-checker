"""
STEP 3: Citation Cross-Checker + Reference Verifier — v10.0
-----------------------------------------------------------
Strict multi-field validation (Title, Author, Exact Year, Venue).
All discrepancies route to MANUAL_REVIEW — no auto-FAKE from heuristics,
no unverified caching, no author-blind REAL verdicts.

v10.0 changes:
  - _author_gate_for_verified: REAL requires author overlap >= 0.60
  - Every _search_* / _lookup_* function gates REAL on author overlap
  - _fetch_url_strict no longer caches; docs-site shortcut removed
  - _lookup_by_arxiv_id caches only with author overlap >= 0.60
  - verify_reference Step 1 (duplicate) -> manual_review, not verified
  - verify_reference Step 9 (ML gate) does not write to cache
  - _check_metadata_consistency returns (is_consistent, issues, flags)
  - Landmark detection requires title similarity >= 0.70
  - Retracted papers are SUSPICIOUS, not REAL
  - Duplicate verdicts inherited from canonical, never upgraded
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

from parser import BibEntry, _extract_surnames
from local_db import search_cache, save_to_cache, get_cache_stats, init_cache_db
from web_search_verifier import verify_with_web_search, _safe_re_sub
from review_queue import is_venue_whitelisted, get_review_decision, get_false_positive
from ai_checker import _is_grey_literature, _is_fabricated_title, _ai_available
from author_journal_verifier import verify_reference_comprehensive_cached, AuthorVerificationResult

# ---------------------------------------------------------------------------
# Configurable verification thresholds
# ---------------------------------------------------------------------------
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("LNI_TITLE_SIM_THRESHOLD", "0.80"))
AUTHOR_OVERLAP_THRESHOLD = float(os.getenv("LNI_AUTHOR_OVERLAP_THRESHOLD", "0.60"))
AUTHOR_MISMATCH_THRESHOLD = float(os.getenv("LNI_AUTHOR_MISMATCH_THRESHOLD", "0.60"))
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("LNI_CONFIDENCE_HIGH", "0.85"))

# Author-Journal verification thresholds
AUTHOR_JOURNAL_FAKE_THRESHOLD = float(os.getenv("LNI_AUTHOR_JOURNAL_FAKE_THRESHOLD", "0.25"))
AUTHOR_JOURNAL_VERIFY_THRESHOLD = float(os.getenv("LNI_AUTHOR_JOURNAL_VERIFY_THRESHOLD", "0.75"))
AUTHOR_JOURNAL_SUSPICIOUS_THRESHOLD = float(os.getenv("LNI_AUTHOR_JOURNAL_SUSPICIOUS_THRESHOLD", "0.45"))

# Minimum author overlap for a "verified" API result
AUTHOR_VERIFY_MIN_OVERLAP = float(os.getenv("LNI_AUTHOR_VERIFY_MIN_OVERLAP", "0.60"))

# Confidence tiers
CONFIDENCE_HIGH = 0.90
CONFIDENCE_MODERATE = 0.65
CONFIDENCE_LOW = 0.40

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
# Author overlap with umlaut tolerance
# ---------------------------------------------------------------------------

def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    """Calculate author surname overlap with umlaut tolerance."""
    if not cited_authors or not correct_authors:
        return None

    def _normalize_surname(s: str) -> str:
        s = s.lower()
        for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss'),
                     ('à','a'),('á','a'),('â','a'),('ã','a'),
                     ('è','e'),('é','e'),('ê','e'),('ë','e'),
                     ('ì','i'),('í','i'),('î','i'),('ï','i'),
                     ('ò','o'),('ó','o'),('ô','o'),('õ','o'),
                     ('ù','u'),('ú','u'),('û','u'),
                     ('ý','y'),('ÿ','y'),('ñ','n'),('ç','c')]:
            s = s.replace(a, b)
        return re.sub(r'[^a-z0-9]', '', s)

    def _extract_surnames(authors_str: str) -> set:
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
                particles = {'von', 'van', 'de', 'del', 'della', 'der',
                             'la', 'le', 'du', 'des', 'di'}
                non_particle = [t for t in tokens
                                if t not in particles and not re.match(r'^[a-z]\.?$', t)]
                surname_part = (
                    non_particle[-1] if non_particle
                    else (tokens[-1] if tokens else '')
                )
            clean = _normalize_surname(surname_part)
            if len(clean) > 2:
                surnames.add(clean)
        return surnames

    cited = _extract_surnames(cited_authors)
    correct = _extract_surnames(correct_authors)

    if not cited or not correct:
        return None

    matches = 0.0
    cited_list = list(cited)[:6]

    for s in cited_list:
        matched = False
        match_weight = 1.0
        for c in correct:
            if s == c:
                matched = True
                break
            if len(s) >= 5 and len(c) >= 5 and s[:5] == c[:5]:
                matched = True
                break
            if len(s) >= 4 and len(c) >= 4 and s[:4] == c[:4]:
                matched = True
                match_weight = 0.9
                break
            if len(s) >= 3 and len(c) >= 3 and s[:3] == c[:3]:
                matched = True
                match_weight = 0.8
                break

        if matched:
            matches += match_weight
        else:
            if s and any(c and s[0] == c[0] for c in correct):
                matches += 0.5

    if len(cited_list) == 1:
        first_char_match = any(
            s and any(c and s[0] == c[0] for c in correct) for s in cited_list
        )
        if first_char_match and matches < 0.7:
            matches = 0.7

    return round(matches / min(len(cited_list), 6), 3)


def _author_gate_for_verified(
    entry: BibEntry,
    matched_authors: Optional[str],
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Decide whether an API match may be returned with status="verified".

    A verified result requires:
      - no cited author list to contradict (entry.authors empty), OR
      - author overlap >= AUTHOR_VERIFY_MIN_OVERLAP (default 0.60)

    Returns (allowed, overlap, reason).
    """
    if not entry.authors:
        return True, None, None
    if not matched_authors:
        return False, None, "Cited authors present but source provided no author list."
    overlap = author_overlap_score(entry.authors, matched_authors)
    if overlap is None:
        return False, None, "Author overlap could not be computed."
    if overlap < AUTHOR_VERIFY_MIN_OVERLAP:
        return False, round(overlap, 4), (
            f"Author overlap {int(overlap*100)}% below the "
            f"{int(AUTHOR_VERIFY_MIN_OVERLAP*100)}% threshold for a verified match."
        )
    return True, round(overlap, 4), None


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str
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
    confidence_tier: str = "moderate"
    author_journal_verification: Optional[Dict] = None
    consistency_issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retraction check
# ---------------------------------------------------------------------------

def _check_retraction(doi: str) -> tuple:
    if not doi:
        return False, None, None
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/10.0 (mailto:{mailto})" if mailto else "LNI-Checker/10.0"
        resp = requests.get(f"https://api.crossref.org/works/{doi}",
                            timeout=5, headers={"User-Agent": ua})
        if resp.status_code != 200:
            return False, None, None
        work = resp.json().get("message", {})
        # Check both 'update-to' and the work's own 'type' being "retraction"
        for u in work.get("update-to", []):
            if u.get("type", "").lower() == "retraction":
                ret_doi = u.get("DOI", "")
                parts = u.get("updated", {}).get("date-parts", [[""]])[0]
                date_str = "-".join(str(p) for p in parts if p) if parts else "unknown"
                return True, ret_doi, f"Retracted {date_str}. Retraction DOI: {ret_doi}"
        if str(work.get("type", "")).lower() == "retraction":
            return True, work.get("DOI"), "This item is itself a retraction notice."
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
        "corrected_volume": str(work.get("volume", "")) or None,
        "corrected_pages": str(work.get("page", "")) or None,
    }


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------

def _check_unpaywall(doi: str) -> Optional[str]:
    if not doi:
        return None
    mailto = os.environ.get("UNPAYWALL_EMAIL",
                            os.environ.get("CROSSREF_MAILTO", "")).strip()
    if not mailto:
        return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                            params={"email": mailto}, timeout=5,
                            headers={"User-Agent": "LNI-Checker/10.0"})
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
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/10.0 (mailto:{mailto})" if mailto else "LNI-Checker/10.0"
        resp = requests.get(f"https://api.crossref.org/works/{entry.doi}",
                            timeout=5, headers={"User-Agent": ua})
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            if sim >= 0.75:
                meta = _extract_corrected_metadata(work)
                is_ret, ret_doi, ret_note = _check_retraction(entry.doi)

                # Retraction is a hard SUSPICIOUS signal, never REAL
                if is_ret:
                    return VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status="suspicious", confidence=0.90,
                        matched_title=title, doi=entry.doi,
                        note=ret_note or "Paper has been retracted.",
                        sources_checked=["CrossRef (DOI)"],
                        correct_authors=meta["corrected_authors"],
                        is_retracted=True, retraction_doi=ret_doi,
                        retraction_note=ret_note,
                        corrected_title=title,
                        corrected_authors=meta["corrected_authors"],
                        corrected_year=meta["corrected_year"],
                        corrected_journal=meta["corrected_journal"],
                        title_match_score=round(sim, 4),
                    )

                allowed, overlap, reason = _author_gate_for_verified(
                    entry, meta["corrected_authors"]
                )
                status = "verified" if allowed else "partial_match"
                confidence = sim if allowed else min(sim, 0.65)
                note_text = (
                    f"DOI verified via CrossRef ({int(sim*100)}%)"
                    if allowed
                    else f"CrossRef title match ({int(sim*100)}%) but {reason}"
                )
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status=status, confidence=confidence,
                    matched_title=title, doi=entry.doi,
                    open_access_url=_check_unpaywall(entry.doi),
                    note=note_text,
                    sources_checked=["CrossRef (DOI)"],
                    correct_authors=meta["corrected_authors"],
                    is_retracted=False,
                    corrected_title=title,
                    corrected_authors=meta["corrected_authors"],
                    corrected_year=meta["corrected_year"],
                    corrected_journal=meta["corrected_journal"],
                    title_match_score=round(sim, 4),
                    author_match_score=overlap,
                )
    except Exception:
        pass
    return None


def _lookup_by_arxiv_id(entry: BibEntry) -> Optional[VerificationResult]:
    arxiv_patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
        r'arXiv:([a-z\-]+/\d{7})',
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
                            timeout=5, headers={"User-Agent": "LNI-Checker/10.0"})
        if resp.status_code == 200:
            m_title = re.search(r'title\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
            title = m_title.group(1) if m_title else None
            m_auth = re.search(r'author\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
            arxiv_authors = m_auth.group(1) if m_auth else None
            m_yr = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?', resp.text, re.IGNORECASE)
            arxiv_year = m_yr.group(1) if m_yr else None

            if title:
                sim = _title_similarity(entry.title or "", title)
                if sim >= 0.75:
                    allowed, overlap, reason = _author_gate_for_verified(
                        entry, arxiv_authors
                    )
                    status = "verified" if allowed else "partial_match"
                    confidence = sim if allowed else min(sim, 0.65)
                    # Only cache when we actually verified the authors
                    if allowed:
                        save_to_cache(
                            title=title,
                            authors=entry.authors or "",
                            year=arxiv_year or entry.year or "",
                            doi=entry.doi or "",
                            url=f"https://arxiv.org/pdf/{arxiv_id}",
                            source="arxiv_verified",
                            confidence=sim,
                        )
                    return VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status=status, confidence=confidence,
                        matched_title=title,
                        open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                        note=(
                            f"arXiv ID {arxiv_id} verified (author overlap "
                            f"{int((overlap or 0)*100)}%)"
                            if allowed
                            else f"arXiv ID {arxiv_id} title match but {reason}"
                        ),
                        sources_checked=["arXiv (ID)"],
                        correct_authors=arxiv_authors,
                        corrected_title=title,
                        corrected_year=arxiv_year,
                        title_match_score=round(sim, 4),
                        author_match_score=overlap,
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
    ua = f"LNI-Checker/10.0 (mailto:{mailto})" if mailto else "LNI-Checker/10.0"

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
                            for a in authors[:5]
                        ) if authors else None
                        allowed, overlap, reason = _author_gate_for_verified(
                            entry, author_str
                        )
                        status = "verified" if allowed else "partial_match"
                        confidence = sim if allowed else min(sim, 0.65)
                        note_text = (
                            f"CrossRef match ({int(sim*100)}%)"
                            if allowed
                            else f"CrossRef title match ({int(sim*100)}%) but {reason}"
                        )
                        return VerificationResult(
                            key=entry.key, title=entry.title,
                            status=status, confidence=confidence,
                            matched_title=title, doi=doi,
                            open_access_url=_check_unpaywall(doi) if doi else None,
                            note=note_text,
                            sources_checked=["CrossRef"],
                            correct_authors=author_str,
                            corrected_title=title,
                            corrected_authors=meta["corrected_authors"],
                            corrected_year=meta["corrected_year"],
                            corrected_journal=meta["corrected_journal"],
                            title_match_score=round(sim, 4),
                            author_match_score=overlap,
                        )
                return None
            elif resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            else:
                return None
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None

    clean_title = entry.title
    for pattern in [r'\.\s*In:\s*.*$', r'\.\s*doi:\s*.*$', r'\.\s*https?://\S+$']:
        clean_title = _safe_re_sub(pattern, '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.strip().strip('.,;:')

    if not clean_title:
        clean_title = entry.title

    _rate_limit("api.semanticscholar.org", 0.25)
    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "LNI-Checker/10.0"}
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
                            "fields": "title,authors,year,venue,publicationVenue,openAccessPdf,externalIds"},
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
                                a.get("name", "") for a in authors[:5]
                            ) if authors else None
                            oa = (paper.get("openAccessPdf") or {}).get("url")
                            doi = paper.get("externalIds", {}).get("DOI")
                            pub_venue = (
                                paper.get("venue")
                                or (paper.get("publicationVenue") or {}).get("name")
                                or None
                            )
                            pub_year = str(paper.get("year")) if paper.get("year") else None
                            allowed, overlap, reason = _author_gate_for_verified(
                                entry, author_str
                            )
                            status = "verified" if allowed else "partial_match"
                            confidence = sim if allowed else min(sim, 0.65)
                            note_text = (
                                f"Semantic Scholar match ({int(sim*100)}%)"
                                if allowed
                                else f"Semantic Scholar title match ({int(sim*100)}%) but {reason}"
                            )
                            return VerificationResult(
                                key=entry.key, title=entry.title,
                                status=status, confidence=confidence,
                                matched_title=title, doi=doi,
                                open_access_url=oa,
                                note=note_text,
                                sources_checked=["Semantic Scholar"],
                                correct_authors=author_str,
                                corrected_title=title,
                                corrected_authors=author_str,
                                corrected_year=pub_year,
                                corrected_journal=pub_venue,
                                title_match_score=round(sim, 4),
                                author_match_score=overlap,
                            )
                    # Second pass: partial matches (title sim 0.50-0.75)
                    for paper in data[:3]:
                        title = paper.get("title", "")
                        if not title:
                            continue
                        sim = _title_similarity(entry.title, title)
                        if 0.50 <= sim < 0.75:
                            authors = paper.get("authors", [])
                            author_str = "; ".join(
                                a.get("name", "") for a in authors[:5]
                            ) if authors else None
                            oa = (paper.get("openAccessPdf") or {}).get("url")
                            doi = paper.get("externalIds", {}).get("DOI")
                            pub_venue = (
                                paper.get("venue")
                                or (paper.get("publicationVenue") or {}).get("name")
                                or None
                            )
                            pub_year = str(paper.get("year")) if paper.get("year") else None
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
                                corrected_year=pub_year,
                                corrected_journal=pub_venue,
                                title_match_score=round(sim, 4),
                            )
                elif resp.status_code in (429, 503):
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.Timeout:
                pass
            except Exception:
                pass

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
        clean_title = _safe_re_sub(pattern, '', clean_title, flags=re.IGNORECASE)
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
            headers={"User-Agent": "LNI-Checker/10.0"})
        if resp.status_code != 200:
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
                    for a in auth_list[:5]
                ) if auth_list else None
                pub_year = str(work.get("publication_year") or "")
                source_info = (work.get("primary_location") or {}).get("source") or {}
                venue_name = source_info.get("display_name") or None
                allowed, overlap, reason = _author_gate_for_verified(
                    entry, author_str
                )
                status = "verified" if allowed else "partial_match"
                confidence = sim if allowed else min(sim, 0.65)
                note_text = (
                    f"OpenAlex match ({int(sim*100)}%)"
                    if allowed
                    else f"OpenAlex title match ({int(sim*100)}%) but {reason}"
                )
                return VerificationResult(
                    key=entry.key, title=entry.title,
                    status=status, confidence=confidence,
                    matched_title=title, doi=doi or None,
                    open_access_url=oa_url,
                    note=note_text,
                    sources_checked=["OpenAlex"],
                    correct_authors=author_str,
                    corrected_title=title,
                    corrected_year=pub_year or None,
                    corrected_journal=venue_name,
                    title_match_score=round(sim, 4),
                    author_match_score=overlap,
                )
        for work in results[:3]:
            title = work.get("title") or ""
            if not title:
                continue
            sim = _title_similarity(entry.title, title)
            if 0.50 <= sim < 0.75:
                auth_list = work.get("authorships", [])
                author_str = "; ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in auth_list[:5]
                ) if auth_list else None
                source_info = (work.get("primary_location") or {}).get("source") or {}
                venue_name = source_info.get("display_name") or None
                pub_year = str(work.get("publication_year") or "")
                return VerificationResult(
                    key=entry.key, title=entry.title,
                    status="partial_match",
                    confidence=sim,
                    matched_title=title,
                    note=f"Partial OpenAlex match ({int(sim*100)}%)",
                    sources_checked=["OpenAlex"],
                    correct_authors=author_str,
                    corrected_year=pub_year or None,
                    corrected_journal=venue_name,
                    title_match_score=round(sim, 4),
                )
    except requests.exceptions.Timeout:
        pass
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# URL fetch
# ---------------------------------------------------------------------------

def _fetch_url_strict(entry: BibEntry) -> Optional[VerificationResult]:
    """
    Fetch the cited URL and compare its title to the citation.

    v10.0: does NOT write to the local cache under any circumstances.
    Docs-site shortcut removed — every URL goes through the same title
    comparison. A URL-only match is never enough for status="verified"
    when the entry has cited authors.
    """
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
                                status="partial_match",
                                confidence=round(sim, 4),
                                matched_title=pdf_title,
                                open_access_url=resp.url,
                                note=(
                                    f"PDF reachable; title matches ({int(sim*100)}%) "
                                    f"but author identity not verified by URL check."
                                ),
                                sources_checked=["url_fetch", "pdf_extract"],
                            )
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title or "",
                            status="url_blocked",
                            confidence=0.0,
                            open_access_url=resp.url,
                            note=(
                                f"PDF reachable but extracted title mismatch "
                                f"(cited: '{(entry.title or '')[:50]}' | PDF: '{pdf_title[:50]}' "
                                f"| sim: {int(sim*100)}%)."
                            ),
                            sources_checked=["url_fetch", "pdf_extract"],
                        )

                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=resp.url,
                        note="PDF reachable but no extractable title.",
                        sources_checked=["url_fetch"],
                    )

                soup = BeautifulSoup(resp.text, "html.parser")

                candidates = []
                if soup.find("title"):
                    t = soup.find("title").get_text().strip()
                    if t:
                        candidates.append(t)
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
                        note="URL alive but bot-protected.",
                        sources_checked=["url_fetch"],
                    )

                if not page_title:
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=url,
                        note="URL reachable but no page title found.",
                        sources_checked=["url_fetch"],
                    )

                if entry.title:
                    best_sim, best_candidate = 0.0, page_title
                    for cand in candidates:
                        s = _title_similarity(entry.title, cand)
                        if s > best_sim:
                            best_sim, best_candidate = s, cand
                    sim = best_sim
                    page_title = best_candidate

                    if sim >= 0.80:
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title or "",
                            status="partial_match",
                            confidence=round(sim, 4),
                            matched_title=page_title,
                            open_access_url=resp.url,
                            note=(
                                f"URL reachable; page title matches ({int(sim*100)}%) "
                                f"but author identity not verified by URL check."
                            ),
                            sources_checked=["url_verify"],
                        )
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="url_blocked",
                        confidence=0.0,
                        open_access_url=url,
                        note=(
                            f"URL reachable but title mismatch "
                            f"(cited: '{(entry.title or '')[:50]}' | page: '{page_title[:50]}' "
                            f"| sim: {int(sim*100)}%)."
                        ),
                        sources_checked=["url_fetch"],
                    )

                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="url_blocked",
                    confidence=0.0,
                    open_access_url=url,
                    note="URL reachable but no cited title to compare.",
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (301, 302, 303, 307, 308):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="url_blocked", confidence=0.0,
                    open_access_url=url,
                    note=f"URL redirects (HTTP {resp.status_code}).",
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (403, 429):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="url_blocked", confidence=0.0,
                    open_access_url=url,
                    note=f"URL reachable but bot-blocked (HTTP {resp.status_code}).",
                    sources_checked=["url_fetch"],
                )

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    if last_status in (404, None) and re.search(r'[a-z0-9][A-Z]', url):
        repaired_url = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', url)
        if repaired_url != url:
            for profile in _profiles:
                session = requests.Session()
                session.headers.update(profile)
                try:
                    resp = session.get(repaired_url, timeout=15, allow_redirects=True)
                    if resp.status_code == 200:
                        repaired_entry = BibEntry(
                            key=entry.key,
                            raw_text=getattr(entry, "raw_text", "")
                        )
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
        note=f"URL unreachable after all attempts ({status_str}).",
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
    r"""Extract citation keys from body text."""
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
    numeric_missing = (
        missing_numeric if bib_keys and all(k.isdigit() for k in bib_keys) else set()
    )
    result.cited_not_in_bib = sorted(
        key for key in (real_cited - bib_keys) | numeric_missing
        if key.lower() not in bib_key_lookup
    )
    result.in_bib_not_cited = sorted(bib_keys - real_cited)
    return result


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(bib_dict: dict,
                    threshold: float = TITLE_SIMILARITY_THRESHOLD) -> List[dict]:
    entries = list(bib_dict.values())
    duplicates = []
    seen_pairs = set()

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
                            "key_a": a.key, "key_b": b.key,
                            "similarity": 1.0,
                            "title_a": a.title or "",
                            "title_b": b.title or "",
                            "reason": "Same DOI",
                        })

    url_map = {}
    for entry in entries:
        if entry.url:
            url_norm = entry.url.lower().strip()
            url_norm = re.sub(r'^https?://(?:www\.)?', '', url_norm)
            url_norm = url_norm.rstrip('/')
            url_norm = re.sub(r'gi-?ev\.at', 'gi.de', url_norm)
            url_map.setdefault(url_norm, []).append(entry)

    for url_norm, matches in url_map.items():
        if len(matches) > 1:
            for i in range(len(matches)):
                for j in range(i + 1, len(matches)):
                    a, b = matches[i], matches[j]
                    pair = tuple(sorted([a.key, b.key]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        duplicates.append({
                            "key_a": a.key, "key_b": b.key,
                            "similarity": 1.0,
                            "title_a": a.title or "",
                            "title_b": b.title or "",
                            "reason": f"Same URL: {url_norm}",
                        })

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
                    "key_a": a.key, "key_b": b.key,
                    "similarity": 1.0,
                    "title_a": a.title, "title_b": b.title,
                    "reason": "Exact title (after normalization)",
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
                    "key_a": a.key, "key_b": b.key,
                    "similarity": round(sim, 3),
                    "title_a": a.title, "title_b": b.title,
                    "reason": "Title similarity + author overlap",
                })
                continue

            if (a.authors and b.authors and a.year and b.year
                    and a.year == b.year
                    and _title_similarity(a.title or "", b.title or "") >= threshold):
                def get_surnames(authors_str):
                    surnames = set()
                    for part in re.split(r';|\band\b|\bund\b', authors_str,
                                         flags=re.IGNORECASE):
                        part = part.strip()
                        if not part or re.match(r'^et\s+al\.?$', part.lower()):
                            continue
                        if ',' in part:
                            surname = part.split(',')[0].strip()
                        else:
                            tokens = part.lower().split()
                            particles = {'von', 'van', 'de', 'del', 'della',
                                         'der', 'la', 'le', 'du', 'des', 'di'}
                            non_particle = [t for t in tokens if t not in particles]
                            surname = (
                                non_particle[-1] if non_particle
                                else (tokens[-1] if tokens else '')
                            )
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
                            "key_a": a.key, "key_b": b.key,
                            "similarity": round(overlap, 3),
                            "title_a": a.title, "title_b": b.title,
                            "reason": f"Same year ({a.year}) + author overlap ({int(overlap*100)}%)",
                        })

    return duplicates


def get_duplicate_map(bib_dict: dict) -> Dict[str, str]:
    """Union-find based duplicate map."""
    duplicates = find_duplicates(bib_dict)
    parent = {key: key for key in bib_dict.keys()}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            keys_list = list(bib_dict.keys())
            x_idx = keys_list.index(px) if px in bib_dict else 999
            y_idx = keys_list.index(py) if py in bib_dict else 999
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
    if entry.entry_type in ("website", "online"):
        if not entry.title:
            return VerificationResult(
                key=entry.key, title="", status="incomplete", confidence=0.0,
                note="Missing required fields: title.",
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
            status="incomplete", confidence=0.0,
            note=f"Missing required fields: {', '.join(missing_fields)}.",
            sources_checked=["structural_validation"],
        )

    return None


def _looks_like_fake_author(authors_str: str) -> bool:
    if not authors_str:
        return False
    first = authors_str.split(";")[0].strip()
    _fake_surnames = {
        "ghost", "fake", "test", "example", "placeholder",
        "unknown", "anonymous", "nobody", "someone", "author",
        "dummy", "sample", "demo", "null", "none",
        "xylander",
    }
    if "," in first:
        surname = first.split(",")[0].strip().lower()
    else:
        parts = first.split()
        surname = parts[-1].lower() if parts else ""
    return surname in _fake_surnames or len(surname) < 2


# ---------------------------------------------------------------------------
# Metadata consistency
# ---------------------------------------------------------------------------

def _check_metadata_consistency(
    entry: BibEntry,
    api_result: Optional[VerificationResult],
) -> Tuple[bool, List[str], Dict[str, bool]]:
    """
    Compare cited metadata against the API-verified record.
    Returns (is_consistent, issues, flags).
    """
    issues: List[str] = []
    flags = {
        "author_mismatch": False,
        "severe_author_mismatch": False,
        "year_mismatch": False,
        "venue_mismatch": False,
        "publisher_mismatch": False,
        "title_mismatch": False,
        "doi_mismatch": False,
    }

    if not entry or not api_result:
        return True, issues, flags

    # Year
    if entry.year and api_result.corrected_year:
        m_c = re.search(r'\d{4}', str(entry.year))
        m_a = re.search(r'\d{4}', str(api_result.corrected_year))
        if m_c and m_a and m_c.group() != m_a.group():
            issues.append(f"Year mismatch: cited {m_c.group()} vs verified {m_a.group()}")
            flags["year_mismatch"] = True

    # Authors
    if entry.authors and api_result.correct_authors:
        overlap = author_overlap_score(entry.authors, api_result.correct_authors)
        cited_list = [a.strip() for a in entry.authors.split(';') if a.strip()]
        api_list = [a.strip() for a in api_result.correct_authors.split(';') if a.strip()]
        api_truncated = len(api_list) <= 3 and len(cited_list) > len(api_list)
        if overlap is not None:
            if overlap < 0.20 and not api_truncated:
                issues.append(f"Severe author mismatch (overlap: {int(overlap*100)}%)")
                flags["severe_author_mismatch"] = True
                flags["author_mismatch"] = True
            elif overlap < 0.60 and not api_truncated:
                issues.append(f"Author mismatch (overlap: {int(overlap*100)}%)")
                flags["author_mismatch"] = True

    # Venue
    cited_venue = (entry.journal or entry.booktitle or "").strip()
    api_venue = (api_result.corrected_journal or "").strip()
    if cited_venue and api_venue:
        c_norm = re.sub(r'[^a-z0-9]', '', cited_venue.lower())
        a_norm = re.sub(r'[^a-z0-9]', '', api_venue.lower())
        if c_norm and a_norm and c_norm not in a_norm and a_norm not in c_norm:
            c_words = set(re.findall(r'[a-z]{3,}', cited_venue.lower()))
            a_words = set(re.findall(r'[a-z]{3,}', api_venue.lower()))
            if not (c_words & a_words):
                issues.append(
                    f"Venue mismatch: cited '{cited_venue[:40]}' vs verified '{api_venue[:40]}'"
                )
                flags["venue_mismatch"] = True

    # Publisher
    if entry.publisher and api_result.corrected_publisher:
        c_norm = re.sub(r'[^a-z0-9]', '', entry.publisher.lower())
        a_norm = re.sub(r'[^a-z0-9]', '', api_result.corrected_publisher.lower())
        if c_norm and a_norm and c_norm not in a_norm and a_norm not in c_norm:
            issues.append(
                f"Publisher mismatch: '{entry.publisher[:40]}' vs "
                f"'{api_result.corrected_publisher[:40]}'"
            )
            flags["publisher_mismatch"] = True

    # Title
    if entry.title and api_result.matched_title:
        sim = _title_similarity(entry.title, api_result.matched_title)
        if sim < 0.75:
            issues.append(f"Title mismatch (similarity: {int(sim*100)}%)")
            flags["title_mismatch"] = True

    # DOI
    if entry.doi and api_result.doi:
        if entry.doi.strip().lower() != api_result.doi.strip().lower():
            issues.append(f"DOI mismatch: cited '{entry.doi}' vs verified '{api_result.doi}'")
            flags["doi_mismatch"] = True

    is_consistent = (len(issues) == 0)
    return is_consistent, issues, flags


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

def _is_landmark_paper_local(title: str, key: str = "") -> Optional[Dict]:
    """Requires title similarity >= 0.70, or (>= 0.55 with matching key pattern)."""
    if not title:
        return None

    try:
        from author_journal_verifier import (
            LANDMARK_PAPERS,
            _normalize_title_for_matching,
            _title_similarity_simple,
        )

        for landmark_title, info in LANDMARK_PAPERS.items():
            sim = _title_similarity_simple(title, landmark_title)
            matched_key = any(
                pattern.lower() in key.lower()
                for pattern in info.get("key_patterns", [])
            )
            if sim >= 0.70 or (sim >= 0.55 and matched_key):
                return {
                    **info,
                    "matched_by": "title_similarity",
                    "similarity": sim,
                    "landmark_title": landmark_title,
                }
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

def _apply_confidence_thresholds(result: "VerificationResult") -> "VerificationResult":
    """Set confidence_tier; do not silently downgrade fabricated/suspicious."""
    if result.confidence >= CONFIDENCE_HIGH:
        result.confidence_tier = "high"
    elif result.confidence >= CONFIDENCE_MODERATE:
        result.confidence_tier = "moderate"
    else:
        result.confidence_tier = "low"
    return result


# ---------------------------------------------------------------------------
# MAIN VERIFICATION FUNCTION
# ---------------------------------------------------------------------------

def verify_reference(
    entry: BibEntry,
    dup_map: dict = None,
    allow_ai_fallback: bool = True,
) -> VerificationResult:
    """
    Verification pipeline:
    0. Professor review override
    1. Duplicate short-circuit (returns manual_review; canonical decides later)
    2. Field validation
    3. Local DB cache (author-aware)
    3.5 Landmark paper
    4. Academic APIs (author-gated)
    5. Author-journal verification
    6. Metadata consistency check
    7. URL fetch
    8. AI / web-search fallback
    9. ML gate (does NOT write to cache)
    """
    try:
        # ── STEP 0: Professor review override ─────────────────────────────────
        review = get_review_decision(entry.title or "", entry.authors or "")
        if review:
            decision = (review.get("decision") or "").lower()
            if decision in ("verified", "real", "accepted"):
                save_to_cache(
                    title=entry.title or "",
                    authors=entry.authors or "",
                    year=entry.year or "",
                    doi=entry.doi or "",
                    url=entry.url or "",
                    source="professor_review",
                    confidence=1.0,
                )
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
                    status="manual_review", confidence=1.0,
                    matched_title=entry.title or "",
                    note="Marked as fake by professor review.",
                    sources_checked=["professor_review"],
                )

        # ── STEP 1: Duplicate short-circuit ───────────────────────────────────
        # Do not assume REAL. The canonical entry decides the verdict in
        # verify_all_references Step 3.
        if dup_map and entry.key in dup_map:
            canonical_key = dup_map[entry.key]
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="manual_review",
                confidence=0.0,
                note=f"Duplicate of [{canonical_key}] — verdict inherited from canonical.",
                sources_checked=["duplicate_detection"],
                is_duplicate=True,
                duplicate_of=canonical_key,
            )

        # ── STEP 2: Field validation ──────────────────────────────────────────
        field_error = _validate_entry_fields(entry)
        if field_error:
            return field_error

        if not entry.title and not entry.doi:
            return VerificationResult(
                key=entry.key, title="",
                status="manual_review", confidence=0.0,
                note="No title or DOI — cannot verify.",
                sources_checked=[],
            )

        # ── STEP 3: Local SQLite DB (author-aware via search_cache) ───────────
        cached = search_cache(entry.title or "", entry.authors or "")
        if cached:
            # Year sanity: if year is present in both, they must match
            if entry.year and cached.year and str(entry.year).strip() != str(cached.year).strip():
                pass  # fall through to full verification
            else:
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=max(cached.confidence, 0.90),
                    matched_title=cached.title, doi=cached.doi,
                    open_access_url=cached.url,
                    note=f"Found in local database (source: {cached.source}).",
                    sources_checked=["local_db"],
                )

        # ── STEP 3.5: Landmark paper (title-similarity gated) ─────────────────
        landmark_info = _is_landmark_paper_local(entry.title or "", entry.key)
        if landmark_info:
            landmark_authors = landmark_info.get("authors", [])
            landmark_year = landmark_info.get("year", "")
            landmark_venue = landmark_info.get("venue", "")

            lm_issues = []
            if entry.year and landmark_year and str(entry.year).strip() != str(landmark_year).strip():
                lm_issues.append(
                    f"Year mismatch: cited {entry.year} vs landmark year {landmark_year}"
                )

            if entry.authors and landmark_authors:
                cited_surnames = _extract_surnames(entry.authors)
                norm_lm_authors = [re.sub(r'[^a-z0-9]', '', a.lower()) for a in landmark_authors]
                matches = [s for s in cited_surnames if s.lower() in norm_lm_authors]
                if cited_surnames:
                    overlap = len(matches) / len(cited_surnames)
                    if overlap < 0.60:
                        lm_issues.append(
                            f"Author mismatch on landmark paper "
                            f"({int(overlap*100)}% overlap)"
                        )

            if lm_issues:
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="manual_review",
                    confidence=0.40,
                    matched_title=landmark_info.get("title", entry.title or ""),
                    correct_authors=(
                        "; ".join(landmark_authors)
                        if isinstance(landmark_authors, list)
                        else str(landmark_authors)
                    ),
                    corrected_year=landmark_year,
                    corrected_journal=landmark_venue,
                    note=f"Suspected landmark mismatch: {'; '.join(lm_issues)}.",
                    sources_checked=["landmark_detection"],
                    consistency_issues=lm_issues,
                )

            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="verified",
                confidence=0.99,
                matched_title=landmark_info.get("title", entry.title or ""),
                correct_authors=(
                    "; ".join(landmark_authors)
                    if isinstance(landmark_authors, list)
                    else str(landmark_authors)
                ),
                corrected_year=landmark_year,
                corrected_journal=landmark_venue,
                open_access_url=landmark_info.get("url", ""),
                note=f"Landmark paper confirmed: {landmark_info.get('title', entry.title or '')[:60]}",
                sources_checked=["landmark_detection"],
            )

        # ── STEP 4: Academic APIs ─────────────────────────────────────────────
        api_result: Optional[VerificationResult] = None

        if entry.doi:
            api_result = _lookup_by_doi(entry)

        if not api_result:
            api_result = _lookup_by_arxiv_id(entry)

        if not api_result:
            best: Optional[VerificationResult] = None
            ex = ThreadPoolExecutor(max_workers=3)
            futures = {
                ex.submit(fn, entry): fn
                for fn in [_search_crossref, _search_semantic_scholar, _search_openalex]
            }

            max_wait = 30
            try:
                for future in as_completed(futures, timeout=max_wait):
                    try:
                        r = future.result(timeout=5)
                    except Exception:
                        r = None

                    if r is None:
                        continue

                    if r.status == "verified" and r.confidence >= 0.75:
                        if best is None or r.confidence > best.confidence:
                            best = r
                        if best.confidence >= 0.85:
                            for f in futures:
                                f.cancel()
                            break
                    elif r.status == "partial_match" and (best is None or r.confidence > best.confidence):
                        best = r
            except Exception:
                for f in futures:
                    f.cancel()
            finally:
                for f in futures:
                    if not f.done():
                        f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)

            api_result = best

        # ── STEP 5: Author-journal verification ───────────────────────────────
        author_journal_result = None
        author_journal_conf = 0.0

        try:
            if entry.authors and (entry.journal or entry.booktitle or entry.publisher):
                venue = entry.journal or entry.booktitle or entry.publisher or ""
                author_journal_result = verify_reference_comprehensive_cached(
                    author_names=entry.authors,
                    venue_name=venue,
                    paper_title=entry.title or "",
                    year=entry.year or "",
                    key=entry.key,
                )
                author_journal_conf = author_journal_result.overall_confidence
        except Exception as e:
            print(f"[Author-Journal] Error for {entry.key}: {e}")

        # ── STEP 6: Metadata consistency ──────────────────────────────────────
        if api_result and api_result.status in ("verified", "partial_match"):
            is_consistent, consistency_issues, flags = _check_metadata_consistency(
                entry, api_result
            )

            if not is_consistent:
                rec = (
                    "Suspected fabrication: title found but cited author/metadata mismatch."
                    if flags.get("severe_author_mismatch")
                    else "Discrepancy detected in cited metadata."
                )
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="manual_review",
                    confidence=0.40,
                    matched_title=api_result.matched_title,
                    correct_authors=api_result.correct_authors,
                    corrected_year=api_result.corrected_year,
                    corrected_journal=api_result.corrected_journal,
                    note=f"{rec} Issues: {'; '.join(consistency_issues)}",
                    sources_checked=api_result.sources_checked or ["academic_db_crosscheck"],
                    consistency_issues=consistency_issues,
                )

            # ONLY save to DB if every cited field matched AND the API result was definitively verified
            if api_result.status == "verified":
                save_to_cache(
                    title=api_result.matched_title or entry.title,
                    authors=entry.authors or "",
                    year=api_result.corrected_year or entry.year or "",
                    doi=api_result.doi or entry.doi or "",
                    url=api_result.open_access_url or "",
                    source=(api_result.sources_checked[0]
                            if api_result.sources_checked else "api"),
                    confidence=api_result.confidence,
                )
            return api_result


        # Author-journal fallback (no direct API title match)
        if author_journal_result:
            if author_journal_conf < AUTHOR_JOURNAL_FAKE_THRESHOLD:
                warn_str = "; ".join(author_journal_result.warnings[:2])
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="manual_review",
                    confidence=0.30,
                    matched_title=entry.title or "",
                    note=f"Author-journal mismatch: {warn_str}.",
                    sources_checked=["author_journal_verification"],
                    author_journal_verification={
                        "overall_confidence": author_journal_conf,
                        "warnings": author_journal_result.warnings,
                    },
                )
            elif (
                author_journal_conf >= AUTHOR_JOURNAL_VERIFY_THRESHOLD
                and author_journal_result.author_exists
                and author_journal_result.journal_exists
                and author_journal_result.author_venue_match
            ):
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="verified",
                    confidence=min(0.95, author_journal_conf),
                    matched_title=entry.title or "",
                    correct_authors=entry.authors or "",
                    note=f"Verified via author-journal check (conf: {author_journal_conf:.2f}).",
                    sources_checked=["author_journal_verification"],
                    author_journal_verification={
                        "overall_confidence": author_journal_conf,
                        "evidence": author_journal_result.evidence,
                    },
                )

        # ── STEP 7: URL fetch ─────────────────────────────────────────────────
        entry_url = (getattr(entry, "url", "") or "").strip()
        if entry_url and entry_url.startswith("http"):
            url_result = _fetch_url_strict(entry)
            # URL-only matches are partial_match now; only pass through if
            # the entry is a website/misc AND has no cited authors to check.
            if url_result and url_result.status == "verified":
                if entry.entry_type in ("website", "online") or not entry.authors:
                    return url_result
                # Otherwise downgrade
                url_result.status = "manual_review"
                url_result.note = (
                    "URL page title verified but author identity requires review."
                )
                return url_result
            elif url_result and url_result.status in ("manual_review", "url_blocked"):
                # Preserve URL fetch failure signal for the caller
                pass

        # ── STEP 8: AI / web-search fallback ──────────────────────────────────
        api_status = api_result.status if api_result else "not_found"
        api_matched_title = api_result.matched_title if api_result else ""

        entry_dict = {
            "title": entry.title or "",
            "authors": entry.authors or "",
            "year": entry.year or "",
            "url": entry_url,
            "publisher": getattr(entry, "publisher", "") or "",
            "entry_type": getattr(entry, "entry_type", "") or "",
            "raw_text": getattr(entry, "raw_text", "") or "",
            "api_status": api_status,
            "api_matched_title": api_matched_title,
            "url_note": "",
            "open_access_url": api_result.open_access_url if api_result else None,
        }

        is_fabricated, fab_confidence = _is_fabricated_title(entry.title or "")
        if is_fabricated and fab_confidence >= 0.80:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="manual_review",
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
            )

        web_result = verify_with_web_search(entry_dict, api_status)
        if (web_result.get("status") == "verified"
                and web_result.get("confidence", 0.0) >= 0.75):
            matched = web_result.get("matched_title") or entry.title
            # Only cache when we have an actual author check; web-search
            # results rarely provide author lists, so don't cache.
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified",
                confidence=web_result["confidence"],
                matched_title=matched,
                open_access_url=web_result.get("open_access_url"),
                note=web_result.get("note", "Verified via web search"),
                sources_checked=web_result.get("sources_checked", ["web_search"]),
            )

        # ── STEP 9: ML gate (does NOT write to cache) ─────────────────────────
        from ai_checker import _local_ml_gate
        entry_dict_for_ml = {
            "title": entry.title or "",
            "authors": entry.authors or "",
            "year": entry.year or "",
            "url": entry_url,
            "journal": getattr(entry, "journal", "") or "",
            "booktitle": getattr(entry, "booktitle", "") or "",
            "publisher": getattr(entry, "publisher", "") or "",
            "entry_type": getattr(entry, "entry_type", "") or "",
            "raw_text": getattr(entry, "raw_text", "") or "",
        }
        ml_gate = _local_ml_gate(entry_dict_for_ml)

        return VerificationResult(
            key=entry.key,
            title=entry.title or "",
            status="manual_review",
            confidence=ml_gate["confidence"],
            matched_title=entry.title or "",
            correct_authors=entry.authors or "",
            note=f"Unconfirmed reference: {ml_gate['reason']}",
            sources_checked=["ml_gate"],
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
    Verify all references. Duplicates inherit the canonical's verdict —
    never upgraded to REAL.
    """
    entries = list(bib_dict.values())

    dup_map = get_duplicate_map(bib_dict)
    skip_keys = set(dup_map.keys())

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
    except Exception:
        pass
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

    # Duplicates inherit the canonical's verdict (never upgraded)
    for dup_key, canonical_key in dup_map.items():
        canonical_result = results_by_key.get(canonical_key)
        if canonical_result is None:
            entry = bib_dict.get(dup_key)
            results.append(VerificationResult(
                key=dup_key,
                title=entry.title if entry else "",
                status="manual_review",
                confidence=0.0,
                note=f"Duplicate of [{canonical_key}] but canonical not found.",
                is_duplicate=True,
                duplicate_of=canonical_key,
            ))
            continue

        dup_result = VerificationResult(
            key=dup_key,
            title=canonical_result.title,
            # Inherit, do not upgrade
            status=canonical_result.status,
            confidence=canonical_result.confidence,
            matched_title=canonical_result.matched_title,
            doi=canonical_result.doi,
            open_access_url=canonical_result.open_access_url,
            note=(
                f"Duplicate of [{canonical_key}] — inherits "
                f"'{canonical_result.status}' verdict."
            ),
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
            consistency_issues=canonical_result.consistency_issues,
        )
        results.append(dup_result)

    key_order = list(bib_dict.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)

    results = [_apply_confidence_thresholds(r) for r in results]

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
    """Compute academic submission score."""
    score = 100
    penalties = []

    if not bib_list and xcheck.cited_not_in_bib:
        deduct = 60
        score -= deduct
        penalties.append({
            "category": "No bibliography found",
            "count": len(xcheck.cited_not_in_bib),
            "deduction": deduct,
        })

    missing = len(xcheck.cited_not_in_bib)
    if missing and bib_list:
        deduct = min(missing * 5, 20)
        score -= deduct
        penalties.append({
            "category": "Missing citations",
            "count": missing,
            "deduction": deduct,
        })

    orphaned = len(xcheck.in_bib_not_cited)
    if orphaned:
        deduct = min(orphaned * 2, 10)
        score -= deduct
        penalties.append({
            "category": "Orphaned entries",
            "count": orphaned,
            "deduction": deduct,
        })

    dup_count = len(duplicates)
    if dup_count:
        deduct = min(dup_count * 3, 10)
        score -= deduct
        penalties.append({
            "category": "Duplicates",
            "count": dup_count,
            "deduction": deduct,
        })

    if professor_confirmed_fakes:
        deduct = min(professor_confirmed_fakes * 10, 60)
        score -= deduct
        penalties.append({
            "category": "Confirmed fake references",
            "count": professor_confirmed_fakes,
            "deduction": deduct,
        })

    key_mismatch_count = sum(
        1 for e in bib_list if getattr(e, "key_consistent", None) is False
    )
    if key_mismatch_count:
        deduct = min(key_mismatch_count * 3, 15)
        score -= deduct
        penalties.append({
            "category": "Key mismatches (author/year)",
            "count": key_mismatch_count,
            "deduction": deduct,
        })

    if retracted_count:
        deduct = min(retracted_count * 8, 25)
        score -= deduct
        penalties.append({
            "category": "Retracted papers cited",
            "count": retracted_count,
            "deduction": deduct,
        })

    incomplete = sum(1 for e in bib_list if getattr(e, "completeness_issues", None))
    if incomplete:
        deduct = min(incomplete * 2, 10)
        score -= deduct
        penalties.append({
            "category": "Incomplete entries",
            "count": incomplete,
            "deduction": deduct,
        })

    score = max(0, score)

    entry_quality_score = None
    if verification_results and bib_list:
        real_count = sum(
            1 for v in verification_results if v.get("ai_verdict") == "REAL"
        )
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
    if retracted_count:
        summary_parts.append(f"{retracted_count} retracted paper(s)")
    if missing and bib_list:
        summary_parts.append(f"{missing} missing citation(s)")
    if orphaned:
        summary_parts.append(f"{orphaned} orphaned entry/entries")

    summary = "; ".join(summary_parts) if summary_parts else "No issues detected."

    return {
        "score": score,
        "grade": grade,
        "penalties": penalties,
        "entry_quality_score": entry_quality_score,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Reference extraction from PDF
# ---------------------------------------------------------------------------

def check_references_from_pdf(pdf_path: str) -> dict:
    """End-to-end pipeline: Extract references from PDF and validate."""
    from extractor import extract, extract_references_from_bibliography
    from parser import parse_raw_references

    result = extract(pdf_path)
    references = result.get("references", [])
    parsed = parse_raw_references(references)

    report = {
        "total_references": len(parsed),
        "pdf_file": pdf_path,
        "references": [],
    }

    for entry in parsed:
        ref_report = {
            "key": entry.key,
            "type": entry.entry_type,
            "authors": entry.authors,
            "title": entry.title,
            "year": entry.year,
            "publisher": entry.publisher,
            "venue": entry.booktitle,
            "pages": entry.pages,
            "url": entry.url,
            "completeness_issues": entry.completeness_issues,
            "needs_ai_parsing": entry.needs_ai_parsing,
            "raw_text": entry.raw_text[:100] + "..." if len(entry.raw_text) > 100 else entry.raw_text,
        }
        report["references"].append(ref_report)

    return report