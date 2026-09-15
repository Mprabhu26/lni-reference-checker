"""
STEP 3: Citation Cross-Checker + Reference Verifier — v11.2
-----------------------------------------------------------
Multi-source, multi-field verification with parallel multi-threading,
robust academic fuzzy matching, and automatic plausibility promotion.
"""

import hashlib
import html
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
from api_config import (
    get_crossref_email,
    get_openalex_email,
    get_semantic_scholar_key,
    get_ncbi_key,
    get_unpaywall_email,
    user_agent as _build_ua,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("LNI_TITLE_SIM_THRESHOLD", "0.80"))
AUTHOR_OVERLAP_THRESHOLD = float(os.getenv("LNI_AUTHOR_OVERLAP_THRESHOLD", "0.60"))
DOI_TITLE_SIM_THRESHOLD = float(os.getenv("LNI_DOI_TITLE_SIM_THRESHOLD", "0.90"))

CONFIDENCE_HIGH = 0.90
CONFIDENCE_MODERATE = 0.65
CONFIDENCE_LOW = 0.40

SKIP_SCRAPERS = os.environ.get("LNI_SKIP_SCRAPERS", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Caches
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

def _normalize_metadata_text(value: Any) -> str:
    """Decode HTML entities and normalize whitespace before metadata comparison."""
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\b([A-Z])\s+and\s+([A-Z])\b", r"\1&\2", text, flags=re.IGNORECASE)


def _format_api_authors(authors: list, limit: int = 15) -> Optional[str]:
    names = []
    for author in (authors or [])[:limit]:
        family = str(author.get("family") or "").strip()
        given = str(author.get("given") or "").strip()
        if not family:
            organization = str(author.get("name") or "").strip()
            # CrossRef sometimes places an author's affiliation or a malformed
            # institutional contributor in the author array. LNI author checks
            # must compare people, not affiliations.
            if organization and "," not in organization and len(organization.split()) <= 4:
                family = organization
        if not family:
            continue
        names.append(f"{family}, {given}" if given else family)
    return "; ".join(names) or None

def _normalize_title(t: str) -> str:
    if not t:
        return ""
    t = _normalize_metadata_text(t).lower()
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
# Author surname extraction
# ---------------------------------------------------------------------------

_PARTICLES = {'von', 'van', 'de', 'del', 'della', 'der', 'den',
              'la', 'le', 'du', 'des', 'di', 'da', 'dos', 'das'}


def _normalize_surname(s: str) -> str:
    s = s.lower()
    for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss'),
                 ('à','a'),('á','a'),('â','a'),('ã','a'),
                 ('è','e'),('é','e'),('ê','e'),('ë','e'),
                 ('ì','i'),('í','i'),('î','i'),('ï','i'),
                 ('ò','o'),('ó','o'),('ô','o'),('õ','o'),
                 ('ù','u'),('ú','u'),('û','u'),
                 ('ý','y'),('ÿ','y'),('ñ','n'),('ç','c'),
                 ('š','s'),('ž','z'),('č','c'),('ř','r'),
                 ('ł','l'),('ą','a'),('ę','e'),('ś','s'),
                 ('ź','z'),('ż','z'),('ő','o'),('ű','u')]:
        s = s.replace(a, b)
    return re.sub(r'[^a-z0-9]', '', s)


def _extract_cited_surnames(authors_str: str) -> List[str]:
    if not authors_str:
        return []
    out: List[str] = []
    for part in re.split(r';|\band\b|\bund\b', authors_str, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        # "et al." means the list continues beyond what is printed. Strip the
        # marker and keep the cited surname(s); do NOT break out and drop them.
        part = re.sub(r'[,;]?\s*et\s+al\.?', '', part, flags=re.IGNORECASE).strip()
        if not part:
            continue
        if ',' in part:
            surname = part.split(',')[0].strip()
        else:
            tokens = [t for t in part.lower().split()
                      if t not in _PARTICLES and not re.match(r'^[a-z]\.?$', t)]
            surname = tokens[-1] if tokens else (part.split()[-1] if part.split() else '')
        clean = _normalize_surname(surname)
        if len(clean) >= 2:
            out.append(clean)
    return out


def _extract_source_surnames(authors_str: str) -> List[str]:
    if not authors_str:
        return []
    out: List[str] = []
    for part in re.split(r';|\band\b|\bund\b|\|', authors_str, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        if ',' in part:
            surname = part.split(',')[0].strip()
        else:
            tokens = [t for t in part.lower().split()
                      if t not in _PARTICLES and not re.match(r'^[a-z]\.?$', t)]
            surname = tokens[-1] if tokens else (part.split()[-1] if part.split() else '')
        clean = _normalize_surname(surname)
        if len(clean) >= 2:
            out.append(clean)
    return out


def _surnames_match(cited: str, source: str) -> bool:
    if cited == source:
        return True
    if len(cited) >= 4 and len(source) >= 4 and cited[:4] == source[:4]:
        return True
    if len(cited) >= 5 and len(source) >= 5 and cited[:5] == source[:5]:
        return True
    if cited.replace(' ', '') == source.replace(' ', ''):
        return True
    min_len = min(len(cited), len(source))
    if min_len >= 5 and cited[:min_len] == source[:min_len]:
        return True
    if len(cited) >= 7 and len(source) >= 7 and abs(len(cited)-len(source)) <= 1:
        mismatches = sum(1 for a, b in zip(cited, source) if a != b)
        if mismatches <= 1:
            return True
    return False


def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    cited = _extract_cited_surnames(cited_authors)
    source = _extract_source_surnames(correct_authors)
    if not cited or not source:
        return None
    matched = 0
    for c in cited:
        if any(_surnames_match(c, s) for s in source):
            matched += 1
    # Denominator is the SMALLER of the two lists. A reference abbreviated with
    # "et al." (or a database record truncated to its first authors) lists
    # fewer names than the other side; only the names actually printed are
    # expected to match, so we must not penalise using the longer count.
    n_cited = len(cited)
    n_source = len(source)
    denom = min(n_cited, n_source)
    if denom <= 0:
        return None
    return round(matched / denom, 4)


# ---------------------------------------------------------------------------
# ROBUST FUZZY MATCHING
# ---------------------------------------------------------------------------

def _full_combination_match(entry: BibEntry, source: dict) -> Tuple[bool, Dict[str, Any]]:
    """
    Strict field matching: ALL provided fields must match.
    A missing field in the citation is treated as an INCOMPLETE entry, not a pass.
    """
    checks: Dict[str, Any] = {
        "title": None,
        "authors": None,
        "year": None,
        "venue": None,
        "doi": None,
    }

    # ========== TITLE VALIDATION ==========
    cited_title = _normalize_metadata_text(entry.title)
    src_title = _normalize_metadata_text(source.get("title"))
    title_ok = False
    title_sim = 0.0

    if cited_title and src_title:
        sim = _title_similarity(cited_title, src_title)
        title_sim = sim
        # Title threshold is LOWER for DOI-matched references (they're authoritative)
        threshold = 0.65 if (entry.doi and source.get("doi") and
                           entry.doi.strip().lower() == source["doi"].strip().lower()) else 0.80
        title_ok = (sim >= threshold)
        if title_ok:
            checks["title"] = (True, f"title match: {sim:.4f} >= {threshold:.2f}")
        else:
            checks["title"] = (False, f"title mismatch: similarity {sim:.4f} < {threshold:.2f}")
    elif not cited_title and not src_title:
        # Both missing: cannot validate, skip
        title_ok = True
        checks["title"] = (True, "no title on either side (skipped)")
    else:
        # ONE side has title, other doesn't: FAIL (incomplete reference)
        title_ok = False
        checks["title"] = (False, "title missing on one side (INCOMPLETE)")
        return (False, checks)  # Early exit: incomplete reference

    # ========== AUTHOR VALIDATION (STRICT) ==========
    cited_authors = _normalize_metadata_text(entry.authors)
    src_authors = _normalize_metadata_text(source.get("authors"))
    authors_ok = False  # CRITICAL: Default to False (was True!)

    if cited_authors and src_authors:
        cited_surnames = _extract_cited_surnames(cited_authors)
        src_surnames = _extract_source_surnames(src_authors)

        if cited_surnames and src_surnames:
            matched_count = sum(1 for c in cited_surnames
                              if any(_surnames_match(c, s) for s in src_surnames))
            n_cited = len(cited_surnames)

            if n_cited <= 2:
                overlap_ratio = matched_count / n_cited if n_cited > 0 else 0
                threshold = 1.0
                reason = f"author mismatch: {matched_count}/{n_cited} cited authors matched; expected 100% match"
            else:
                overlap_ratio = matched_count / n_cited if n_cited > 0 else 0
                threshold = 0.60
                reason = f"author mismatch: {matched_count}/{n_cited} cited authors matched; expected >= {threshold*100:.0f}%"

            authors_ok = (overlap_ratio >= threshold)
            if authors_ok:
                checks["authors"] = (True, f"author match: {matched_count}/{n_cited} surnames matched")
            else:
                checks["authors"] = (False, reason)
        else:
            authors_ok = False
            checks["authors"] = (False, "author mismatch: cannot extract surnames from one or both sides")
    elif not cited_authors and not src_authors:
        authors_ok = True
        checks["authors"] = (True, "no authors on either side (skipped)")
    else:
        authors_ok = False
        checks["authors"] = (False, "author mismatch: authors missing on one side (INCOMPLETE)")
        return (False, checks)

    # ========== YEAR VALIDATION ==========
    cited_year = _normalize_metadata_text(entry.year)
    src_year = _normalize_metadata_text(source.get("year"))
    year_ok = True

    if cited_year and src_year:
        m_c = re.search(r'\d{4}', cited_year)
        m_s = re.search(r'\d{4}', src_year)
        if m_c and m_s:
            diff = abs(int(m_c.group()) - int(m_s.group()))
            year_ok = (diff == 0)
            if year_ok:
                checks["year"] = (True, f"year match: {m_c.group()} vs {m_s.group()}")
            else:
                checks["year"] = (False, f"year mismatch: {m_c.group()} vs {m_s.group()} (diff={diff})")
        else:
            checks["year"] = (True, "year not comparable (skipped)")
    elif not cited_year and not src_year:
        checks["year"] = (True, "no year on either side (skipped)")
    else:
        checks["year"] = (True, "year missing on one side (skipped)")

    # ========== VENUE VALIDATION (OPTIONAL) ==========
    cited_venue = _normalize_metadata_text(entry.journal or entry.booktitle)
    src_venue = _normalize_metadata_text(source.get("venue"))
    venue_ok = True

    if cited_venue and src_venue:
        v_sim = _title_similarity(cited_venue, src_venue)
        venue_ok = (v_sim >= 0.60)
        checks["venue"] = (venue_ok, round(v_sim, 4))
    elif not cited_venue and not src_venue:
        # Both missing: skip
        checks["venue"] = (True, "no venue on either side (skipped)")
    else:
        # ONE side has venue: skip (not critical for identification)
        checks["venue"] = (True, "venue missing on one side (skipped)")

    # ========== DOI VALIDATION (OPTIONAL) ==========
    cited_doi = (entry.doi or "").strip().lower()
    src_doi = (source.get("doi") or "").strip().lower()
    doi_ok = True

    if cited_doi and src_doi:
        doi_ok = (cited_doi == src_doi or src_doi.endswith(cited_doi)
                  or cited_doi.endswith(src_doi))
        checks["doi"] = (doi_ok, "doi match" if doi_ok else "doi mismatch")
    elif not cited_doi and not src_doi:
        # Both missing: skip
        checks["doi"] = (True, "no doi on either side (skipped)")
    else:
        # ONE side has DOI: skip (not critical for identification)
        checks["doi"] = (True, "doi missing on one side (skipped)")

    # ========== FINAL VERDICT ==========
    # CRITICAL RULE:
    #   - Title + Year + Authors ALL must pass (if provided)
    #   - Venue and DOI are OPTIONAL (nice-to-have)
    #   - If ANY required field FAILS, the match is REJECTED
    is_matched = title_ok and year_ok and authors_ok

    if not is_matched:
        for key in ["title", "authors", "year"]:
            if key in checks and isinstance(checks[key], tuple) and not _check_ok(checks[key]):
                checks[key] = (False, checks[key][1])

    return (is_matched, checks)

def _get_session(retries: int = 3) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


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
    field_checks: Optional[Dict[str, Any]] = None


def _check_retraction(doi: str) -> tuple:
    if not doi:
        return False, None, None
    try:
        resp = requests.get(f"https://api.crossref.org/works/{doi}",
                            timeout=5, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return False, None, None
        work = resp.json().get("message", {})
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


def _check_unpaywall(doi: str) -> Optional[str]:
    if not doi:
        return None
    email = get_unpaywall_email()
    if not email:
        return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                            params={"email": email}, timeout=5,
                            headers={"User-Agent": _build_ua()})
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.doi:
        return None
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', entry.doi.strip())
    _rate_limit("crossref.org", 0.2)
    try:
        resp = requests.get(f"https://api.crossref.org/works/{doi}",
                            timeout=8, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        work = resp.json().get("message", {})
        title = (work.get("title") or [""])[0]
        authors = work.get("author", [])
        author_str = _format_api_authors(authors, limit=10)
        issued = work.get("issued", {}).get("date-parts", [[None]])[0]
        year = str(issued[0]) if issued and issued[0] else None
        container = (work.get("container-title") or [""])[0]

        source_record = {
            "title": title,
            "authors": author_str,
            "year": year,
            "venue": container or None,
            "doi": doi,
            "url": f"https://doi.org/{doi}",
        }

        ok, checks = _full_combination_match(entry, source_record)

        is_ret, ret_doi, ret_note = _check_retraction(doi)
        if is_ret:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="suspicious", confidence=0.90,
                matched_title=title, doi=doi,
                note=ret_note or "Paper has been retracted.",
                sources_checked=["CrossRef (DOI)"],
                correct_authors=author_str,
                is_retracted=True, retraction_doi=ret_doi,
                retraction_note=ret_note,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=container or None,
                title_match_score=checks.get("title", (0,))[1] if isinstance(checks.get("title"), tuple) else None,
                field_checks=checks,
            )

        if not ok:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="partial_match", confidence=0.55,
                matched_title=title, doi=doi,
                open_access_url=_check_unpaywall(doi),
                note=f"DOI {doi} resolves but some cited fields differ: {_fails_summary(checks)}",
                sources_checked=["CrossRef (DOI)"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=container or None,
                field_checks=checks,
                consistency_issues=[_field_mismatch_label(k) for k, v in checks.items() if v and not _check_ok(v)],
            )

        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="verified", confidence=0.98,
            matched_title=title, doi=doi,
            open_access_url=_check_unpaywall(doi),
            note=f"DOI confirmed via CrossRef; all cited fields match.",
            sources_checked=["CrossRef (DOI)"],
            correct_authors=author_str,
            corrected_title=title, corrected_authors=author_str,
            corrected_year=year, corrected_journal=container or None,
            field_checks=checks,
        )
    except Exception:
        return None


def _field_mismatch_label(key: str) -> str:
    labels = {
        "title": "title mismatch",
        "authors": "author mismatch",
        "year": "year mismatch",
        "venue": "venue mismatch",
        "doi": "doi mismatch",
    }
    return labels.get(key, f"{key} mismatch")


def _fails_summary(checks: Dict[str, Any]) -> str:
    out = []
    for k, v in checks.items():
        if v is None:
            continue
        if not _check_ok(v):
            label = _field_mismatch_label(k)
            detail = v[1] if isinstance(v, tuple) else str(v)
            out.append(f"{label}: {detail}")
    return "; ".join(out) if out else "unknown"


def _check_ok(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, tuple):
        return bool(v[0])
    if isinstance(v, str):
        return False
    return bool(v)


def _lookup_by_arxiv_id(entry: BibEntry) -> Optional[VerificationResult]:
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
        r'arXiv:([a-z\-]+/\d{7})',
    ]
    arxiv_id = None
    for fv in [entry.url or "", entry.doi or "", entry.raw_text or ""]:
        for pat in patterns:
            m = re.search(pat, fv, re.IGNORECASE)
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
                            timeout=8, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        m_title = re.search(r'title\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
        title = m_title.group(1) if m_title else None
        m_auth = re.search(r'author\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
        authors = m_auth.group(1) if m_auth else None
        m_year = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?', resp.text, re.IGNORECASE)
        year = m_year.group(1) if m_year else None

        source_record = {
            "title": title, "authors": authors, "year": year,
            "venue": "arXiv", "doi": None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
        ok, checks = _full_combination_match(entry, source_record)
        if not ok:
            return None

        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="verified", confidence=0.90,
            matched_title=title,
            open_access_url=f"https://arxiv.org/abs/{arxiv_id}",
            note=f"arXiv {arxiv_id} matches all cited fields.",
            sources_checked=["arXiv (ID)"],
            correct_authors=authors,
            corrected_title=title, corrected_year=year,
            field_checks=checks,
        )
    except Exception:
        return None


def _search_openalex(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.openalex.org", 0.2)
    params = {
        "search": entry.title[:200],
        "per-page": 10,
        "mailto": get_openalex_email() or get_crossref_email() or "",
    }
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=10,
            headers={"User-Agent": _build_ua()},
        )
        if resp.status_code != 200:
            return None
        for work in resp.json().get("results", [])[:10]:
            title = work.get("title") or ""
            if not title:
                continue
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author") or {}
                name = (author.get("display_name") or "").strip()
                if name:
                    if "," in name:
                        family, given = [part.strip() for part in name.split(",", 1)]
                    else:
                        name_parts = name.split()
                        family = name_parts[-1]
                        given = " ".join(name_parts[:-1])
                    authors.append({"family": family, "given": given})
            author_str = _format_api_authors(authors)
            year = str(work.get("publication_year")) if work.get("publication_year") else None
            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue = source.get("display_name") or None
            doi_url = work.get("doi") or ""
            doi = re.sub(r"^https?://doi.org/", "", doi_url, flags=re.IGNORECASE) or None
            record = {
                "title": title,
                "authors": author_str,
                "year": year,
                "venue": venue,
                "doi": doi,
                "url": doi_url or work.get("id"),
            }
            ok, checks = _full_combination_match(entry, record)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key,
                title=entry.title or "",
                status="verified",
                confidence=0.94,
                matched_title=title,
                doi=doi,
                open_access_url=(work.get("open_access") or {}).get("oa_url"),
                note="OpenAlex: every cited field matched.",
                sources_checked=["OpenAlex"],
                correct_authors=author_str,
                corrected_title=title,
                corrected_authors=author_str,
                corrected_year=year,
                corrected_journal=venue,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_crossref(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("crossref.org", 0.2)
    params = {"query.title": entry.title, "rows": 30}
    if entry.authors:
        first = entry.authors.split(';')[0].split(',')[0].strip()
        first = re.sub(r'\s+et\s+al\.?$', '', first, flags=re.IGNORECASE)
        if first and len(first) > 2:
            params["query.author"] = first
    # Use venue to narrow search if available
    if entry.journal or entry.booktitle:
        venue = entry.journal or entry.booktitle or ""
        if venue and len(venue) > 5:
            params["query.container-title"] = venue
    # Try first with venue, then without if 0 results
    for attempt in range(2):
        if attempt == 1:
            # Second attempt: remove venue query, keep only title + author
            params.pop("query.container-title", None)
            print(f"  [CrossRef DEBUG] Retrying WITHOUT venue for: {entry.title[:60]!r}")

        resp = requests.get("https://api.crossref.org/works", params=params,
                            timeout=15, headers={"User-Agent": _build_ua()})
        if resp.status_code == 429:
            print(f"  [CrossRef DEBUG] Rate-limited (429) for: {entry.title[:60]!r}")
            return None
        if resp.status_code != 200:
            print(f"  [CrossRef DEBUG] HTTP {resp.status_code} for: {entry.title[:60]!r}")
            return None
        _items = resp.json().get("message", {}).get("items", [])
        if not _items:
            print(f"  [CrossRef DEBUG] 0 results for: {entry.title[:60]!r} (attempt {attempt+1})")
            if attempt == 0:
                continue  # Try again without venue
            else:
                return None  # Both attempts failed
        # Got results, break out of retry loop
        print(f"  [CrossRef DEBUG] {len(_items)} items returned (attempt {attempt+1})")
        break

    best_partial = None
    best_partial_score = 0.0
    try:
        for item in _items:
            title = (item.get("title") or [""])[0]
            if not title:
                continue
            authors = item.get("author", [])
            author_str = _format_api_authors(authors, limit=15)
            issued = item.get("issued", {}).get("date-parts", [[None]])[0]
            year = str(issued[0]) if issued and issued[0] else None
            container = (item.get("container-title") or [""])[0]
            doi = item.get("DOI", "")

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": container or None, "doi": doi,
                   "url": f"https://doi.org/{doi}" if doi else None}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                fails = [k for k, v in checks.items() if v and not _check_ok(v)]
                title_check = checks.get("title")
                title_score = title_check[1] if isinstance(title_check, tuple) and isinstance(title_check[1], (int, float)) else _title_similarity(entry.title, title)
                print(f"  [CrossRef DEBUG] PARTIAL CANDIDATE title={title[:60]!r} year={year} fails={fails}")
                if title_score >= 0.75 and title_score > best_partial_score:
                    best_partial_score = title_score
                    best_partial = VerificationResult(
                        key=entry.key, title=entry.title or "",
                        status="partial_match", confidence=round(max(0.45, min(0.79, title_score * 0.75)), 2),
                        matched_title=title, doi=doi or None,
                        open_access_url=_check_unpaywall(doi) if doi else None,
                        note=f"CrossRef found a likely paper by title, but metadata differs: {_fails_summary(checks)}",
                        sources_checked=["CrossRef"],
                        correct_authors=author_str,
                        corrected_title=title, corrected_authors=author_str,
                        corrected_year=year, corrected_journal=container or None,
                        field_checks=checks,
                        consistency_issues=[_field_mismatch_label(k) for k, v in checks.items() if v and not _check_ok(v)],
                    )
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.92,
                matched_title=title, doi=doi or None,
                open_access_url=_check_unpaywall(doi) if doi else None,
                note="CrossRef: every cited field matched.",
                sources_checked=["CrossRef"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=container or None,
                field_checks=checks,
            )
    except Exception:
        return best_partial
    return best_partial


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    clean_title = entry.title
    for pat in [r'\.\s*In:\s*.*$', r'\.\s*doi:\s*.*$', r'\.\s*https?://\S+$']:
        clean_title = _safe_re_sub(pat, '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.strip().strip('.,;:') or entry.title

    _rate_limit("api.semanticscholar.org", 0.25)
    headers = {"User-Agent": _build_ua()}
    ss_key = get_semantic_scholar_key()
    if ss_key:
        headers["x-api-key"] = ss_key
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": clean_title[:200], "limit": 8,
                    "fields": "title,authors,year,venue,publicationVenue,openAccessPdf,externalIds"},
            timeout=12, headers=headers)
        if resp.status_code != 200:
            print(f"  [SS DEBUG] HTTP {resp.status_code} for: {clean_title[:60]!r}")
            return None
        results = resp.json().get("data", [])
        print(f"  [SS DEBUG] {len(results)} results for: {clean_title[:60]!r}")
        for paper in results[:8]:
            title = paper.get("title", "")
            if not title:
                continue
            authors = paper.get("authors", [])
            author_str = "; ".join(a.get("name", "") for a in authors[:15]) if authors else None
            oa = (paper.get("openAccessPdf") or {}).get("url")
            doi = (paper.get("externalIds") or {}).get("DOI")
            venue = (paper.get("venue")
                     or (paper.get("publicationVenue") or {}).get("name")
                     or None)
            year = str(paper.get("year")) if paper.get("year") else None

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": venue, "doi": doi, "url": oa}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                print(f"  [SS DEBUG] NO MATCH title={title[:50]!r} year={year} venue={venue!r}")
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.90,
                matched_title=title, doi=doi, open_access_url=oa,
                note="Semantic Scholar: every cited field matched.",
                sources_checked=["Semantic Scholar"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=venue,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_dblp(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("dblp.org", 0.5)
    try:
        s = _get_session()
        resp = s.get(
            "https://dblp.org/search/publ/api",
            params={"q": entry.title[:200], "format": "json", "h": 10},
            timeout=12, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
        print(f"  [DBLP DEBUG] {len(hits)} results for: {entry.title[:60]!r}")
        for hit in hits:
            info = hit.get("info", {})
            title = info.get("title", "").rstrip('.')
            if not title:
                continue
            authors_field = info.get("authors", {}).get("author", [])
            if isinstance(authors_field, dict):
                authors_field = [authors_field]
            author_str = "; ".join(
                a.get("text", "") if isinstance(a, dict) else str(a)
                for a in authors_field
            ) if authors_field else None
            year = str(info.get("year", "")) or None
            venue = info.get("venue") or info.get("booktitle") or None
            doi = info.get("doi") or None
            ee = info.get("ee") or None
            url = info.get("url") or None

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": venue, "doi": doi, "url": ee or url}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.92,
                matched_title=title, doi=doi, open_access_url=ee or url,
                note="DBLP: every cited field matched.",
                sources_checked=["DBLP"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=venue,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_arxiv(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("export.arxiv.org", 3.0)
    try:
        s = _get_session()
        resp = s.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{entry.title[:150]}"',
                    "max_results": 8},
            timeout=15, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        entries = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        for e in entries:
            m_title = re.search(r'<title>(.*?)</title>', e, re.DOTALL)
            title = re.sub(r'\s+', ' ', m_title.group(1)).strip() if m_title else ""
            if not title:
                continue
            authors = re.findall(r'<name>(.*?)</name>', e)
            author_str = "; ".join(authors) if authors else None
            m_pub = re.search(r'<published>(\d{4})', e)
            year = m_pub.group(1) if m_pub else None
            m_id = re.search(r'<id>(.*?)</id>', e)
            url = m_id.group(1) if m_id else None

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": "arXiv", "doi": None, "url": url}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.88,
                matched_title=title, open_access_url=url,
                note="arXiv: every cited field matched.",
                sources_checked=["arXiv"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_pubmed(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("eutils.ncbi.nlm.nih.gov", 0.4)
    try:
        s = _get_session()
        ncbi_params: dict = {"db": "pubmed", "term": entry.title[:180],
                             "retmode": "json", "retmax": 5}
        ncbi_key = get_ncbi_key()
        if ncbi_key:
            ncbi_params["api_key"] = ncbi_key
        r1 = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                   params=ncbi_params,
                   timeout=12, headers={"User-Agent": _build_ua()})
        if r1.status_code != 200:
            return None
        ids = r1.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        r2 = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                   params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                   timeout=12, headers={"User-Agent": _build_ua()})
        if r2.status_code != 200:
            return None
        result = r2.json().get("result", {})
        for pmid in ids:
            doc = result.get(pmid, {})
            title = (doc.get("title") or "").rstrip('.')
            if not title:
                continue
            authors = [a.get("name", "") for a in doc.get("authors", [])]
            author_str = "; ".join(authors) if authors else None
            year = (doc.get("pubdate") or "").split(" ")[0] if doc.get("pubdate") else None
            venue = doc.get("fulljournalname") or doc.get("source") or None
            doi = None
            for aid in doc.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": venue, "doi": doi,
                   "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.90,
                matched_title=title, doi=doi,
                open_access_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                note="PubMed: every cited field matched.",
                sources_checked=["PubMed"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=venue,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_datacite(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.datacite.org", 0.5)
    try:
        s = _get_session()
        resp = s.get("https://api.datacite.org/dois",
                     params={"query": entry.title[:200], "page[size]": 8},
                     timeout=15,
                     headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        for item in resp.json().get("data", [])[:8]:
            attrs = item.get("attributes", {})
            titles = attrs.get("titles") or []
            title = titles[0].get("title", "") if titles else ""
            if not title:
                continue
            creators = attrs.get("creators") or []
            author_str = "; ".join(
                c.get("name", "") or f"{c.get('familyName','')}, {c.get('givenName','')}".strip(', ')
                for c in creators[:15]
            ) if creators else None
            year = str(attrs.get("publicationYear") or "") or None
            publisher = attrs.get("publisher") or None
            doi = attrs.get("doi") or None

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": publisher, "doi": doi,
                   "url": f"https://doi.org/{doi}" if doi else None}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.88,
                matched_title=title, doi=doi,
                open_access_url=f"https://doi.org/{doi}" if doi else None,
                note="DataCite: every cited field matched.",
                sources_checked=["DataCite"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=publisher,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_openaire(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.openaire.eu", 0.5)
    try:
        s = _get_session()
        resp = s.get(
            "https://api.openaire.eu/search/publications",
            params={"title": entry.title[:180], "size": 8, "format": "json"},
            timeout=15, headers={"User-Agent": _build_ua()})
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = (data.get("response", {})
                       .get("results", {})
                       .get("result", []))
        if isinstance(results, dict):
            results = [results]
        for r in results[:8]:
            try:
                meta = r["metadata"]["oaf:entity"]["oaf:result"]
                title_obj = meta.get("title", {})
                if isinstance(title_obj, list):
                    title_obj = title_obj[0]
                title = title_obj.get("$", "") if isinstance(title_obj, dict) else str(title_obj)
                if not title:
                    continue
                creators = meta.get("creator", [])
                if isinstance(creators, dict):
                    creators = [creators]
                author_str = "; ".join(
                    c.get("$", "") if isinstance(c, dict) else str(c)
                    for c in creators
                ) if creators else None
                date_obj = meta.get("dateofacceptance", {})
                year = None
                if isinstance(date_obj, dict):
                    year = (date_obj.get("$", "") or "")[:4]
                doi = None
                pid = meta.get("pid", [])
                if isinstance(pid, dict):
                    pid = [pid]
                for p in pid:
                    if isinstance(p, dict) and p.get("@classid") == "doi":
                        doi = p.get("$", "")
                        break

                rec = {"title": title, "authors": author_str, "year": year,
                       "venue": None, "doi": doi, "url": None}
                ok, checks = _full_combination_match(entry, rec)
                if not ok:
                    continue
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=0.85,
                    matched_title=title, doi=doi,
                    note="OpenAIRE: every cited field matched.",
                    sources_checked=["OpenAIRE"],
                    correct_authors=author_str,
                    corrected_title=title, corrected_authors=author_str,
                    corrected_year=year,
                    field_checks=checks,
                )
            except Exception:
                continue
    except Exception:
        return None
    return None


def _search_base(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("base-search.net", 2.0)
    try:
        s = _get_session()
        resp = s.get(
            "https://www.base-search.net/Search/Results",
            params={"lookfor": entry.title[:150], "type": "tit"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LNI-Checker/11.0)"})
        if resp.status_code != 200:
            return None
        blocks = re.findall(
            r'<span class="record-title[^"]*">(.*?)</span>',
            resp.text, re.DOTALL)
        for raw in blocks[:8]:
            title = re.sub(r'<[^>]+>', '', raw)
            title = re.sub(r'\s+', ' ', title).strip()
            if not title:
                continue
            rec = {"title": title, "authors": None, "year": None,
                   "venue": None, "doi": None, "url": None}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.75,
                matched_title=title,
                note="BASE: title match only (source returns limited metadata).",
                sources_checked=["BASE"],
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_google_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if SKIP_SCRAPERS or not entry.title:
        return None
    _rate_limit("scholar.google.com", 5.0)
    try:
        s = _get_session(retries=0)
        resp = s.get(
            "https://scholar.google.com/scholar",
            params={"q": entry.title[:180]},
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0.0.0 Safari/537.36"})
        if resp.status_code != 200:
            return None
        results = re.findall(
            r'<h3 class="gs_rt".*?>(.*?)</h3>.*?<div class="gs_a">(.*?)</div>',
            resp.text, re.DOTALL)
        for raw_title, raw_meta in results[:8]:
            title = re.sub(r'<[^>]+>', '', raw_title).strip()
            if not title:
                continue
            meta_clean = re.sub(r'<[^>]+>', '', raw_meta).strip()
            author_str = None
            venue = None
            year = None
            if ' - ' in meta_clean:
                parts = [p.strip() for p in meta_clean.split(' - ')]
                if parts:
                    author_str = parts[0]
                if len(parts) >= 2:
                    mid = parts[1]
                    m = re.search(r'(\d{4})', mid)
                    if m:
                        year = m.group(1)
                    venue = re.sub(r',\s*\d{4}', '', mid).strip(', ').strip()

            rec = {"title": title, "authors": author_str, "year": year,
                   "venue": venue, "doi": None, "url": None}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.85,
                matched_title=title,
                note="Google Scholar: every cited field matched.",
                sources_checked=["Google Scholar"],
                correct_authors=author_str,
                corrected_title=title, corrected_authors=author_str,
                corrected_year=year, corrected_journal=venue,
                field_checks=checks,
            )
    except Exception:
        return None
    return None


def _search_researchgate(entry: BibEntry) -> Optional[VerificationResult]:
    if SKIP_SCRAPERS or not entry.title:
        return None
    _rate_limit("www.researchgate.net", 5.0)
    try:
        s = _get_session(retries=0)
        resp = s.get(
            "https://www.researchgate.net/search/publication",
            params={"q": entry.title[:180]},
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0.0.0 Safari/537.36"})
        if resp.status_code != 200:
            return None
        candidates = re.findall(r'<h3[^>]*>(.*?)</h3>', resp.text, re.DOTALL)
        for raw in candidates[:8]:
            title = re.sub(r'<[^>]+>', '', raw).strip()
            if not title or len(title) < 10:
                continue
            rec = {"title": title, "authors": None, "year": None,
                   "venue": None, "doi": None, "url": None}
            ok, checks = _full_combination_match(entry, rec)
            if not ok:
                continue
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified", confidence=0.75,
                matched_title=title,
                note="ResearchGate: title match only (scraped).",
                sources_checked=["ResearchGate"],
                field_checks=checks,
            )
    except Exception:
        return None
    return None


_SOURCE_CHAIN = [
    ("DOI", _lookup_by_doi),
    ("arXiv-ID", _lookup_by_arxiv_id),
    ("OpenAlex", _search_openalex),
    ("CrossRef", _search_crossref),
    ("SemanticScholar", _search_semantic_scholar),
    ("DBLP", _search_dblp),
    ("arXiv", _search_arxiv),
    ("PubMed", _search_pubmed),
    ("DataCite", _search_datacite),
    ("OpenAIRE", _search_openaire),
    ("BASE", _search_base),
    ("GoogleScholar", _search_google_scholar),
    ("ResearchGate", _search_researchgate),
]

_SOURCE_PRIORITY_ORDER = [
    "CrossRef (DOI)", "CrossRef", "OpenAlex", "Semantic Scholar", "DBLP",
    "arXiv (ID)", "arXiv", "PubMed", "DataCite", "OpenAIRE",
    "BASE", "Google Scholar", "ResearchGate",
]


def _source_priority(r: "VerificationResult"):
    src_name = r.sources_checked[0] if r.sources_checked else ""
    try:
        return (_SOURCE_PRIORITY_ORDER.index(src_name), -r.confidence)
    except ValueError:
        return (len(_SOURCE_PRIORITY_ORDER), -r.confidence)


def future_map_name(futures: Dict, future, default: str) -> str:
    for fut, name in futures.items():
        if fut is future:
            return name
    return default


def _try_all_sources(entry: BibEntry) -> Tuple[Optional[VerificationResult], List[str]]:
    """
    Walk source chain with STRICT timeout management.
    - Per-source timeout: 8 seconds (hard limit)
    - Total collective timeout: 60 seconds (not 40)
    - Early exit on FIRST verified result
    - Graceful degradation on timeouts (skip, don't cascade)
    """
    tried = []
    verified_results = []
    partial_matches = []

    # Optimized source chain: fastest sources first (reduces average time)
    optimized_chain = [
        ("DOI", _lookup_by_doi),                # 2-3 sec if DOI present
        ("arXiv-ID", _lookup_by_arxiv_id),     # 3-5 sec if arXiv ID present
        ("OpenAlex", _search_openalex),        # ~2 sec (fast API)
        ("CrossRef", _search_crossref),        # ~2 sec (fast API)
        ("SemanticScholar", _search_semantic_scholar),  # ~3 sec
        ("DBLP", _search_dblp),               # ~3 sec
        ("arXiv", _search_arxiv),             # ~5 sec (has rate limit)
        ("PubMed", _search_pubmed),           # ~3 sec
        ("DataCite", _search_datacite),       # ~2 sec
        ("OpenAIRE", _search_openaire),       # ~2 sec
        ("BASE", _search_base),               # ~4-5 sec (scraping)
        ("GoogleScholar", _search_google_scholar),  # Skip by default
        ("ResearchGate", _search_researchgate),    # Skip by default
    ]

    def query_source(item):
        name, fn = item
        try:
            return name, fn(entry)
        except requests.exceptions.Timeout:
            return name, "timeout"  # Return marker, not None
        except Exception:
            return name, None

    # Reduce to 2 workers (even fewer than before) to avoid cascade timeouts
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="verifier") as executor:
        futures = {executor.submit(query_source, item): item[0] for item in optimized_chain}

        # CRITICAL: Collective timeout is PER BATCH, not wall clock
        try:
            for future in as_completed(futures, timeout=60):  # Increased to 60 sec
                try:
                    name, result = future.result(timeout=8)  # Per-future timeout: 8 sec
                    tried.append(name)

                    # Skip timeout markers entirely
                    if result == "timeout":
                        continue

                    if result is not None:
                        if result.is_retracted:
                            # Early exit on retraction
                            executor.shutdown(wait=False, cancel_futures=True)
                            return result, tried
                        elif result.status == "verified":
                            verified_results.append(result)
                            # CRITICAL: Exit immediately on first verified result
                            # Do NOT wait for remaining sources
                            executor.shutdown(wait=False, cancel_futures=True)
                            return verified_results[0], tried
                        elif result.status == "partial_match":
                            partial_matches.append(result)

                except TimeoutError:
                    tried.append(f"{name} (timeout)")
                    continue  # Skip and try next
                except Exception:
                    tried.append(f"{name} (error)")
                    continue

        except TimeoutError:
            # 60-second wall clock exceeded; return best result so far
            pass

    # Fallback chain: return best available result
    if verified_results:
        priority_order = ["CrossRef (DOI)", "CrossRef", "OpenAlex",
                        "Semantic Scholar", "DBLP", "arXiv (ID)",
                        "arXiv", "PubMed", "DataCite", "OpenAIRE",
                        "BASE", "Google Scholar", "ResearchGate"]
        def _sort_key(r):
            src_name = r.sources_checked[0] if r.sources_checked else ""
            try:
                return (priority_order.index(src_name), -r.confidence)
            except ValueError:
                return (len(priority_order), -r.confidence)
        verified_results.sort(key=_sort_key)
        return verified_results[0], tried

    if partial_matches:
        partial_matches.sort(key=lambda r: -r.confidence)
        return partial_matches[0], tried

    return None, tried


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


def extract_citations_from_body(body: str) -> set:
    keys = set()
    if not body:
        return keys

    def normalize_brackets(m):
        return '[' + re.sub(r'\s+', '', m.group(1)) + ']'

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
        return '[' + re.sub(r'\s+', '', m.group(1)) + ']'

    body_clean = re.sub(r'\[([A-Za-z0-9\s\n,;\-]+)\]', normalize_brackets, body)
    for m in re.finditer(
        r'(.{0,80})(\[[A-Za-z][A-Za-z0-9+]{0,40}(?:[,;]\s*[A-Za-z][A-Za-z0-9+]{0,40})*\]|\[\d[\d,;\s\-]*\])(.{0,80})',
        body_clean,
    ):
        pre, cite, post = m.group(1), m.group(2), m.group(3)
        for k in re.split(r'\s*[,;]\s*', cite[1:-1]):
            k = k.strip()
            if k:
                contexts.setdefault(k, []).append(f"...{pre}{cite}{post}...".strip())
    return contexts


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
    numeric_cited = {k[6:-2] for k in cited_keys
                     if k.startswith('__NUM_') and k.endswith('__')}
    real_cited.update(numeric_cited & bib_keys)
    real_cited = {bib_key_lookup.get(key.lower(), key) for key in real_cited}
    missing_numeric = numeric_cited - bib_keys
    result.correctly_used = sorted(real_cited & bib_keys)
    numeric_missing = (missing_numeric if bib_keys and all(k.isdigit() for k in bib_keys)
                       else set())
    result.cited_not_in_bib = sorted(
        key for key in (real_cited - bib_keys) | numeric_missing
        if key.lower() not in bib_key_lookup
    )
    result.in_bib_not_cited = sorted(bib_keys - real_cited)
    return result


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
                            "key_a": a.key, "key_b": b.key, "similarity": 1.0,
                            "title_a": a.title or "", "title_b": b.title or "",
                            "reason": "Same DOI",
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
                w = t.lower().split()
                return len(w) < 3 or (len(w) == 1 and t.isupper())
            if is_generic_title(a.title) or is_generic_title(b.title):
                continue
            sim = _title_similarity(a.title, b.title)
            if sim < threshold:
                continue
            if a.authors and b.authors:
                overlap = author_overlap_score(a.authors, b.authors)
                if overlap is None or overlap < AUTHOR_OVERLAP_THRESHOLD:
                    continue
            seen_pairs.add(pair)
            duplicates.append({
                "key_a": a.key, "key_b": b.key, "similarity": round(sim, 3),
                "title_a": a.title, "title_b": b.title,
                "reason": "Title similarity + author overlap",
            })
    return duplicates


def get_duplicate_map(bib_dict: dict) -> Dict[str, str]:
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


def detect_self_citations(bib_dict: dict, body: str) -> List[dict]:
    self_cites = []
    body_lower = body.lower()
    self_signals = re.findall(
        r'(?:we|our|this paper|this work|the author|the authors|i )(?:.{0,60})'
        r'(\[[A-Z][A-Za-z+]{0,5}\d{2}[^\]]*\])',
        body_lower)
    self_keys = set()
    for match in self_signals:
        for k in re.split(r',\s*', match[1:-1]):
            self_keys.add(k.strip().upper())
    for key, entry in bib_dict.items():
        if key.upper() in self_keys:
            self_cites.append({
                "key": key, "title": entry.title or "",
                "reason": "Citation appears near self-referential language",
                "matched_author": entry.authors or "",
            })
    return self_cites


def _validate_entry_fields(entry: BibEntry) -> Optional[VerificationResult]:
    if entry.entry_type in ("website", "online"):
        if not entry.title:
            return VerificationResult(
                key=entry.key, title="", status="incomplete", confidence=0.0,
                note="Missing required fields: title.",
                sources_checked=["structural_validation"])
        return None

    has_title = entry.title and entry.title.strip()
    has_authors = entry.authors and entry.authors.strip()
    has_year = entry.year and entry.year.strip()

    if not has_title:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="incomplete", confidence=0.0,
            note="Missing title - cannot identify reference.",
            sources_checked=["structural_validation"])

    if not (has_authors or has_year):
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="incomplete", confidence=0.0,
            note="Missing both authors and year - cannot identify reference.",
            sources_checked=["structural_validation"])

    return None


def _is_landmark_paper_local(title: str, key: str = "") -> Optional[Dict]:
    if not title:
        return None
    try:
        from author_journal_verifier import (
            LANDMARK_PAPERS, _title_similarity_simple,
        )
        for landmark_title, info in LANDMARK_PAPERS.items():
            sim = _title_similarity_simple(title, landmark_title)
            matched_key = any(p.lower() in key.lower()
                              for p in info.get("key_patterns", []))
            if sim >= 0.70 or (sim >= 0.55 and matched_key):
                return {**info, "matched_by": "title_similarity",
                        "similarity": sim, "landmark_title": landmark_title}
    except Exception:
        pass
    return None


def _apply_confidence_thresholds(result: "VerificationResult") -> "VerificationResult":
    if result.confidence >= CONFIDENCE_HIGH:
        result.confidence_tier = "high"
    elif result.confidence >= CONFIDENCE_MODERATE:
        result.confidence_tier = "moderate"
    else:
        result.confidence_tier = "low"
    return result


def verify_reference(
    entry: BibEntry,
    dup_map: dict = None,
    allow_ai_fallback: bool = True,
) -> VerificationResult:
    """
    Main entry point for reference verification.

    Flow:
    1. Check professor review (highest priority)
    2. Check duplicate map
    3. Validate entry fields (critical: title+authors+year)
    4. Check local cache
    5. Check landmarks
    6. Try all academic databases (with timeout protection)
    7. If databases fail: hand to AI for final assessment
    8. NEVER auto-promote "complete but unconfirmed" to REAL
    """
    try:
        # ===== PROFESSOR REVIEW =====
        review = get_review_decision(entry.title or "", entry.authors or "")
        if review:
            decision = (review.get("decision") or "").lower()
            if decision in ("verified", "real", "accepted"):
                save_to_cache(entry.title or "", entry.authors or "",
                              entry.year or "", entry.doi or "", entry.url or "",
                              source="professor_review", confidence=1.0)
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=1.0,
                    matched_title=entry.title or "",
                    correct_authors=entry.authors or "",
                    note="Confirmed by professor review.",
                    sources_checked=["professor_review"])
            if decision in ("rejected", "fake"):
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="suspicious", confidence=1.0,
                    matched_title=entry.title or "",
                    note="Marked as suspicious by professor review.",
                    sources_checked=["professor_review"])

        # ===== DUPLICATE CHECK =====
        if dup_map and entry.key in dup_map:
            canonical = dup_map[entry.key]
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="manual_review", confidence=0.0,
                note=f"Duplicate of [{canonical}] — verdict inherited.",
                sources_checked=["duplicate_detection"],
                is_duplicate=True, duplicate_of=canonical)

        # ===== STRUCTURAL VALIDATION =====
        # This is a FILTER, not a verdict. We proceed even if structure is poor.
        fe = _validate_entry_fields(entry)
        if fe:
            return fe

        if not entry.title and not entry.doi:
            return VerificationResult(
                key=entry.key, title="", status="incomplete", confidence=0.0,
                note="No title or DOI — cannot verify.", sources_checked=[])

        # ===== LOCAL CACHE CHECK =====
        cached = search_cache(entry.title or "", entry.authors or "")
        if cached:
            rec = {"title": cached.title, "authors": cached.authors,
                   "year": cached.year, "venue": None,
                   "doi": cached.doi, "url": cached.url}
            ok, checks = _full_combination_match(entry, rec)
            if ok:
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified",
                    confidence=max(cached.confidence, 0.90),
                    matched_title=cached.title, doi=cached.doi,
                    open_access_url=cached.url,
                    note=f"Local DB ({cached.source}): all required fields matched.",
                    sources_checked=["local_db"],
                    correct_authors=cached.authors,
                    corrected_title=cached.title,
                    corrected_authors=cached.authors,
                    corrected_year=cached.year,
                    field_checks=checks)

        # ===== LANDMARK PAPERS =====
        lm = _is_landmark_paper_local(entry.title or "", entry.key)
        if lm:
            lm_authors = lm.get("authors", [])
            lm_year = lm.get("year", "")
            lm_venue = lm.get("venue", "")
            rec = {"title": lm.get("title", entry.title or ""),
                   "authors": "; ".join(lm_authors) if isinstance(lm_authors, list)
                              else str(lm_authors),
                   "year": lm_year, "venue": lm_venue,
                   "doi": None, "url": lm.get("url", "")}
            ok, checks = _full_combination_match(entry, rec)
            if ok:
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="verified", confidence=0.99,
                    matched_title=rec["title"],
                    correct_authors=rec["authors"],
                    corrected_year=lm_year, corrected_journal=lm_venue,
                    open_access_url=lm.get("url", ""),
                    note=f"Landmark paper confirmed: {rec['title'][:60]}",
                    sources_checked=["landmark_detection"],
                    field_checks=checks)

        # ===== FABRICATION CHECK =====
        is_fab, fab_conf = _is_fabricated_title(entry.title or "")
        if is_fab and fab_conf >= 0.95:
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="suspicious", confidence=fab_conf,
                note="Title matches known fabrication patterns.",
                sources_checked=["fabrication_detector"])

        # ===== ACADEMIC DATABASE CHAIN =====
        src_result, tried = _try_all_sources(entry)
        if src_result is not None:
            if src_result.status == "verified":
                # Cache the result for future lookups
                save_to_cache(
                    title=src_result.matched_title or entry.title or "",
                    authors=entry.authors or "",
                    year=src_result.corrected_year or entry.year or "",
                    doi=src_result.doi or entry.doi or "",
                    url=src_result.open_access_url or "",
                    source=(src_result.sources_checked[0]
                            if src_result.sources_checked else "api"),
                    confidence=src_result.confidence)
            return src_result

        # ===== WEB SEARCH FALLBACK =====
        entry_dict = {
            "title": entry.title or "",
            "authors": entry.authors or "",
            "year": entry.year or "",
            "url": getattr(entry, "url", "") or "",
            "publisher": getattr(entry, "publisher", "") or "",
            "entry_type": getattr(entry, "entry_type", "") or "",
            "raw_text": getattr(entry, "raw_text", "") or "",
            "api_status": "not_found",
            "api_matched_title": "",
            "url_note": "",
            "open_access_url": None,
        }

        web_result = verify_with_web_search(entry_dict, "not_found")
        if (web_result.get("status") == "verified"
                and web_result.get("confidence", 0.0) >= 0.75):
            return VerificationResult(
                key=entry.key, title=entry.title or "",
                status="verified",
                confidence=web_result["confidence"],
                matched_title=web_result.get("matched_title") or entry.title,
                open_access_url=web_result.get("open_access_url"),
                note=web_result.get("note", "Verified via web search."),
                sources_checked=web_result.get("sources_checked", ["web_search"]),
                field_checks={"web_search": (True, "web search confirmed")})

        # ===== AI FALLBACK ASSESSMENT =====
        # At this point all academic databases and web search failed to
        # confirm the reference. The AI audit is run ONCE, one layer up, by
        # app.py -> ai_checker.ai_verify_references on every unresolved
        # reference. It must NOT be duplicated here, and it must NOT import
        # names that do not exist in ai_checker (older code imported
        # `_is_ai_available` / `run_verification_assessment`, which are not
        # defined - that ImportError was caught by the outer except and made
        # EVERY database-unconfirmed reference return manual_review).
        #
        # Escalate cleanly to manual_review. app.py will hand this entry to
        # the real AI auditor and let the professor decide the final verdict.
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="manual_review",
            confidence=0.0,
              note=f"No exact metadata match was found. Sources tried: {', '.join(tried) or 'none'}. "
                  f"AI review was requested because the academic-source checks did not establish identity; "
                  f"manual review is required only to resolve the remaining uncertainty.",
              sources_checked=tried or ["api_exhausted"])

    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="manual_review", confidence=0.0,
            note=f"Verification error: {str(e)[:120]}",
            sources_checked=["error"])


def verify_all_references(bib_dict: dict) -> List[VerificationResult]:
    entries = list(bib_dict.values())
    dup_map = get_duplicate_map(bib_dict)
    skip_keys = set(dup_map.keys())
    unique_entries = [e for e in entries if e.key not in skip_keys]

    results = []
    results_by_key = {}

    worker_count = min(8, max(1, len(unique_entries)))
    ex = ThreadPoolExecutor(max_workers=worker_count)
    future_map = {ex.submit(verify_reference, e, dup_map): e for e in unique_entries}

    completed_keys = set()
    # 300s was far too long when many references each hang on a slow source.
    # Per-reference verification is already bounded (see _try_all_sources), so
    # a 120s wall-clock cap for the whole batch is comfortable.
    batch_timeout = 120
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
                    note=f"Verification error: {exc}")
            results.append(result)
            results_by_key[result.key] = result
    except Exception:
        pass
    finally:
        for future in future_map:
            if not future.done():
                future.cancel()
        ex.shutdown(wait=False, cancel_futures=True)  # never wait on hung tasks

    for e in unique_entries:
        if e.key not in completed_keys:
            r = VerificationResult(
                key=e.key, title=e.title or "",
                status="manual_review", confidence=0.0,
                note=f"Verification timed out after {batch_timeout} seconds.")
            results.append(r)
            results_by_key[e.key] = r

    for dup_key, canonical_key in dup_map.items():
        canonical_result = results_by_key.get(canonical_key)
        if canonical_result is None:
            entry = bib_dict.get(dup_key)
            results.append(VerificationResult(
                key=dup_key, title=entry.title if entry else "",
                status="manual_review", confidence=0.0,
                note=f"Duplicate of [{canonical_key}] but canonical not found.",
                is_duplicate=True, duplicate_of=canonical_key))
            continue
        dup_result = VerificationResult(
            key=dup_key, title=canonical_result.title,
            status=canonical_result.status,
            confidence=canonical_result.confidence,
            matched_title=canonical_result.matched_title,
            doi=canonical_result.doi,
            open_access_url=canonical_result.open_access_url,
            note=f"Duplicate of [{canonical_key}] — inherits '{canonical_result.status}'.",
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
            is_duplicate=True, duplicate_of=canonical_key,
            consistency_issues=canonical_result.consistency_issues,
            field_checks=canonical_result.field_checks)
        results.append(dup_result)

    key_order = list(bib_dict.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)
    results = [_apply_confidence_thresholds(r) for r in results]
    return results


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
    score = 100
    penalties = []

    if not bib_list and xcheck.cited_not_in_bib:
        score -= 60
        penalties.append({"category": "No bibliography found",
                          "count": len(xcheck.cited_not_in_bib), "deduction": 60})

    missing = len(xcheck.cited_not_in_bib)
    if missing and bib_list:
        d = min(missing * 5, 20)
        score -= d
        penalties.append({"category": "Missing citations", "count": missing, "deduction": d})

    orphaned = len(xcheck.in_bib_not_cited)
    if orphaned:
        d = min(orphaned * 2, 10)
        score -= d
        penalties.append({"category": "Orphaned entries", "count": orphaned, "deduction": d})

    dup_count = len(duplicates)
    if dup_count:
        d = min(dup_count * 3, 10)
        score -= d
        penalties.append({"category": "Duplicates", "count": dup_count, "deduction": d})

    if professor_confirmed_fakes:
        d = min(professor_confirmed_fakes * 10, 60)
        score -= d
        penalties.append({"category": "Confirmed fake references",
                          "count": professor_confirmed_fakes, "deduction": d})

    key_mismatch = sum(1 for e in bib_list if getattr(e, "key_consistent", None) is False)
    if key_mismatch:
        d = min(key_mismatch * 3, 15)
        score -= d
        penalties.append({"category": "Key mismatches (author/year)",
                          "count": key_mismatch, "deduction": d})

    if retracted_count:
        d = min(retracted_count * 8, 25)
        score -= d
        penalties.append({"category": "Retracted papers cited",
                          "count": retracted_count, "deduction": d})

    incomplete = sum(1 for e in bib_list if getattr(e, "completeness_issues", None))
    if incomplete:
        d = min(incomplete * 2, 10)
        score -= d
        penalties.append({"category": "Incomplete entries",
                          "count": incomplete, "deduction": d})

    score = max(0, score)
    grade = ("A" if score >= 90 else "B" if score >= 80 else
             "C" if score >= 60 else "D" if score >= 50 else "F")

    summary_parts = []
    if professor_confirmed_fakes:
        summary_parts.append(f"{professor_confirmed_fakes} confirmed fake")
    if retracted_count:
        summary_parts.append(f"{retracted_count} retracted")
    if missing and bib_list:
        summary_parts.append(f"{missing} missing citation(s)")
    if orphaned:
        summary_parts.append(f"{orphaned} orphaned")
    summary = "; ".join(summary_parts) if summary_parts else "No issues detected."

    return {"score": score, "grade": grade, "penalties": penalties,
            "entry_quality_score": None, "summary": summary}


def check_references_from_pdf(pdf_path: str) -> dict:
    from extractor import extract
    from parser import parse_raw_references

    result = extract(pdf_path)
    references = result.get("references", [])
    parsed = parse_raw_references(references)
    report = {"total_references": len(parsed), "pdf_file": pdf_path, "references": []}
    for entry in parsed:
        report["references"].append({
            "key": entry.key, "type": entry.entry_type,
            "authors": entry.authors, "title": entry.title, "year": entry.year,
            "publisher": entry.publisher, "venue": entry.booktitle,
            "pages": entry.pages, "url": entry.url,
            "completeness_issues": entry.completeness_issues,
            "needs_ai_parsing": entry.needs_ai_parsing,
            "raw_text": entry.raw_text[:100] + "..." if len(entry.raw_text) > 100 else entry.raw_text,
        })
    return report
