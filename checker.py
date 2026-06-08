"""
STEP 3: Citation Cross-Checker + Reference Verifier — v7.0
----------------------------------------------------------
CORRECTED FLOW v7.0:
  1. SQLite DB check ONLY (no browser/disk cache)
  2. If URL exists → fetch directly (bypass APIs)
  3. Only then check academic APIs
  4. AI as last resort
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
from web_search_verifier import verify_with_web_search
from review_queue import is_venue_whitelisted, get_review_decision, get_false_positive

# ---------------------------------------------------------------------------
# Persistent disk cache - DISABLED (only SQLite DB)
# ---------------------------------------------------------------------------

_DISK_CACHE_DIR: str = os.environ.get("LNI_CACHE_DIR", ".lni_cache")
_DISK_CACHE_LOCK = threading.Lock()
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


def _normalize_title(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    diacritic_map = [
        ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
        ('à', 'a'),  ('á', 'a'),  ('â', 'a'),  ('ã', 'a'),
        ('è', 'e'),  ('é', 'e'),  ('ê', 'e'),  ('ë', 'e'),
        ('ì', 'i'),  ('í', 'i'),  ('î', 'i'),  ('ï', 'i'),
        ('ò', 'o'),  ('ó', 'o'),  ('ô', 'o'),  ('õ', 'o'),
        ('ù', 'u'),  ('ú', 'u'),  ('û', 'u'),
        ('ý', 'y'),  ('ÿ', 'y'),
        ('ñ', 'n'),  ('ç', 'c'),  ('ø', 'o'),  ('å', 'aa'),
        ('æ', 'ae'), ('œ', 'oe'),
    ]
    for src, dst in diacritic_map:
        t = t.replace(src, dst)
    t = re.sub(r'&[a-z]+;', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'[{}]', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    stop = {
        'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 'its',
        'using', 'based', 'towards', 'toward', 'via', 'approach',
        'der', 'die', 'das', 'und', 'fur', 'fuer', 'von', 'mit', 'im', 'an',
        'zu', 'zur', 'zum', 'eine', 'ein', 'des', 'dem', 'den',
        'is', 'are', 'was', 'were', 'be', 'by', 'at', 'or', 'not',
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
        fuzzy_score = max(score_a, score_b)
    except ImportError:
        from difflib import SequenceMatcher
        fuzzy_score = SequenceMatcher(None, t1, t2).ratio()
    words1 = set(t1.split())
    words2 = set(t2.split())
    sig1 = {w for w in words1 if len(w) >= 5}
    sig2 = {w for w in words2 if len(w) >= 5}
    if sig1 and sig2:
        overlap = len(sig1 & sig2) / max(len(sig1), len(sig2))
    else:
        overlap = 0.0
    combined = 0.75 * fuzzy_score + 0.25 * overlap
    return round(min(combined, 1.0), 4)


def _extract_surnames(s: str) -> List[str]:
    out = []
    for part in re.split(r';|\band\b|\bund\b', s, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        part_lower = part.lower()
        for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
                     ('à', 'a'), ('é', 'e'), ('è', 'e'), ('ñ', 'n')]:
            part_lower = part_lower.replace(a, b)
        if re.match(r'^et\s+al\.?$', part_lower.strip()):
            continue
        if ',' in part_lower:
            surname_part = part_lower.split(',')[0].strip()
        else:
            tokens = part_lower.split()
            particles = {'von', 'van', 'de', 'del', 'della', 'der', 'la', 'le', 'du', 'des', 'di'}
            non_particle = [t for t in tokens if t not in particles and not re.match(r'^[a-z]\.?$', t)]
            surname_part = non_particle[-1] if non_particle else (tokens[-1] if tokens else '')
        surname_clean = re.sub(r'[^a-z0-9]', '', surname_part)
        if len(surname_clean) > 2:
            out.append(surname_clean)
    return out


def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    if not cited_authors or not correct_authors:
        return None
    cited = _extract_surnames(cited_authors)
    correct = _extract_surnames(correct_authors)
    if not cited or not correct:
        return None
    correct_set = set(correct)
    matches = 0
    for s in cited[:6]:
        if s in correct_set:
            matches += 1
            continue
        if any(s.startswith(c[:4]) or c.startswith(s[:4]) for c in correct_set if len(c) >= 4):
            matches += 0.8
            continue
    return round(matches / min(len(cited[:6]), 6), 3)


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


def _check_retraction(doi: str) -> tuple:
    if not doi:
        return False, None, None
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/7.0 (mailto:{mailto})" if mailto else "LNI-Checker/7.0"
        resp = requests.get(f"https://api.crossref.org/works/{doi}", timeout=5, headers={"User-Agent": ua})
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
    authors = work.get("author", [])
    author_str = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in authors[:5]) if authors else None
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


def _fetch_url_with_browser_headers(url: str, title: str = "", authors: str = "", year: str = "") -> tuple:
    """
    Simple URL fetch with browser headers.
    Just returns the URL status and content.
    AI will decide if it's real or not.
    """
    if not url or not url.startswith("http"):
        return False, "", {}, "No valid URL provided"

    # Real browser user agents to bypass simple blocks
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    ]

    headers_base = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    for ua in user_agents:
        headers = {**headers_base, "User-Agent": ua}
        try:
            resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            # Get the page content (first 5000 chars for AI to analyze)
            content = resp.text[:5000] if resp.text else ""
            
            if resp.status_code == 200:
                return True, content, {"url": resp.url}, f"URL reachable (HTTP 200)"
            elif resp.status_code in (301, 302, 307, 308):
                return True, content, {"url": resp.url}, f"URL redirects to: {resp.headers.get('Location', 'unknown')}"
            else:
                return True, content, {"url": resp.url}, f"URL responded with HTTP {resp.status_code}"
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            continue
    
    return False, "", {}, f"URL not reachable"


def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.doi:
        return None
    _rate_limit("crossref.org", 0.2)
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/7.0 (mailto:{mailto})" if mailto else "LNI-Checker/7.0"
        resp = requests.get(f"https://api.crossref.org/works/{entry.doi}", timeout=5, headers={"User-Agent": ua})
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            if sim >= 0.95:
                meta = _extract_corrected_metadata(work)
                is_retracted, ret_doi, ret_note = _check_retraction(entry.doi)
                status = "verified" if not is_retracted else "retracted"
                return VerificationResult(
                    key=entry.key, title=entry.title or "", status=status, confidence=0.95,
                    matched_title=title, doi=entry.doi, open_access_url=_check_unpaywall(entry.doi),
                    note=f"DOI verified via CrossRef (match: {int(sim*100)}%)", sources_checked=["CrossRef (DOI)"],
                    correct_authors=meta["corrected_authors"], is_retracted=is_retracted,
                    corrected_title=title, corrected_authors=meta["corrected_authors"],
                    corrected_year=meta["corrected_year"], corrected_journal=meta["corrected_journal"],
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
        resp = requests.get(f"https://arxiv.org/bibtex/{arxiv_id}", timeout=5, headers={"User-Agent": "LNI-Checker/7.0"})
        if resp.status_code == 200:
            bibtex = resp.text
            title_match = re.search(r'title\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
            title = title_match.group(1) if title_match else None
            if title and _title_similarity(entry.title or "", title) >= 0.95:
                return VerificationResult(
                    key=entry.key, title=entry.title or "", status="verified", confidence=0.92,
                    matched_title=title, open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    note=f"arXiv ID {arxiv_id} verified", sources_checked=["arXiv (ID)"],
                )
    except Exception:
        pass
    return None


def _search_crossref(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("crossref.org", 0.2)
    try:
        params = {"query.title": entry.title, "rows": 3}
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            params["query.author"] = first_author
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/7.0 (mailto:{mailto})" if mailto else "LNI-Checker/7.0"
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=6, headers={"User-Agent": ua})
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            for item in items[:3]:
                title = (item.get("title") or [""])[0]
                sim = _title_similarity(entry.title, title)
                if sim >= 0.95:
                    doi = item.get("DOI", "")
                    authors = item.get("author", [])
                    author_str = "; ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors[:3]]) if authors else None
                    meta = _extract_corrected_metadata(item)
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified", confidence=sim,
                        matched_title=title, doi=doi, open_access_url=_check_unpaywall(doi) if doi else None,
                        note=f"Found on CrossRef (match: {int(sim*100)}%)", sources_checked=["CrossRef"],
                        correct_authors=author_str, corrected_title=title,
                        corrected_authors=meta["corrected_authors"], corrected_year=meta["corrected_year"],
                    )
    except Exception:
        pass
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.semanticscholar.org", 0.2)
    try:
        headers = {"User-Agent": "LNI-Checker/7.0"}
        resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", 
                           params={"query": entry.title, "limit": 3, "fields": "title,authors,year,openAccessPdf,externalIds"}, 
                           timeout=6, headers=headers)
        if resp.status_code == 200:
            papers = resp.json().get("data", [])
            for paper in papers[:3]:
                title = paper.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.95:
                    authors = paper.get("authors", [])
                    author_str = "; ".join([a.get("name", "") for a in authors[:3]]) if authors else None
                    oa = (paper.get("openAccessPdf") or {}).get("url")
                    doi = paper.get("externalIds", {}).get("DOI")
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified", confidence=sim,
                        matched_title=title, doi=doi, open_access_url=oa,
                        note=f"Found on Semantic Scholar (match: {int(sim*100)}%)", sources_checked=["Semantic Scholar"],
                        correct_authors=author_str,
                    )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------

def _validate_doi_format(doi: str) -> tuple:
    if not doi:
        return False, "Empty DOI"
    doi = doi.strip()
    if re.match(r'^10\.\d{4,9}/', doi):
        return True, "Valid DOI format"
    return False, "Invalid DOI format"


def _check_year_plausibility(year: str) -> tuple:
    if not year:
        return True, "No year"
    m = re.search(r'\d{4}', str(year))
    if not m:
        return False, f"Cannot parse year: {year}"
    y = int(m.group())
    import datetime
    current_year = datetime.datetime.now().year
    if y < 1800:
        return False, f"Year {y} is implausibly old"
    if y > current_year + 1:
        return False, f"Year {y} is in the future"
    return True, f"Year {y} is plausible"


def _check_unpaywall(doi: str) -> Optional[str]:
    if not doi:
        return None
    mailto = os.environ.get("UNPAYWALL_EMAIL", os.environ.get("CROSSREF_MAILTO", "")).strip()
    if not mailto:
        return None
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": mailto},
            timeout=5,
            headers={"User-Agent": "LNI-Checker/7.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            best = data.get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LNI style check
# ---------------------------------------------------------------------------

def check_lni_macros(body: str) -> List[dict]:
    suggestions = []
    if not body:
        return suggestions

    lines = body.splitlines()
    for i, line in enumerate(lines):
        ctx = line.strip()[:120]

        if re.search(r'\[\d[\d,\s\-]*\]', line):
            suggestions.append({
                "type": "numeric_citation",
                "message": "Numeric citation style detected. LNI requires author-year keys e.g. [AB20].",
                "context": ctx,
                "line": i + 1,
            })

        for m in re.finditer(r'\\cite\{([^}]+)\}', line):
            key = m.group(1).strip()
            if not re.match(r'^[A-Z]{1,4}\d{2}', key):
                suggestions.append({
                    "type": "cite_key_format",
                    "message": f"Citation key '{key}' may not follow LNI convention (e.g. AB20).",
                    "context": ctx,
                    "line": i + 1,
                })

        if re.search(r'\bet\s+al\.', line, re.IGNORECASE):
            suggestions.append({
                "type": "et_al_in_text",
                "message": "Found 'et al.' in text — acceptable in citations but bibliography must list all authors.",
                "context": ctx,
                "line": i + 1,
            })

    return suggestions


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def extract_citations_from_body(body: str) -> set:
    keys = set()
    if not body:
        return keys

    for m in re.finditer(r'\[([A-Z][A-Za-z+]{0,5}\d{2}(?:,\s*[A-Z][A-Za-z+]{0,5}\d{2})*)\]', body):
        for k in re.split(r',\s*', m.group(1)):
            keys.add(k.strip())

    for m in re.finditer(r'\\(?:cite|citet|citep|Cite)\{([^}]+)\}', body):
        for k in m.group(1).split(','):
            keys.add(k.strip())

    if re.search(r'\[\d[\d,\s\-]*\]', body):
        keys.add('__numeric_citations__')

    return keys


def extract_citation_contexts(body: str) -> List[dict]:
    contexts = []
    if not body:
        return contexts

    for m in re.finditer(
        r'(.{0,80})'
        r'(\[[A-Z][A-Za-z+]{0,5}\d{2}[^\]]*\]|\[\d[\d,\s\-]*\])'
        r'(.{0,80})',
        body,
    ):
        pre, cite, post = m.group(1), m.group(2), m.group(3)
        inner = cite[1:-1]
        for k in re.split(r',\s*', inner):
            k = k.strip()
            if k:
                contexts.append({
                    "key": k,
                    "context": f"...{pre}{cite}{post}...".strip(),
                })

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
    if not bib_dict:
        return result

    bib_keys = set(bib_dict.keys())

    real_cited = {
        k for k in cited_keys
        if k and not k.startswith('__')
    }

    if '__numeric_citations__' in cited_keys:
        result.correctly_used = sorted(bib_keys)
        return result

    result.correctly_used = sorted(real_cited & bib_keys)
    result.cited_not_in_bib = sorted(real_cited - bib_keys)
    result.in_bib_not_cited = sorted(bib_keys - real_cited)
    return result


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(bib_dict: dict) -> List[dict]:
    entries = list(bib_dict.values())
    duplicates = []
    seen_pairs = set()

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if not a.title or not b.title:
                continue
            sim = _title_similarity(a.title, b.title)
            if sim >= 0.92:
                pair = tuple(sorted([a.key, b.key]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    duplicates.append({
                        "key_a": a.key,
                        "key_b": b.key,
                        "similarity": round(sim, 3),
                        "title_a": a.title,
                        "title_b": b.title,
                    })

    return duplicates


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
        inner = match[1:-1]
        for k in re.split(r',\s*', inner):
            self_keys.add(k.strip().upper())

    for key, entry in bib_dict.items():
        if key.upper() in self_keys:
            self_cites.append({
                "key": key,
                "title": entry.title or "",
                "reason": "Citation appears near self-referential language",
            })

    return self_cites


# ---------------------------------------------------------------------------
# MAIN VERIFICATION FUNCTION - CORRECTED FLOW
# ---------------------------------------------------------------------------

def verify_reference(entry) -> "VerificationResult":
    """
    CORRECTED FLOW v7.0:
      1. SQLite DB check ONLY
      2. If URL exists → fetch directly with browser headers
      3. Then check academic APIs (only for papers without URLs)
      4. AI as last resort
      5. AUTO-SAVE to DB for ALL verified references
    """
    if not entry.title and not entry.doi:
        return VerificationResult(
            key=entry.key, title=entry.title or "", status="not_checked",
            confidence=0.0, note="No title or DOI to verify against",
        )

    # ── STEP 1: SQLite DB check ONLY ─────────────────────────────────────────
    cached = search_cache(entry.title or "", entry.authors or "")
    if cached:
        return VerificationResult(
            key=entry.key, title=entry.title or "", status="verified",
            confidence=cached.confidence,
            matched_title=cached.title,
            doi=cached.doi,
            open_access_url=cached.url,
            note=f"Found in local database (source: {cached.source})",
            sources_checked=["local_db"],
        )

    # ── STEP 2: If URL exists, fetch directly ────────────────────────────────
    entry_url = (getattr(entry, 'url', '') or '').strip()
    result = None
    
    if entry_url and entry_url.startswith("http"):
        reachable, page_content, meta, url_note = _fetch_url_with_browser_headers(
            entry_url, entry.title or "", entry.authors or "", entry.year or ""
        )
        
        if reachable:
            result = VerificationResult(
                key=entry.key, title=entry.title or "",
                status="partial_match", confidence=0.50,
                matched_title=None,
                open_access_url=entry_url,
                note=f"URL reachable. AI will verify content. {url_note}",
                sources_checked=["url_fetch"],
                web_evidence=page_content[:1000],
            )
        else:
            result = VerificationResult(
                key=entry.key, title=entry.title or "",
                status="not_found", confidence=0.0,
                matched_title=None,
                note=f"URL not reachable: {url_note}",
                sources_checked=["url_fetch"],
            )
    else:
        # ── STEP 3: Academic API pipeline ────────────────────────────────────
        if entry.doi:
            result = _lookup_by_doi(entry)

        if not result:
            result = _lookup_by_arxiv_id(entry)

        if not result:
            searchers = [_search_crossref, _search_semantic_scholar]
            best = None
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {ex.submit(fn, entry): fn.__name__ for fn in searchers}
                for future in as_completed(futures):
                    try:
                        r = future.result()
                    except Exception:
                        r = None
                    if r is None:
                        continue
                    if r.status == "verified" and r.confidence >= 0.95:
                        best = r
                        for f in futures:
                            f.cancel()
                        break
                    if best is None or (r.confidence > best.confidence):
                        best = r
            result = best
        
        # ── AUTO-SAVE API-verified academic papers to DB ─────────────────────
        if result and result.status == "verified" and result.confidence >= 0.85:
            if entry.title:
                save_to_cache(
                    title=result.matched_title or entry.title,
                    authors=entry.authors or "",
                    year=entry.year or "",
                    doi=result.doi or entry.doi or "",
                    url=result.open_access_url or entry.url or "",
                    source=result.sources_checked[0] if result.sources_checked else "api",
                    confidence=result.confidence,
                )

    # ── STEP 4: AI fallback via web_search_verifier ──────────────────────────
    if not result or result.status in ("not_found", "error", "partial_match"):
        entry_dict = {
            "title": entry.title or "",
            "authors": entry.authors or "",
            "year": entry.year or "",
            "url": entry_url,
        }
        api_status = result.status if result else "not_found"
        web = verify_with_web_search(entry_dict, api_status)
        if web.get("status") == "verified":
            result = VerificationResult(
                key=entry.key, title=entry.title or "", status="verified",
                confidence=web.get("confidence", 0.7),
                matched_title=web.get("matched_title"),
                open_access_url=web.get("open_access_url"),
                note=web.get("note", "Verified via web search"),
                sources_checked=web.get("sources_checked", ["web_search"]),
                web_evidence=web.get("note"),
            )
            # Save to DB if AI confirmed it's real
            if result.matched_title:
                save_to_cache(
                    title=result.matched_title,
                    authors=entry.authors or "",
                    year=entry.year or "",
                    doi=result.doi or "",
                    url=result.open_access_url or "",
                    source="web_search",
                    confidence=result.confidence,
                )

    if result is None:
        result = VerificationResult(
            key=entry.key, title=entry.title or "", status="not_found",
            confidence=0.0, note="Not found in any source",
            sources_checked=[],
        )

    return result


def verify_all_references(bib_dict: dict) -> List[VerificationResult]:
    entries = list(bib_dict.values())
    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {ex.submit(verify_reference, e): e for e in entries}
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:
                e = future_map[future]
                results.append(VerificationResult(
                    key=e.key, title=e.title or "", status="error",
                    confidence=0.0, note=f"Verification error: {exc}",
                ))

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
) -> dict:
    score = 100
    penalties = []

    missing = len(xcheck.cited_not_in_bib)
    if missing:
        deduct = min(missing * 5, 20)
        score -= deduct
        penalties.append({"reason": f"{missing} citation(s) missing from bibliography", "deduction": deduct})

    orphaned = len(xcheck.in_bib_not_cited)
    if orphaned:
        deduct = min(orphaned * 2, 10)
        score -= deduct
        penalties.append({"reason": f"{orphaned} bibliography entry/entries never cited", "deduction": deduct})

    style_count = len(style_suggestions)
    if style_count:
        deduct = min(style_count * 2, 15)
        score -= deduct
        penalties.append({"reason": f"{style_count} LNI style issue(s)", "deduction": deduct})

    dup_count = len(duplicates)
    if dup_count:
        deduct = min(dup_count * 3, 10)
        score -= deduct
        penalties.append({"reason": f"{dup_count} duplicate reference(s)", "deduction": deduct})

    if professor_confirmed_fakes:
        deduct = min(professor_confirmed_fakes * 15, 40)
        score -= deduct
        penalties.append({"reason": f"{professor_confirmed_fakes} confirmed fake reference(s)", "deduction": deduct})

    if retracted_count:
        deduct = min(retracted_count * 5, 15)
        score -= deduct
        penalties.append({"reason": f"{retracted_count} retracted paper(s) cited", "deduction": deduct})

    incomplete = sum(1 for e in bib_list if getattr(e, 'completeness_issues', None))
    if incomplete:
        deduct = min(incomplete * 2, 10)
        score -= deduct
        penalties.append({"reason": f"{incomplete} incomplete bibliography entry/entries", "deduction": deduct})

    score = max(0, score)

    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"

    return {"score": score, "grade": grade, "penalties": penalties}