"""
STEP 3: Citation Cross-Checker + Reference Verifier — v8.0
----------------------------------------------------------
PIPELINE v8.0 (strict 4-step, no shortcuts):
  1. SQLite local DB           → ≥95% match → REAL, done.
  2. Academic APIs             → ≥95% title+author+year → REAL, save, done.
                                 anything else → SUSPICIOUS, continue.
  3. URL fetch                 → only if suspicious AND entry has a URL.
                                 HTTP 200 + ≥95% title match → REAL, save, done.
                                 anything else → stays SUSPICIOUS.
  4. AI (web search + LLM)     → only remaining suspicious entries.
                                 ≥95% confidence REAL → save, done.
                                 else → stays SUSPICIOUS (never auto-FAKE here).

KEY RULE: only save to DB when confidence ≥ 0.95. FAKE is never set by this
module — it is a professor-only manual action.
"""

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
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from local_db import search_cache, save_to_cache, get_cache_stats, init_cache_db
from web_search_verifier import verify_with_web_search
from review_queue import is_venue_whitelisted, get_review_decision, get_false_positive

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
# Author overlap
# ---------------------------------------------------------------------------

def _extract_surnames(s: str) -> List[str]:
    out = []
    for part in re.split(r';|\band\b|\bund\b', s, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        part_lower = part.lower()
        for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss'),
                     ('à','a'),('é','e'),('è','e'),('ñ','n')]:
            part_lower = part_lower.replace(a, b)
        if re.match(r'^et\s+al\.?$', part_lower.strip()):
            continue
        if ',' in part_lower:
            surname_part = part_lower.split(',')[0].strip()
        else:
            tokens = part_lower.split()
            particles = {'von','van','de','del','della','der','la','le','du','des','di'}
            non_particle = [t for t in tokens
                            if t not in particles and not re.match(r'^[a-z]\.?$', t)]
            surname_part = non_particle[-1] if non_particle else (tokens[-1] if tokens else '')
        clean = re.sub(r'[^a-z0-9]', '', surname_part)
        if len(clean) > 2:
            out.append(clean)
    return out


def author_overlap_score(cited_authors: str, correct_authors: str) -> Optional[float]:
    if not cited_authors or not correct_authors:
        return None
    cited   = _extract_surnames(cited_authors)
    correct = _extract_surnames(correct_authors)
    if not cited or not correct:
        return None
    correct_set = set(correct)
    matches = 0.0
    for s in cited[:6]:
        if s in correct_set:
            matches += 1.0
            continue
        if any(s.startswith(c[:4]) or c.startswith(s[:4])
               for c in correct_set if len(c) >= 4):
            matches += 0.8
    return round(matches / min(len(cited[:6]), 6), 3)


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


# ---------------------------------------------------------------------------
# Retraction check
# ---------------------------------------------------------------------------

def _check_retraction(doi: str) -> tuple:
    if not doi:
        return False, None, None
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/8.0 (mailto:{mailto})" if mailto else "LNI-Checker/8.0"
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
                            headers={"User-Agent": "LNI-Checker/8.0"})
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Academic API lookups — all require ≥ 0.95 similarity to return "verified"
# ---------------------------------------------------------------------------

def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.doi:
        return None
    _rate_limit("crossref.org", 0.2)
    try:
        mailto = os.environ.get("CROSSREF_MAILTO","").strip()
        ua = f"LNI-Checker/8.0 (mailto:{mailto})" if mailto else "LNI-Checker/8.0"
        resp = requests.get(f"https://api.crossref.org/works/{entry.doi}",
                            timeout=5, headers={"User-Agent": ua})
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            if sim >= 0.85:
                meta = _extract_corrected_metadata(work)
                is_ret, ret_doi, ret_note = _check_retraction(entry.doi)
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
                            timeout=5, headers={"User-Agent": "LNI-Checker/8.0"})
        if resp.status_code == 200:
            m = re.search(r'title\s*=\s*[{"](.*?)[}"]', resp.text, re.IGNORECASE)
            title = m.group(1) if m else None
            if title and _title_similarity(entry.title or "", title) >= 0.85:
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
    try:
        params = {"query.title": entry.title, "rows": 5}
        if entry.authors:
            # Extract first author surname, remove "et al."
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            first_author = re.sub(r'\s+et\s+al\.?$', '', first_author, flags=re.IGNORECASE)
            if first_author and len(first_author) > 2:
                params["query.author"] = first_author
        
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/8.0 (mailto:{mailto})" if mailto else "LNI-Checker/8.0"
        
        # Longer timeout for academic papers (10 seconds instead of 8)
        timeout = 10
        
        resp = requests.get("https://api.crossref.org/works",
                            params=params, timeout=timeout,
                            headers={"User-Agent": ua})
        
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            for item in items[:5]:
                title = (item.get("title") or [""])[0]
                if not title:
                    continue
                sim = _title_similarity(entry.title, title)
                if sim >= 0.85:
                    doi = item.get("DOI", "")
                    meta = _extract_corrected_metadata(item)
                    authors = item.get("author", [])
                    author_str = "; ".join(
                        f"{a.get('family','')}, {a.get('given','')}"
                        for a in authors[:3]
                    ) if authors else None
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified",
                        confidence=sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=_check_unpaywall(doi) if doi else None,
                        note=f"CrossRef match ({int(sim*100)}%)",
                        sources_checked=["CrossRef"],
                        correct_authors=author_str,
                        corrected_title=title,
                        corrected_authors=meta["corrected_authors"],
                        corrected_year=meta["corrected_year"],
                        corrected_journal=meta["corrected_journal"],
                    )
    except requests.exceptions.Timeout:
        # Return a special result indicating timeout - not a failure
        print(f"CrossRef timeout for {entry.key}, will use AI fallback")
        return VerificationResult(
            key=entry.key,
            title=entry.title or "",
            status="timeout",
            confidence=0.0,
            note="CrossRef API timeout - will try AI verification",
            sources_checked=["CrossRef_timeout"],
        )
    except Exception as e:
        print(f"CrossRef search error for {entry.key}: {e}")
    
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.semanticscholar.org", 0.2)
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": entry.title, "limit": 5,
                    "fields": "title,authors,year,openAccessPdf,externalIds"},
            timeout=8, headers={"User-Agent": "LNI-Checker/8.0"})
        if resp.status_code == 200:
            for paper in resp.json().get("data", [])[:5]:
                title = paper.get("title", "")
                if not title:
                    continue
                sim = _title_similarity(entry.title, title)
                if sim >= 0.85:
                    authors = paper.get("authors", [])
                    author_str = "; ".join(
                        a.get("name", "") for a in authors[:3]
                    ) if authors else None
                    oa = (paper.get("openAccessPdf") or {}).get("url")
                    doi = paper.get("externalIds", {}).get("DOI")
                    # RETURN IMMEDIATELY when match found
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title,
                        status="verified",
                        confidence=sim,
                        matched_title=title,
                        doi=doi,
                        open_access_url=oa,
                        note=f"Semantic Scholar match ({int(sim*100)}%)",
                        sources_checked=["Semantic Scholar"],
                        correct_authors=author_str,
                    )
    except Exception as e:
        print(f"Semantic Scholar search error for {entry.key}: {e}")
    
    return None


# ---------------------------------------------------------------------------
# URL fetch — used ONLY for suspicious entries that have a URL
# ---------------------------------------------------------------------------

def _fetch_url_strict(entry: BibEntry) -> Optional[VerificationResult]:
    from bs4 import BeautifulSoup

    url = (getattr(entry, "url", "") or "").strip()

    if not url or not url.startswith("http"):
        return None

    # Clean URL
    url = re.sub(r'\s+', '', url)
    url = re.sub(r',?\s*Stand:.*$', '', url, flags=re.IGNORECASE)
    url = re.sub(r'Stand:.*$', '', url, flags=re.IGNORECASE)

    # Rotate two browser profiles to defeat simple bot-blocks
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
            # Warm-up HEAD to acquire cookies (helps Cloudflare / auth redirects)
            try:
                session.head(url, timeout=8, allow_redirects=True)
            except Exception:
                pass

            resp = session.get(url, timeout=15, allow_redirects=True)
            last_status = resp.status_code

            if resp.status_code == 200:
                content_type = resp.headers.get('Content-Type', '').lower()

                # PDF → verified without title comparison
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    return VerificationResult(
                        key=entry.key,
                        title=entry.title or "",
                        status="verified",
                        confidence=0.90,
                        matched_title=entry.title,
                        open_access_url=resp.url,
                        note="PDF verified (HTTP 200)",
                        sources_checked=["url_verify"],
                    )

                # HTML → extract and compare title
                soup = BeautifulSoup(resp.text, "html.parser")
                page_title = ""
                if soup.find("title"):
                    page_title = soup.find("title").get_text().strip()
                if not page_title:
                    meta = soup.find("meta", property="og:title")
                    if meta:
                        page_title = meta.get("content", "")
                if not page_title:
                    h1 = soup.find("h1")
                    if h1:
                        page_title = h1.get_text().strip()

                # Challenge / bot-block pages count as "URL alive, bot-blocked"
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

                if page_title and entry.title:
                    sim = _title_similarity(entry.title, page_title)
                    if sim >= 0.85:
                        return VerificationResult(
                            key=entry.key,
                            title=entry.title or "",
                            status="verified",
                            confidence=round(sim, 4),
                            matched_title=page_title,
                            open_access_url=resp.url,
                            note=f"URL verified (title match {int(sim*100)}%)",
                            sources_checked=["url_verify"],
                        )

                # Page reachable but title didn't match well enough
                return VerificationResult(
                    key=entry.key,
                    title=entry.title or "",
                    status="url_blocked",
                    confidence=0.0,
                    open_access_url=url,
                    note=(
                        f"URL reachable (HTTP 200) but title similarity too low "
                        f"(page: '{page_title[:60]}'). Escalating to AI."
                    ),
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (301, 302, 303, 307, 308):
                # Pure redirect without following — shouldn't happen (allow_redirects=True)
                # but guard anyway
                return VerificationResult(
                    key=entry.key, title=entry.title or "",
                    status="url_blocked", confidence=0.0,
                    open_access_url=url,
                    note=f"URL redirects (HTTP {resp.status_code}). Escalating to AI.",
                    sources_checked=["url_fetch"],
                )

            elif resp.status_code in (403, 429):
                # Server alive, blocking bots → URL almost certainly valid
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

    # All attempts exhausted — still pass URL to AI rather than giving up
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
    keys = set()
    if not body:
        return keys
    for m in re.finditer(
        r'\[([A-Z][A-Za-z+]{0,5}\d{2}(?:,\s*[A-Z][A-Za-z+]{0,5}\d{2})*)\]', body
    ):
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
        r'(.{0,80})(\[[A-Z][A-Za-z+]{0,5}\d{2}[^\]]*\]|\[\d[\d,\s\-]*\])(.{0,80})',
        body,
    ):
        pre, cite, post = m.group(1), m.group(2), m.group(3)
        for k in re.split(r',\s*', cite[1:-1]):
            k = k.strip()
            if k:
                contexts.append({"key": k,
                                  "context": f"...{pre}{cite}{post}...".strip()})
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
    real_cited = {k for k in cited_keys if k and not k.startswith('__')}
    if '__numeric_citations__' in cited_keys:
        result.correctly_used = sorted(bib_keys)
        return result
    result.correctly_used    = sorted(real_cited & bib_keys)
    result.cited_not_in_bib  = sorted(real_cited - bib_keys)
    result.in_bib_not_cited  = sorted(bib_keys - real_cited)
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
                        "key_a": a.key, "key_b": b.key,
                        "similarity": round(sim, 3),
                        "title_a": a.title, "title_b": b.title,
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
# MAIN VERIFICATION FUNCTION — strict 4-step pipeline
# ---------------------------------------------------------------------------

def verify_reference(entry: BibEntry) -> VerificationResult:
    """
    4-step pipeline:
      1. Local SQLite DB  → ≥95% → REAL, done.
      2. Academic APIs    → ≥95% title+author+year → REAL, save, done. Else SUSPICIOUS.
      3. URL fetch        → only if suspicious + URL present.
                           HTTP 200 + ≥95% title → REAL, save, done.
      4. AI/web search    → only remaining suspicious.
                           ≥95% confidence REAL → save, done. Else SUSPICIOUS.

    FAKE is never set here. That is a professor-only manual action.
    """
    if not entry.title and not entry.doi:
        return VerificationResult(
            key=entry.key, title="",
            status="suspicious", confidence=0.0,
            note="No title or DOI — cannot verify.",
            sources_checked=[],
        )

    # ── STEP 1: Local SQLite DB ───────────────────────────────────────────────
    cached = search_cache(entry.title or "", entry.authors or "")
    if cached and cached.confidence >= 0.95:
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="verified", confidence=cached.confidence,
            matched_title=cached.title, doi=cached.doi,
            open_access_url=cached.url,
            note=f"Found in local database (source: {cached.source})",
            sources_checked=["local_db"],
        )

    # ── STEP 1b: Fast-path for grey literature ────────────────────────────────
    # Grey literature (industry reports, blogs, government docs) is never indexed
    # in CrossRef or Semantic Scholar. Skip academic APIs entirely and go straight
    # to URL fetch → AI verification.
        
    _entry_dict_for_grey = {
        "title":      entry.title or "",
        "authors":    getattr(entry, "authors", "") or "",
        "year":       entry.year or "",
        "url":        (getattr(entry, "url", "") or "").strip(),
        "publisher":  getattr(entry, "publisher", "") or "",
        "entry_type": getattr(entry, "entry_type", "") or "",
        "raw_text":   getattr(entry, "raw_text", "") or "",
    }
    from ai_checker import _is_grey_literature
    _is_grey, _grey_reason = _is_grey_literature(_entry_dict_for_grey)

    if _is_grey:
        entry_url = _entry_dict_for_grey["url"]
        url_blocked = False
        url_note = ""

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
                url_note = "URL fetch returned no result or 404."
        else:
            url_note = "No URL in grey literature entry."

        # Build full raw reference for AI context
        raw_ref = f"{entry.authors or ''}: {entry.title or ''}. {getattr(entry, 'publisher', '') or getattr(entry, 'journal', '') or ''}, {entry.year or ''}. {entry_url}"
        
        # Go straight to AI with full raw reference text
        grey_dict = {
            **_entry_dict_for_grey,
            "api_status":        "not_found",
            "api_matched_title": "",
            "url_note":          url_note,
            "url_blocked":       url_blocked,
            "raw_text":          raw_ref,  # Pass full reference
        }
        web_result = verify_with_web_search(grey_dict, "not_found")
        
        # Lower confidence threshold for grey literature (0.75 instead of 0.95)
        if (web_result.get("status") == "verified"
                and web_result.get("confidence", 0) >= 0.75):
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
        # AI couldn't confirm — return suspicious but informative
        return VerificationResult(
            key=entry.key, title=entry.title or "",
            status="suspicious",
            confidence=web_result.get("confidence", 0.5),
            note=f"Grey literature ({_grey_reason}). {web_result.get('note', 'Could not be auto-verified — manual check recommended.')}",
            sources_checked=["grey_lit", "ai_attempted"],
        )

    # ── STEP 2: Academic APIs ─────────────────────────────────────────────────
    api_result: Optional[VerificationResult] = None

    # DOI is strongest — try first
    if entry.doi:
        api_result = _lookup_by_doi(entry)

    # arXiv ID embedded in URL or DOI
    if not api_result:
        api_result = _lookup_by_arxiv_id(entry)

    # CrossRef + Semantic Scholar in parallel
    if not api_result:
        best: Optional[VerificationResult] = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(fn, entry): fn for fn in
                       [_search_crossref, _search_semantic_scholar]}
            for future in as_completed(futures):
                try:
                    r = future.result()
                except Exception:
                    r = None
                if r is None:
                    continue
                if r.status == "verified" and r.confidence >= 0.85:
                    best = r
                    for f in futures:
                        f.cancel()
                    break
                if best is None or r.confidence > best.confidence:
                    best = r
        api_result = best

    # Strict match: title ≥ 0.95 AND year within 1 year AND author overlap ≥ 0.5
    if api_result and api_result.status == "verified" and api_result.confidence >= 0.85:
        year_ok = True
        if entry.year and api_result.corrected_year:
            try:
                year_ok = abs(int(entry.year) - int(api_result.corrected_year)) <= 1
            except (ValueError, TypeError):
                year_ok = True  # unparseable year — don't penalise

        author_ok = True
        if entry.authors and api_result.correct_authors:
            overlap = author_overlap_score(entry.authors, api_result.correct_authors)
            if overlap is not None:
                author_ok = overlap >= 0.50

        if year_ok and author_ok:
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

    # API didn't confirm → entry is now SUSPICIOUS
    sources_tried = list(api_result.sources_checked) if api_result else []

    # ── STEP 3: URL fetch (only for entries with a URL) ───────────────────────
    entry_url = (getattr(entry, "url", "") or "").strip()
    url_note = ""
    url_blocked = False  # True = server alive but bot-blocked; hand off to AI with URL

    if entry_url and entry_url.startswith("http"):
        sources_tried.append("url_fetch")
        url_result = _fetch_url_strict(entry)
        if url_result:
            if url_result.status == "verified":
                # HTTP 200 + title confirmed
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
            elif url_result.status == "url_blocked":
                # Server is alive but blocked automated access — AI can still verify
                url_blocked = True
                url_note = url_result.note or "URL reachable but bot-blocked."
            else:
                url_note = "URL fetch did not confirm (no 200 or title mismatch)."
        else:
            url_note = "URL fetch returned no result."
    else:
        url_note = "No URL in entry."

    # ── STEP 4: AI / web-search fallback ─────────────────────────────────────
        # ── STEP 4: AI / web-search fallback ─────────────────────────────────────
    entry_dict = {
        "title":      entry.title or "",
        "authors":    entry.authors or "",
        "year":       entry.year or "",
        "url":        entry_url,
        "publisher":  getattr(entry, "publisher", "") or "",
        "entry_type": getattr(entry, "entry_type", "") or "",
        "raw_text":   getattr(entry, "raw_text", "") or "",
        "api_status": api_result.status if api_result else "not_found",
        "api_matched_title": api_result.matched_title if api_result else "",
        "url_note":   url_note,
        "url_blocked": url_blocked,
    }
    sources_tried.append("web_search+ai")

    web_result = verify_with_web_search(entry_dict, "not_found")
    
    # Lower threshold for AI verification - 0.80 is sufficient
    if (web_result.get("status") == "verified"
            and web_result.get("confidence", 0) >= 0.80):
        matched = web_result.get("matched_title") or entry.title
        save_to_cache(
            title=matched,
            authors=entry.authors or "",
            year=entry.year or "",
            doi="",
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
            sources_checked=list(dict.fromkeys(sources_tried)),
        )

    # Nothing confirmed it → SUSPICIOUS
    return VerificationResult(
        key=entry.key, title=entry.title or "",
        status="suspicious",
        confidence=web_result.get("confidence", 0.0),
        matched_title=api_result.matched_title if api_result else None,
        note=f"Not confirmed by any source. {web_result.get('note', url_note)}",
        sources_checked=list(dict.fromkeys(sources_tried)),
    )


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
                    key=e.key, title=e.title or "",
                    status="suspicious", confidence=0.0,
                    note=f"Verification error: {exc}",
                ))
    return results


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score(
    bib_list: list,
    xcheck: CrossCheckResult,
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
        penalties.append({"category": "Missing citations", "count": missing,
                           "deduction": deduct})

    orphaned = len(xcheck.in_bib_not_cited)
    if orphaned:
        deduct = min(orphaned * 2, 10)
        score -= deduct
        penalties.append({"category": "Orphaned entries", "count": orphaned,
                           "deduction": deduct})

    style_count = len(style_suggestions)
    if style_count:
        deduct = min(style_count * 2, 15)
        score -= deduct
        penalties.append({"category": "Style issues", "count": style_count,
                           "deduction": deduct})

    dup_count = len(duplicates)
    if dup_count:
        deduct = min(dup_count * 3, 10)
        score -= deduct
        penalties.append({"category": "Duplicates", "count": dup_count,
                           "deduction": deduct})

    if professor_confirmed_fakes:
        deduct = min(professor_confirmed_fakes * 15, 40)
        score -= deduct
        penalties.append({"category": "Confirmed fake references",
                           "count": professor_confirmed_fakes, "deduction": deduct})

    if retracted_count:
        deduct = min(retracted_count * 5, 15)
        score -= deduct
        penalties.append({"category": "Retracted papers cited",
                           "count": retracted_count, "deduction": deduct})

    incomplete = sum(1 for e in bib_list if getattr(e, "completeness_issues", None))
    if incomplete:
        deduct = min(incomplete * 2, 10)
        score -= deduct
        penalties.append({"category": "Incomplete entries", "count": incomplete,
                           "deduction": deduct})

    score = max(0, score)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 50 else "F"
    return {"score": score, "grade": grade, "penalties": penalties}