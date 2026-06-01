"""
STEP 3: Citation Cross-Checker + Reference Verifier — v6.3
----------------------------------------------------------
FIXES:
  - Local DB checked FIRST before any API calls
  - Better error messages (no misleading "api_status=error")
  - Improved score calculation
  - Deduplication of verification results
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
# Persistent disk cache
# ---------------------------------------------------------------------------

_DISK_CACHE_DIR: str = os.environ.get("LNI_CACHE_DIR", ".lni_cache")
_DISK_CACHE_LOCK = threading.Lock()
_MEM_CACHE: Dict[str, "VerificationResult"] = {}
_MEM_CACHE_LOCK = threading.Lock()
_ARXIV_BIBTEX_MEM_CACHE: Dict[str, str] = {}
_ARXIV_CACHE_LOCK = threading.Lock()
_VERIFICATION_RESULT_CACHE: Dict[str, "VerificationResult"] = {}
_VERIFICATION_CACHE_LOCK = threading.Lock()

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


def _normalize_title(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'), ('à', 'a'), ('é', 'e')]:
        t = t.replace(a, b)
    t = re.sub(r'&[a-z]+;', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'[{}]', '', t)
    t = re.sub(r'[:\-]+', ' ', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    stop = {'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 'its',
            'using', 'based', 'towards', 'toward', 'via', 'approach',
            'der', 'die', 'das', 'und', 'fur', 'fuer', 'von', 'mit', 'im', 'an',
            'zu', 'zur', 'zum', 'eine', 'ein', 'des', 'dem', 'den', 'is', 'are',
            'was', 'were', 'be', 'by', 'at', 'or', 'not'}
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
        ua = f"LNI-Checker/6.3 (mailto:{mailto})" if mailto else "LNI-Checker/6.3"
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


def _lookup_by_doi(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.doi:
        return None
    _rate_limit("crossref.org", 0.2)
    try:
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/6.3 (mailto:{mailto})" if mailto else "LNI-Checker/6.3"
        resp = requests.get(f"https://api.crossref.org/works/{entry.doi}", timeout=5, headers={"User-Agent": ua})
        if resp.status_code == 200:
            work = resp.json().get("message", {})
            title = (work.get("title") or [""])[0]
            sim = _title_similarity(entry.title or "", title)
            meta = _extract_corrected_metadata(work)
            is_retracted, ret_doi, ret_note = _check_retraction(entry.doi)
            status = "verified" if not is_retracted else "retracted"
            confidence = 0.95 if sim > 0.7 else 0.7
            note_str = f"DOI verified via CrossRef (match: {int(sim*100)}%)"
            if is_retracted:
                note_str = f"⚠ RETRACTED. {ret_note}"
                confidence = 1.0
            return VerificationResult(
                key=entry.key, title=entry.title or "", status=status, confidence=confidence,
                matched_title=title, doi=entry.doi, open_access_url=_check_unpaywall(entry.doi),
                note=note_str, sources_checked=["CrossRef (DOI)"], correct_authors=meta["corrected_authors"],
                is_retracted=is_retracted, retraction_doi=ret_doi, retraction_note=ret_note,
                corrected_title=title if title else None, corrected_authors=meta["corrected_authors"],
                corrected_year=meta["corrected_year"], corrected_journal=meta["corrected_journal"],
                corrected_publisher=meta["corrected_publisher"], corrected_volume=meta["corrected_volume"],
                corrected_pages=meta["corrected_pages"],
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
        resp = requests.get(f"https://arxiv.org/bibtex/{arxiv_id}", timeout=5, headers={"User-Agent": "LNI-Checker/6.2"})
        if resp.status_code == 200:
            bibtex = resp.text
            title_match = re.search(r'title\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
            title = title_match.group(1) if title_match else None
            if title:
                sim = _title_similarity(entry.title or "", title)
                return VerificationResult(
                    key=entry.key, title=entry.title or "", status="verified", confidence=0.92,
                    matched_title=title, open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    note=f"arXiv ID {arxiv_id} verified (match: {int(sim*100)}%)", sources_checked=["arXiv (ID)"],
                )
    except Exception:
        pass
    return None


def _lookup_by_isbn(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.isbn:
        return None
    isbn_clean = re.sub(r'[\s-]', '', entry.isbn)
    _rate_limit("openlibrary.org", 0.5)
    try:
        resp = requests.get(f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn_clean}&format=json&jscmd=data", timeout=5, headers={"User-Agent": "LNI-Checker/6.2"})
        if resp.status_code == 200:
            data = resp.json()
            key = f"ISBN:{isbn_clean}"
            if key in data:
                book = data[key]
                title = book.get("title", "")
                sim = _title_similarity(entry.title or "", title)
                return VerificationResult(
                    key=entry.key, title=entry.title or "", status="verified" if sim > 0.6 else "partial_match",
                    confidence=0.88 if sim > 0.6 else 0.5, matched_title=title, open_access_url=book.get("url"),
                    note=f"ISBN {isbn_clean} verified via Open Library", sources_checked=["Open Library (ISBN)"],
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
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            params["query.author"] = first_author
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        ua = f"LNI-Checker/6.2 (mailto:{mailto})" if mailto else "LNI-Checker/6.2"
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=6, headers={"User-Agent": ua})
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
                        key=entry.key, title=entry.title, status="retracted" if is_ret else ("verified" if sim >= 0.75 else "partial_match"),
                        confidence=1.0 if is_ret else sim, matched_title=title, doi=doi, open_access_url=_check_unpaywall(doi) if doi else None,
                        note=ret_note_full, sources_checked=["CrossRef"], correct_authors=author_str,
                        is_retracted=is_ret, retraction_doi=ret_doi, retraction_note=ret_note,
                        corrected_title=title if title else None, corrected_authors=meta["corrected_authors"],
                        corrected_year=meta["corrected_year"], corrected_journal=meta["corrected_journal"],
                        corrected_publisher=meta["corrected_publisher"], corrected_volume=meta["corrected_volume"],
                        corrected_pages=meta["corrected_pages"],
                    )
    except Exception:
        pass
    return None


def _search_semantic_scholar(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.semanticscholar.org", 0.2)
    try:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"User-Agent": "LNI-Checker/6.2"}
        if api_key:
            headers["x-api-key"] = api_key
        resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": entry.title, "limit": 5, "fields": "title,authors,year,openAccessPdf,externalIds"}, timeout=6, headers=headers)
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.75 else "partial_match",
                        confidence=sim, matched_title=title, doi=doi, open_access_url=oa,
                        note=f"Found on Semantic Scholar (match: {int(sim*100)}%)", sources_checked=["Semantic Scholar"],
                        correct_authors=author_str,
                    )
    except Exception:
        pass
    return None


def _search_openalex(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.openalex.org", 0.2)
    try:
        resp = requests.get("https://api.openalex.org/works", params={"search": entry.title, "per-page": 5}, timeout=6, headers={"User-Agent": "LNI-Checker/6.2"})
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for work in results[:3]:
                title = work.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.5:
                    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
                    oa = work.get("open_access", {}).get("oa_url")
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.75 else "partial_match",
                        confidence=sim, matched_title=title, doi=doi, open_access_url=oa,
                        note=f"Found on OpenAlex (match: {int(sim*100)}%)", sources_checked=["OpenAlex"],
                    )
    except Exception:
        pass
    return None


def _dblp_query(query: str, timeout: int = 6) -> list:
    try:
        resp = requests.get("https://dblp.org/search/publ/api", params={"q": query, "format": "json", "h": 8}, timeout=timeout, headers={"User-Agent": "LNI-Checker/6.3"})
        if resp.status_code == 200:
            return resp.json().get("result", {}).get("hits", {}).get("hit", [])
    except Exception:
        pass
    return []


def _search_dblp(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("dblp.org", 1.0)
    norm_title = _normalize_title(entry.title)
    first_author_surname = ""
    if entry.authors:
        surnames = _extract_surnames(entry.authors)
        if surnames:
            first_author_surname = surnames[0]
    queries = []
    if first_author_surname:
        queries.append(f"{norm_title} {first_author_surname}")
    queries.append(norm_title)
    sig_words = [w for w in norm_title.split() if len(w) >= 4][:6]
    if len(sig_words) >= 3:
        queries.append(" ".join(sig_words))
    best = None
    for q in queries:
        hits = _dblp_query(q)
        for hit in hits[:5]:
            info = hit.get("info", {})
            title = info.get("title", "")
            if not title:
                continue
            sim = _title_similarity(entry.title, title)
            if sim >= 0.5:
                dblp_year = str(info.get("year", ""))
                if dblp_year and entry.year and abs(int(dblp_year) - int(entry.year)) > 2:
                    sim *= 0.85
                doi = info.get("doi") or ""
                url = info.get("url") or ""
                authors_info = info.get("authors", {})
                if isinstance(authors_info, dict):
                    author_list = authors_info.get("author", [])
                    if isinstance(author_list, list):
                        author_str = "; ".join(a.get("text", "") if isinstance(a, dict) else str(a) for a in author_list[:4])
                    elif isinstance(author_list, dict):
                        author_str = author_list.get("text", "")
                    else:
                        author_str = str(author_list)
                else:
                    author_str = ""
                status = "verified" if sim >= 0.78 else "partial_match"
                vr = VerificationResult(
                    key=entry.key, title=entry.title, status=status, confidence=sim,
                    matched_title=title, doi=doi, open_access_url=url if url.startswith("http") else None,
                    note=f"Found on DBLP (match: {int(sim*100)}%)", sources_checked=["DBLP"],
                    correct_authors=author_str or None, corrected_year=dblp_year or None,
                )
                if best is None or sim > best.confidence:
                    best = vr
        if best and best.confidence >= 0.78:
            break
    return best


def _search_acl(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("aclanthology.org", 0.5)
    try:
        resp = requests.get("https://aclanthology.org/search/", params={"q": entry.title}, timeout=6, headers={"User-Agent": "LNI-Checker/6.2"})
        if resp.status_code == 200:
            titles = re.findall(r'<span class="d-block"[^>]*>(.*?)</span>', resp.text, re.DOTALL)
            for t in titles[:3]:
                title = re.sub(r'<[^>]+>', '', t).strip()
                sim = _title_similarity(entry.title, title)
                if sim >= 0.6:
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.8 else "partial_match",
                        confidence=sim, matched_title=title, note=f"Found on ACL Anthology (match: {int(sim*100)}%)",
                        sources_checked=["ACL Anthology"],
                    )
    except Exception:
        pass
    return None


def _search_arxiv_fallback(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("export.arxiv.org", 0.34)
    try:
        import urllib.parse
        query = f'ti:"{urllib.parse.quote(entry.title)}"'
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            query += f' AND au:{urllib.parse.quote(first_author)}'
        resp = requests.get("https://export.arxiv.org/api/query", params={"search_query": query, "max_results": 3}, timeout=8, headers={"User-Agent": "LNI-Checker/6.2"})
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
                            key=entry.key, title=entry.title, status="verified" if sim >= 0.8 else "partial_match",
                            confidence=sim, matched_title=title, open_access_url=oa,
                            note=f"Found on arXiv (match: {int(sim*100)}%)", sources_checked=["arXiv (search)"],
                        )
    except Exception:
        pass
    return None


def _search_openreview(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("openreview.net", 0.5)
    try:
        resp = requests.get("https://api2.openreview.net/notes/search", params={"term": entry.title, "limit": 5, "offset": 0}, timeout=7, headers={"User-Agent": "LNI-Checker/6.3"})
        if resp.status_code != 200:
            return None
        notes = resp.json().get("notes", [])
        for note in notes[:5]:
            content = note.get("content", {})
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
                    key=entry.key, title=entry.title, status="verified" if sim >= 0.78 else "partial_match",
                    confidence=sim, matched_title=found_title, open_access_url=oa_url,
                    note=f"Found on OpenReview (match: {int(sim * 100)}%)", sources_checked=["OpenReview"],
                    correct_authors=author_str,
                )
    except Exception:
        pass
    return None


def _search_ieee(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("ieeexplore.ieee.org", 1.0)
    try:
        api_key = os.environ.get("IEEE_API_KEY", "").strip()
        params = {"querytext": entry.title[:200], "max_records": 5, "format": "json", "apikey": api_key or "none"}
        resp = requests.get("https://ieeexploreapi.ieee.org/api/v1/search/articles", params=params, timeout=7, headers={"User-Agent": "LNI-Checker/6.3"})
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            for art in articles[:5]:
                title = art.get("title", "")
                if not title:
                    continue
                sim = _title_similarity(entry.title, title)
                if sim >= 0.55:
                    doi = art.get("doi", "")
                    authors_raw = art.get("authors", {}).get("authors", [])
                    author_str = "; ".join(a.get("full_name", "") for a in authors_raw[:4]) if authors_raw else None
                    year = str(art.get("publication_year", ""))
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title, doi=doi or None, open_access_url=f"https://doi.org/{doi}" if doi else None,
                        note=f"Found on IEEE Xplore (match: {int(sim*100)}%)", sources_checked=["IEEE Xplore"],
                        correct_authors=author_str, corrected_year=year or None,
                    )
    except Exception:
        pass
    return None


def _search_core(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("api.core.ac.uk", 0.5)
    try:
        api_key = os.environ.get("CORE_API_KEY", "").strip()
        headers = {"User-Agent": "LNI-Checker/6.3"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get("https://api.core.ac.uk/v3/search/works", params={"q": entry.title[:200], "limit": 5}, timeout=7, headers=headers)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for r in results[:5]:
                title = r.get("title", "")
                if not title:
                    continue
                sim = _title_similarity(entry.title, title)
                if sim >= 0.6:
                    doi = r.get("doi", "")
                    oa_url = r.get("downloadUrl") or (r.get("links") or [{}])[0].get("url") if r.get("links") else None
                    authors_raw = r.get("authors", [])
                    author_str = "; ".join(a.get("name", "") for a in authors_raw[:4]) if isinstance(authors_raw, list) else None
                    year = str(r.get("yearPublished", ""))
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title, doi=doi or None, open_access_url=oa_url,
                        note=f"Found on CORE (match: {int(sim*100)}%)", sources_checked=["CORE"],
                        correct_authors=author_str, corrected_year=year or None,
                    )
    except Exception:
        pass
    return None


def _search_springer(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    api_key = os.environ.get("SPRINGER_API_KEY", "").strip()
    if not api_key:
        return None
    _rate_limit("api.springernature.com", 0.5)
    try:
        resp = requests.get("https://api.springernature.com/meta/v2/json", params={"q": f'title:"{entry.title[:150]}"', "api_key": api_key, "p": 5}, timeout=7, headers={"User-Agent": "LNI-Checker/6.3"})
        if resp.status_code == 200:
            records = resp.json().get("records", [])
            for rec in records[:5]:
                title = rec.get("title", "")
                if not title:
                    continue
                sim = _title_similarity(entry.title, title)
                if sim >= 0.6:
                    doi = rec.get("doi", "")
                    creators = rec.get("creators", [])
                    author_str = "; ".join(c.get("creator", "") for c in creators[:4]) if creators else None
                    year = str(rec.get("publicationDate", ""))[:4]
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title, doi=doi or None, open_access_url=f"https://doi.org/{doi}" if doi else None,
                        note=f"Found on Springer (match: {int(sim*100)}%)", sources_checked=["Springer"],
                        correct_authors=author_str, corrected_year=year or None,
                    )
    except Exception:
        pass
    return None


def _search_opengrey(entry: BibEntry) -> Optional[VerificationResult]:
    """Search OpenGrey for European grey literature (reports, theses, conference proceedings)."""
    if not entry.title:
        return None
    _rate_limit("opengrey.eu", 0.5)
    try:
        # OpenGrey simple search
        query = entry.title
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            query += f" {first_author}"
        
        resp = requests.get(
            "https://www.opengrey.eu/search/index.php",
            params={
                "q": query,
                "l": "en",
                "document_type": "",
                "sm": "all"
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200:
            # Parse HTML response for results
            import re as regex_module
            titles = regex_module.findall(r'<h4[^>]*>([^<]+)</h4>', resp.text)
            for title in titles[:2]:
                title_clean = regex_module.sub(r'<[^>]+>', '', title).strip()
                sim = _title_similarity(entry.title, title_clean)
                if sim >= 0.65:
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title_clean,
                        note=f"Found on OpenGrey (European grey literature, match: {int(sim*100)}%)",
                        sources_checked=["OpenGrey (grey literature)"],
                    )
    except Exception:
        pass
    return None


def _search_open_library(entry: BibEntry) -> Optional[VerificationResult]:
    """Search Open Library for books and published works."""
    if not entry.title:
        return None
    _rate_limit("openlibrary.org", 0.5)
    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={
                "title": entry.title,
                "limit": 5
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            for doc in data.get("docs", [])[:3]:
                title = doc.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.65:
                    author = doc.get("author_name", [""])[0] if doc.get("author_name") else ""
                    year = str(doc.get("first_publish_year", ""))
                    
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title,
                        note=f"Found on Open Library (published book, match: {int(sim*100)}%)",
                        sources_checked=["Open Library"],
                        correct_authors=author or None,
                        corrected_year=year if year != "0" else None,
                    )
    except Exception:
        pass
    return None


def _search_internet_archive(entry: BibEntry) -> Optional[VerificationResult]:
    """Search Internet Archive for digitized books, texts, and documents."""
    if not entry.title:
        return None
    _rate_limit("archive.org", 0.5)
    try:
        resp = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f'title:"{entry.title}"',
                "fl": "identifier,title,creator,date",
                "output": "json",
                "rows": 5
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            for doc in data.get("response", {}).get("docs", [])[:3]:
                title = doc.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.65:
                    creator = doc.get("creator", [""])[0] if isinstance(doc.get("creator"), list) else doc.get("creator", "")
                    year = doc.get("date", "")[:4] if doc.get("date") else ""
                    identifier = doc.get("identifier", "")
                    url = f"https://archive.org/details/{identifier}" if identifier else None
                    
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title,
                        open_access_url=url,
                        note=f"Found on Internet Archive (digitized, match: {int(sim*100)}%)",
                        sources_checked=["Internet Archive"],
                        correct_authors=creator or None,
                        corrected_year=year if year else None,
                    )
    except Exception:
        pass
    return None


def _search_dnb(entry: BibEntry) -> Optional[VerificationResult]:
    """Search Deutsche Nationalbibliothek (German National Library) for German literature."""
    if not entry.title:
        return None
    _rate_limit("dnb.de", 0.5)
    try:
        # DNB uses SRU protocol
        resp = requests.get(
            "https://services.dnb.de/sru/",
            params={
                "version": "1.1",
                "operation": "searchRetrieve",
                "query": f'tit="{entry.title}"',
                "recordSchema": "MARC21-xml",
                "maximumRecords": 5
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200 and b"numberOfRecords" in resp.content:
            # Parse response for record count and title
            import re as regex_module
            records = regex_module.findall(rb'<datafield tag="245"[^>]*>.*?<subfield[^>]*>([^<]+)</subfield>', resp.content)
            
            if records:
                title_match = records[0].decode('utf-8', errors='ignore')
                sim = _title_similarity(entry.title, title_match)
                if sim >= 0.65:
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title_match,
                        note=f"Found on Deutsche Nationalbibliothek (German National Library, match: {int(sim*100)}%)",
                        sources_checked=["Deutsche Nationalbibliothek"],
                    )
    except Exception:
        pass
    return None


def _search_google_books(entry: BibEntry) -> Optional[VerificationResult]:
    """Search Google Books for published books and book previews."""
    if not entry.title:
        return None
    _rate_limit("googleapis.com", 1.0)
    try:
        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": entry.title,
                "maxResults": 5,
                "key": "AIzaSyA7Nk3PJs6VlP1pqBj3S7bpJNMsKWrCUIs"  # Free tier key
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", [])[:3]:
                vol_info = item.get("volumeInfo", {})
                title = vol_info.get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.65:
                    authors = "; ".join(vol_info.get("authors", [])[:3]) if vol_info.get("authors") else None
                    year = vol_info.get("publishedDate", "")[:4] if vol_info.get("publishedDate") else None
                    preview_url = item.get("volumeInfo", {}).get("previewLink")
                    
                    return VerificationResult(
                        key=entry.key, title=entry.title,
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title,
                        open_access_url=preview_url,
                        note=f"Found on Google Books (match: {int(sim*100)}%)",
                        sources_checked=["Google Books"],
                        correct_authors=authors,
                        corrected_year=year,
                    )
    except Exception:
        pass
    return None


def _search_pubmed(entry: BibEntry) -> Optional[VerificationResult]:
    """Search PubMed Central for biomedical and life sciences literature."""
    if not entry.title:
        return None
    _rate_limit("ncbi.nlm.nih.gov", 0.5)
    try:
        # Search via NCBI E-utils
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pmc",
                "term": entry.title,
                "retmax": 5,
                "rettype": "json"
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3", "Email": "reference-checker@research.org"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            
            if ids:
                # Get details for first result
                details_resp = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={
                        "db": "pmc",
                        "id": ids[0],
                        "rettype": "json"
                    },
                    timeout=8,
                    headers={"User-Agent": "LNI-Checker/6.3"}
                )
                
                if details_resp.status_code == 200:
                    detail_data = details_resp.json()
                    doc = detail_data.get("result", {}).get(ids[0], {})
                    title = doc.get("title", "")
                    sim = _title_similarity(entry.title, title)
                    
                    if sim >= 0.65:
                        authors = doc.get("authors", [])
                        author_str = "; ".join([a.get("name", "") for a in authors[:3]]) if authors else None
                        year = doc.get("epublish_date", doc.get("pdate", ""))[:4] if doc.get("epublish_date") or doc.get("pdate") else None
                        
                        return VerificationResult(
                            key=entry.key, title=entry.title,
                            status="verified" if sim >= 0.78 else "partial_match",
                            confidence=sim, matched_title=title,
                            note=f"Found on PubMed Central (biomedical, match: {int(sim*100)}%)",
                            sources_checked=["PubMed Central"],
                            correct_authors=author_str,
                            corrected_year=year,
                        )
    except Exception:
        pass
    return None



    """Search Zenodo for grey literature, reports, and preprints."""
    if not entry.title:
        return None
    _rate_limit("zenodo.org", 0.5)
    try:
        # Search Zenodo API
        query = entry.title
        if entry.authors:
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            query += f" {first_author}"
        
        resp = requests.get(
            "https://zenodo.org/api/records/",
            params={
                "q": query,
                "size": 5,
                "sort": "-mostrecent"
            },
            timeout=8,
            headers={"User-Agent": "LNI-Checker/6.3"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            for record in data.get("hits", {}).get("hits", []):
                title = record.get("metadata", {}).get("title", "")
                sim = _title_similarity(entry.title, title)
                if sim >= 0.65:
                    year = record.get("metadata", {}).get("publication_date", "")[:4]
                    doi = record.get("metadata", {}).get("doi")
                    url = record.get("links", {}).get("html") or record.get("links", {}).get("self")
                    
                    return VerificationResult(
                        key=entry.key, title=entry.title, 
                        status="verified" if sim >= 0.78 else "partial_match",
                        confidence=sim, matched_title=title,
                        doi=doi, open_access_url=url,
                        note=f"Found on Zenodo (grey literature archive, match: {int(sim*100)}%)",
                        sources_checked=["Zenodo (grey literature)"],
                        corrected_year=year or None,
                    )
    except Exception:
        pass
    return None


def _search_opengrey(entry: BibEntry) -> Optional[VerificationResult]:
    """Search OpenGrey for European grey literature (technical reports, etc.)."""
    if not entry.title:
        return None
    _rate_limit("opengrey.eu", 0.5)
    try:
        # OpenGrey doesn't have a public API, but we can check if it exists via web search
        # For now, return None to avoid rate limiting
        # In future: could implement web scraping or use their search interface
        pass
    except Exception:
        pass
    return None



    if not entry.title:
        return None
    _rate_limit("scholar.google.com", 2.0)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept-Language": "en-US,en;q=0.9"}
        resp = requests.get("https://scholar.google.com/scholar", params={"q": entry.title, "hl": "en", "num": 3}, timeout=8, headers=headers)
        if resp.status_code == 200:
            titles = re.findall(r'class="gs_rt"[^>]*>(?:<[^>]+>)*([^<]+)', resp.text)
            for title in titles[:2]:
                title_clean = re.sub(r'\[[^\]]+\]', '', title).strip()
                sim = _title_similarity(entry.title, title_clean)
                if sim >= 0.6:
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="partial_match", confidence=sim,
                        matched_title=title_clean, note=f"Found on Google Scholar (match: {int(sim*100)}%)",
                        sources_checked=["Google Scholar"],
                    )
    except Exception:
        pass
    return None


def _search_duckduckgo(entry: BibEntry) -> Optional[VerificationResult]:
    if not entry.title:
        return None
    _rate_limit("duckduckgo.com", 1.0)
    try:
        resp = requests.get("https://html.duckduckgo.com/html/", params={"q": f'"{entry.title}"'}, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            if snippets:
                combined = " ".join(snippets[:2]).lower()
                title_words = set(w for w in entry.title.lower().split() if len(w) > 3)
                matches = sum(1 for w in title_words if w in combined)
                coverage = matches / len(title_words) if title_words else 0
                if coverage >= 0.5:
                    return VerificationResult(
                        key=entry.key, title=entry.title, status="partial_match" if coverage >= 0.6 else "not_found",
                        confidence=coverage, web_evidence=snippets[0][:200], note=f"Web search found evidence (coverage: {int(coverage*100)}%)",
                        sources_checked=["Web (DDG)"],
                    )
    except Exception:
        pass
    return None


def _verify_website(entry: BibEntry) -> VerificationResult:
    """Verify a website/grey-literature reference by checking URL reachability."""
    url = entry.url or ""
    title = entry.title or ""
    
    if not url:
        # No URL but we have a title — try to classify as grey literature by title/publisher
        if entry.publisher or title:
            grey_indicators = [
                "bitkom", "flexera", "gartner", "forrester", "statista", "idc",
                "mckinsey", "deloitte", "pwc", "cloud report", "state of the cloud",
                "industry report", "market report", "whitepaper",
            ]
            combined = (title + " " + (entry.publisher or "")).lower()
            if any(g in combined for g in grey_indicators):
                return VerificationResult(
                    key=entry.key, title=title or "(grey literature)",
                    status="partial_match", confidence=0.6,
                    note="Grey literature source — no URL to verify. Recommend manual check.",
                    sources_checked=["grey_lit_classification"],
                )
        return VerificationResult(
            key=entry.key, title=title or "(website)", status="error",
            confidence=0.0, note="No URL provided for website",
            sources_checked=[],
        )
    
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    # Extract domain for rate limiting
    try:
        domain = url.split("/")[2]
    except IndexError:
        domain = url
    
    # Known trustworthy grey-literature domains — give high confidence even without live check
    trusted_grey_domains = {
        "bitkom.org", "flexera.com", "info.flexera.com", "gartner.com",
        "forrester.com", "mckinsey.com", "deloitte.com", "statista.com",
        "idc.com", "accenture.com", "pwc.com", "37signals.com",
        "bsi.bund.de", "ec.europa.eu", "nist.gov", "basecamp.com",
    }
    is_trusted = any(td in domain for td in trusted_grey_domains)
    
    try:
        _rate_limit(domain, 0.5)
        # Try HEAD first (faster), fallback to GET
        try:
            resp = requests.head(url, timeout=8, allow_redirects=True,
                                  headers={"User-Agent": "Mozilla/5.0 (LNI Reference Checker)"})
        except Exception:
            resp = None
        
        if resp is None or resp.status_code in (405, 403, 401):
            # HEAD not allowed — try GET with a range request
            resp = requests.get(url, timeout=10, allow_redirects=True, stream=True,
                                 headers={"User-Agent": "Mozilla/5.0 (LNI Reference Checker)",
                                          "Range": "bytes=0-1023"})
        
        if resp.status_code < 400:
            return VerificationResult(
                key=entry.key, title=title or url, status="verified",
                confidence=0.92, open_access_url=url,
                note=f"URL reachable (HTTP {resp.status_code})",
                sources_checked=["url_check"],
            )
        elif resp.status_code in (403, 401):
            # Access restricted — we can't confirm it exists
            if is_trusted:
                return VerificationResult(
                    key=entry.key, title=title or url, status="partial_match",
                    confidence=0.55, open_access_url=url,
                    note=f"URL access restricted (HTTP {resp.status_code}) — couldn't be validated",
                    sources_checked=["url_check"],
                )
            # Unknown domain with access restriction
            return VerificationResult(
                key=entry.key, title=title or url, status="partial_match",
                confidence=0.3, note=f"URL access restricted (HTTP {resp.status_code}) — couldn't be validated",
                sources_checked=["url_check"],
            )
        elif resp.status_code in (404, 410):
            note = f"URL not found (HTTP {resp.status_code}) — may have moved or expired"
            # For trusted domains, still give partial credit — report may have moved
            if is_trusted:
                return VerificationResult(
                    key=entry.key, title=title or url, status="partial_match",
                    confidence=0.55, note=note + f" (trusted publisher: {domain})",
                    sources_checked=["url_check"],
                )
            return VerificationResult(
                key=entry.key, title=title or url, status="not_found",
                confidence=0.1, note=note, sources_checked=["url_check"],
            )
        else:
            confidence = 0.6 if is_trusted else 0.2
            return VerificationResult(
                key=entry.key, title=title or url, status="partial_match",
                confidence=confidence,
                note=f"URL returned HTTP {resp.status_code}",
                sources_checked=["url_check"],
            )
    except requests.exceptions.SSLError:
        # SSL errors on known domains still indicate the domain exists
        confidence = 0.65 if is_trusted else 0.3
        return VerificationResult(
            key=entry.key, title=title or url, status="partial_match",
            confidence=confidence, open_access_url=url,
            note="URL has SSL issue but domain is reachable",
            sources_checked=["url_check"],
        )
    except Exception as e:
        # Network error — for trusted publishers, give benefit of the doubt
        if is_trusted:
            return VerificationResult(
                key=entry.key, title=title or url, status="partial_match",
                confidence=0.65, open_access_url=url,
                note=f"URL check failed (network error) but publisher '{domain}' is trusted",
                sources_checked=["url_check", "trusted_publisher"],
            )
        return VerificationResult(
            key=entry.key, title=title or url, status="error",
            confidence=0.0, note=f"URL check failed: {str(e)[:100]}",
            sources_checked=["url_check"],
        )


def _check_unpaywall(doi: str) -> Optional[str]:
    email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
    if not email or email == "lni-checker@uni-project.de":
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


def _validate_doi_format(doi: str) -> tuple:
    if not doi:
        return True, ""
    doi = doi.strip()
    if not doi.startswith("10."):
        return False, f"DOI does not start with '10.': {doi[:30]}"
    m = re.match(r"^10\.([0-9]{4,})/(\S+)$", doi)
    if not m:
        return False, f"Malformed DOI structure: {doi[:40]}"
    registrant, suffix = m.group(1), m.group(2)
    if int(registrant) < 1000:
        return False, f"Implausible DOI registrant: 10.{registrant}"
    if re.match(r"^(fake|test|example|placeholder)", suffix.lower()):
        return False, f"Fake-looking DOI suffix: {suffix[:30]}"
    return True, ""


def _check_year_plausibility(year: str) -> tuple:
    if not year:
        return True, ""
    try:
        y = int(str(year).strip()[:4])
    except ValueError:
        return False, f"Non-numeric year: {year}"
    import datetime
    current_year = datetime.date.today().year
    if y > current_year + 1:
        return False, f"Future year: {y} (current: {current_year})"
    if y < 1800:
        return False, f"Implausibly old publication year: {y}"
    return True, ""


def verify_reference(entry: BibEntry) -> VerificationResult:
    """
    Verify a single reference - LOCAL DB CHECKED FIRST, then APIs, then AI.
    Pipeline: local DB → professor review → API checks → web search → AI
    """
    # Route website entries and any entry that has a URL (grey literature)
    # to the URL-based verifier. Also run API checks in parallel for academic sources.
    if entry.entry_type == "website":
        return _verify_website(entry)
    
    # For non-website entries that have a URL, run URL check in parallel with APIs
    has_url = bool(entry.url)
    
    # STEP 1: Check local SQLite DB FIRST (fastest, most reliable)
    cached_paper = search_cache(entry.title or "", entry.authors or "")
    if cached_paper:
        sim = _title_similarity(entry.title or "", cached_paper.title)
        if sim >= 0.7:
            return VerificationResult(
                key=entry.key, title=entry.title or "", status="verified", confidence=cached_paper.confidence,
                matched_title=cached_paper.title, doi=cached_paper.doi, open_access_url=cached_paper.url,
                note=f"Found in local verified database (from {cached_paper.source}) - {int(sim*100)}% title match",
                sources_checked=["local_db"], correct_authors=cached_paper.authors,
            )
    
    # STEP 2: Check memory cache
    cached = _get_cached(entry)
    if cached:
        result = copy.copy(cached)
        result.note = (result.note or "") + " [cached]"
        return result
    
    # STEP 3: Check professor review decisions (manual override)
    review = get_review_decision(entry.title or "", entry.authors or "")
    if review:
        if review.get("decision") == "verified":
            return VerificationResult(
                key=entry.key, title=entry.title or "", status="verified", confidence=0.99,
                matched_title=entry.title, doi=review.get("verified_doi"), open_access_url=review.get("verified_url"),
                note=f"Professor verified: {review.get('professor_note', 'Manually approved')}",
                sources_checked=["professor_review"], correct_authors=entry.authors,
            )
        elif review.get("decision") == "rejected":
            return VerificationResult(
                key=entry.key, title=entry.title or "", status="not_found", confidence=0.0,
                note=f"Professor marked as rejected: {review.get('professor_note', '')}",
                sources_checked=["professor_review"],
            )
    
    # STEP 4: Check false positives (paper was flagged but professor said REAL)
    fp_record = get_false_positive(entry.title or "", entry.authors or "")
    if fp_record:
        return VerificationResult(
            key=entry.key, title=entry.title or "", status="verified", confidence=0.95,
            matched_title=entry.title, note=f"Professor previously corrected this as REAL: {fp_record.get('notes', '')}",
            sources_checked=["professor_false_positive_correction"], correct_authors=entry.authors,
        )
    
    # STEP 5: For entries with a URL, run URL verification in parallel with APIs
    # This handles cases like He22 (podcast/blog) that have a URL but were classified as 'unknown'
    if has_url:
        url_result = _verify_website(entry)
        if url_result.status == "verified" and url_result.confidence >= 0.85:
            # URL confirmed reachable — save to local DB and return
            save_to_cache(
                title=entry.title or "", authors=entry.authors or "",
                year=entry.year or "", doi=entry.doi or "",
                url=entry.url or "", source="url_check", confidence=url_result.confidence,
            )
            return url_result
        # URL exists but with lower confidence — still run API checks, then merge
        url_partial = url_result
    else:
        url_partial = None

    # STEP 6: Run API checks (CrossRef, Semantic Scholar, etc.)
    api_result = _run_api_checks(entry)
    
    # If we have a URL partial result, merge it with API result
    if url_partial and url_partial.status in ("verified", "partial_match"):
        if api_result.status == "verified":
            # Both confirmed — boost confidence
            api_result.confidence = min(api_result.confidence + 0.05, 0.98)
            api_result.open_access_url = api_result.open_access_url or url_partial.open_access_url
            api_result.note = (api_result.note or "") + f" | URL also reachable"
        elif api_result.status in ("not_found", "error"):
            # API found nothing but URL works — treat as partial match (grey literature)
            url_partial.note = (url_partial.note or "") + " | Not found in academic databases (expected for grey literature)"
            return url_partial
    
    return api_result


def _run_api_checks(entry: BibEntry) -> VerificationResult:
    """Run all API checks (only called if not found in local DB)."""
    all_results: List[VerificationResult] = []
    
    # Pre-flight checks
    _doi_valid, _doi_reason = _validate_doi_format(entry.doi or "")
    _year_plaus, _year_reason = _check_year_plausibility(entry.year or "")
    _preflight_flags: List[str] = []
    if not _doi_valid:
        _preflight_flags.append(f"Invalid DOI format: {_doi_reason}")
    if not _year_plaus:
        _preflight_flags.append(f"Implausible year: {_year_reason}")

    # PHASE 1: Fast identifier lookups
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_lookup_by_doi, entry), executor.submit(_lookup_by_arxiv_id, entry), executor.submit(_lookup_by_isbn, entry)]
        for future in as_completed(futures, timeout=5):
            try:
                r = future.result()
                if r:
                    all_results.append(r)
            except Exception:
                pass
    
    # PHASE 2: Title/author search (academic databases)
    if not any(r.status == "verified" for r in all_results):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(_search_crossref, entry), 
                executor.submit(_search_semantic_scholar, entry),
                executor.submit(_search_openalex, entry), 
                executor.submit(_search_dblp, entry),
                executor.submit(_search_acl, entry), 
                executor.submit(_search_ieee, entry),
                executor.submit(_search_core, entry), 
                executor.submit(_search_springer, entry),
                executor.submit(_search_pubmed, entry),  # ← NEW: Biomedical
                executor.submit(_search_open_library, entry),  # ← NEW: Books
            ]
            for future in as_completed(futures, timeout=10):
                try:
                    r = future.result()
                    if r:
                        all_results.append(r)
                except Exception:
                    pass
    
    # PHASE 3: Deep search (including grey literature)
    best_so_far = max(all_results, key=lambda r: r.confidence, default=None) if all_results else None
    if not best_so_far or best_so_far.confidence < 0.6:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_search_arxiv_fallback, entry), 
                executor.submit(_search_openreview, entry),
                executor.submit(_search_google_scholar, entry), 
                executor.submit(_search_zenodo, entry),  # ← Grey literature
                executor.submit(_search_opengrey, entry),  # ← European grey lit
                executor.submit(_search_internet_archive, entry),  # ← Digitized books
                executor.submit(_search_dnb, entry),  # ← German National Library
                executor.submit(_search_google_books, entry),  # ← Published books
                executor.submit(_search_duckduckgo, entry)
            ]
            for future in as_completed(futures, timeout=15):
                try:
                    r = future.result()
                    if r:
                        all_results.append(r)
                except Exception:
                    pass
    
    # PHASE 4: Web search + LLM fallback (only for ambiguous cases)
    if not all_results or (best_so_far and best_so_far.confidence < 0.5):
        try:
            web_result = verify_with_web_search({"title": entry.title, "authors": entry.authors, "year": entry.year}, best_so_far.status if best_so_far else "not_found")
            if web_result.get("status") == "verified":
                save_to_cache(title=entry.title or "", authors=entry.authors or "", year=entry.year or "", doi=web_result.get("doi", ""), url=web_result.get("open_access_url", ""), source="web_search", confidence=web_result.get("confidence", 0.8))
                return VerificationResult(
                    key=entry.key, title=entry.title or "", status="verified", confidence=web_result.get("confidence", 0.8),
                    matched_title=web_result.get("matched_title"), open_access_url=web_result.get("open_access_url"),
                    note=web_result.get("note", "Verified via web search"), sources_checked=["web_search", "llm_verification"]
                )
        except Exception:
            pass
    
    # Aggregate results
    if not all_results:
        note = "No results found in any academic database."
        if _preflight_flags:
            note += f" Issues: {'; '.join(_preflight_flags)}"
        return VerificationResult(
            key=entry.key, title=entry.title or "", status="not_found", confidence=0.0,
            note=note, sources_checked=[],
        )
    
    priority = {"verified": 3, "partial_match": 2, "not_found": 1, "error": 0}
    all_results.sort(key=lambda r: (priority.get(r.status, 0), r.confidence), reverse=True)
    best = all_results[0]
    
    all_sources = []
    for r in all_results:
        for src in r.sources_checked:
            if src not in all_sources:
                all_sources.append(src)
    best.sources_checked = all_sources
    
    if _preflight_flags:
        flag_str = " | ".join(_preflight_flags)
        best.note = f"{best.note or ''} ⚠ {flag_str}".strip()
        if not _doi_valid and best.status == "not_found":
            best.confidence = min(best.confidence, 0.2)
    
    for r in all_results:
        if not best.web_evidence and r.web_evidence:
            best.web_evidence = r.web_evidence
        if not best.correct_authors and r.correct_authors:
            best.correct_authors = r.correct_authors
        if not best.doi and r.doi:
            best.doi = r.doi
        if not best.open_access_url and r.open_access_url:
            best.open_access_url = r.open_access_url
        if r.is_retracted:
            best.is_retracted = True
            best.retraction_doi = best.retraction_doi or r.retraction_doi
            best.retraction_note = best.retraction_note or r.retraction_note
            best.status = "retracted"
    
    verified_count = sum(1 for r in all_results if r.status == "verified")
    if verified_count >= 2 and best.status == "verified":
        best.confidence = min(best.confidence + 0.08, 0.98)
        best.note = f"Confirmed by {verified_count} independent sources. {best.note}"
    
    if best.status == "verified":
        save_to_cache(title=entry.title or "", authors=entry.authors or "", year=entry.year or "", doi=best.doi or "", url=best.open_access_url or "", source="api", confidence=best.confidence)
    
    # At the end of verify_reference function, before returning best
# Auto-save verified REAL papers to local database
    if best.status == "verified" and best.confidence >= 0.7:
        try:
            from local_db import save_to_cache
            save_to_cache(
            title=entry.title or "",
            authors=entry.authors or "",
            year=entry.year or "",
            doi=best.doi or "",
            url=best.open_access_url or entry.url or "",
            source="api_verified",
            confidence=best.confidence
        )
        except Exception as e:
            print(f"Auto-save to DB failed: {e}")
    _put_cache(entry, best)
    return best


def verify_all_references(bib_entries: dict) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_key = {executor.submit(verify_reference, entry): key for key, entry in bib_entries.items()}
        for future in as_completed(future_to_key, timeout=120):
            try:
                results.append(future.result())
            except Exception as e:
                key = future_to_key[future]
                results.append(VerificationResult(key=key, title=bib_entries[key].title or "", status="error", confidence=0.0, note=f"Verification crashed: {str(e)[:100]}", sources_checked=[]))
    key_order = list(bib_entries.keys())
    results.sort(key=lambda r: key_order.index(r.key) if r.key in key_order else 999)
    return results


def extract_citations_from_body(body_text: str) -> set:
    keys = set()
    _LNI_KEY = r'[A-Za-z]{2,6}\d{2}[a-z]?'
    filtered = re.sub(r'(?:e\.?g\.?|z\.?[Bb]\.?|cf\.?|i\.?e\.?|for example|e\.g\.,|z\.B\.,|vgl\.|see e\.g\.)' + r'\s*[\(\[]?' + _LNI_KEY + r'(?:[;,]\s*' + _LNI_KEY + r')*[\)\]]?', '[EXAMPLE_REF]', body_text, flags=re.IGNORECASE)
    filtered = re.sub(r'\{\d+,?\d*\}[^\[]*(?=\[)', ' ', filtered)
    lni_bracket_matches = re.findall(r'\[' + _LNI_KEY + r'(?:[;,]\s*' + _LNI_KEY + r')*\]', filtered)
    for match in lni_bracket_matches:
        for key in re.split(r'[;,]\s*', match.strip('[]')):
            key = key.strip()
            if re.match(r'^' + _LNI_KEY + r'$', key):
                keys.add(key)
    numeric_bracket_matches = re.findall(r'\[(\d{1,3}(?:[;,]\s*\d{1,3})*)\]', filtered)
    if numeric_bracket_matches:
        keys.add('__numeric_citations__')
    return keys


def extract_citation_contexts(body_text: str) -> dict:
    contexts = {}
    _LNI_KEY = r'[A-Za-z]{2,6}\d{2}[a-z]?'
    multi_lni = _LNI_KEY + r'(?:[;,]\s*' + _LNI_KEY + r')*'
    multi_num = r'\d+(?:[;,]\s*\d+)*'
    for m in re.finditer(r'([^.]{0,80})\[(' + multi_lni + r'|' + multi_num + r')\]([^.]{0,80})', body_text):
        snippet = (m.group(1) + '[' + m.group(2) + ']' + m.group(3)).strip()
        for key in re.split(r'[;,]\s*', m.group(2)):
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
    bib_keys = set(str(k) for k in bib_entries.keys())
    lni_cited = {k for k in cited_keys if not str(k).startswith('__') and not str(k).isdigit()}
    numeric_cited = {k for k in cited_keys if str(k).isdigit() or str(k).startswith('__NUM_')}
    bib_lni_keys = {k for k in bib_keys if not k.isdigit()}
    bib_numeric_keys = {k for k in bib_keys if k.isdigit()}
    r = CrossCheckResult()
    if bib_numeric_keys and numeric_cited:
        r.cited_not_in_bib = sorted(numeric_cited - bib_numeric_keys)
        r.in_bib_not_cited = sorted(bib_numeric_keys - numeric_cited)
        r.correctly_used = sorted(numeric_cited & bib_numeric_keys)
    else:
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
                dupes.append({"key_a": a.key, "key_b": b.key, "title_a": a.title, "title_b": b.title, "similarity": round(score, 2), "type": "exact" if score >= 0.97 else "near-duplicate"})
    return dupes


def check_lni_macros(body_text: str) -> list:
    suggestions = []
    for pattern, message in [(r'\be\.g\.', r"Use LNI macro '\eg' instead of 'e.g.'"), (r'\bi\.e\.', r"Use LNI macro '\ie' instead of 'i.e.'"), (r'\bcf\.', r"Use LNI macro '\cf' instead of 'cf.'"), (r'\bet al\.', r"Use LNI macro '\etal' instead of 'et al.'")]:
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


def compute_score(bib_list, xcheck, verification_results, style_suggestions, duplicates, ai_fake_count=0, retracted_count=0, year_mismatches=0, author_mismatches=0):
    """Compute a 0-100 score with detailed per-category penalty breakdown."""
    score, penalties = 100, []
    rows = [
        ("Retracted papers cited", retracted_count, 25, 50),
        ("Missing from bibliography", len(xcheck.cited_not_in_bib), 10, 30),
        ("Likely fabricated references", ai_fake_count, 15, 45),
        ("Year mismatch (>1 yr off)", year_mismatches, 8, 24),
        ("Author mismatch", author_mismatches, 5, 20),
        ("Cited nowhere in text", len(xcheck.in_bib_not_cited), 5, 20),
        ("Incomplete LNI entries", sum(1 for e in bib_list if e.completeness_issues), 5, 20),
        ("Duplicate entries", len(duplicates), 5, 10),
        ("LNI style violations", len(style_suggestions), 2, 6),
    ]
    for label, count, per_item, cap in rows:
        p = min(count * per_item, cap)
        if p:
            penalties.append({"category": label, "count": count, "deduction": p, "per_item": per_item, "cap": cap})
        score -= p
    score = max(0, min(100, score))
    if score >= 95: grade = "A+"
    elif score >= 90: grade = "A"
    elif score >= 85: grade = "A-"
    elif score >= 80: grade = "B+"
    elif score >= 75: grade = "B"
    elif score >= 70: grade = "B-"
    elif score >= 65: grade = "C+"
    elif score >= 60: grade = "C"
    elif score >= 55: grade = "C-"
    elif score >= 50: grade = "D"
    else: grade = "F"
    return {"score": score, "grade": grade, "penalties": penalties, "max_score": 100}