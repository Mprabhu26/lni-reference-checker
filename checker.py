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
    """
    Deterministic title normaliser used for ALL similarity comparisons.
    Must be applied to BOTH sides before any threshold check so that
    casing ('Is' vs 'is'), punctuation ('.', ',', ':'), and diacritics
    (é, ö, ñ …) never cause a real paper to fall below the 0.95 cutoff.
    """
    if not t:
        return ""
    # ── 1. Lower-case first so all subsequent replacements are uniform ──────
    t = t.lower().strip()
    # ── 2. Full diacritic / umlaut map (covers Latin-1 + common accents) ───
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
    # ── 3. Strip LaTeX / HTML entities ──────────────────────────────────────
    t = re.sub(r'&[a-z]+;', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'[{}]', '', t)
    # ── 4. Collapse all punctuation to spaces (covers -, :, ., ,, /, etc.) ─
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    # ── 5. Remove stop-words (keep content words only) ──────────────────────
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
                        key=entry.key, title=entry.title, status="retracted" if is_ret else ("verified" if sim >= 0.95 else "partial_match"),
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
        api_key = ""
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                status = "verified" if sim >= 0.95 else "partial_match"
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                            key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                    key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
                        key=entry.key, title=entry.title, status="verified" if sim >= 0.95 else "partial_match",
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
    