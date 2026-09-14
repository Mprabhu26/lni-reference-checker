"""
Dynamic Author-Journal Verifier — v3.0
--------------------------------------
Verifies that a cited author exists and has published in the cited venue.

v3.0 changes:
  - _is_landmark_paper no longer matches on key substrings alone. It requires
    title similarity >= 0.70, or >= 0.55 with an exact key match. This closes
    the landmark-hijack loophole where a fabricated entry could be promoted
    to REAL simply because it reused a well-known citation key (e.g. [VS17]).
  - Removed the reverse-substring check (`key_lower in pattern.lower()`) which
    allowed arbitrary short keys to match long landmark key patterns.
  - verify_reference_comprehensive now accepts a landmark match only after the
    citation's own authors have been checked against the landmark's author list.
  - No other behavioural changes: rate limiting, caching, Semantic Scholar and
    CrossRef lookups, and the overall confidence formula are unchanged.
"""

import re
import time
import threading
import requests
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuthorVerificationResult:
    """Result of author-journal verification"""
    author_exists: bool
    journal_exists: bool
    author_venue_match: bool
    overall_confidence: float
    warnings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    found_authors: List[str] = field(default_factory=list)
    found_journals: List[str] = field(default_factory=list)
    match_details: Dict[str, float] = field(default_factory=dict)
    is_landmark: bool = False
    landmark_name: str = ""


# ---------------------------------------------------------------------------
# LANDMARK PAPERS DATABASE
# ---------------------------------------------------------------------------

# These are universally-known papers. Entries are only consulted when the
# citation's own title is similar to the landmark title (threshold below).
LANDMARK_PAPERS = {
    "attention is all you need": {
        "authors": ["vaswani", "shazeer", "parmar", "uszkoreit",
                    "jones", "gomez", "kaiser", "polosukhin"],
        "venue": "neurips",
        "year": "2017",
        "key_patterns": ["vs17", "vsp17"],
        "url": "https://arxiv.org/abs/1706.03762",
        "title": "Attention Is All You Need",
    },
    "imagenet classification with deep convolutional neural networks": {
        "authors": ["krizhevsky", "sutskever", "hinton"],
        "venue": "neurips",
        "year": "2012",
        "key_patterns": ["ksh12", "ks12"],
        "url": "https://papers.nips.cc/paper/2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
    },
    "deep residual learning for image recognition": {
        "authors": ["he", "zhang", "ren", "sun"],
        "venue": "cvpr",
        "year": "2015",
        "key_patterns": ["hzrs15", "he15"],
        "url": "https://arxiv.org/abs/1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
    },
    "bert pre-training of deep bidirectional transformers": {
        "authors": ["devlin", "chang", "lee", "toutanova"],
        "venue": "naacl",
        "year": "2018",
        "key_patterns": ["dclt18", "de18"],
        "url": "https://arxiv.org/abs/1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    },
    "adam a method for stochastic optimization": {
        "authors": ["kingma", "ba"],
        "venue": "iclr",
        "year": "2014",
        "key_patterns": ["kb14", "kingma2014"],
        "url": "https://arxiv.org/abs/1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
    },
}


# ---------------------------------------------------------------------------
# Title normalisation + landmark matching (strict)
# ---------------------------------------------------------------------------

def _normalize_title_for_matching(title: str) -> str:
    """Normalize a title for landmark matching."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r'[^\w\s]', '', t)
    stop = {
        'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 'by', 'at',
    }
    words = [w for w in t.split() if w not in stop and len(w) > 2]
    return ' '.join(words)


def _title_similarity_simple(title1: str, title2: str) -> float:
    """Simple Jaccard similarity over normalized title tokens."""
    if not title1 or not title2:
        return 0.0
    t1 = _normalize_title_for_matching(title1)
    t2 = _normalize_title_for_matching(title2)
    if not t1 or not t2:
        return 0.0
    s1 = set(t1.split())
    s2 = set(t2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _is_landmark_paper(title: str, key: str = "") -> Optional[Dict]:
    """
    Check whether a paper matches a known landmark.

    v3.0 rules:
      - Title similarity >= 0.70 always qualifies.
      - Title similarity >= 0.55 AND exact key match (not substring) qualifies.
      - A key match alone never qualifies. A fabricated entry that merely
        reuses a well-known key (e.g. [VS17]) is not a landmark.

    Returns the landmark record with a `matched_by` field, or None.
    """
    if not title:
        return None

    key_lower = (key or "").strip().lower()

    for landmark_title, info in LANDMARK_PAPERS.items():
        sim = _title_similarity_simple(title, landmark_title)

        if sim >= 0.70:
            return {
                **info,
                "matched_by": "title_similarity",
                "similarity": sim,
                "landmark_title": landmark_title,
            }

        if sim >= 0.55 and key_lower:
            key_matched = any(
                pattern.lower() == key_lower
                for pattern in info.get("key_patterns", [])
            )
            if key_matched:
                return {
                    **info,
                    "matched_by": "title+key",
                    "similarity": sim,
                    "landmark_title": landmark_title,
                }

    return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_RATE_LOCKS: Dict[str, threading.Lock] = {}
_RATE_LAST: Dict[str, float] = {}
_RATE_META_LOCK = threading.Lock()


def _rate_limit(host: str, min_interval: float) -> None:
    with _RATE_META_LOCK:
        if host not in _RATE_LOCKS:
            _RATE_LOCKS[host] = threading.Lock()
    with _RATE_LOCKS[host]:
        elapsed = time.time() - _RATE_LAST.get(host, 0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _RATE_LAST[host] = time.time()


def _get_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Author name normalization
# ---------------------------------------------------------------------------

def _normalize_author(author: str) -> str:
    if not author:
        return ""
    author = re.sub(r'\s+et\s+al\.?$', '', author, flags=re.IGNORECASE)
    author = re.sub(r',\s*eds?\.?$', '', author, flags=re.IGNORECASE)
    author = re.sub(r'\s+', ' ', author).strip()
    return author


def _extract_surname(author: str) -> str:
    if not author:
        return ""
    if ',' in author:
        surname = author.split(',')[0].strip()
    else:
        parts = author.split()
        surname = parts[-1] if parts else author
    particles = {
        'van', 'von', 'de', 'del', 'della', 'der', 'la', 'le', 'du', 'des', 'di',
    }
    surname_lower = surname.lower()
    if surname_lower in particles:
        return surname
    for p in particles:
        if surname_lower.startswith(p + ' '):
            surname = surname[len(p):].strip()
            break
    return surname


def _extract_initials(author: str) -> str:
    if not author:
        return ""
    if ',' in author:
        given = author.split(',')[1].strip() if ',' in author else ""
        if given:
            return given[0].upper()
    parts = author.split()
    if parts:
        return parts[0][0].upper()
    return ""


def _author_matches(existing_author: str, query_author: str) -> float:
    if not existing_author or not query_author:
        return 0.0

    ex_norm = _normalize_author(existing_author.lower())
    q_norm = _normalize_author(query_author.lower())

    if ex_norm == q_norm:
        return 1.0

    ex_surname = _extract_surname(existing_author).lower()
    q_surname = _extract_surname(query_author).lower()

    if ex_surname == q_surname:
        ex_init = _extract_initials(existing_author).lower()
        q_init = _extract_initials(query_author).lower()
        if ex_init and q_init and ex_init == q_init:
            return 0.95
        return 0.85

    if len(ex_surname) >= 3 and len(q_surname) >= 3:
        if ex_surname[:3] == q_surname[:3]:
            return 0.70

    if ex_surname and q_surname and ex_surname[0] == q_surname[0]:
        return 0.50

    return 0.0


def _author_overlap_vs_landmark(cited_authors: str,
                                landmark_authors: List[str]) -> float:
    """
    Surname-overlap ratio between the citation's authors and the landmark's
    canonical author list. Used to reject a landmark match when the cited
    author list has been fabricated.
    """
    if not cited_authors or not landmark_authors:
        return 0.0

    def _surnames(s: str) -> Set[str]:
        out: Set[str] = set()
        for part in re.split(r';|\band\b|\bund\b', s, flags=re.IGNORECASE):
            part = part.strip()
            if not part or re.match(r'^et\s+al\.?$', part.lower()):
                continue
            sur = part.split(',')[0].strip() if ',' in part else part.split()[-1].strip()
            sur = re.sub(r'[^a-z]', '', sur.lower())
            if sur:
                out.add(sur)
        return out

    cited = _surnames(cited_authors)
    landmark = {re.sub(r'[^a-z]', '', a.lower()) for a in landmark_authors}
    if not cited or not landmark:
        return 0.0

    matched = sum(
        1 for s in cited
        if any(s == lm or (len(s) >= 3 and len(lm) >= 3 and s[:3] == lm[:3])
               for lm in landmark)
    )
    return matched / len(cited)


# ---------------------------------------------------------------------------
# Semantic Scholar API — author lookup
# ---------------------------------------------------------------------------

def _search_semantic_scholar_author(author_name: str) -> Tuple[bool, List[str], float]:
    if not author_name or len(author_name) < 2:
        return False, [], 0.0

    surname = _extract_surname(author_name)
    if not surname or len(surname) < 2:
        return False, [], 0.0

    _rate_limit("semanticscholar.org", 0.25)

    try:
        session = _get_session()
        headers = {"User-Agent": "LNI-Checker/10.0"}

        resp = session.get(
            "https://api.semanticscholar.org/graph/v1/author/search",
            params={
                "query": surname,
                "limit": 5,
                "fields": "name,paperCount,citationCount",
            },
            timeout=10,
            headers=headers,
        )

        if resp.status_code == 200:
            data = resp.json()
            authors = data.get("data", [])
            if not authors:
                return False, [], 0.0

            best_score = 0.0
            best_names: List[str] = []
            for a in authors[:5]:
                a_name = a.get("name", "")
                if not a_name:
                    continue
                score = _author_matches(a_name, author_name)
                if score > best_score:
                    best_score = score
                    best_names = [a_name]
                elif score == best_score and best_score > 0.5:
                    best_names.append(a_name)

            if best_score >= 0.70:
                return True, best_names, min(0.95, best_score + 0.1)
            if best_score >= 0.40:
                return True, best_names, best_score
            return False, [], best_score

        if resp.status_code == 429:
            time.sleep(2)
            return _search_semantic_scholar_author(author_name)

        return False, [], 0.0

    except Exception as e:
        print(f"[Semantic Scholar] Author search error: {e}")
        return False, [], 0.0


# ---------------------------------------------------------------------------
# CrossRef API — venue lookup
# ---------------------------------------------------------------------------

def _search_crossref_venue(venue_name: str) -> Tuple[bool, List[str], float]:
    if not venue_name or len(venue_name) < 2:
        return False, [], 0.0

    _rate_limit("crossref.org", 0.2)

    try:
        session = _get_session()
        headers = {"User-Agent": "LNI-Checker/10.0"}

        resp = session.get(
            "https://api.crossref.org/journals",
            params={"query": venue_name, "rows": 5},
            timeout=10,
            headers=headers,
        )

        if resp.status_code == 200:
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                return False, [], 0.0

            best_score = 0.0
            best_names: List[str] = []
            venue_lower = venue_name.lower()

            for item in items[:5]:
                title = item.get("title", [""])[0] or ""
                if not title:
                    continue

                title_lower = title.lower()
                if venue_lower in title_lower or title_lower in venue_lower:
                    score = min(
                        0.95,
                        0.7 + len(venue_lower) / max(len(title_lower), 1) * 0.3,
                    )
                else:
                    query_words = set(venue_lower.split())
                    title_words = set(title_lower.split())
                    common = query_words & title_words
                    score = (
                        len(common) / max(len(query_words), 1) * 0.8
                        if common else 0.0
                    )

                if score > best_score:
                    best_score = score
                    best_names = [title]
                elif score == best_score and best_score > 0.5:
                    best_names.append(title)

            if best_score >= 0.70:
                return True, best_names, min(0.95, best_score + 0.05)
            if best_score >= 0.40:
                return True, best_names, best_score
            return False, [], best_score

        if resp.status_code == 429:
            time.sleep(2)
            return _search_crossref_venue(venue_name)

        return False, [], 0.0

    except Exception as e:
        print(f"[CrossRef] Venue search error: {e}")
        return False, [], 0.0


# ---------------------------------------------------------------------------
# Author ↔ venue match
# ---------------------------------------------------------------------------

def _check_author_venue_match(
    author_name: str, venue_name: str
) -> Tuple[bool, float, List[str]]:
    if not author_name or not venue_name:
        return False, 0.0, []

    surname = _extract_surname(author_name)
    if not surname or len(surname) < 2:
        return False, 0.0, []

    _rate_limit("semanticscholar.org", 0.25)

    try:
        session = _get_session()
        headers = {"User-Agent": "LNI-Checker/10.0"}

        resp = session.get(
            "https://api.semanticscholar.org/graph/v1/author/search",
            params={
                "query": surname,
                "limit": 3,
                "fields": "name,paperCount",
            },
            timeout=10,
            headers=headers,
        )

        if resp.status_code != 200:
            return False, 0.0, []

        authors = resp.json().get("data", [])
        if not authors:
            return False, 0.0, []

        best_author_id = None
        best_score = 0.0
        for a in authors[:3]:
            a_name = a.get("name", "")
            if not a_name:
                continue
            score = _author_matches(a_name, author_name)
            if score > best_score:
                best_score = score
                best_author_id = a.get("authorId")

        if not best_author_id or best_score < 0.5:
            return False, 0.0, []

        resp = session.get(
            f"https://api.semanticscholar.org/graph/v1/author/{best_author_id}/papers",
            params={"limit": 20, "fields": "title,venue,year"},
            timeout=10,
            headers=headers,
        )

        if resp.status_code != 200:
            return False, 0.0, []

        papers = resp.json().get("data", [])
        venue_lower = venue_name.lower()
        matches = []

        for paper in papers:
            paper_venue = paper.get("venue", "") or ""
            if not paper_venue:
                continue
            paper_venue_lower = paper_venue.lower()
            if venue_lower in paper_venue_lower or paper_venue_lower in venue_lower:
                matches.append({
                    "title": paper.get("title", ""),
                    "venue": paper_venue,
                    "year": paper.get("year", ""),
                })

        if matches:
            return True, 0.90, [f"Found {len(matches)} paper(s) in this venue"]

        venue_words = set(venue_lower.split())
        for paper in papers:
            paper_venue = paper.get("venue", "") or ""
            if not paper_venue:
                continue
            paper_words = set(paper_venue.lower().split())
            if len(venue_words & paper_words) >= 2:
                return True, 0.70, [f"Similar venue match: {paper_venue}"]

        return False, 0.20, ["Author not found in this venue"]

    except Exception as e:
        print(f"[Author-Venue] Check error: {e}")
        return False, 0.0, []


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_reference_comprehensive(
    author_names: str,
    venue_name: str,
    paper_title: str = "",
    year: str = "",
    key: str = "",
) -> AuthorVerificationResult:
    """Comprehensive verification of author + journal/venue."""

    warnings: List[str] = []
    evidence: List[str] = []

    # ── STEP 1: Landmark detection (strict, author-gated) ───────────────────
    landmark_info = _is_landmark_paper(paper_title, key)
    if landmark_info:
        landmark_authors = landmark_info.get("authors", []) or []
        landmark_year = landmark_info.get("year", "") or ""

        # Require that the citation's own authors overlap with the landmark's
        # canonical author list. Without this, a fabricated entry that happens
        # to share a landmark's title (or key) would be promoted to REAL.
        if author_names:
            overlap = _author_overlap_vs_landmark(author_names, landmark_authors)
            if overlap < 0.60:
                warnings.append(
                    f"Landmark title/key matched "
                    f"({landmark_info.get('landmark_title', '')}) "
                    f"but cited authors overlap only {int(overlap*100)}% "
                    f"with the landmark's canonical authors."
                )
                return AuthorVerificationResult(
                    author_exists=False,
                    journal_exists=True,
                    author_venue_match=False,
                    overall_confidence=0.20,
                    warnings=warnings,
                    evidence=[
                        f"Landmark title similarity "
                        f"{int(landmark_info.get('similarity', 0) * 100)}%",
                        f"Cited authors: {author_names[:80]}",
                        f"Landmark authors: {', '.join(landmark_authors)}",
                    ],
                    found_authors=landmark_authors,
                    found_journals=[landmark_info.get("venue", "")],
                    is_landmark=False,
                    match_details={
                        "landmark": True,
                        "author_overlap": overlap,
                    },
                )

        evidence.append(
            f"Landmark paper detected: "
            f"{landmark_info.get('landmark_title', '')}"
        )
        evidence.append(f"Authors: {', '.join(landmark_authors)}")
        evidence.append(
            f"Venue: {landmark_info.get('venue', '')} ({landmark_year})"
        )

        return AuthorVerificationResult(
            author_exists=True,
            journal_exists=True,
            author_venue_match=True,
            overall_confidence=0.99,
            warnings=[],
            evidence=evidence,
            found_authors=landmark_authors,
            found_journals=[landmark_info.get("venue", "")],
            is_landmark=True,
            landmark_name=landmark_info.get("landmark_title", ""),
            match_details={"landmark": True},
        )

    # ── STEP 2: Parse authors ───────────────────────────────────────────────
    authors: List[str] = []
    if author_names:
        for a in re.split(r';\s*|\s+and\s+', author_names):
            a = a.strip()
            if a and not re.match(r'^et\s+al\.?$', a, re.IGNORECASE):
                authors.append(a)

    if not authors:
        return AuthorVerificationResult(
            author_exists=False,
            journal_exists=False,
            author_venue_match=False,
            overall_confidence=0.0,
            warnings=["No authors provided"],
            evidence=[],
        )

    first_author = authors[0]

    # ── STEP 3: Check author exists ─────────────────────────────────────────
    author_exists = False
    found_authors: List[str] = []
    author_conf = 0.0

    if first_author:
        exists, found, conf = _search_semantic_scholar_author(first_author)
        author_exists = exists
        found_authors = found
        author_conf = conf

        if author_exists and conf >= 0.7:
            evidence.append(
                f"Author '{first_author}' found in Semantic Scholar "
                f"(conf: {conf:.2f})"
            )
        elif author_exists and conf >= 0.4:
            evidence.append(
                f"Author '{first_author}' partially matched (conf: {conf:.2f})"
            )
            warnings.append(
                f"Author '{first_author}' partially matched (low confidence)"
            )
        else:
            warnings.append(
                f"Author '{first_author}' not found in Semantic Scholar"
            )

    # ── STEP 4: Check venue exists ──────────────────────────────────────────
    venue_exists = False
    found_venues: List[str] = []
    venue_conf = 0.0

    if venue_name:
        exists, found, conf = _search_crossref_venue(venue_name)
        venue_exists = exists
        found_venues = found
        venue_conf = conf

        if venue_exists and conf >= 0.7:
            evidence.append(
                f"Venue '{venue_name}' found in CrossRef (conf: {conf:.2f})"
            )
        elif venue_exists and conf >= 0.4:
            evidence.append(
                f"Venue '{venue_name}' partially matched (conf: {conf:.2f})"
            )
            warnings.append(
                f"Venue '{venue_name}' partially matched (low confidence)"
            )
        else:
            warnings.append(f"Venue '{venue_name}' not found in CrossRef")

    # ── STEP 5: Check author-venue match ────────────────────────────────────
    author_venue_match = False
    venue_match_conf = 0.0

    if author_exists and venue_exists and first_author and venue_name:
        match, conf, _ev = _check_author_venue_match(first_author, venue_name)
        author_venue_match = match
        venue_match_conf = conf

        if match and conf >= 0.7:
            evidence.append(
                f"Author has published in {venue_name} (conf: {conf:.2f})"
            )
        elif match and conf >= 0.4:
            evidence.append(
                f"Author possibly published in related venue (conf: {conf:.2f})"
            )
            warnings.append("Author may not be in this exact venue")
        else:
            warnings.append(f"Author not found in venue {venue_name}")

    # ── STEP 6: Compute overall confidence ──────────────────────────────────
    weights = {"author": 0.35, "venue": 0.30, "match": 0.35}
    overall_conf = 0.0

    overall_conf += weights["author"] * (author_conf if author_exists else 0.1)
    overall_conf += weights["venue"] * (venue_conf if venue_exists else 0.1)
    overall_conf += weights["match"] * (venue_match_conf if author_venue_match else 0.1)

    # ── STEP 7: Apply penalties ─────────────────────────────────────────────
    if not author_exists:
        overall_conf -= 0.2
        warnings.append("Author does not exist in academic databases")
    if not venue_exists:
        overall_conf -= 0.15
        warnings.append("Venue does not exist in academic databases")
    if author_exists and venue_exists and not author_venue_match:
        overall_conf -= 0.25
        warnings.append("Author has not published in this venue")

    # ── STEP 8: Boost when both exist but match is inconclusive ─────────────
    if author_exists and venue_exists and overall_conf < 0.6:
        overall_conf = 0.65
        evidence.append("Author and venue exist (boosted confidence)")

    overall_conf = max(0.0, min(1.0, overall_conf))

    return AuthorVerificationResult(
        author_exists=author_exists,
        journal_exists=venue_exists,
        author_venue_match=author_venue_match,
        overall_confidence=overall_conf,
        warnings=warnings,
        evidence=evidence,
        found_authors=found_authors,
        found_journals=found_venues,
        is_landmark=False,
        match_details={
            "author_confidence": author_conf,
            "venue_confidence": venue_conf,
            "match_confidence": venue_match_conf,
        },
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_VERIFICATION_CACHE: Dict[str, AuthorVerificationResult] = {}
_CACHE_LOCK = threading.Lock()


def get_cached_verification(
    author: str, venue: str, title: str = ""
) -> Optional[AuthorVerificationResult]:
    key = f"{author}|{venue}|{title[:30]}"
    with _CACHE_LOCK:
        return _VERIFICATION_CACHE.get(key)


def cache_verification(
    author: str, venue: str, title: str, result: AuthorVerificationResult
) -> None:
    key = f"{author}|{venue}|{title[:30]}"
    with _CACHE_LOCK:
        _VERIFICATION_CACHE[key] = result


def verify_reference_comprehensive_cached(
    author_names: str,
    venue_name: str,
    paper_title: str = "",
    year: str = "",
    key: str = "",
) -> AuthorVerificationResult:
    cached = get_cached_verification(author_names, venue_name, paper_title)
    if cached:
        return cached

    result = verify_reference_comprehensive(
        author_names, venue_name, paper_title, year, key
    )
    cache_verification(author_names, venue_name, paper_title, result)
    return result