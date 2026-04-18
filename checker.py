"""
STEP 3: Citation Cross-Checker + Fake Reference Detector
---------------------------------------------------------
1. Extracts all in-text citation keys like [Ez10], [ABC01] from body text
2. Cross-checks: are all bib entries cited? Are all citations in bib?
3. Verifies references exist using CrossRef and Semantic Scholar APIs
4. Attempts to fetch full text (open access) via Unpaywall
"""

import re
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
from parser import BibEntry


# ─── In-text citation extraction ────────────────────────────────────────────

def extract_citations_from_body(body_text: str) -> set[str]:
    """
    Find all citation keys used in the body text.
    LNI format: [Ez10], [AB00], [Wa14a], [ABC01], etc.
    Also handles multiple citations like [Ez10, AB00] or [Ez10][AB00]
    """
    # Match individual keys inside brackets
    raw_matches = re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?(?:,\s*[A-Za-z]{2,6}\d{2}[a-z]?)*)\]', body_text)

    keys = set()
    for match in raw_matches:
        # Handle comma-separated multi-citations [Ez10, AB00]
        for key in re.split(r',\s*', match):
            key = key.strip()
            if re.match(r'^[A-Za-z]{2,6}\d{2}[a-z]?$', key):
                keys.add(key)

    return keys


# ─── Cross-checking ──────────────────────────────────────────────────────────

@dataclass
class CrossCheckResult:
    cited_not_in_bib: list[str] = field(default_factory=list)   # in text but no bib entry
    in_bib_not_cited: list[str] = field(default_factory=list)   # in bib but never cited
    correctly_used: list[str] = field(default_factory=list)     # in both


def cross_check(bib_entries: dict, cited_keys: set[str]) -> CrossCheckResult:
    """Compare bibliography entries against in-text citations."""
    result = CrossCheckResult()

    bib_keys = set(bib_entries.keys())

    result.cited_not_in_bib = sorted(cited_keys - bib_keys)
    result.in_bib_not_cited = sorted(bib_keys - cited_keys)
    result.correctly_used = sorted(cited_keys & bib_keys)

    return result


# ─── Fake reference detection ────────────────────────────────────────────────

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str           # "verified", "not_found", "partial_match", "error"
    confidence: float     # 0.0 to 1.0
    matched_title: Optional[str] = None
    doi: Optional[str] = None
    open_access_url: Optional[str] = None
    note: Optional[str] = None


def verify_reference(entry: BibEntry) -> VerificationResult:
    """
    Try to verify a reference exists using CrossRef, then Semantic Scholar.
    Returns a VerificationResult with confidence score.
    """
    if not entry.title:
        return VerificationResult(
            key=entry.key,
            title="(no title parsed)",
            status="error",
            confidence=0.0,
            note="Could not extract title from entry — check parsing"
        )

    # Skip verification for websites — just check URL reachability
    if entry.entry_type == "website":
        return _verify_website(entry)

    # Try CrossRef first
    result = _query_crossref(entry)
    if result.status == "verified":
        return result

    # Fallback: Semantic Scholar
    result2 = _query_semantic_scholar(entry)
    if result2.status in ("verified", "partial_match"):
        return result2

    return result  # Return CrossRef result even if not found


def _query_crossref(entry: BibEntry) -> VerificationResult:
    """Query CrossRef API by title + author."""
    try:
        params = {
            "query.title": entry.title,
            "rows": 3,
            "select": "DOI,title,author,published-print,is-referenced-by-count"
        }
        if entry.authors:
            # Use first author surname only
            first_author = entry.authors.split(';')[0].split(',')[0].strip()
            params["query.author"] = first_author

        resp = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=10,
            headers={"User-Agent": "LNI-Checker/1.0 (student-project)"}
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("message", {}).get("items", [])
        if not items:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="not_found", confidence=0.0,
                note="No results from CrossRef"
            )

        best = items[0]
        found_titles = best.get("title", [])
        found_title = found_titles[0] if found_titles else ""
        doi = best.get("DOI", "")

        # Calculate simple title similarity
        similarity = _title_similarity(entry.title, found_title)

        if similarity >= 0.75:
            oa_url = _check_unpaywall(doi) if doi else None
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="verified", confidence=similarity,
                matched_title=found_title, doi=doi,
                open_access_url=oa_url,
                note="Found via CrossRef"
            )
        elif similarity >= 0.4:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="partial_match", confidence=similarity,
                matched_title=found_title, doi=doi,
                note="Partial title match — verify manually"
            )
        else:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="not_found", confidence=similarity,
                matched_title=found_title,
                note="Title did not match CrossRef results"
            )

    except requests.exceptions.Timeout:
        return VerificationResult(
            key=entry.key, title=entry.title,
            status="error", confidence=0.0, note="CrossRef API timeout"
        )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title,
            status="error", confidence=0.0, note=f"CrossRef error: {str(e)}"
        )


def _query_semantic_scholar(entry: BibEntry) -> VerificationResult:
    """Query Semantic Scholar API as fallback."""
    try:
        query = entry.title
        if entry.authors:
            query += " " + entry.authors.split(';')[0].split(',')[0].strip()

        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3, "fields": "title,authors,year,openAccessPdf"},
            timeout=10,
            headers={"User-Agent": "LNI-Checker/1.0"}
        )
        resp.raise_for_status()
        data = resp.json()

        papers = data.get("data", [])
        if not papers:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="not_found", confidence=0.0,
                note="Not found in Semantic Scholar either"
            )

        best = papers[0]
        found_title = best.get("title", "")
        similarity = _title_similarity(entry.title, found_title)

        oa = best.get("openAccessPdf")
        oa_url = oa.get("url") if oa else None

        status = "verified" if similarity >= 0.75 else (
            "partial_match" if similarity >= 0.4 else "not_found"
        )

        return VerificationResult(
            key=entry.key, title=entry.title,
            status=status, confidence=similarity,
            matched_title=found_title,
            open_access_url=oa_url,
            note="Found via Semantic Scholar"
        )

    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title,
            status="error", confidence=0.0,
            note=f"Semantic Scholar error: {str(e)}"
        )


def _verify_website(entry: BibEntry) -> VerificationResult:
    """Check if a website URL is reachable."""
    url = entry.url
    if not url:
        return VerificationResult(
            key=entry.key, title=entry.title or "(website)",
            status="error", confidence=0.0, note="No URL found"
        )
    try:
        if not url.startswith("http"):
            url = "https://" + url
        resp = requests.head(url, timeout=8, allow_redirects=True)
        if resp.status_code < 400:
            return VerificationResult(
                key=entry.key, title=entry.title or url,
                status="verified", confidence=1.0,
                open_access_url=url, note=f"URL reachable (HTTP {resp.status_code})"
            )
        else:
            return VerificationResult(
                key=entry.key, title=entry.title or url,
                status="not_found", confidence=0.0,
                note=f"URL returned HTTP {resp.status_code}"
            )
    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title or url,
            status="error", confidence=0.0, note=f"URL check failed: {str(e)}"
        )


def _check_unpaywall(doi: str) -> Optional[str]:
    """Check Unpaywall for open access full text URL."""
    import os
    contact_email = os.environ.get("UNPAYWALL_EMAIL", "lni-checker@uni-project.de")
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={contact_email}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("is_oa"):
                best_loc = data.get("best_oa_location")
                if best_loc:
                    return best_loc.get("url_for_pdf") or best_loc.get("url")
    except Exception:
        pass
    return None


def _title_similarity(title1: str, title2: str) -> float:
    """
    Compute title similarity using rapidfuzz token_set_ratio when available,
    falling back to difflib SequenceMatcher.  Both handle abbreviations,
    word reordering, and minor spelling differences far better than Jaccard.
    Returns a float in [0.0, 1.0].
    """
    if not title1 or not title2:
        return 0.0

    def _normalize(t: str) -> str:
        t = t.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        stopwords = {'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with',
                     'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an'}
        return ' '.join(w for w in t.split() if w not in stopwords)

    t1 = _normalize(title1)
    t2 = _normalize(title2)

    if not t1 or not t2:
        return 0.0

    try:
        from rapidfuzz import fuzz
        # token_set_ratio handles word reordering and partial matches well
        return fuzz.token_set_ratio(t1, t2) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


def verify_all_references(bib_entries: dict, delay: float = 0.5) -> list[VerificationResult]:
    """Verify all bibliography entries with rate-limiting delay."""
    results = []
    for key, entry in bib_entries.items():
        result = verify_reference(entry)
        results.append(result)
        time.sleep(delay)  # Be polite to APIs
    return results

def check_lni_macros(body_text: str) -> list[dict]:
    """
    Scans body text for manual abbreviations that should be LNI macros.
    Returns a list of suggestion dictionaries.
    """
    suggestions = []
    
    # Mapping of plain text to the recommended LNI macro
    macro_rules = {
        r'\be\.g\.': r'\eg',
        r'\bi\.e\.': r'\ie',
        r'\bcf\.': r'\cf',
        r'\bet al\.': r'\etal',
    }
    
    for pattern, macro in macro_rules.items():
        matches = re.findall(pattern, body_text, re.IGNORECASE)
        if matches:
            suggestions.append({
                "type": "Style",
                "message": f"Found '{matches[0]}'. In LNI, use the macro '{macro}' for correct spacing.",
                "count": len(matches)
            })
            
    return suggestions