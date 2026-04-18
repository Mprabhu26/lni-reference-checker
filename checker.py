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

    # Fallback 1: Semantic Scholar
    result2 = _query_semantic_scholar(entry)
    if result2.status in ("verified", "partial_match"):
        return result2

    # Fallback 2: Google Scholar (scrape, no API key needed)
    result3 = _query_google_scholar(entry)
    if result3.status in ("verified", "partial_match"):
        return result3

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


def _query_google_scholar(entry: BibEntry) -> VerificationResult:
    """
    Query Google Scholar as a last-resort fallback by scraping the search page.
    No API key required. Returns a VerificationResult based on title matching
    the first result snippet. Rate-limited and may be blocked under heavy use.
    """
    try:
        query = entry.title or ""
        if entry.authors:
            query += " " + entry.authors.split(';')[0].split(',')[0].strip()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; LNI-Checker/1.0; "
                "+https://github.com/lni-reference-checker)"
            )
        }
        resp = requests.get(
            "https://scholar.google.com/scholar",
            params={"q": query, "hl": "en", "num": 3},
            headers=headers,
            timeout=10,
        )

        if resp.status_code == 429:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="error", confidence=0.0,
                note="Google Scholar rate-limited (429) — try again later"
            )
        if resp.status_code != 200:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="error", confidence=0.0,
                note=f"Google Scholar returned HTTP {resp.status_code}"
            )

        # Extract result titles from <h3 class="gs_rt"> tags (no external parser needed)
        found_titles = re.findall(r'class="gs_rt"[^>]*>.*?<(?:a[^>]*>)?(.*?)</(?:a|h3)>', resp.text)
        # Strip any remaining HTML tags from extracted titles
        found_titles = [re.sub(r'<[^>]+>', '', t).strip() for t in found_titles if t.strip()]

        if not found_titles:
            return VerificationResult(
                key=entry.key, title=entry.title,
                status="not_found", confidence=0.0,
                note="No results found on Google Scholar"
            )

        best_title = found_titles[0]
        similarity = _title_similarity(entry.title, best_title)
        status = (
            "verified"      if similarity >= 0.75 else
            "partial_match" if similarity >= 0.40 else
            "not_found"
        )

        return VerificationResult(
            key=entry.key, title=entry.title,
            status=status, confidence=similarity,
            matched_title=best_title,
            note="Found via Google Scholar"
        )

    except Exception as e:
        return VerificationResult(
            key=entry.key, title=entry.title,
            status="error", confidence=0.0,
            note=f"Google Scholar error: {str(e)}"
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


# ─── Duplicate detection ─────────────────────────────────────────────────────

def find_duplicates(bib_entries: dict, similarity_threshold: float = 0.85) -> list[dict]:
    """
    Detect duplicate or near-duplicate bibliography entries.
    Compares every pair of entries using title similarity.
    Returns a list of duplicate groups, each with the two conflicting keys,
    their titles, and the similarity score.
    """
    entries = list(bib_entries.values())
    duplicates = []
    seen_pairs = set()

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a = entries[i]
            b = entries[j]

            # Skip if either has no title
            if not a.title or not b.title:
                continue

            pair = tuple(sorted([a.key, b.key]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            score = _title_similarity(a.title, b.title)
            if score >= similarity_threshold:
                duplicates.append({
                    "key_a":      a.key,
                    "key_b":      b.key,
                    "title_a":    a.title,
                    "title_b":    b.title,
                    "similarity": round(score, 2),
                    "type":       "exact" if score >= 0.97 else "near-duplicate",
                })

    return duplicates


# ─── Quality scoring / grading rubric ────────────────────────────────────────

def compute_score(bib_list: list, xcheck, verification_results: list,
                  style_suggestions: list, duplicates: list) -> dict:
    """
    Compute a 0–100 reference quality score with per-category breakdown.

    Penalty weights (total deductions cannot exceed 100):
      - Missing citations (cited but not in bib):  -10 pts each, max -30
      - Orphaned entries (in bib but not cited):   -5  pts each, max -20
      - Incomplete entries (missing fields):        -5  pts each, max -20
      - Fake / unverified references:              -8  pts each, max -24
      - Duplicate entries:                         -5  pts each, max -10
      - Style issues (wrong macros):               -2  pts each, max -6
    """
    score = 100

    penalties = []

    # Missing citations
    missing = len(xcheck.cited_not_in_bib)
    p = min(missing * 10, 30)
    if p:
        penalties.append({"category": "Missing from bibliography", "count": missing, "deduction": p})
    score -= p

    # Orphaned entries
    orphaned = len(xcheck.in_bib_not_cited)
    p = min(orphaned * 5, 20)
    if p:
        penalties.append({"category": "Cited nowhere in text", "count": orphaned, "deduction": p})
    score -= p

    # Incomplete entries
    incomplete = sum(1 for e in bib_list if e.completeness_issues)
    p = min(incomplete * 5, 20)
    if p:
        penalties.append({"category": "Incomplete entries", "count": incomplete, "deduction": p})
    score -= p

    # Fake / not found references
    fake = sum(1 for vr in verification_results if vr.status == "not_found")
    p = min(fake * 8, 24)
    if p:
        penalties.append({"category": "Unverified / fake references", "count": fake, "deduction": p})
    score -= p

    # Duplicates
    p = min(len(duplicates) * 5, 10)
    if p:
        penalties.append({"category": "Duplicate entries", "count": len(duplicates), "deduction": p})
    score -= p

    # Style issues
    p = min(len(style_suggestions) * 2, 6)
    if p:
        penalties.append({"category": "LNI style violations", "count": len(style_suggestions), "deduction": p})
    score -= p

    score = max(score, 0)

    grade = (
        "A" if score >= 90 else
        "B" if score >= 75 else
        "C" if score >= 60 else
        "D" if score >= 45 else
        "F"
    )

    return {
        "score":    score,
        "grade":    grade,
        "penalties": penalties,
        "max_score": 100,
    }



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