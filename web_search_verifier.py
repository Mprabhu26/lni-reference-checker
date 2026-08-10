"""
Web Search Verifier - RefChecker-style hallucination detection
Performs LLM-powered web search for references not found in APIs
"""
import json
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=False)
except ImportError:
    pass

from ddgs import DDGS


@dataclass
class WebVerificationResult:
    """Result from web search + LLM verification"""
    found: bool
    verdict: str  # "REAL", "FAKE", "UNCERTAIN"
    confidence: float
    found_url: Optional[str] = None
    found_title: Optional[str] = None
    found_authors: Optional[str] = None
    found_year: Optional[str] = None
    explanation: str = ""


def _is_german_academic_venue_for_web(title: str, authors: str, venue: str = "") -> bool:
    """Detect German academic venues for web search fallback."""
    text_to_check = f"{title} {authors} {venue}".lower()
    german_hints = [
        'informatik', 'gi', 'lni', 'gesellschaft für', 'gesellschaft fur',
        'informatik spektrum', 'datenbank', 'wirtschaftsinformatik', 'btw',
        'mensch und computer', 'delfi', 'fachtagung', 'dagstuhl',
        'universität', 'hochschule', 'lecture notes in informatics'
    ]
    return any(hint in text_to_check for hint in german_hints)


def search_web_for_paper(title: str, authors: str = "") -> List[Dict]:
    """Search the web for a paper using DuckDuckGo (free, no API key)."""
    query = f'"{title}"'
    if authors:
        first_author = authors.split(';')[0].split(',')[0].strip()
        query += f" {first_author}"
    
    results = []
    last_exc = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", "")[:500]
                    })
            break
        except Exception as e:
            last_exc = e
            import time as _time
            _time.sleep(1.5 * (attempt + 1))
    if not results and last_exc:
        print(f"Web search error after 3 attempts: {last_exc}")
    return results


def _call_llm_for_verification(prompt: str) -> dict:
    """Call the AI for web-search verification."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ai_checker import _call_ai_json
    return _call_ai_json(prompt, max_tokens=800)


def _title_similarity_simple(title1: str, title2: str) -> float:
    """Simple title similarity for fallback."""
    if not title1 or not title2:
        return 0.0
    
    def normalize(t: str) -> set:
        t = t.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        stop = {'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with',
                'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu'}
        return {w for w in t.split() if w not in stop and len(w) > 2}
    
    s1 = normalize(title1)
    s2 = normalize(title2)
    
    if not s1 or not s2:
        return 0.0
    
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def llm_verify_with_web_search(
    cited_title: str,
    cited_authors: str,
    cited_year: str,
    web_results: List[Dict]
) -> WebVerificationResult:
    """Use LLM to analyze web search results and determine if the paper is real."""
    
    if not web_results:
        return WebVerificationResult(
            found=False,
            verdict="UNCERTAIN",
            confidence=0.3,
            explanation="No web search results found"
        )
    
    web_summary = []
    for i, r in enumerate(web_results[:3]):
        web_summary.append(f"Result {i+1}:\n  URL: {r['url']}\n  Title: {r['title']}\n  Snippet: {r['body'][:200]}")
    
    prompt = f"""You are an expert Academic Reference Verification Agent. Your job is to determine if a cited reference is REAL or FAKE.

### CRITERIA FOR FAKE:

A reference is FAKE if ANY of these apply:

1. **The publisher does not exist** - Use your knowledge of real academic publishers (Springer, Elsevier, Wiley, IEEE, ACM, MIT Press, Cambridge, Oxford, Nature, Science, etc.). If the publisher name sounds made up, generic, or doesn't match any real publisher, it's FAKE.

2. **The author names are suspicious** - Look for:
   - Names that are common words in German/English (e.g., "Azubi" means trainee, "Gans" means goose)
   - Names from pop culture (e.g., "Corleone" from The Godfather)
   - Overly generic names that sound like placeholders

3. **The title sounds fictional or playful** - Titles containing "Magic", "Hypothetical", "Fake", "Imaginary", "Made Up".

4. **No real-world evidence exists** - If web search returns no credible matches and the metadata looks suspicious.

### CRITERIA FOR REAL:

A reference is REAL if:
- It matches a known landmark paper (e.g., Vaswani et al. 2017 "Attention Is All You Need")
- The publisher is a recognized academic publisher
- A valid DOI or ISBN is present
- Web search confirms the paper exists on real academic sites (arXiv, IEEE, ACM, Springer, etc.)

### REFERENCE TO ANALYZE:

Title: {cited_title}
Authors: {cited_authors}
Year: {cited_year}

### WEB SEARCH RESULTS:

{chr(10).join(web_summary) if web_summary else "No web search results found."}

Return ONLY valid JSON:
{{"verdict": "REAL or FAKE or UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "specific explanation", "found_url": null or "url", "found_title": null or "title"}}"""

    try:
        result = _call_llm_for_verification(prompt)
        verdict = result.get("verdict", "UNCERTAIN")
        confidence = result.get("confidence", 0.5)
        
        return WebVerificationResult(
            found=verdict == "REAL",
            verdict=verdict,
            confidence=confidence,
            found_url=result.get("found_url"),
            found_title=result.get("found_title"),
            explanation=result.get("reasoning", "")
        )
    except Exception as e:
        return WebVerificationResult(
            found=False,
            verdict="UNCERTAIN",
            confidence=0.3,
            explanation=f"LLM analysis failed: {str(e)[:100]}"
        )


def verify_grey_literature_by_url_direct(url: str, title: str = "") -> dict:
    """
    Direct URL verification - ONLY trusts the URL if the page title matches
    the cited title with at least 25% similarity.
    """
    if not url:
        return {
            "status": "not_found",
            "confidence": 0.0,
            "note": "No URL provided",
            "sources_checked": []
        }
    
    # CRITICAL FIX v8.5: Blacklist fake/example/test URLs that will mislead LLM
    # These are placeholders or non-existent sites commonly used in test docs.
    # When a URL is unreachable AND on this list, mark as FAKE immediately
    # rather than falling back to LLM verification (which may hallucinate matches).
    _FAKE_URL_PATTERNS = {
        'example.com', 'test.com', 'sample.com', 'dummy.com', 'placeholder.com',
        'docs.example.com', 'api.example.com', 'www.example.com',
        'example.org', 'example.net',
        'techblog', 'medium.com/@',  # fake Medium handles
    }
    url_lower = url.lower()
    for pattern in _FAKE_URL_PATTERNS:
        if pattern in url_lower:
            return {
                "status": "fake",
                "confidence": 0.95,
                "note": f"URL uses placeholder domain: {pattern}",
                "sources_checked": ["url_pattern_check"],
            }
    

    url = re.sub(r'\s+', '', url)

    browser_profiles = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-GB;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
    ]

    def _extract_title(html: str) -> str:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _sim(a: str, b: str) -> float:
        a_set = set(re.sub(r'[^\w\s]', '', a.lower()).split())
        b_set = set(re.sub(r'[^\w\s]', '', b.lower()).split())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    for profile in browser_profiles:
        session = requests.Session()
        session.headers.update(profile)
        try:
            resp = session.get(url, timeout=15, allow_redirects=True)

            if resp.status_code == 200:
                page_title = _extract_title(resp.text)
                
                # CRITICAL FIX: Only trust URL if title matches
                if title and page_title:
                    similarity = _sim(title, page_title)
                    if similarity >= 0.25:
                        return {
                            "status": "verified",
                            "confidence": 0.85,
                            "matched_title": page_title,
                            "open_access_url": url,
                            "note": f"URL verified: title match ({int(similarity*100)}%)",
                            "sources_checked": ["url_verify"]
                        }
                    else:
                        # URL exists but title doesn't match - don't trust it
                        return {
                            "status": "partial_match",
                            "confidence": 0.35,
                            "matched_title": page_title,
                            "open_access_url": url,
                            "note": f"URL reachable but title mismatch (similarity {int(similarity*100)}%)",
                            "sources_checked": ["url_verify"]
                        }
                else:
                    # No title to compare - don't trust the URL
                    return {
                        "status": "partial_match",
                        "confidence": 0.30,
                        "open_access_url": url,
                        "note": "URL reachable but no title to verify against",
                        "sources_checked": ["url_verify"]
                    }
        except Exception:
            continue

    return {
        "status": "not_found",
        "confidence": 0.0,
        "note": "URL not reachable",
        "sources_checked": []
    }


def llm_verify_grey_literature_by_knowledge(
    raw_text: str,
    title: str,
    authors: str,
    year: str,
    url: str,
    url_note: str,
) -> dict:
    """AI knowledge verification for grey literature."""
    
    prompt = f"""You are verifying an academic or grey literature reference.

The URL check returned: {url_note}

Based on your knowledge, does this document exist?

CITED REFERENCE:
- Title: "{title}"
- Authors: "{authors or '(organisation as author)'}"
- Year: "{year}"
- URL from citation: {url or '(none)'}

FULL REFERENCE TEXT:
{raw_text[:500]}

Return ONLY valid JSON:
{{"verdict": "REAL or FAKE or UNCERTAIN", "confidence": 0.0-1.0, "explanation": "one-sentence reasoning"}}
"""
    try:
        result = _call_llm_for_verification(prompt)
        verdict = result.get("verdict", "UNCERTAIN").upper()
        confidence = float(result.get("confidence", 0.5))
        
        # STRICT: Only mark as REAL if AI is confident enough
        if verdict == "REAL" and confidence >= 0.60:
            return {
                "status": "verified",
                "web_verified": True,
                "confidence": confidence,
                "matched_title": title,
                "open_access_url": url,
                "note": f"AI confirmed real: {result.get('explanation', '')}",
                "sources_checked": ["ai_knowledge"],
            }
        
        return {
            "status": "suspicious",
            "web_verified": False,
            "confidence": 0.3,
            "note": "AI could not verify this reference with sufficient confidence",
            "sources_checked": ["ai_knowledge"],
        }
        
    except Exception as e:
        return {
            "status": "suspicious",
            "web_verified": False,
            "confidence": 0.3,
            "note": f"AI verification failed: {str(e)[:100]}",
            "sources_checked": [],
        }


def verify_with_web_search(entry: dict, api_status: str) -> dict:
    """Main entry point for Step 4 verification."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from ai_checker import _is_grey_literature

    title = entry.get("title", "")
    authors = entry.get("authors", "")
    year = entry.get("year", "")
    original_url = entry.get("url", "")
    raw_text = entry.get("raw_text", "")

    if not title:
        return {"status": api_status, "web_verified": False, "note": "No title to search for"}

    is_grey, grey_reason = _is_grey_literature(entry)

    # ── TRY URL VERIFICATION FIRST ──
    # CRITICAL: This will ONLY return "verified" if the page title matches
    url_result = None
    if original_url and original_url.startswith("http"):
        url_result = verify_grey_literature_by_url_direct(original_url, title)
        if url_result and url_result.get("status") == "verified":
            return {
                "status": "verified",
                "web_verified": True,
                "confidence": url_result.get("confidence", 0.7),
                "matched_title": url_result.get("matched_title", title),
                "open_access_url": original_url,
                "note": url_result.get("note", "URL verified with title match"),
                "sources_checked": ["url_verify"],
            }
        # NEW FIX v8.5: If URL check detected it's FAKE, return immediately
        if url_result and url_result.get("status") == "fake":
            return {
                "status": "fake",
                "web_verified": False,
                "confidence": 0.95,
                "note": url_result.get("note", "URL is a placeholder/fake domain"),
                "sources_checked": ["url_pattern_check"],
            }

    # ── GREY LITERATURE → AI knowledge check ──
    if is_grey:
        url_note = "No URL provided"
        if url_result:
            url_note = url_result.get("note", "URL fetch attempted")
        elif original_url:
            url_note = "URL provided but fetch failed"
        
        return llm_verify_grey_literature_by_knowledge(
            raw_text=raw_text,
            title=title,
            authors=authors,
            year=year,
            url=original_url,
            url_note=url_note,
        )

    # ── ACADEMIC PAPER → Web search + LLM ──
    web_results = []
    try:
        web_results = search_web_for_paper(title, authors)
    except Exception as e:
        print(f"Web search failed for '{title}': {e}")

    if web_results:
        result = llm_verify_with_web_search(title, authors, year, web_results)
        if result.verdict == "REAL" and result.confidence >= 0.60:
            # NEW: Check if the matched title is actually similar to what was cited
            match_sim = _title_similarity_simple(title, result.found_title or title)
            if match_sim < 0.90:
                # Title doesn't match — don't trust this result
                print(f"[verify_with_web_search] Web search title mismatch: "
                      f"cited='{title}' vs found='{result.found_title}' ({match_sim*100:.0f}%)")
                # Don't return verified; continue to fallback
            else:
                return {
                    "status": "verified",
                    "web_verified": True,
                    "confidence": result.confidence * match_sim,  # Reduce confidence by title match quality
                    "matched_title": result.found_title or title,
                    "open_access_url": result.found_url or original_url or None,
                    "note": f"Web search confirmed: {result.explanation}",
                    "sources_checked": ["web_search", "llm_analysis"],
                }

    # ── FALLBACK → AI knowledge check ──
    url_note = "No URL provided"
    if url_result:
        url_note = url_result.get("note", "URL fetch attempted")
    elif original_url:
        url_note = "URL provided but fetch failed"
    
    return llm_verify_grey_literature_by_knowledge(
        raw_text=raw_text,
        title=title,
        authors=authors,
        year=year,
        url=original_url,
        url_note=url_note + ("; web search inconclusive" if web_results else "; web search returned no results"),
    )