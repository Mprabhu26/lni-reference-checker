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

# Load .env silently — keys set here are picked up by the rest of the app too
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=False)
except ImportError:
    pass  # python-dotenv not installed; keys must be set in the environment directly

# Web search library (NEW: ddgs instead of duckduckgo-search)
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


def search_web_for_paper(title: str, authors: str = "") -> List[Dict]:
    """
    Search the web for a paper using DuckDuckGo (free, no API key).
    Returns list of result URLs with snippets.
    """
    query = f'"{title}"'
    if authors:
        first_author = authors.split(';')[0].split(',')[0].strip()
        query += f" {first_author}"
    
    results = []
    last_exc = None
    for attempt in range(3):  # up to 3 retries on transient failures
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", "")[:500]
                    })
            break  # success
        except Exception as e:
            last_exc = e
            import time as _time
            _time.sleep(1.5 * (attempt + 1))  # back-off: 1.5s, 3s, 4.5s
    if not results and last_exc:
        print(f"Web search error after 3 attempts: {last_exc}")
    return results


def _call_llm_for_verification(prompt: str) -> dict:
    """
    Call the AI for web-search verification.
    Uses the same Groq setup as the rest of the app (ai_checker._call_ai_json).
    """
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
        # Remove punctuation
        t = re.sub(r'[^\w\s]', ' ', t)
        # Remove common short words
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
    """
    Use LLM to analyze web search results and determine if the paper is real.
    This mimics RefChecker's Stage 2 + Stage 3.
    """
    
    if not web_results:
        return WebVerificationResult(
            found=False,
            verdict="UNCERTAIN",
            confidence=0.3,
            explanation="No web search results found"
        )
    
    # Prepare web results for LLM
    web_summary = []
    for i, r in enumerate(web_results[:3]):
        web_summary.append(f"Result {i+1}:\n  URL: {r['url']}\n  Title: {r['title']}\n  Snippet: {r['body'][:200]}")
    
    prompt = f"""You are verifying if an academic paper reference is real or fabricated.

CITED REFERENCE:
- Title: {cited_title}
- Authors: {cited_authors}
- Year: {cited_year}

WEB SEARCH RESULTS:
{chr(10).join(web_summary)}

TASK: Determine if this paper actually exists. Look for a DEDICATED page about this paper (not just a citation in another paper's reference list).

Return ONLY valid JSON, no markdown:
{{
  "verdict": "REAL or FAKE or UNCERTAIN",
  "confidence": 0.0-1.0,
  "found_url": "the best URL that confirms this paper exists (or null)",
  "found_title": "the actual title from that page (or null)",
  "found_authors": "the actual authors from that page (or null)",
  "found_year": "the actual year from that page (or null)",
  "explanation": "brief reasoning"
}}

RULES:
- REAL: Found a dedicated page with matching title and authors
- FAKE: Searched thoroughly and found no evidence this paper exists
- UNCERTAIN: Found something similar but not an exact match, or page not accessible
"""

    try:
        result = _call_llm_for_verification(prompt)
        
        verdict = result.get("verdict", "UNCERTAIN")
        confidence = result.get("confidence", 0.5)
        
        # If LLM says REAL, do a final title similarity check as safety
        if verdict == "REAL" and result.get("found_title"):
            sim = _title_similarity_simple(cited_title, result["found_title"])
            if sim < 0.5:
                verdict = "UNCERTAIN"
                confidence = sim
                result["explanation"] = f"Title mismatch: '{cited_title}' vs '{result['found_title']}' (similarity: {sim:.0%})"
        
        return WebVerificationResult(
            found=verdict == "REAL",
            verdict=verdict,
            confidence=confidence,
            found_url=result.get("found_url"),
            found_title=result.get("found_title"),
            found_authors=result.get("found_authors"),
            found_year=result.get("found_year"),
            explanation=result.get("explanation", "")
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
    Direct URL verification for grey literature.
    Uses a requests.Session with full browser headers to defeat simple bot detection.
    """
    if not url:
        return {
            "status": "not_found",
            "confidence": 0.0,
            "note": "No URL provided for grey literature reference",
            "sources_checked": []
        }

    # Fix common URL artefacts from PDF extraction
    url = re.sub(r'\s+', '', url)

    # Cloudflare / challenge page titles — if we see these the URL exists but is gated
    _CHALLENGE_TITLES = {"just a moment", "access denied", "attention required",
                         "403 forbidden", "404 not found", "please wait"}

    browser_profiles = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Referer": "https://www.google.com/",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-GB;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
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

    last_status = None
    for profile in browser_profiles:
        session = requests.Session()
        session.headers.update(profile)
        try:
            # Warm up the session with a HEAD to get cookies (helps with Cloudflare)
            try:
                session.head(url, timeout=8, allow_redirects=True)
            except Exception:
                pass

            resp = session.get(url, timeout=15, allow_redirects=True)
            last_status = resp.status_code

            if resp.status_code == 200:
                page_title = _extract_title(resp.text)

                # If it's a challenge page, treat as partial (URL exists but gated)
                if page_title.lower() in _CHALLENGE_TITLES or "cloudflare" in resp.text[:500].lower():
                    return {
                        "status": "partial_match",
                        "confidence": 0.60,
                        "note": f"URL reachable (HTTP 200) but gated by bot-protection. Manual verification recommended.",
                        "open_access_url": url,
                        "sources_checked": ["url_fetch"]
                    }

                if title and page_title:
                    similarity = _sim(title, page_title)
                    if similarity >= 0.3:
                        return {
                            "status": "verified",
                            "confidence": 0.85,
                            "matched_title": page_title,
                            "open_access_url": url,
                            "note": f"URL verified: page title '{page_title[:80]}' (similarity {int(similarity*100)}%)",
                            "sources_checked": ["url_fetch"]
                        }

                return {
                    "status": "partial_match",
                    "confidence": 0.60,
                    "matched_title": page_title or None,
                    "open_access_url": url,
                    "note": f"URL reachable (HTTP 200). Page title: '{page_title[:60]}'",
                    "sources_checked": ["url_fetch"]
                }

            elif resp.status_code in (403, 429):
                # Server is alive and blocking bots — URL almost certainly exists
                return {
                    "status": "partial_match",
                    "confidence": 0.60,
                    "note": f"URL reachable but server blocked automated access (HTTP {resp.status_code}). Manual verification recommended.",
                    "open_access_url": url,
                    "sources_checked": ["url_fetch"]
                }

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    # All attempts exhausted
    status_str = f"HTTP {last_status}" if last_status else "connection failed"
    return {
        "status": "not_found",
        "confidence": 0.0,
        "note": f"URL not reachable after multiple attempts ({status_str}). Manual verification required.",
        "sources_checked": []
    }


def verify_with_web_search(entry: dict, api_status: str) -> dict:
    """
    Main entry point. Only call this when API lookup returned nothing.
    Returns updated verification result.
    """
    title = entry.get("title", "")
    authors = entry.get("authors", "")
    year = entry.get("year", "")
    
    if not title:
        return {"status": api_status, "web_verified": False, "note": "No title to search for"}
    
    # Step 1: Search the web
    web_results = search_web_for_paper(title, authors)
    
    if not web_results:
        return {
            "status": api_status,
            "web_verified": False,
            "note": "Web search found no results",
            "web_attempted": True
        }
    
    # Step 2: LLM analysis of web results
    llm_result = llm_verify_with_web_search(title, authors, year, web_results)
    
    # Step 3: If LLM found a real paper, return verified
    if llm_result.verdict == "REAL" and llm_result.found_title:
        return {
            "status": "verified",  # Upgrade from not_found to verified!
            "web_verified": True,
            "confidence": llm_result.confidence,
            "matched_title": llm_result.found_title,
            "matched_authors": llm_result.found_authors,
            "matched_year": llm_result.found_year,
            "open_access_url": llm_result.found_url,
            "note": f"Found via web search: {llm_result.explanation}",
            "sources_checked": ["web_search", "llm_verification"]
        }
    
    return {
        "status": api_status,
        "web_verified": False,
        "confidence": llm_result.confidence,
        "note": f"Web search: {llm_result.explanation}",
        "web_attempted": True
    }