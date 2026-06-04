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
    Call LLM (reuses your Groq/Gemini setup from ai_checker.py)
    """
    import sys
    
    # Try to import your existing AI function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from ai_checker import _call_ai_json
        return _call_ai_json(prompt, max_tokens=800)
    except (ImportError, AttributeError) as import_err:
        # Fallback: try direct Groq call
        print(f"Could not import _call_ai_json: {import_err}, falling back to direct API calls")
        
        import requests
        
        groq_key = os.environ.get("AI_API_KEY", "")
        if groq_key:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 800,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    # Extract JSON from response
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    return json.loads(content)
            except Exception as groq_err:
                print(f"Groq fallback failed: {groq_err}")
        
        # Fallback to Gemini
        gemini_key = os.environ.get("AI_API_KEY_GEMINI", "")
        if gemini_key:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.1}
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    parts = resp.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                    return json.loads(text)
            except Exception as gemini_err:
                print(f"Gemini fallback failed: {gemini_err}")
        
        raise RuntimeError("No LLM available for web search verification. Set GROQ_API_KEY or GEMINI_API_KEY")


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