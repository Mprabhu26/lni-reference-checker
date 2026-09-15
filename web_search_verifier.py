"""
Web Search Verifier — v3.0
--------------------------
RefChecker-style hallucination detection.

v3.0 changes:
  - Removed ALL _save_to_cache calls. Web-search results are ranking signals,
    not verification results; only checker.verify_reference may cache, and only
    after author-overlap verification.
  - Landmark detection now delegates to author_journal_verifier._is_landmark_paper
    (the strict, title-similarity-required matcher). The local LANDMARK_PAPERS
    table that shipped here has been removed — it was a second, divergent list.
  - URL-only title match returns status="partial_match", never "verified".
  - url_blocked is a hard negative signal (SUSPICIOUS), never REAL.
  - og:title is no longer accepted as page-title evidence (user-controlled).
  - _search_web_with_timeout now actually uses a signal-based timeout on POSIX.
"""

import json
import re
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=False)
except ImportError:
    pass

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WebVerificationResult:
    """Result from web search + LLM verification"""
    found: bool
    verdict: str          # "REAL", "FAKE", "UNCERTAIN"
    confidence: float
    found_url: Optional[str] = None
    found_title: Optional[str] = None
    found_authors: Optional[str] = None
    found_year: Optional[str] = None
    explanation: str = ""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _title_similarity_simple(title1: str, title2: str) -> float:
    """Simple Jaccard title similarity for web-search fallback."""
    if not title1 or not title2:
        return 0.0

    def normalize(t: str) -> set:
        t = t.lower()
        # Fix common OCR artifacts that appear in PDF-extracted titles
        t = re.sub(r'image net', 'imagenet', t)
        t = re.sub(r'pre train ing', 'pretraining', t)
        t = re.sub(r'net work', 'network', t)
        t = re.sub(r'over fit ting', 'overfitting', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        stop = {
            'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with',
            'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu',
            'from', 'into', 'through', 'during',
        }
        return {w for w in t.split() if w not in stop and len(w) > 2}

    s1 = normalize(title1)
    s2 = normalize(title2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0


def _safe_re_sub(pattern, replacement, text: str, flags: int = 0) -> str:
    """Accept either a compiled pattern or a string pattern."""
    if hasattr(pattern, "sub"):
        try:
            return pattern.sub(replacement, text)
        except TypeError:
            return re.sub(pattern.pattern, replacement, text, flags=flags)
    return re.sub(pattern, replacement, text, flags=flags)


def _safe_re_search(pattern, text: str, flags: int = 0):
    """Accept either a compiled pattern or a string pattern."""
    if hasattr(pattern, "search"):
        try:
            return pattern.search(text)
        except TypeError:
            return re.search(pattern.pattern, text, flags=flags)
    return re.search(pattern, text, flags=flags)


def _make_requests_session(timeout: int = 5) -> requests.Session:
    """Create a requests session with one retry and a short backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=1,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _url_is_reachable(url: str, timeout: float = 4.0) -> bool:
    """Quick liveness check before presenting a URL to the user as evidence."""
    if not url or not url.startswith("http"):
        return False
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        session = _make_requests_session(timeout=timeout)
        resp = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code < 400:
            return True
        resp = session.get(
            url, headers=headers, timeout=timeout,
            allow_redirects=True, stream=True,
        )
        return resp.status_code < 400
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_web_for_paper(title: str, authors: str = "") -> List[Dict]:
    """Search the web for a paper using DuckDuckGo. Fails fast on timeout."""
    if DDGS is None:
        return []

    clean_title = title
    for pattern in [
        r'\.\s*In:\s*.*$', r'\.\s*doi:\s*.*$', r'https?://\S+',
        r'Stand:\s*[\d./-]+', r'accessed\s+[\d./-]+',
        r'\.\s*[A-Z][a-z]+\.\s*\d{4}',
    ]:
        clean_title = _safe_re_sub(pattern, '', clean_title, flags=re.IGNORECASE)

    clean_title = _safe_re_sub(r'pre train ing', 'pretraining',
                               clean_title, flags=re.IGNORECASE)
    clean_title = _safe_re_sub(r'net work', 'network',
                               clean_title, flags=re.IGNORECASE)
    clean_title = _safe_re_sub(r'over fit ting', 'overfitting',
                               clean_title, flags=re.IGNORECASE)
    clean_title = _safe_re_sub(r'image net', 'imagenet',
                               clean_title, flags=re.IGNORECASE)
    clean_title = re.sub(r'[ ]{2,}', ' ', clean_title).strip().strip('.,;:')
    if not clean_title:
        clean_title = title

    queries = []
    if authors:
        first_author = authors.split(';')[0].split(',')[0].strip()
        first_author = re.sub(r'\s+et\s+al\.?$', '', first_author,
                              flags=re.IGNORECASE)
        if first_author and len(first_author) > 2:
            queries.append(f'"{clean_title}" {first_author}')
            short_title = ' '.join(clean_title.split()[:5])
            queries.append(f'"{short_title}" {first_author}')

    if len(clean_title) > 20:
        queries.append(f'"{clean_title}"')

    short_title = ' '.join(clean_title.split()[:6])
    if short_title and short_title != clean_title:
        queries.append(f'"{short_title}"')

    queries = list(dict.fromkeys(queries))

    results: List[Dict] = []
    for query in queries[:2]:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    result_title = r.get("title", "")
                    if any(
                        skip in result_title.lower()
                        for skip in ['amazon', 'ebay', 'facebook', 'twitter', 'search results']
                    ):
                        continue
                    results.append({
                        "title": result_title,
                        "url": r.get("href", ""),
                        "body": r.get("body", "")[:500],
                    })
                if results:
                    break
        except Exception:
            continue

    seen_urls = set()
    unique_results = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    return unique_results


def _search_web_with_timeout(title: str, authors: str = "",
                             timeout: float = 8.0) -> List[Dict]:
    """
    Wrap search_web_for_paper with a real timeout on POSIX via SIGALRM.
    On Windows (or when called off the main thread), fall back to running the
    search directly — the DDGS client has its own internal timeouts.
    """
    # On Windows (or any platform without SIGALRM) run the DDGS search in a
    # worker thread and enforce a hard wall-clock timeout. Otherwise a single
    # slow DuckDuckGo query can hang the whole verifier indefinitely.
    if not hasattr(signal, "SIGALRM"):
        return _ddgs_with_thread_timeout(title, authors, timeout)

    # SIGALRM can only fire on the main thread of the main interpreter.
    if threading.current_thread() is not threading.main_thread():
        return _ddgs_with_thread_timeout(title, authors, timeout)

    def _handle(signum, frame):
        raise TimeoutError("Web search timed out")

    prev = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(int(timeout))
    try:
        return search_web_for_paper(title, authors)
    except Exception:
        return []
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _ddgs_with_thread_timeout(title: str, authors: str = "",
                              timeout: float = 8.0) -> List[Dict]:
    """Run search_web_for_paper in a worker thread with a hard timeout.

    Used on platforms without POSIX SIGALRM (e.g. Windows) and when called
    from a non-main thread. Returns whatever finished before `timeout`
    seconds, or [] if the search did not complete in time. NEVER blocks on
    teardown: the executor is shut down with wait=False so a hung DDGS
    socket is abandoned rather than waited for.
    """
    _ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddgs-timeout")
    try:
        fut = _ex.submit(search_web_for_paper, title, authors)
        try:
            return fut.result(timeout=timeout) or []
        except TimeoutError:
            fut.cancel()
            return []
        except Exception:
            return []
    finally:
        # wait=False => do NOT wait for a possibly-stuck worker.
        try:
            _ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def llm_verify_with_web_search(
    title: str,
    authors: str,
    year: str,
    web_results: List[Dict],
) -> WebVerificationResult:
    """
    Heuristic verification of a paper against web-search results.
    (No LLM call here; the LLM pass happens upstream in ai_checker.)
    """
    if not web_results:
        return WebVerificationResult(
            found=False, verdict="UNCERTAIN", confidence=0.3,
            explanation="No web search results found.",
        )

    for result in web_results:
        result_title = result.get("title", "")
        sim = _title_similarity_simple(title, result_title)
        if sim >= 0.70:
            return WebVerificationResult(
                found=True,
                verdict="REAL",
                confidence=min(0.85, 0.65 + sim * 0.20),
                found_title=result_title,
                found_url=result.get("url", ""),
                explanation=(
                    f"Web search found matching paper "
                    f"(title similarity: {int(sim*100)}%)"
                ),
            )

    year_found = any(year in r.get("body", "") for r in web_results if year)
    authors_found = any(
        authors.split(';')[0][:10] in r.get("body", "")
        for r in web_results if authors
    )
    if year_found or authors_found:
        return WebVerificationResult(
            found=True,
            verdict="REAL",
            confidence=0.70,
            found_title=web_results[0].get("title", ""),
            found_url=web_results[0].get("url", ""),
            explanation="Web search found paper with matching year/authors metadata.",
        )

    return WebVerificationResult(
        found=False,
        verdict="UNCERTAIN",
        confidence=0.45,
        explanation="Web search found results but could not confirm this specific paper.",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_with_web_search(entry: dict, api_status: str) -> Dict:
    """
    Web-search fallback for a single reference.

    Contract (v3.0):
      - Never writes to the local cache.
      - Never returns status="verified" purely on URL-title match.
      - Landmark detection delegates to the strict matcher in
        author_journal_verifier (title similarity required).
      - url_blocked is a hard SUSPICIOUS signal.

    Returns a dict with at least:
      status           : "verified" | "partial_match" | "suspicious" | "url_blocked"
      confidence       : float 0..1
      note             : str
      sources_checked  : list[str]
    """
    title = (entry.get("title") or "").strip()
    authors = (entry.get("authors") or "").strip()
    year = (entry.get("year") or "").strip()
    original_url = (entry.get("url") or "").strip()
    api_matched_title = entry.get("api_matched_title", "")
    api_status_check = entry.get("api_status", "not_found")
    open_access_url = entry.get("open_access_url")

    # ── Guard: no title at all ──────────────────────────────────────────────
    if not title:
        return {
            "status": "suspicious",
            "web_verified": False,
            "confidence": 0.0,
            "note": "No title to verify.",
            "sources_checked": ["validation"],
        }

    # ── Landmark detection (strict, delegated) ─────────────────────────────
    # The strict matcher requires title similarity >= 0.70 (or >= 0.55 with an
    # exact key match). It does NOT save to the cache.
    try:
        from author_journal_verifier import _is_landmark_paper as _shared_landmark
        landmark_info = _shared_landmark(title, key="")
    except Exception:
        landmark_info = None

    if landmark_info:
        landmark_title = landmark_info.get("title") or landmark_info.get("landmark_title") or title
        return {
            "status": "verified",
            "web_verified": True,
            "confidence": 0.95,
            "matched_title": landmark_title,
            "open_access_url": open_access_url or landmark_info.get("url", ""),
            "note": (
                f"Landmark paper matched "
                f"(sim {int(landmark_info.get('similarity', 0) * 100)}%): "
                f"{landmark_title[:60]}"
            ),
            "sources_checked": ["landmark_detection"],
        }

    # ── GI URL normalization (no cache write) ──────────────────────────────
    if original_url and ("gi.de" in original_url.lower() or "gi-ev.at" in original_url.lower()):
        correct_url = re.sub(r'gi-?ev\.at', 'gi.de', original_url.lower())
        if not correct_url.startswith("http"):
            correct_url = "https://" + correct_url
        return {
            "status": "partial_match",
            "web_verified": False,
            "confidence": 0.60,
            "matched_title": title,
            "open_access_url": correct_url,
            "note": (
                f"GI URL normalized from {original_url} to {correct_url}; "
                f"author identity not verified by URL check."
            ),
            "sources_checked": ["url_correction"],
        }

    # ── URL verification ───────────────────────────────────────────────────
    url_already_dead = bool(entry.get("url_blocked", False))

    if original_url and original_url.startswith("http") and not url_already_dead:
        try:
            session = _make_requests_session(timeout=3)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
            resp = session.get(original_url, headers=headers,
                               timeout=3, allow_redirects=True)

            if resp.status_code == 200:
                page_title = ""
                # Only trust the real <title> element. og:title is author-controlled.
                match = re.search(r'<title[^>]*>([^<]+)</title>',
                                  resp.text, re.IGNORECASE)
                if match:
                    page_title = match.group(1).strip()

                if page_title and title:
                    sim = _title_similarity_simple(title, page_title)
                    if sim >= 0.65:
                        # URL-only title match is NOT author verification.
                        return {
                            "status": "partial_match",
                            "web_verified": False,
                            "confidence": round(sim, 4),
                            "matched_title": page_title,
                            "open_access_url": original_url,
                            "note": (
                                f"URL reachable and title matches ({int(sim*100)}%) "
                                f"but author identity not verified by URL check."
                            ),
                            "sources_checked": ["url_verify"],
                        }
                    return {
                        "status": "suspicious",
                        "web_verified": False,
                        "confidence": 0.35,
                        "note": (
                            f"URL reachable but page title does not match cited title "
                            f"(sim: {int(sim*100)}%)."
                        ),
                        "sources_checked": ["url_verify"],
                    }

                return {
                    "status": "suspicious",
                    "web_verified": False,
                    "confidence": 0.35,
                    "note": "URL reachable (HTTP 200) but no page title found.",
                    "sources_checked": ["url_verify"],
                }

            # Non-200: hard negative signal
            return {
                "status": "url_blocked",
                "web_verified": False,
                "confidence": 0.0,
                "note": f"URL returned HTTP {resp.status_code}.",
                "sources_checked": ["url_verify"],
            }
        except Exception:
            # Network failure — fall through to web search
            pass

    if url_already_dead:
        url_note = entry.get("url_note", "URL check failed")
        dead_url_note = f"URL check failed: {url_note}. "
    else:
        dead_url_note = ""

    # ── Web search fallback ────────────────────────────────────────────────
    web_results = _search_web_with_timeout(title, authors)

    if web_results:
        result = llm_verify_with_web_search(title, authors, year, web_results)

        min_confidence = 0.70 if url_already_dead else 0.55

        if result.verdict == "REAL" and result.confidence >= min_confidence:
            found_title = result.found_title or title
            title_sim = _title_similarity_simple(title, found_title)

            if title_sim < 0.55:
                return {
                    "status": "suspicious",
                    "web_verified": False,
                    "confidence": 0.35,
                    "matched_title": found_title,
                    "note": (
                        dead_url_note +
                        f"Web search found \"{found_title}\" but titles are only "
                        f"{int(title_sim*100)}% similar — likely a different document."
                    ),
                    "sources_checked": ["web_search", "llm_analysis"],
                }

            # Ground the returned URL in actual search results
            result_urls = {r.get("url", "") for r in web_results if r.get("url")}
            candidate_url = result.found_url if result.found_url in result_urls else None
            hallucinated_url = bool(result.found_url) and candidate_url is None

            verified_url = None
            if candidate_url:
                verified_url = candidate_url if _url_is_reachable(candidate_url) else None

            note_text = dead_url_note + result.explanation
            if hallucinated_url:
                note_text += (
                    " (Note: the suggested source link was not among the actual "
                    "search results and has been discarded.)"
                )
            elif candidate_url and not verified_url:
                note_text += (
                    " (Note: the source link found in search results is currently "
                    "unreachable and has been omitted.)"
                )

            if title_sim >= 0.85:
                final_confidence = min(0.90, result.confidence + 0.05)
            elif title_sim >= 0.70:
                final_confidence = min(0.85, result.confidence)
            else:
                final_confidence = min(0.75, result.confidence)

            # NOTE: no cache write. The caller (checker.verify_reference) is
            # responsible for caching, and only after the full pipeline —
            # including author overlap — has passed.
            return {
                "status": "partial_match",
                "web_verified": False,
                "confidence": final_confidence,
                "matched_title": found_title,
                "open_access_url": verified_url or (None if url_already_dead else original_url),
                "note": (
                    note_text
                    + " Web-search hit is treated as partial evidence only; "
                    "author verification is required upstream."
                ),
                "sources_checked": ["web_search", "llm_analysis"],
            }

    # ── Fallback: if the API already produced a title match, surface it as
    # a partial match. Never as "verified" — author checks belong upstream.
    if api_status_check == "verified" and api_matched_title:
        sim = _title_similarity_simple(title, api_matched_title)
        if sim >= 0.50:
            return {
                "status": "partial_match",
                "web_verified": False,
                "confidence": 0.70 + sim * 0.20,
                "matched_title": api_matched_title,
                "open_access_url": open_access_url or original_url or None,
                "note": f"API title match (similarity {int(sim*100)}%); author verification pending.",
                "sources_checked": ["api_fallback"],
            }

    # ── Nothing found ──────────────────────────────────────────────────────
    return {
        "status": "suspicious",
        "web_verified": False,
        "confidence": 0.40,
        "note": (
            dead_url_note + "Manual review required."
            if dead_url_note
            else "Manual review required — no confirming evidence found via web search."
        ),
        "sources_checked": ["none"],
    }