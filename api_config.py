"""
api_config.py — Central API Configuration for LNI Reference Checker
====================================================================
All external API credentials and endpoints are controlled here via
environment variables (set them in your .env file).

Users add their own keys: nothing is hardcoded. Each API section
describes what the key unlocks and where to get one.

Priority order used by checker.py:
  DOI (CrossRef) → arXiv-ID → OpenAlex → CrossRef search →
  Semantic Scholar → DBLP → arXiv search → PubMed → DataCite →
  OpenAIRE → BASE → Google Scholar → ResearchGate → AI fallback
"""

import os
import time
import requests
from typing import Optional, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# API Definitions
# Each entry:  key → { env_var, description, get_url, rate_limit_s, required }
# ---------------------------------------------------------------------------

API_REGISTRY = {
    # ── Free APIs (no key needed, just politeness header) ──────────────────
    "crossref": {
        "env_var": "CROSSREF_MAILTO",           # optional but strongly recommended
        "description": "CrossRef — DOI lookup and title search. Free; "
                       "set CROSSREF_MAILTO=you@email.com for the 'polite pool' "
                       "(higher rate limits). https://www.crossref.org/",
        "base_url": "https://api.crossref.org",
        "rate_limit_s": 0.2,
        "key_required": False,
        "test_url": "https://api.crossref.org/works?query=test&rows=1",
    },
    "openalex": {
        "env_var": "OPENALEX_EMAIL",            # optional but polite
        "description": "OpenAlex — open academic graph, title/author search. "
                       "Free; set OPENALEX_EMAIL=you@email.com for polite pool. "
                       "https://openalex.org/",
        "base_url": "https://api.openalex.org",
        "rate_limit_s": 0.1,
        "key_required": False,
        "test_url": "https://api.openalex.org/works?search=test&per-page=1",
    },
    "dblp": {
        "env_var": None,
        "description": "DBLP — CS bibliography index. No key needed. "
                       "https://dblp.org/",
        "base_url": "https://dblp.org",
        "rate_limit_s": 0.5,
        "key_required": False,
        "test_url": "https://dblp.org/search/publ/api?q=test&format=json&h=1",
    },
    "arxiv": {
        "env_var": None,
        "description": "arXiv — preprint server. No key needed. "
                       "https://arxiv.org/",
        "base_url": "https://export.arxiv.org",
        "rate_limit_s": 3.0,
        "key_required": False,
        "test_url": "http://export.arxiv.org/api/query?search_query=ti:test&max_results=1",
    },
    "pubmed": {
        "env_var": "NCBI_API_KEY",              # optional; raises rate limit 3→10 req/s
        "description": "PubMed/NCBI — biomedical literature. Optional API key "
                       "raises limit from 3 to 10 req/s. "
                       "https://www.ncbi.nlm.nih.gov/account/",
        "base_url": "https://eutils.ncbi.nlm.nih.gov",
        "rate_limit_s": 0.4,
        "key_required": False,
        "test_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=test&retmode=json&retmax=1",
    },
    "datacite": {
        "env_var": None,
        "description": "DataCite — research dataset DOIs. No key needed. "
                       "https://datacite.org/",
        "base_url": "https://api.datacite.org",
        "rate_limit_s": 0.5,
        "key_required": False,
        "test_url": "https://api.datacite.org/dois?query=test&page[size]=1",
    },
    "openaire": {
        "env_var": None,
        "description": "OpenAIRE — EU open access publications. No key needed. "
                       "https://api.openaire.eu/",
        "base_url": "https://api.openaire.eu",
        "rate_limit_s": 0.5,
        "key_required": False,
        "test_url": "https://api.openaire.eu/search/publications?title=test&size=1&format=json",
    },
    # ── APIs that benefit from a key ───────────────────────────────────────
    "semantic_scholar": {
        "env_var": "SEMANTIC_SCHOLAR_API_KEY",  # optional; raises limit 1→100 req/s
        "description": "Semantic Scholar — AI-powered academic search. "
                       "Free key at https://www.semanticscholar.org/product/api "
                       "raises rate limit from 1 to 100 req/s.",
        "base_url": "https://api.semanticscholar.org",
        "rate_limit_s": 0.25,
        "key_required": False,
        "test_url": "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1&fields=title",
    },
    "unpaywall": {
        "env_var": "UNPAYWALL_EMAIL",           # required — must supply email
        "description": "Unpaywall — open-access PDF finder. "
                       "Requires your email as API key (set UNPAYWALL_EMAIL). "
                       "https://unpaywall.org/products/api",
        "base_url": "https://api.unpaywall.org",
        "rate_limit_s": 0.2,
        "key_required": True,                   # email IS the key
        "test_url": None,                       # skipped (needs DOI in path)
    },
    # ── AI fallback (used only when all academic APIs fail) ───────────────
    "ai_llm": {
        "env_var": "AI_API_KEY",
        "description": "LLM fallback for ambiguous references. "
                       "Set AI_API_KEY + AI_BASE_URL + AI_MODEL to use any "
                       "OpenAI-compatible provider (OpenAI, Groq, Mistral, "
                       "Together, local Ollama, etc.).\n"
                       "  AI_BASE_URL=https://api.groq.com/openai/v1\n"
                       "  AI_MODEL=llama-3.3-70b-versatile\n"
                       "  AI_API_KEY=gsk_...",
        "base_url": None,                       # dynamic from env
        "rate_limit_s": 1.0,
        "key_required": True,
        "test_url": None,
    },
}


# ---------------------------------------------------------------------------
# Accessors — used by checker.py and app.py
# ---------------------------------------------------------------------------

def get_crossref_email() -> Optional[str]:
    return (os.environ.get("CROSSREF_MAILTO") or
            os.environ.get("OPENALEX_EMAIL") or "").strip() or None


def get_openalex_email() -> Optional[str]:
    return (os.environ.get("OPENALEX_EMAIL") or
            os.environ.get("CROSSREF_MAILTO") or "").strip() or None


def get_semantic_scholar_key() -> Optional[str]:
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None


def get_ncbi_key() -> Optional[str]:
    return os.environ.get("NCBI_API_KEY", "").strip() or None


def get_unpaywall_email() -> Optional[str]:
    return (os.environ.get("UNPAYWALL_EMAIL") or
            get_crossref_email() or "").strip() or None


def get_ai_config() -> Tuple[str, str, str]:
    """Returns (base_url, model, api_key) — all empty strings if not configured."""
    return (
        os.environ.get("AI_BASE_URL", "").rstrip("/"),
        os.environ.get("AI_MODEL", ""),
        os.environ.get("AI_API_KEY", ""),
    )


def user_agent(version: str = "11.1") -> str:
    """Build a polite User-Agent string, including email if set."""
    email = get_crossref_email()
    base = f"LNI-Checker/{version}"
    return f"{base} (mailto:{email})" if email else base


# ---------------------------------------------------------------------------
# Health check — tests reachability of each API
# ---------------------------------------------------------------------------

def check_api_reachability(timeout: float = 5.0) -> dict:
    """
    Probe each API endpoint and return a status dict.
    Called by /status/apis — does NOT block the main pipeline.

    Returns:
    {
        "crossref":          { "configured": True, "reachable": True,  "key_set": False, "note": "..." },
        "semantic_scholar":  { "configured": True, "reachable": False, "key_set": True,  "note": "403 Forbidden — check key" },
        ...
    }
    """
    results = {}

    ua = user_agent()
    headers = {"User-Agent": ua}

    for name, cfg in API_REGISTRY.items():
        test_url = cfg.get("test_url")
        env_var = cfg.get("env_var")
        key_set = bool(env_var and os.environ.get(env_var, "").strip())
        key_required = cfg.get("key_required", False)
        configured = (not key_required) or key_set

        entry: dict = {
            "configured": configured,
            "key_set": key_set,
            "env_var": env_var,
            "description": cfg["description"],
        }

        if not test_url:
            entry["reachable"] = None   # untestable without request-specific params
            entry["note"] = "Reachability not auto-tested (requires runtime params)"
            results[name] = entry
            continue

        if not configured:
            entry["reachable"] = False
            entry["note"] = f"Missing required env var: {env_var}"
            results[name] = entry
            continue

        # Add key to headers/params where needed
        req_headers = dict(headers)
        req_params: dict = {}
        if name == "semantic_scholar":
            key = get_semantic_scholar_key()
            if key:
                req_headers["x-api-key"] = key
        elif name == "pubmed":
            key = get_ncbi_key()
            if key:
                req_params["api_key"] = key
        elif name == "openalex":
            email = get_openalex_email()
            if email:
                req_params["mailto"] = email
        elif name == "crossref":
            # email is embedded in UA string already
            pass

        try:
            t0 = time.time()
            r = requests.get(test_url, headers=req_headers, params=req_params,
                             timeout=timeout)
            elapsed = round(time.time() - t0, 2)

            if r.status_code == 200:
                entry["reachable"] = True
                entry["note"] = f"OK ({elapsed}s)"
            elif r.status_code == 403:
                entry["reachable"] = False
                entry["note"] = (
                    f"403 Forbidden — likely blocked by network proxy. "
                    f"In your own deployment this will work with the right key/email."
                )
            elif r.status_code == 429:
                entry["reachable"] = True    # reachable, just rate-limited
                entry["note"] = f"429 Rate limited — API is reachable but slow down requests"
            else:
                entry["reachable"] = False
                entry["note"] = f"HTTP {r.status_code} ({elapsed}s)"
        except requests.exceptions.ConnectionError:
            entry["reachable"] = False
            entry["note"] = "Connection refused or DNS failure"
        except requests.exceptions.Timeout:
            entry["reachable"] = False
            entry["note"] = f"Timed out after {timeout}s"
        except Exception as exc:
            entry["reachable"] = False
            entry["note"] = f"Error: {str(exc)[:80]}"

        results[name] = entry

    # AI config is special — check if base_url+model+key are all set
    base_url, model, api_key = get_ai_config()
    ai_entry = results.get("ai_llm", {})
    ai_entry["configured"] = bool(base_url and model and api_key)
    ai_entry["note"] = (
        f"Provider: {base_url.split('/')[2] if '//' in base_url else base_url or 'not set'}, "
        f"Model: {model or 'not set'}"
    ) if base_url else "Not configured — set AI_BASE_URL, AI_MODEL, AI_API_KEY in .env"
    results["ai_llm"] = ai_entry

    return results


def get_active_api_summary() -> dict:
    """
    Quick summary (no network calls) of what's configured.
    Used at startup to print a health summary.
    """
    summary = {}
    for name, cfg in API_REGISTRY.items():
        env_var = cfg.get("env_var")
        key_required = cfg.get("key_required", False)
        key_set = bool(env_var and os.environ.get(env_var, "").strip())
        summary[name] = {
            "key_required": key_required,
            "key_set": key_set,
            "ready": (not key_required) or key_set,
        }
    return summary


def print_startup_summary() -> None:
    """Print a human-readable API config summary at server startup."""
    summary = get_active_api_summary()
    print("\n" + "═" * 60)
    print("  LNI Reference Checker — API Configuration")
    print("═" * 60)
    for name, info in summary.items():
        if info["ready"]:
            status = "✓ ready"
            if info["key_set"]:
                status += " (key configured)"
            else:
                status += " (no key needed)"
        else:
            env_var = API_REGISTRY[name].get("env_var", "")
            status = f"✗ NOT configured — set {env_var} in .env"
        print(f"  {name:<20} {status}")
    base_url, model, _ = get_ai_config()
    if base_url and model:
        provider = base_url.split("/")[2] if "//" in base_url else base_url
        print(f"  {'ai_llm':<20} ✓ ready ({provider}, {model})")
    else:
        print(f"  {'ai_llm':<20} ✗ NOT configured — see .env.example")
    print("═" * 60)
    print("  Tip: copy .env.example → .env and fill in your keys")
    print("═" * 60 + "\n")
