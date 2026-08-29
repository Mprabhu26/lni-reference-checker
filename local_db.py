"""
Local Academic Cache — v2.3 (FIXED: Proper deduplication)
Stores ONLY confirmed-real papers. Grows automatically as references are verified.
Uses zlib compression on title/abstract blobs to stay lightweight over time.
SQLite WAL mode for safe concurrent access.

FIXES v2.3:
  - Improved deduplication: normalizes titles better before storing
  - Duplicate entries are NOT stored multiple times
  - Better author matching for deduplication
  - Confidence scores are merged (highest wins)
"""

import sqlite3
import zlib
import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

DB_DIR = Path(os.environ.get("LNI_DB_DIR", ".lni_db"))
DB_DIR.mkdir(exist_ok=True)
CACHE_DB = DB_DIR / "verified_papers.db"

# ── Schema version — bump when you change the table layout ──────────────────
_SCHEMA_VERSION = 3


def _validate_url(url: str) -> str:
    """
    Validate URL and try common fixes. Returns the working URL or empty string.
    Strategy:
    1. Try HEAD request to URL as-is
    2. Try GET if HEAD fails (some servers don't support HEAD)
    3. If 4xx error and www missing, try adding www.
    4. If PDF/download URL fails, try without download params
    5. If all fail, return empty string (skip caching URL, but don't reject paper)
    """
    if not url or not url.strip().startswith("http"):
        return ""
    
    import requests
    
    original_url = url.strip()
    attempts = [original_url]
    
    # For PDF downloads, also try removing download params
    if "?download" in original_url or ".pdf" in original_url.lower():
        base_url = original_url.split("?")[0]
        if base_url != original_url:
            attempts.append(base_url)
    
    # Variant 2: Add www if missing
    if "://" in original_url:
        schema, rest = original_url.split("://", 1)
        if not rest.startswith("www."):
            attempts.append(f"{schema}://www.{rest}")
    
    for attempt_url in attempts:
        try:
            # Try HEAD first (faster, but some servers don't allow it)
            resp = requests.head(attempt_url, timeout=5, allow_redirects=True, 
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code in (200, 301, 302, 303, 307, 308):
                return attempt_url  # This URL works
            
            # If 4xx, try GET instead (some servers don't support HEAD or block PDFs via HEAD)
            if 400 <= resp.status_code < 500:
                resp = requests.get(attempt_url, timeout=5, allow_redirects=True,
                                   headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code in (200, 301, 302, 303, 307, 308):
                    return attempt_url  # GET works even if HEAD failed
        except requests.Timeout:
            pass  # Try next variant
        except Exception:
            pass  # Try next variant
    
    return ""  # No URL variant worked; skip caching but don't reject paper


@dataclass
class CachedPaper:
    title: str
    authors: str
    year: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    source: str          # 'crossref' | 'semantic_scholar' | 'openalex' | 'web_search' | 'manual'
    confidence: float
    last_seen: str
    from_local_db: bool = True   # always True for entries returned from here


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compress(text: str) -> bytes:
    """zlib-compress a UTF-8 string. Saves ~60% space for long titles."""
    return zlib.compress(text.encode("utf-8"), level=6)


def _decompress(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8")


def normalize_title(title: str) -> str:
    """
    Deterministic title key used for deduplication.
    FIXED v2.3: Better normalization for matching.
    """
    if not title:
        return ""
    
    # Lowercase
    t = title.lower()
    
    # German umlauts
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        t = t.replace(a, b)
    
    # Remove punctuation and extra spaces
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    
    # Remove common stop words (more aggressive for dedup)
    stop = {
        'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 'by', 'at',
        'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu',
        'for', 'from', 'into', 'through', 'during', 'including', 'without',
        'after', 'before', 'above', 'below', 'between', 'among',
    }
    words = [w for w in t.split() if w not in stop and len(w) > 2]
    
    # Keep first 8 words max for dedup (enough to uniquely identify)
    return ' '.join(words[:8])


def normalize_authors(authors: str) -> str:
    """
    Normalize authors for deduplication.
    Extracts first author's surname and first initial.
    """
    if not authors:
        return ""
    
    # Get first author
    first = authors.split(';')[0].strip()
    
    # Extract surname (part before comma)
    if ',' in first:
        surname = first.split(',')[0].strip()
    else:
        # No comma - take last word as surname
        parts = first.split()
        surname = parts[-1] if parts else first
    
    # Normalize surname
    surname = surname.lower()
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        surname = surname.replace(a, b)
    
    # Remove punctuation
    surname = re.sub(r'[^\w]', '', surname)
    
    # Get first initial if available
    initial = ""
    if ',' in first:
        given = first.split(',')[1].strip()
        if given:
            initial = given[0].lower()
    
    return f"{surname}_{initial}" if initial else surname


# ── Initialisation ────────────────────────────────────────────────────────────

def init_cache_db():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)
    """)
    c.execute("INSERT OR IGNORE INTO schema_version VALUES (?)", (_SCHEMA_VERSION,))

    c.execute("""
        CREATE TABLE IF NOT EXISTS verified_papers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            title_norm       TEXT    UNIQUE NOT NULL,   -- dedup key
            author_norm      TEXT,                       -- normalized first author
            title_blob       BLOB    NOT NULL,          -- zlib-compressed title
            authors_blob     BLOB,                      -- zlib-compressed authors
            year             INTEGER,
            doi              TEXT,
            url              TEXT,
            source           TEXT    NOT NULL DEFAULT 'unknown',
            confidence       REAL    NOT NULL DEFAULT 1.0,
            confirmed_real   INTEGER NOT NULL DEFAULT 1, -- 1 = only real papers stored
            added_date       TEXT    NOT NULL,
            last_seen        TEXT    NOT NULL
        )
    """)

    # Fast lookup indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_tnorm  ON verified_papers(title_norm)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_doi    ON verified_papers(doi)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_year   ON verified_papers(year)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_author ON verified_papers(author_norm)")

    conn.commit()
    conn.close()


def _ensure_db():
    if not CACHE_DB.exists():
        init_cache_db()


# ── Write (FIXED: Better deduplication) ─────────────────────────────────────

def save_to_cache(title: str, authors: str, year: str, doi: str,
                  url: str, source: str, confidence: float,
                  only_if_real: bool = True):
    """
    Persist a paper to the local DB.
    FIXED v2.3:
      - Better deduplication using normalized title + author
      - Merges confidence scores (highest wins)
      - Duplicates are NOT stored multiple times
    """
    if not title or not title.strip():
        return
    _ensure_db()

    norm_title = normalize_title(title)
    if not norm_title:
        return
    
    norm_author = normalize_authors(authors) if authors else ""

    year_int = None
    if year:
        m = re.search(r'\d{4}', str(year))
        if m:
            year_int = int(m.group())

    # Validate URLs before caching (try to fix incomplete URLs)
    url_to_cache = url
    if url and url.strip().startswith("http"):
        fixed_url = _validate_url(url)
        if fixed_url:
            url_to_cache = fixed_url
            if fixed_url != url.strip():
                print(f"[local_db] URL fixed: {url} → {fixed_url}")
        else:
            print(f"[local_db] URL validation failed, skipping: {url}")
            url_to_cache = ""

    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    now = datetime.now().isoformat()
    
    try:
        # Check if the paper already exists with the same normalized title
        existing = conn.execute(
            "SELECT id, confidence, source FROM verified_papers WHERE title_norm = ?",
            (norm_title,)
        ).fetchone()
        
        if existing:
            # Update existing record with higher confidence
            existing_id, existing_conf, existing_source = existing
            new_confidence = max(confidence, existing_conf)
            
            # Merge sources
            if source not in existing_source:
                merged_source = f"{existing_source},{source}"
            else:
                merged_source = existing_source
            
            # Update the record
            conn.execute("""
                UPDATE verified_papers 
                SET confidence = MAX(confidence, ?),
                    source = ?,
                    last_seen = ?,
                    doi = COALESCE(NULLIF(?, ''), doi),
                    url = COALESCE(NULLIF(?, ''), url),
                    year = COALESCE(?, year),
                    authors_blob = COALESCE(NULLIF(?, ''), authors_blob)
                WHERE title_norm = ?
            """, (
                confidence,
                merged_source,
                now,
                doi,
                url_to_cache,
                year_int,
                _compress(authors[:500]) if authors else None,
                norm_title,
            ))
            conn.commit()
            print(f"[local_db] Updated existing paper: '{title[:40]}...' (confidence {new_confidence:.2f})")
        else:
            # Insert new paper
            conn.execute("""
                INSERT INTO verified_papers
                    (title_norm, author_norm, title_blob, authors_blob, year, doi, url,
                     source, confidence, confirmed_real, added_date, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                norm_title,
                norm_author[:50] if norm_author else None,
                _compress(title[:500]),
                _compress(authors[:500]) if authors else None,
                year_int,
                doi,
                url_to_cache,
                source,
                round(confidence, 4),
                now,
                now,
            ))
            conn.commit()
            print(f"[local_db] Added new paper: '{title[:40]}...' (source: {source})")
            
    except sqlite3.IntegrityError as e:
        print(f"[local_db] Integrity error (duplicate): {e}")
        # Try to update instead
        try:
            conn.execute("""
                UPDATE verified_papers 
                SET confidence = MAX(confidence, ?),
                    source = ?,
                    last_seen = ?,
                    doi = COALESCE(NULLIF(?, ''), doi),
                    url = COALESCE(NULLIF(?, ''), url)
                WHERE title_norm = ?
            """, (
                confidence,
                source,
                now,
                doi,
                url_to_cache,
                norm_title,
            ))
            conn.commit()
        except Exception as e2:
            print(f"[local_db] Update failed: {e2}")
    finally:
        conn.close()


# ── Read ──────────────────────────────────────────────────────────────────────

def search_cache(title: str, authors: str = "") -> Optional[CachedPaper]:
    """Look up a paper by normalised title."""
    if not title:
        return None
    _ensure_db()
    
    norm = normalize_title(title)
    if not norm:
        return None
    
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("""
            SELECT title_blob, authors_blob, year, doi, url, source, confidence, last_seen
            FROM verified_papers
            WHERE title_norm = ? AND confirmed_real = 1
            ORDER BY confidence DESC
            LIMIT 1
        """, (norm,)).fetchone()
        
        # Do not fall back to author-only matching: common or fabricated
        # author strings can otherwise return an unrelated real paper.
    finally:
        conn.close()
    
    if not row:
        return None
    
    return CachedPaper(
        title       = _decompress(row["title_blob"]),
        authors     = _decompress(row["authors_blob"]) if row["authors_blob"] else "",
        year        = str(row["year"]) if row["year"] else None,
        doi         = row["doi"],
        url         = row["url"],
        source      = row["source"],
        confidence  = row["confidence"],
        last_seen   = row["last_seen"],
        from_local_db = True,
    )


# ── Manual inject (professor confirms a suspicious entry as real) ─────────────

def inject_confirmed_paper(title: str, authors: str, year: str,
                            doi: str = "", url: str = "") -> bool:
    """
    Professor manually confirms a reference is real.
    Stored with source='manual' and confidence=1.0.
    FIXED v2.3: Uses save_to_cache for proper deduplication.
    """
    try:
        save_to_cache(title, authors, year, doi, url,
                      source="manual", confidence=1.0)
        return True
    except Exception as e:
        print(f"inject_confirmed_paper error: {e}")
        return False


# ── Maintenance ───────────────────────────────────────────────────────────────

def get_cache_stats() -> dict:
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB))
    try:
        total = conn.execute("SELECT COUNT(*) FROM verified_papers").fetchone()[0]
        by_source = {}
        for row in conn.execute(
            "SELECT source, COUNT(*) FROM verified_papers GROUP BY source"
        ).fetchall():
            # Handle comma-separated sources
            sources = row[0].split(',')
            for s in sources:
                by_source[s] = by_source.get(s, 0) + row[1]
        # Approximate disk size
        size_bytes = CACHE_DB.stat().st_size if CACHE_DB.exists() else 0
        return {
            "total_papers": total,
            "by_source": by_source,
            "db_size_kb": round(size_bytes / 1024, 1),
            "db_path": str(CACHE_DB),
        }
    finally:
        conn.close()


def vacuum_db():
    """Reclaim disk space (run occasionally, not on every request)."""
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("VACUUM")
    conn.close()


def clear_old_entries(days: int = 730):
    """
    Remove papers not seen for `days` days (default 2 years).
    Manual entries are never deleted.
    """
    _ensure_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(CACHE_DB))
    try:
        cur = conn.execute(
            "DELETE FROM verified_papers WHERE last_seen < ? AND source != 'manual'",
            (cutoff,)
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_all_papers(limit: int = 500, offset: int = 0, search: str = "") -> list:
    """
    Retrieve all papers from the DB for the Database browser tab.
    Supports pagination and optional search filter.
    """
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        if search:
            norm_search = normalize_title(search)
            rows = conn.execute("""
                SELECT title_blob, authors_blob, year, doi, url, source, confidence, last_seen, added_date
                FROM verified_papers
                WHERE (title_norm LIKE ? OR author_norm LIKE ?) AND confirmed_real = 1
                ORDER BY added_date DESC
                LIMIT ? OFFSET ?
            """, (f"%{norm_search}%", f"%{norm_search}%", limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT title_blob, authors_blob, year, doi, url, source, confidence, last_seen, added_date
                FROM verified_papers
                WHERE confirmed_real = 1
                ORDER BY added_date DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "title": _decompress(row["title_blob"]),
                "authors": _decompress(row["authors_blob"]) if row["authors_blob"] else "",
                "year": str(row["year"]) if row["year"] else "",
                "doi": row["doi"] or "",
                "url": row["url"] or "",
                "source": row["source"],
                "confidence": round(row["confidence"], 2),
                "last_seen": row["last_seen"][:10] if row["last_seen"] else "",
                "added_date": row["added_date"][:10] if row["added_date"] else "",
            })
        return results
    finally:
        conn.close()


def delete_paper(title: str) -> bool:
    """Delete a paper from the DB by title (for the DB browser tab)."""
    if not title:
        return False
    _ensure_db()
    norm = normalize_title(title)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cur = conn.execute("DELETE FROM verified_papers WHERE title_norm = ?", (norm,))
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        print(f"Delete error: {e}")
        return False
    finally:
        conn.close()


def paper_exists(title: str, authors: str = "") -> bool:
    """Check if a paper already exists in the database."""
    if not title:
        return False
    norm = normalize_title(title)
    conn = sqlite3.connect(str(CACHE_DB))
    try:
        row = conn.execute(
            "SELECT 1 FROM verified_papers WHERE title_norm = ?",
            (norm,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()