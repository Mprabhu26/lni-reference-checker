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


def _db_author_overlap(cited: str, cached: str) -> float:
    """Check author surname overlap inside DB queries without circular imports."""
    if not cited or not cached:
        return 0.0
    def _surnames(s):
        out = set()
        for p in re.split(r';|\band\b|\bund\b', s, flags=re.IGNORECASE):
            p = p.strip()
            if not p or re.match(r'^et\s+al\.?$', p.lower()):
                continue
            sur = p.split(',')[0].strip() if ',' in p else p.split()[-1].strip()
            clean = re.sub(r'[^a-zA-Z0-9]', '', sur.lower())
            if len(clean) > 2:
                out.add(clean)
        return out
    s_cited = _surnames(cited)
    s_cached = _surnames(cached)
    if not s_cited or not s_cached:
        return 0.0
    
    # Overlap measured against authors provided by user (supports "et al." or partial citation)
    matched = len(s_cited & s_cached)
    return round(matched / len(s_cited), 3)


def _validate_url(url: str) -> str:
    """
    Validate URL and try common fixes. Returns the working URL or empty string.
    """
    if not url or not url.strip().startswith("http"):
        return ""
    
    import requests
    
    original_url = url.strip()
    attempts = [original_url]
    
    if "?download" in original_url or ".pdf" in original_url.lower():
        base_url = original_url.split("?")[0]
        if base_url != original_url:
            attempts.append(base_url)
    
    if "://" in original_url:
        schema, rest = original_url.split("://", 1)
        if not rest.startswith("www."):
            attempts.append(f"{schema}://www.{rest}")
    
    for attempt_url in attempts:
        try:
            resp = requests.head(attempt_url, timeout=5, allow_redirects=True, 
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code in (200, 301, 302, 303, 307, 308):
                return attempt_url
            
            if 400 <= resp.status_code < 500:
                resp = requests.get(attempt_url, timeout=5, allow_redirects=True,
                                   headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code in (200, 301, 302, 303, 307, 308):
                    return attempt_url
        except requests.Timeout:
            pass
        except Exception:
            pass
    
    return ""


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
    from_local_db: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compress(text: str) -> bytes:
    """zlib-compress a UTF-8 string. Saves ~60% space for long titles."""
    return zlib.compress(text.encode("utf-8"), level=6)


def _decompress(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8")


def normalize_title(title: str) -> str:
    """Deterministic title key used for deduplication."""
    if not title:
        return ""
    
    t = title.lower()
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        t = t.replace(a, b)
    
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    
    stop = {
        'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 'by', 'at',
        'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu',
        'for', 'from', 'into', 'through', 'during', 'including', 'without',
        'after', 'before', 'above', 'below', 'between', 'among',
    }
    words = [w for w in t.split() if w not in stop and len(w) > 2]
    return ' '.join(words[:8])


def normalize_authors(authors: str) -> str:
    """Normalize authors for deduplication (first author surname and initial)."""
    if not authors:
        return ""
    
    first = authors.split(';')[0].strip()
    if ',' in first:
        surname = first.split(',')[0].strip()
    else:
        parts = first.split()
        surname = parts[-1] if parts else first
    
    surname = surname.lower()
    for a, b in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        surname = surname.replace(a, b)
    
    surname = re.sub(r'[^\w]', '', surname)
    
    initial = ""
    if ',' in first:
        given = first.split(',')[1].strip()
        if given:
            initial = given[0].lower()
    
    return f"{surname}_{initial}" if initial else surname


# ── Initialisation ────────────────────────────────────────────────────────────

def init_cache_db():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)
    """)
    c.execute("INSERT OR IGNORE INTO schema_version VALUES (?)", (_SCHEMA_VERSION,))

    c.execute("""
        CREATE TABLE IF NOT EXISTS verified_papers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            title_norm       TEXT    UNIQUE NOT NULL,
            author_norm      TEXT,
            title_blob       BLOB    NOT NULL,
            authors_blob     BLOB,
            year             INTEGER,
            doi              TEXT,
            url              TEXT,
            source           TEXT    NOT NULL DEFAULT 'unknown',
            confidence       REAL    NOT NULL DEFAULT 1.0,
            confirmed_real   INTEGER NOT NULL DEFAULT 1,
            added_date       TEXT    NOT NULL,
            last_seen        TEXT    NOT NULL
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_tnorm  ON verified_papers(title_norm)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_doi    ON verified_papers(doi)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_year   ON verified_papers(year)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_author ON verified_papers(author_norm)")
    
    # Purge entries cached under legacy unverified heuristics
    c.execute("""
        DELETE FROM verified_papers 
        WHERE source IN ('ml_gate', 'arxiv_verified', 'url_verify', 'url_verify_pdf', 'url_verify_docs')
           OR (source = 'web_search_verified' AND confidence < 0.75)
    """)

    conn.commit()
    conn.close()


def _ensure_db():
    if not CACHE_DB.exists():
        init_cache_db()


# ── Write ───────────────────────────────────────────────────────────────────

def save_to_cache(title: str, authors: str, year: str, doi: str,
                  url: str, source: str, confidence: float,
                  only_if_real: bool = True):
    """Persist a multi-field verified paper to the local DB."""
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

    url_to_cache = url
    if url and url.strip().startswith("http"):
        fixed_url = _validate_url(url)
        if fixed_url:
            url_to_cache = fixed_url
        else:
            url_to_cache = ""

    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    now = datetime.now().isoformat()
    
    try:
        existing = conn.execute(
            "SELECT id, confidence, source FROM verified_papers WHERE title_norm = ?",
            (norm_title,)
        ).fetchone()
        
        if existing:
            existing_id, existing_conf, existing_source = existing
            new_confidence = max(confidence, existing_conf)
            merged_source = f"{existing_source},{source}" if source not in existing_source else existing_source
            
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
        else:
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
            
    except sqlite3.IntegrityError:
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
        except Exception:
            pass
    finally:
        conn.close()


# ── Read ──────────────────────────────────────────────────────────────────────

def search_cache(title: str, authors: str = "") -> Optional[CachedPaper]:
    """Look up a paper by normalised title with author overlap verification."""
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
    finally:
        conn.close()
    
    if not row:
        return None
    
    cached_authors = _decompress(row["authors_blob"]) if row["authors_blob"] else ""

    # Strict author checking:
    if authors and cached_authors:
        if _db_author_overlap(authors, cached_authors) < 0.60:
            return None  # Author mismatch: reject cache hit
    elif not authors and cached_authors:
        return None  # Do not give free pass to anonymous entries
    
    return CachedPaper(
        title       = _decompress(row["title_blob"]),
        authors     = cached_authors,
        year        = str(row["year"]) if row["year"] else None,
        doi         = row["doi"],
        url         = row["url"],
        source      = row["source"],
        confidence  = row["confidence"],
        last_seen   = row["last_seen"],
        from_local_db = True,
    )


# ── Manual inject ─────────────────────────────────────────────────────────────

def inject_confirmed_paper(title: str, authors: str, year: str,
                            doi: str = "", url: str = "") -> bool:
    """Professor manually confirms a reference is real."""
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
            sources = row[0].split(',')
            for s in sources:
                by_source[s] = by_source.get(s, 0) + row[1]
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
    """Reclaim disk space."""
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("VACUUM")
    conn.close()


def clear_old_entries(days: int = 730):
    """Remove papers not seen for `days` days (default 2 years)."""
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
    """Retrieve all papers from the DB for the Database browser tab."""
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
    """Delete a paper from the DB by title."""
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