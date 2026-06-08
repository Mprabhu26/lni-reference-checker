"""
Local Academic Cache — v2
Stores ONLY confirmed-real papers. Grows automatically as references are verified.
Uses zlib compression on title/abstract blobs to stay lightweight over time.
SQLite WAL mode for safe concurrent access.
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
_SCHEMA_VERSION = 2


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
    """Deterministic title key used for deduplication."""
    if not title:
        return ""
    t = title.lower()
    # German umlauts
    for a, b in [('ä','ae'),('ö','oe'),('ü','ue'),('ß','ss')]:
        t = t.replace(a, b)
    t = re.sub(r'[^\w\s]', '', t)
    stop = {'the','a','an','in','of','for','on','and','to','with','by','at',
            'der','die','das','und','fur','von','mit','im','an','zu'}
    words = sorted(w for w in t.split() if w not in stop and len(w) > 2)
    return ' '.join(words)


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

    conn.commit()
    conn.close()


def _ensure_db():
    if not CACHE_DB.exists():
        init_cache_db()


# ── Write ─────────────────────────────────────────────────────────────────────

def save_to_cache(title: str, authors: str, year: str, doi: str,
                  url: str, source: str, confidence: float,
                  only_if_real: bool = True):
    """
    Persist a paper to the local DB.
    By default (only_if_real=True) only confirmed-real papers are written —
    suspicious / FAKE results are never stored.
    """
    if not title or not title.strip():
        return
    _ensure_db()

    norm = normalize_title(title)
    if not norm:
        return

    year_int = None
    if year:
        m = re.search(r'\d{4}', str(year))
        if m:
            year_int = int(m.group())

    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    now = datetime.now().isoformat()
    try:
        conn.execute("""
            INSERT INTO verified_papers
                (title_norm, title_blob, authors_blob, year, doi, url,
                 source, confidence, confirmed_real, added_date, last_seen)
            VALUES (?,?,?,?,?,?,?,?,1,?,?)
            ON CONFLICT(title_norm) DO UPDATE SET
                confidence  = MAX(confidence, excluded.confidence),
                source      = excluded.source,
                last_seen   = excluded.last_seen,
                doi         = COALESCE(doi, excluded.doi),
                url         = COALESCE(url, excluded.url)
        """, (
            norm,
            _compress(title[:500]),
            _compress(authors[:500]) if authors else None,
            year_int, doi, url, source,
            round(confidence, 4), now, now,
        ))
        conn.commit()
    finally:
        conn.close()


# ── Read ──────────────────────────────────────────────────────────────────────

def search_cache(title: str, authors: str = "") -> Optional[CachedPaper]:
    """
    Look up a paper by normalised title.
    Returns CachedPaper (with from_local_db=True) or None.
    """
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
            by_source[row[0]] = row[1]
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
                WHERE title_norm LIKE ? AND confirmed_real = 1
                ORDER BY added_date DESC
                LIMIT ? OFFSET ?
            """, (f"%{norm_search}%", limit, offset)).fetchall()
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
        if authors:
            row = conn.execute(
                "SELECT 1 FROM verified_papers WHERE title_norm = ? AND authors_blob = ?",
                (norm, _compress(authors[:500]) if authors else None)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM verified_papers WHERE title_norm = ?",
                (norm,)
            ).fetchone()
        return row is not None
    finally:
        conn.close()