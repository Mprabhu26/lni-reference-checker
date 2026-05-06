"""
Local Academic Cache - Builds database from papers you verify
No external downloads needed - grows automatically
"""

import sqlite3
import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# Database directory
DB_DIR = Path(os.environ.get("LNI_DB_DIR", ".lni_db"))
DB_DIR.mkdir(exist_ok=True)

CACHE_DB = DB_DIR / "verified_papers.db"


@dataclass
class CachedPaper:
    """Paper from local cache"""
    title: str
    authors: str
    year: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    source: str  # where it was found (api, web_search, etc.)
    confidence: float
    last_seen: str


def init_cache_db():
    """Initialize the cache database"""
    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()
    
    # Create table for verified papers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verified_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            url TEXT,
            source TEXT,
            confidence REAL,
            last_seen TEXT,
            title_normalized TEXT UNIQUE
        )
    """)
    
    # Create indexes for fast searching
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title_norm ON verified_papers(title_normalized)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON verified_papers(doi)")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Cache database initialized at {CACHE_DB}")


def normalize_title(title: str) -> str:
    """Normalize title for matching"""
    if not title:
        return ""
    # Lowercase
    t = title.lower()
    # Remove punctuation
    t = re.sub(r'[^\w\s]', '', t)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'in', 'of', 'for', 'on', 'and', 'to', 'with', 
                  'by', 'at', 'from', 'into', 'through', 'during', 'including',
                  'der', 'die', 'das', 'und', 'fur', 'von', 'mit', 'im', 'an', 'zu'}
    words = [w for w in t.split() if w not in stop_words and len(w) > 2]
    # Sort for consistent matching
    return ' '.join(sorted(words))


def save_to_cache(title: str, authors: str, year: str, doi: str, url: str, source: str, confidence: float):
    """Save a verified paper to cache"""
    if not title:
        return
    
    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()
    
    title_norm = normalize_title(title)
    year_int = int(year) if year and year.isdigit() else None
    
    cursor.execute("""
        INSERT OR REPLACE INTO verified_papers 
        (title, authors, year, doi, url, source, confidence, last_seen, title_normalized)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        title[:500],
        authors[:500] if authors else "",
        year_int,
        doi,
        url,
        source,
        confidence,
        datetime.now().isoformat(),
        title_norm
    ))
    
    conn.commit()
    conn.close()


def search_cache(title: str, authors: str = "") -> Optional[CachedPaper]:
    """Search local cache for a paper"""
    if not CACHE_DB.exists():
        return None
    
    if not title:
        return None
    
    conn = sqlite3.connect(str(CACHE_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    title_norm = normalize_title(title)
    
    # Try exact normalized match first
    cursor.execute("""
        SELECT title, authors, year, doi, url, source, confidence, last_seen
        FROM verified_papers
        WHERE title_normalized = ?
        ORDER BY confidence DESC
        LIMIT 1
    """, (title_norm,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return CachedPaper(
            title=result['title'],
            authors=result['authors'] or "",
            year=str(result['year']) if result['year'] else None,
            doi=result['doi'],
            url=result['url'],
            source=result['source'],
            confidence=result['confidence'],
            last_seen=result['last_seen']
        )
    
    return None


def get_cache_stats() -> dict:
    """Get statistics about local cache"""
    stats = {"total_papers": 0, "by_source": {}}
    
    if CACHE_DB.exists():
        conn = sqlite3.connect(str(CACHE_DB))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM verified_papers")
        stats["total_papers"] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT source, COUNT(*) as count 
            FROM verified_papers 
            GROUP BY source
        """)
        for row in cursor.fetchall():
            stats["by_source"][row[0]] = row[1]
        
        conn.close()
    
    return stats


def clear_old_entries(days: int = 365):
    """Remove entries older than specified days"""
    if not CACHE_DB.exists():
        return
    
    from datetime import datetime, timedelta
    
    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()
    
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cursor.execute("DELETE FROM verified_papers WHERE last_seen < ?", (cutoff,))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"✅ Cleared {deleted} old entries from cache")