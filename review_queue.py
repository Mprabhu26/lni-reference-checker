"""
Professor Review Queue - Manual override for false positives/negatives
Stores professor decisions and whitelists legitimate venues
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Database directory
DB_DIR = Path(".lni_db")
DB_DIR.mkdir(exist_ok=True)

REVIEW_DB = DB_DIR / "review_queue.db"


def init_review_db():
    """Initialize review database tables"""
    conn = sqlite3.connect(str(REVIEW_DB))
    cursor = conn.cursor()
    
    # Table for professor review decisions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_title TEXT NOT NULL,
            paper_authors TEXT,
            decision TEXT NOT NULL,  -- 'verified', 'rejected', 'needs_review'
            professor_note TEXT,
            verified_url TEXT,
            verified_doi TEXT,
            decision_date TEXT,
            UNIQUE(paper_title, paper_authors)
        )
    """)
    
    # Table for whitelisted venues (German conferences, journals, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS whitelist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            venue_name TEXT NOT NULL UNIQUE,
            venue_type TEXT,  -- 'conference', 'journal', 'proceedings', 'series'
            country TEXT,
            trust_level TEXT,  -- 'high', 'medium', 'low'
            added_date TEXT,
            added_by TEXT
        )
    """)
    
    # Table for false positive tracking (papers marked as real but AI said fake)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS false_positives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_title TEXT NOT NULL,
            paper_authors TEXT,
            ai_verdict TEXT,
            professor_correction TEXT,
            correction_date TEXT,
            notes TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Pre-populate German venue whitelist
    _init_default_whitelist()


def _init_default_whitelist():
    """Add known German academic venues to whitelist"""
    german_venues = [
        ("Lecture Notes in Informatics (LNI)", "series", "Germany", "high"),
        ("GI Jahrestagung", "conference", "Germany", "high"),
        ("INFORMATIK", "conference", "Germany", "high"),
        ("BTW (Datenbanksysteme für Business, Technologie und Web)", "conference", "Germany", "high"),
        ("SKILL (Studierendenkonferenz Informatik)", "conference", "Germany", "high"),
        ("Software Engineering & Management", "conference", "Germany", "high"),
        ("Datenbanksysteme für Business", "conference", "Germany", "medium"),
        ("Technologie und Web (BTW)", "conference", "Germany", "medium"),
        ("Ausgezeichnete Informatikdissertationen", "series", "Germany", "high"),
        ("Informatik Spektrum", "journal", "Germany", "high"),
        ("it - Information Technology", "journal", "Germany", "medium"),
        ("Praxis der Informationsverarbeitung und Kommunikation", "journal", "Germany", "medium"),
        ("Zeitschrift für Informatik", "journal", "Germany", "medium"),
    ]
    
    conn = sqlite3.connect(str(REVIEW_DB))
    cursor = conn.cursor()
    
    for venue, venue_type, country, trust in german_venues:
        cursor.execute("""
            INSERT OR IGNORE INTO whitelist (venue_name, venue_type, country, trust_level, added_date)
            VALUES (?, ?, ?, ?, ?)
        """, (venue, venue_type, country, trust, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def add_review_decision(title: str, authors: str, decision: str, note: str = "", 
                        verified_url: str = "", verified_doi: str = "") -> bool:
    """Add or update a professor's review decision"""
    try:
        conn = sqlite3.connect(str(REVIEW_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO review_decisions 
            (paper_title, paper_authors, decision, professor_note, verified_url, verified_doi, decision_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title[:500], authors[:300] if authors else "", decision, note[:1000], 
              verified_url, verified_doi, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving review decision: {e}")
        return False


def get_review_decision(title: str, authors: str = "") -> Optional[Dict]:
    """Get professor's review decision for a paper"""
    conn = sqlite3.connect(str(REVIEW_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if authors:
        cursor.execute("""
            SELECT * FROM review_decisions 
            WHERE paper_title = ? AND paper_authors = ?
            ORDER BY decision_date DESC LIMIT 1
        """, (title, authors))
    else:
        cursor.execute("""
            SELECT * FROM review_decisions 
            WHERE paper_title = ?
            ORDER BY decision_date DESC LIMIT 1
        """, (title,))
    
    result = cursor.fetchone()
    conn.close()
    
    return dict(result) if result else None


def is_venue_whitelisted(venue: str) -> Dict:
    """Check if a venue is whitelisted (German conference/journal)"""
    if not venue:
        return {"whitelisted": False, "trust": None}
    
    conn = sqlite3.connect(str(REVIEW_DB))
    cursor = conn.cursor()
    
    # Check for partial matches
    cursor.execute("""
        SELECT * FROM whitelist 
        WHERE ? LIKE '%' || venue_name || '%' OR venue_name LIKE '%' || ? || '%'
        LIMIT 1
    """, (venue, venue))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"whitelisted": True, "trust": result[4], "venue": result[1]}
    return {"whitelisted": False, "trust": None}


def add_false_positive(title: str, authors: str, ai_verdict: str, notes: str = ""):
    """Record a false positive (paper AI flagged as fake but actually real)"""
    conn = sqlite3.connect(str(REVIEW_DB))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO false_positives 
        (paper_title, paper_authors, ai_verdict, professor_correction, correction_date, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (title[:500], authors[:300] if authors else "", ai_verdict, "REAL", 
          datetime.now().isoformat(), notes[:500]))
    
    conn.commit()
    conn.close()


def get_pending_reviews(limit: int = 20) -> List[Dict]:
    """Get papers that need professor review (AI was uncertain)"""
    conn = sqlite3.connect(str(REVIEW_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # This would typically join with your main verification results
    # For now, returns structure
    cursor.execute("""
        SELECT * FROM review_decisions 
        WHERE decision = 'needs_review'
        ORDER BY decision_date DESC
        LIMIT ?
    """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [dict(r) for r in results]


def get_review_stats() -> Dict:
    """Get statistics about review decisions"""
    conn = sqlite3.connect(str(REVIEW_DB))
    cursor = conn.cursor()
    
    stats = {}
    
    cursor.execute("SELECT decision, COUNT(*) FROM review_decisions GROUP BY decision")
    for row in cursor.fetchall():
        stats[row[0]] = row[1]
    
    cursor.execute("SELECT COUNT(*) FROM whitelist")
    stats["whitelisted_venues"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM false_positives")
    stats["false_positives_recorded"] = cursor.fetchone()[0]
    
    conn.close()
    return stats