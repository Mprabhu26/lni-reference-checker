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
        # === GI / LNI series (high trust) ===
        ("Lecture Notes in Informatics (LNI)", "series", "Germany", "high"),
        ("LNI", "series", "Germany", "high"),
        ("GI Jahrestagung", "conference", "Germany", "high"),
        ("INFORMATIK", "conference", "Germany", "high"),
        ("Gesellschaft fur Informatik", "organization", "Germany", "high"),
        ("Gesellschaft für Informatik", "organization", "Germany", "high"),

        # === German DB / IS conferences ===
        ("BTW", "conference", "Germany", "high"),
        ("Datenbanksysteme fur Business Technologie und Web", "conference", "Germany", "high"),
        ("Datenbanksysteme für Business, Technologie und Web", "conference", "Germany", "high"),
        ("MKWI", "conference", "Germany", "high"),
        ("Multikonferenz Wirtschaftsinformatik", "conference", "Germany", "high"),
        ("WI", "conference", "Germany", "high"),
        ("Wirtschaftsinformatik", "conference", "Germany", "high"),
        ("EMISA", "conference", "Germany", "medium"),
        ("Modellierung", "conference", "Germany", "medium"),

        # === German SE / SWE conferences ===
        ("Software Engineering", "conference", "Germany", "high"),
        ("Software Engineering & Management", "conference", "Germany", "high"),
        ("SE&M", "conference", "Germany", "high"),
        ("SKILL", "conference", "Germany", "high"),
        ("Studierendenkonferenz Informatik", "conference", "Germany", "high"),
        ("SANER", "conference", "Germany", "medium"),
        ("ICSME", "conference", "Germany", "medium"),

        # === German AI / ML / KI conferences ===
        ("KI", "conference", "Germany", "high"),
        ("Kunstliche Intelligenz", "conference", "Germany", "high"),
        ("Künstliche Intelligenz", "conference", "Germany", "high"),
        ("LWDA", "conference", "Germany", "high"),
        ("Lernen Wissen Daten Analysen", "conference", "Germany", "high"),
        ("KDML", "conference", "Germany", "medium"),
        ("FGWM", "conference", "Germany", "medium"),

        # === German HCI / usability conferences ===
        ("Mensch und Computer", "conference", "Germany", "high"),
        ("MuC", "conference", "Germany", "high"),
        ("GI Mensch", "conference", "Germany", "high"),
        ("UP", "conference", "Germany", "medium"),
        ("Usability Professionals", "conference", "Germany", "medium"),

        # === German e-learning / education ===
        ("DeLFI", "conference", "Germany", "high"),
        ("DELFI", "conference", "Germany", "high"),
        ("GMW", "conference", "Germany", "medium"),
        ("Gesellschaft fur Medien in der Wissenschaft", "conference", "Germany", "medium"),
        ("Gesellschaft für Medien in der Wissenschaft", "conference", "Germany", "medium"),
        ("INFOS", "conference", "Germany", "medium"),

        # === German security / biometrics ===
        ("BIOSIG", "conference", "Germany", "high"),
        ("Sicherheit", "conference", "Germany", "high"),
        ("GI Sicherheit", "conference", "Germany", "high"),
        ("TRUST", "conference", "Germany", "medium"),
        ("D-A-CH Security", "conference", "Germany", "medium"),

        # === German health informatics ===
        ("MIK", "conference", "Germany", "high"),
        ("Medizinische Informatik", "conference", "Germany", "high"),
        ("GMDS", "conference", "Germany", "high"),
        ("eHealth", "conference", "Germany", "medium"),
        ("GI Health", "conference", "Germany", "medium"),

        # === German networking / distributed systems ===
        ("KuVS", "conference", "Germany", "medium"),
        ("KIVS", "conference", "Germany", "medium"),
        ("Kommunikation in Verteilten Systemen", "conference", "Germany", "medium"),
        ("GI Betriebssysteme", "conference", "Germany", "medium"),

        # === German geographic / spatial information ===
        ("AGIT", "conference", "Germany", "medium"),
        ("GeoInformatik", "conference", "Germany", "medium"),
        ("GeNeMe", "conference", "Germany", "medium"),

        # === German journals (GI-published or closely affiliated) ===
        ("Informatik Spektrum", "journal", "Germany", "high"),
        ("it - Information Technology", "journal", "Germany", "high"),
        ("it Information Technology", "journal", "Germany", "high"),
        ("Praxis der Informationsverarbeitung und Kommunikation", "journal", "Germany", "high"),
        ("PIK", "journal", "Germany", "high"),
        ("Datenbank-Spektrum", "journal", "Germany", "high"),
        ("Datenbank Spektrum", "journal", "Germany", "high"),
        ("WIRTSCHAFTSINFORMATIK", "journal", "Germany", "high"),
        ("Business and Information Systems Engineering", "journal", "Germany", "high"),
        ("BISE", "journal", "Germany", "high"),
        ("Electronic Markets", "journal", "Germany", "high"),
        ("Journal of Business Economics", "journal", "Germany", "medium"),
        ("Zeitschrift fur Betriebswirtschaft", "journal", "Germany", "medium"),
        ("Softwaretechnik-Trends", "journal", "Germany", "medium"),

        # === Ausgezeichnete Dissertationen series ===
        ("Ausgezeichnete Informatikdissertationen", "series", "Germany", "high"),
        ("Dissertationspreis", "series", "Germany", "high"),

        # === Austrian / Swiss German-language venues (also commonly in LNI) ===
        ("OCG", "conference", "Austria", "medium"),
        ("Österreichische Computer Gesellschaft", "organization", "Austria", "medium"),
        ("Osterreichische Computer Gesellschaft", "organization", "Austria", "medium"),
        ("Informatiktage", "conference", "Germany", "medium"),
        ("INFOS Informatik", "conference", "Germany", "medium"),

        # === Generic LNI workshop series keywords ===
        ("Workshop", "conference", "Germany", "low"),    # low trust — many non-GI workshops
        ("GI Workshop", "conference", "Germany", "medium"),
        ("GI-Workshop", "conference", "Germany", "medium"),
        ("Fachtagung", "conference", "Germany", "medium"),
        ("Fachgesprach", "conference", "Germany", "medium"),
        ("Fachgespräch", "conference", "Germany", "medium"),
        ("Dagstuhl", "conference", "Germany", "high"),
        ("Schloss Dagstuhl", "conference", "Germany", "high"),
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
                        verified_url: str = "", verified_doi: str = "",
                        ai_had_said: str = "") -> bool:
    """Add or update a professor's review decision.

    If decision='verified' and ai_had_said='FAKE' (or SUSPICIOUS), also write a
    false_positives record so future checks don't re-flag the paper.
    """
    if not REVIEW_DB.exists():
        init_review_db()

    try:
        conn = sqlite3.connect(str(REVIEW_DB))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO review_decisions
            (paper_title, paper_authors, decision, professor_note, verified_url, verified_doi, decision_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title[:500], authors[:300] if authors else "", decision, note[:1000],
              verified_url, verified_doi, datetime.now().isoformat()))

        # If the professor is verifying something the AI called FAKE/SUSPICIOUS,
        # record it as a false positive so it won't be flagged again.
        if decision == "verified" and ai_had_said in ("FAKE", "SUSPICIOUS", "fake", "suspicious"):
            cursor.execute("""
                INSERT INTO false_positives
                (paper_title, paper_authors, ai_verdict, professor_correction, correction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title[:500], authors[:300] if authors else "", ai_had_said.upper(),
                  "REAL", datetime.now().isoformat(), note[:500]))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving review decision: {e}")
        return False


def _ensure_db():
    """Initialize the review DB if it does not exist yet."""
    if not REVIEW_DB.exists():
        init_review_db()


def get_review_decision(title: str, authors: str = "") -> Optional[Dict]:
    """Get professor's review decision for a paper"""
    _ensure_db()
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
    
    _ensure_db()
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
    _ensure_db()
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



def get_false_positive(title: str, authors: str = "") -> Optional[Dict]:
    """Check if this paper was previously corrected as a false positive.

    Returns the record if a professor marked it as REAL after the AI flagged it,
    so subsequent checks skip re-flagging it.
    """
    if not title or not REVIEW_DB.exists():
        return None

    try:
        conn = sqlite3.connect(str(REVIEW_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if authors:
            cursor.execute(
                """SELECT * FROM false_positives
                   WHERE paper_title = ? AND paper_authors = ?
                   ORDER BY correction_date DESC LIMIT 1""",
                (title, authors),
            )
        else:
            cursor.execute(
                """SELECT * FROM false_positives
                   WHERE paper_title = ?
                   ORDER BY correction_date DESC LIMIT 1""",
                (title,),
            )

        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
    except Exception:
        return None


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