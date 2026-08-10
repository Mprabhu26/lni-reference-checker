"""
Fix the database schema - add missing author_norm column
Run: python fix_db.py
"""

import sqlite3
from pathlib import Path

DB_DIR = Path(".lni_db")
CACHE_DB = DB_DIR / "verified_papers.db"

def fix_db():
    if not CACHE_DB.exists():
        print("Database not found. Run the app once to create it.")
        return
    
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()
    
    # Check if author_norm column exists
    c.execute("PRAGMA table_info(verified_papers)")
    columns = [row[1] for row in c.fetchall()]
    
    if "author_norm" not in columns:
        print("Adding author_norm column...")
        c.execute("ALTER TABLE verified_papers ADD COLUMN author_norm TEXT")
        print("✅ Added author_norm column")
    else:
        print("✅ author_norm column already exists")
    
    # Check if we need to recreate indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_author ON verified_papers(author_norm)")
    print("✅ Created index on author_norm")
    
    conn.commit()
    conn.close()
    print("✅ Database fixed successfully!")

if __name__ == "__main__":
    fix_db()