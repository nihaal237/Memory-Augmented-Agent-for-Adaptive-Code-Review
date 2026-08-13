import sqlite3
from datetime import datetime

DB_PATH = "agent_memory.db"

def init_db():
    """Creates the SQLite database and the memory table if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            convention_text TEXT NOT NULL,
            source_pr_number INTEGER,
            category TEXT,
            created_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            times_confirmed INTEGER DEFAULT 1,
            active INTEGER DEFAULT 1
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ SQLite database initialized:", DB_PATH)


if __name__ == "__main__":
    init_db()