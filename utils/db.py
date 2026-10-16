import sqlite3
import datetime
import pandas as pd

DB_FILE = "history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            is_healthy INTEGER,
            damage_pct REAL DEFAULT 0.0,
            severity_grade TEXT DEFAULT 'N/A'
        )
    ''')
    # Migrate if columns don't exist yet
    c.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in c.fetchall()]
    if "damage_pct" not in columns:
        try:
            c.execute("ALTER TABLE predictions ADD COLUMN damage_pct REAL DEFAULT 0.0")
        except Exception:
            pass
    if "severity_grade" not in columns:
        try:
            c.execute("ALTER TABLE predictions ADD COLUMN severity_grade TEXT DEFAULT 'N/A'")
        except Exception:
            pass
    conn.commit()
    conn.close()

def add_prediction(filename, predicted_class, confidence, is_healthy, damage_pct=0.0, severity_grade="N/A"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO predictions (timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp, filename, predicted_class, confidence, int(is_healthy), float(damage_pct), str(severity_grade))
    )
    conn.commit()
    conn.close()

def clear_prediction_history():
    """Clears all records in the predictions table."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

def get_recent_predictions(limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            "timestamp": row[0],
            "filename": row[1],
            "class": row[2],
            "conf": row[3],
            "is_healthy": bool(row[4]),
            "damage_pct": row[5] if len(row) > 5 and row[5] is not None else 0.0,
            "severity_grade": row[6] if len(row) > 6 and row[6] is not None else "N/A"
        }
        for row in rows
    ]

# Initialize on import
init_db()

def get_summary_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), SUM(is_healthy) FROM predictions")
    row = c.fetchone()
    conn.close()
    
    total = row[0] if row[0] is not None else 0
    healthy = row[1] if row[1] is not None else 0
    diseased = total - healthy
    
    return {
        "total": total,
        "healthy": healthy,
        "diseased": diseased
    }

def get_disease_distribution():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT predicted_class, COUNT(*) FROM predictions WHERE is_healthy = 0 GROUP BY predicted_class ORDER BY COUNT(*) DESC")
    rows = c.fetchall()
    conn.close()
    return {row[0].replace('_', ' '): row[1] for row in rows}

def get_daily_trends():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT substr(timestamp, 1, 10) as date, COUNT(*) FROM predictions GROUP BY date ORDER BY date")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def get_all_predictions_df():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df
# Minor update for database utils
