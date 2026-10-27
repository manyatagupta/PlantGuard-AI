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
            severity_grade TEXT DEFAULT 'N/A',
            farm_zone TEXT DEFAULT 'General Field',
            notes TEXT DEFAULT ''
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
    if "farm_zone" not in columns:
        try:
            c.execute("ALTER TABLE predictions ADD COLUMN farm_zone TEXT DEFAULT 'General Field'")
        except Exception:
            pass
    if "notes" not in columns:
        try:
            c.execute("ALTER TABLE predictions ADD COLUMN notes TEXT DEFAULT ''")
        except Exception:
            pass
    conn.commit()
    conn.close()

def add_prediction(filename, predicted_class, confidence, is_healthy, damage_pct=0.0, severity_grade="N/A", farm_zone="General Field", notes=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO predictions (timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade, farm_zone, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, filename, predicted_class, confidence, int(is_healthy), float(damage_pct), str(severity_grade), str(farm_zone), str(notes))
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

def get_recent_predictions(limit=5, farm_zone=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if farm_zone and farm_zone != "All Zones":
        c.execute("SELECT timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade, farm_zone, notes FROM predictions WHERE farm_zone = ? ORDER BY id DESC LIMIT ?", (farm_zone, limit))
    else:
        c.execute("SELECT timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade, farm_zone, notes FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
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
            "severity_grade": row[6] if len(row) > 6 and row[6] is not None else "N/A",
            "farm_zone": row[7] if len(row) > 7 and row[7] is not None else "General Field",
            "notes": row[8] if len(row) > 8 and row[8] is not None else ""
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

def search_prediction_history(
    query: str = None, 
    farm_zone: str = None, 
    severity_grade: str = None, 
    is_healthy: bool = None,
    limit: int = 100
) -> list:
    """
    Performs dynamic multi-parameter search across historical disease records.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    sql = "SELECT id, timestamp, filename, predicted_class, confidence, is_healthy, damage_pct, severity_grade, farm_zone, notes FROM predictions WHERE 1=1"
    params = []
    
    if query:
        sql += " AND (predicted_class LIKE ? OR filename LIKE ? OR notes LIKE ?)"
        q_wildcard = f"%{query}%"
        params.extend([q_wildcard, q_wildcard, q_wildcard])
        
    if farm_zone and farm_zone != "All Zones":
        sql += " AND farm_zone = ?"
        params.append(farm_zone)
        
    if severity_grade and severity_grade != "All Grades":
        sql += " AND severity_grade LIKE ?"
        params.append(f"%{severity_grade}%")
        
    if is_healthy is not None:
        sql += " AND is_healthy = ?"
        params.append(1 if is_healthy else 0)
        
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    c.execute(sql, tuple(params))
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "filename": r[2],
            "class": r[3],
            "conf": r[4],
            "is_healthy": bool(r[5]),
            "damage_pct": r[6] if len(r) > 6 and r[6] is not None else 0.0,
            "severity_grade": r[7] if len(r) > 7 and r[7] is not None else "N/A",
            "farm_zone": r[8] if len(r) > 8 and r[8] is not None else "General Field",
            "notes": r[9] if len(r) > 9 and r[9] is not None else ""
        }
        for r in rows
    ]

def get_zone_health_summary() -> list:
    """
    Computes aggregated health diagnostics grouped by farm lot/zone.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT 
            farm_zone, 
            COUNT(*) as total_scans,
            SUM(is_healthy) as healthy_scans,
            AVG(damage_pct) as avg_damage
        FROM predictions 
        GROUP BY farm_zone
        ORDER BY total_scans DESC
    """)
    rows = c.fetchall()
    conn.close()
    
    summary = []
    for r in rows:
        zone = r[0] if r[0] else "General Field"
        total = r[1]
        healthy = r[2] if r[2] is not None else 0
        diseased = total - healthy
        avg_dmg = round(r[3], 1) if r[3] is not None else 0.0
        health_idx = round((healthy / total) * 100.0, 1) if total > 0 else 100.0
        summary.append({
            "zone": zone,
            "total_scans": total,
            "healthy": healthy,
            "diseased": diseased,
            "health_index": health_idx,
            "avg_damage_pct": avg_dmg
        })
    return summary

def export_history_json() -> str:
    """
    Serializes historical prediction records into JSON string for backup and data portability.
    """
    import json
    records = get_recent_predictions(limit=1000)
    return json.dumps(records, indent=2)
