import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "legalvalidate.db")

def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH):
    """Initializes the SQLite database tables if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                document_type TEXT,
                is_legal INTEGER,
                classification TEXT,
                risk_count INTEGER,
                summary TEXT,
                full_result_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename_a TEXT NOT NULL,
                filename_b TEXT NOT NULL,
                comparison_count INTEGER,
                full_result_json TEXT NOT NULL
            )
        """)
        conn.commit()

def save_analysis(filename: str, final_state: dict, db_path: str = DB_PATH) -> int:
    """Saves a completed single document review result to the database."""
    init_db(db_path)
    timestamp = datetime.now().isoformat(timespec='seconds')
    document_type = final_state.get("document_type", "Unknown")
    is_legal = 1 if final_state.get("is_legal") else 0
    classification = "Legal" if is_legal else "Non-Legal"
    risks = final_state.get("risks", [])
    risk_count = len(risks) if isinstance(risks, list) else 0
    summary = final_state.get("summary", "") or ""
    full_result_json = json.dumps(final_state)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (timestamp, filename, document_type, is_legal, classification, risk_count, summary, full_result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, filename, document_type, is_legal, classification, risk_count, summary, full_result_json))
        conn.commit()
        return cursor.lastrowid

def save_comparison(filename_a: str, filename_b: str, comp_results: dict, db_path: str = DB_PATH) -> int:
    """Saves a completed dual-document comparison result to the database."""
    init_db(db_path)
    timestamp = datetime.now().isoformat(timespec='seconds')
    comparisons = comp_results.get("comparisons", [])
    comparison_count = len(comparisons) if isinstance(comparisons, list) else 0
    full_result_json = json.dumps(comp_results)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO comparisons (timestamp, filename_a, filename_b, comparison_count, full_result_json)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, filename_a, filename_b, comparison_count, full_result_json))
        conn.commit()
        return cursor.lastrowid

def get_all_analyses(db_path: str = DB_PATH) -> list:
    """Retrieves all past single document analyses from the database, ordered newest first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, filename, document_type, is_legal, classification, risk_count, summary FROM analyses ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_analysis_by_id(analysis_id: int, db_path: str = DB_PATH) -> dict:
    """Retrieves a specific single document analysis by ID including full_result_json."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        if row:
            res = dict(row)
            res["full_result"] = json.loads(res["full_result_json"])
            return res
        return None

def get_all_comparisons(db_path: str = DB_PATH) -> list:
    """Retrieves all past dual document comparisons from the database, ordered newest first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, filename_a, filename_b, comparison_count FROM comparisons ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_comparison_by_id(comp_id: int, db_path: str = DB_PATH) -> dict:
    """Retrieves a specific comparison record by ID including full_result_json."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM comparisons WHERE id = ?", (comp_id,))
        row = cursor.fetchone()
        if row:
            res = dict(row)
            res["full_result"] = json.loads(res["full_result_json"])
            return res
        return None
