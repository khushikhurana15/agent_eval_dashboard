# backend/database.py
#
# SQLite has no separate server — it's just a file (results.db) that lives
# in the project root. get_connection() opens that file; init_db() creates
# the two tables the first time the app starts (CREATE TABLE IF NOT EXISTS
# is safe to call every startup — it's a no-op if the tables already exist).

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "results.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    # row_factory lets us read columns by name (row["accuracy"]) instead of
    # by position (row[3]) — much less error-prone as the schema grows.
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # One row per "Run new eval" click. Powers the trend chart and the
    # top-of-dashboard summary cards.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            total_questions INTEGER,
            completed_questions INTEGER,
            stopped_early_reason TEXT,
            accuracy REAL,
            tool_selection_correct_rate REAL,
            hallucination_rate REAL,
            avg_latency_seconds REAL
        )
    """)

    # One row per question, per run. Powers the expandable results table.
    # tools_used and reasoning_trace are stored as JSON text (SQLite has no
    # native list/object type) and parsed back into Python on the way out.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES eval_runs(id),
            question_id TEXT NOT NULL,
            category TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_tool TEXT,
            tools_used TEXT,
            tool_correct INTEGER,
            rag_confidence_distance REAL,
            rag_gated INTEGER,
            final_answer TEXT,
            answer_correct INTEGER,
            hallucination_flag INTEGER,
            passed INTEGER,
            latency_seconds REAL,
            reasoning_trace TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()