# backend/database.py
#
# Migrated from SQLite to Postgres for deployment: a deployed backend's
# filesystem is usually ephemeral (resets on restart/redeploy), so results
# stored in a local SQLite file would vanish. Postgres via a hosted provider
# (Neon, Supabase, Render Postgres) persists independently of the app's
# filesystem. DATABASE_URL is the standard env var name all of these
# providers use, so no provider-specific code is needed here.
#
# For LOCAL development without a hosted Postgres, run one via Docker:
#   docker run -e POSTGRES_PASSWORD=localdev -p 5432:5432 postgres
# and set DATABASE_URL=postgresql://postgres:localdev@localhost:5432/postgres

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# This module is imported first (by main.py, before agent/generator modules
# that also call load_dotenv()), and DATABASE_URL is read at import time —
# so .env needs to be loaded right here, not assumed to already be loaded
# by something imported earlier.
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # One row per "Run new eval" click. Powers the trend chart and the
    # top-of-dashboard summary cards.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id SERIAL PRIMARY KEY,
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
    # tools_used and reasoning_trace are stored as JSON text (kept as TEXT,
    # same as the SQLite version, rather than switching to Postgres's native
    # JSONB — no query needs to filter inside these fields, so there's no
    # benefit to the native type here, and TEXT keeps main.py's
    # json.loads()/json.dumps() calls unchanged).
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id SERIAL PRIMARY KEY,
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
    cur.close()
    conn.close()