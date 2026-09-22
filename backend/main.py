# backend/main.py
#
# Local dev: uvicorn backend.main:app --reload  (from the project root)
# Then open http://127.0.0.1:8000/docs for the interactive API explorer.
#
# Deployed: the platform's start command runs the same app; DATABASE_URL and
# FRONTEND_ORIGIN are read from environment variables instead of being
# hardcoded, so the exact same code runs in both places.

import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db, get_connection
from .eval_runner import run_full_eval

app = FastAPI(title="Agent Eval & Observability Dashboard API")

# In dev, the React dev server runs on a different port (5173) than this API
# (8000) — browsers block cross-port requests unless the server explicitly
# allows them. In production, the deployed frontend's real domain needs to
# be added here too, via the FRONTEND_ORIGIN env var (comma-separated if
# there's more than one, e.g. a Render static site *and* a custom domain).
_extra_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", *_extra_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimum time between eval runs. Without this, anyone visiting a publicly
# deployed dashboard could hammer "Run new eval" repeatedly, burning through
# the LLM provider's token/request quota (and, on Groq, real money) for no
# reason. 6 hours is arbitrary but generous for a demo — tune via env var.
MIN_SECONDS_BETWEEN_RUNS = int(os.getenv("MIN_SECONDS_BETWEEN_RUNS", str(6 * 60 * 60)))


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    """Used by the hosting platform's health check, not meant for humans."""
    return {"status": "ok"}


def _seconds_since_last_run(cur) -> float | None:
    """None if there has never been a run (i.e. no cooldown to enforce)."""
    cur.execute("SELECT started_at FROM eval_runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    last_started = datetime.fromisoformat(row["started_at"])
    if last_started.tzinfo is None:
        # Defensive: eval_runner.py always writes tz-aware timestamps, but
        # guard against a naive one anyway rather than crashing the whole
        # rate-limit check (and with it, the ability to run evals at all).
        last_started = last_started.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_started).total_seconds()


@app.post("/api/eval/run")
def trigger_eval_run():
    """
    Runs the full golden dataset against the agent and returns the new
    run's id once finished. Each question is several LLM calls, so 27-30
    questions can take a couple of minutes — the frontend's "Run new eval"
    button should show a loading state while this is in flight.

    Rate-limited (see MIN_SECONDS_BETWEEN_RUNS) so a public deployment can't
    be spammed into burning through API quota.
    """
    conn = get_connection()
    cur = conn.cursor()
    elapsed = _seconds_since_last_run(cur)
    cur.close()
    conn.close()

    if elapsed is not None and elapsed < MIN_SECONDS_BETWEEN_RUNS:
        wait_minutes = round((MIN_SECONDS_BETWEEN_RUNS - elapsed) / 60)
        raise HTTPException(
            status_code=429,
            detail=(
                f"A run happened recently. Please wait about {wait_minutes} "
                "more minute(s) before starting another one."
            ),
        )

    run_id = run_full_eval()
    return {"run_id": run_id, "status": "completed"}


@app.get("/api/eval/runs")
def list_runs():
    """Every completed run's summary metrics, oldest first — feeds the trend chart."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eval_runs WHERE finished_at IS NOT NULL ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/api/eval/runs/latest")
def latest_run():
    """Most recent completed run's summary — feeds the top metric cards."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eval_runs WHERE finished_at IS NOT NULL ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No completed eval runs yet")
    return dict(row)


@app.get("/api/eval/runs/{run_id}")
def run_detail(run_id: int):
    """Every individual question result for one run, with full reasoning
    trace — feeds the expandable results table."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eval_runs WHERE id = %s", (run_id,))
    run_row = cur.fetchone()
    if not run_row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    cur.execute("SELECT * FROM eval_results WHERE run_id = %s ORDER BY id ASC", (run_id,))
    result_rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in result_rows:
        item = dict(row)
        item["tools_used"] = json.loads(item["tools_used"] or "[]")
        item["reasoning_trace"] = json.loads(item["reasoning_trace"] or "[]")
        results.append(item)

    return {"run": dict(run_row), "results": results}