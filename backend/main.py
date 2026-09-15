# backend/main.py
#
# Run this from the PROJECT ROOT (not inside backend/), same as your
# `python -m agent.agent_core` convention, so the "agent" package resolves:
#
#     uvicorn backend.main:app --reload
#
# Then open http://127.0.0.1:8000/docs for the interactive API explorer.

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db, get_connection
from .eval_runner import run_full_eval

app = FastAPI(title="Agent Eval & Observability Dashboard API")

# The React dev server runs on a different port (5173 for Vite, 3000 for
# create-react-app) than this API (8000) — browsers block cross-port
# requests by default unless the server explicitly allows them. This is
# what CORS middleware does.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/api/eval/run")
def trigger_eval_run():
    """
    Runs the full golden dataset against the agent and returns the new
    run's id once finished. Each question is several LLM calls, so 27-30
    questions can take a couple of minutes — the frontend's "Run new eval"
    button should show a loading state while this is in flight.
    """
    run_id = run_full_eval()
    return {"run_id": run_id, "status": "completed"}


@app.get("/api/eval/runs")
def list_runs():
    """Every completed run's summary metrics, oldest first — feeds the trend chart."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM eval_runs WHERE finished_at IS NOT NULL ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/api/eval/runs/latest")
def latest_run():
    """Most recent completed run's summary — feeds the top metric cards."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM eval_runs WHERE finished_at IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No completed eval runs yet")
    return dict(row)


@app.get("/api/eval/runs/{run_id}")
def run_detail(run_id: int):
    """Every individual question result for one run, with full reasoning
    trace — feeds the expandable results table."""
    conn = get_connection()
    run_row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not run_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    result_rows = conn.execute(
        "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id ASC", (run_id,)
    ).fetchall()
    conn.close()

    results = []
    for row in result_rows:
        item = dict(row)
        item["tools_used"] = json.loads(item["tools_used"] or "[]")
        item["reasoning_trace"] = json.loads(item["reasoning_trace"] or "[]")
        results.append(item)

    return {"run": dict(run_row), "results": results}