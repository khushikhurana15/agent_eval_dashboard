# backend/eval_runner.py
#
# Orchestrates one full eval run: for each golden_dataset.json question,
# call the real agent, capture its tool choice + confidence + latency,
# score it, and write a row to eval_results. At the end, aggregate into
# one eval_runs summary row.

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.agent_core import run_agent, parse_agent_trace
from agent.tools.rag_tool import eval_log as rag_eval_log

from .database import get_connection
from .scoring import score_question

# Defaults to the real 27-question golden_dataset.json, but can be pointed
# at a smaller file (e.g. for quota-safe testing) via an env var —
# no need to edit or temporarily rename the real dataset file:
#   GOLDEN_DATASET_FILE=golden_dataset_test.json uvicorn backend.main:app --reload
GOLDEN_DATASET_PATH = Path(__file__).parent.parent / os.getenv(
    "GOLDEN_DATASET_FILE", "golden_dataset.json"
)


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    return data["questions"]


def _bool_to_int(value):
    """SQLite has no real boolean type — store True/False/None as 1/0/NULL."""
    if value is None:
        return None
    return 1 if value else 0


def run_single_question(golden_item):
    question = golden_item["question"]

    # Snapshot eval_log's length BEFORE calling the agent. eval_log is a
    # module-level list shared across the whole process, so this is the
    # only safe way to know which entries belong to THIS question — slicing
    # from before_len onward, rather than reading the whole list.
    before_len = len(rag_eval_log)

    start = time.perf_counter()
    messages = run_agent(question)
    latency_seconds = time.perf_counter() - start

    final_answer, steps = parse_agent_trace(messages)
    tools_used = [s["tool_name"] for s in steps]

    new_rag_entries = rag_eval_log[before_len:]
    rag_confidence_distance = new_rag_entries[-1]["rag_confidence_distance"] if new_rag_entries else None
    rag_gated = new_rag_entries[-1]["gated"] if new_rag_entries else None

    scores = score_question(golden_item, tools_used, final_answer, rag_gated)

    expected_tool_display = golden_item.get("expected_tool") or ", ".join(
        golden_item.get("acceptable_tools", [])
    )

    return {
        "question_id": golden_item["id"],
        "category": golden_item["category"],
        "question": question,
        "expected_tool": expected_tool_display,
        "tools_used": tools_used,
        "tool_correct": scores["tool_correct"],
        "rag_confidence_distance": rag_confidence_distance,
        "rag_gated": rag_gated,
        "final_answer": final_answer,
        "answer_correct": scores["answer_correct"],
        "hallucination_flag": scores["hallucination_flag"],
        "passed": scores["passed"],
        "latency_seconds": latency_seconds,
        "reasoning_trace": steps,
    }


def _is_quota_error(exc: Exception) -> bool:
    """
    Detects a Gemini free-tier daily quota exhaustion (RESOURCE_EXHAUSTED /
    HTTP 429). Once this happens, every remaining question is guaranteed to
    fail the same way — there's no point burning minutes waiting through
    each one's internal retry before giving up.
    """
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def run_full_eval():
    golden_questions = load_golden_dataset()
    conn = get_connection()
    cur = conn.cursor()

    started_at = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT INTO eval_runs (started_at, total_questions) VALUES (?, ?)",
        (started_at, len(golden_questions)),
    )
    run_id = cur.lastrowid
    conn.commit()

    results = []
    stopped_early_reason = None

    for item in golden_questions:
        try:
            result = run_single_question(item)
        except Exception as e:
            if _is_quota_error(e):
                # Stop the whole run now rather than attempting (and waiting
                # out the retry delay for) every remaining question when
                # they're all guaranteed to fail identically. Include the
                # REAL exception text — the pipeline now spans two providers
                # (Groq for reasoning/RAG synthesis, Google for embeddings),
                # so a quota error could come from either. Hardcoding one
                # provider's name here would hide which one actually failed.
                stopped_early_reason = (
                    f"Stopped after {len(results)}/{len(golden_questions)} questions "
                    f"due to a quota/rate-limit error: {type(e).__name__}: {e}"
                )
                break

            # A non-quota error (e.g. a bug specific to this one question)
            # shouldn't kill the whole run — log it and keep going.
            result = {
                "question_id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected_tool": item.get("expected_tool", ""),
                "tools_used": [],
                "tool_correct": False,
                "rag_confidence_distance": None,
                "rag_gated": None,
                "final_answer": f"ERROR: {e}",
                "answer_correct": False,
                "hallucination_flag": False,
                "passed": False,
                "latency_seconds": None,
                "reasoning_trace": [],
            }
        results.append(result)

        cur.execute(
            """
            INSERT INTO eval_results (
                run_id, question_id, category, question, expected_tool,
                tools_used, tool_correct, rag_confidence_distance, rag_gated,
                final_answer, answer_correct, hallucination_flag, passed,
                latency_seconds, reasoning_trace, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, result["question_id"], result["category"], result["question"],
                result["expected_tool"], json.dumps(result["tools_used"]),
                _bool_to_int(result["tool_correct"]), result["rag_confidence_distance"],
                _bool_to_int(result["rag_gated"]), result["final_answer"],
                _bool_to_int(result["answer_correct"]), _bool_to_int(result["hallucination_flag"]),
                _bool_to_int(result["passed"]), result["latency_seconds"],
                json.dumps(result["reasoning_trace"]), datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()  # commit after each question so partial progress isn't lost

    # ---- aggregate into the run-level summary row ----
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    tool_correct_count = sum(1 for r in results if r["tool_correct"])
    hallucination_count = sum(1 for r in results if r["hallucination_flag"])
    latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]

    accuracy = passed_count / total if total else 0
    tool_rate = tool_correct_count / total if total else 0
    hallucination_rate = hallucination_count / total if total else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    cur.execute(
        """
        UPDATE eval_runs
        SET finished_at = ?, completed_questions = ?, stopped_early_reason = ?,
            accuracy = ?, tool_selection_correct_rate = ?,
            hallucination_rate = ?, avg_latency_seconds = ?
        WHERE id = ?
        """,
        (
            datetime.now(timezone.utc).isoformat(), total, stopped_early_reason,
            accuracy, tool_rate, hallucination_rate, avg_latency, run_id,
        ),
    )
    conn.commit()
    conn.close()

    return run_id