# Agent Eval & Observability Dashboard

A tool that tests, logs, and visualizes the decisions made by an LLM-based research agent — built to demonstrate both AI evaluation skills and full-stack engineering.

The agent itself (a multi-source research assistant with access to a PDF knowledge base, web search, and a calculator) already existed as a separate project. This dashboard wraps it with an evaluation and observability layer: a golden dataset of test questions, an eval runner that scores the agent's behavior, and a React dashboard to visualize results and trends over time.

## Architecture

```
golden_dataset.json  →  eval runner (FastAPI)  →  your agent (Groq + tools)
                              ↓
                         SQLite (results.db)
                              ↓
                      REST API  →  React dashboard
```

- **`agent/`** — the research agent (LangChain `create_agent`, Groq LLM) and its three tools: `rag_tool` (hybrid confidence-gated PDF search), `web_search_tool`, `calculator_tool`.
- **`generator.py` / `retriever.py` / `embedder.py` / `loader.py`** — the RAG pipeline `rag_tool` calls into. Note: the LLM (reasoning + RAG answer synthesis) runs on Groq, but **embeddings stay on Google's Gemini embedding model**, because the existing `chroma_db/` vector store was built with it — switching embedding models would silently break retrieval.
- **`golden_dataset.json`** — 27 hand-written test questions across 5 categories, each tagged with an expected tool, an expected answer (or summary), and a scoring method.
- **`backend/`** — FastAPI service that runs the golden dataset against the agent, scores each answer, and stores results in SQLite.
- **`frontend/`** — React + Recharts dashboard: summary metrics, an accuracy/hallucination trend chart, and an expandable results table with full reasoning traces.

## Setup

### 1. Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key       # agent reasoning + RAG answer synthesis
GOOGLE_API_KEY=your_google_key   # embeddings only (query-time vector search)
```

### 2. Backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Runs at `http://127.0.0.1:8000`. Visit `/docs` for the interactive API explorer.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173`. The Vite dev server proxies `/api/*` requests to the backend, so no CORS setup is needed locally.

## The golden dataset

27 questions across 5 categories, grounded in the actual PDF content loaded into the agent's vector store (not generic guesses):

| Category | Count | What it tests |
|---|---|---|
| `easy_rag` | 10 | Clearly covered by the PDFs — agent should use `rag_tool` and answer correctly |
| `web_search` | 5 | Live/current info not in the PDFs — agent should use `web_search_tool` (answers are non-deterministic, so only tool choice is scored) |
| `calculator` | 4 | Pure arithmetic — exact-match scored |
| `ambiguous` | 5 | Borderline cases where more than one tool choice is reasonable, or retrieval may be low-confidence |
| `unanswerable` | 3 | No tool can genuinely answer these — agent should decline rather than guess |

## Scoring methodology

No RAGAS or LLM-judge — simple, transparent heuristics, on the theory that for a 27-question eval suite, being able to explain *exactly* why a score came out the way it did matters more than marginal precision:

- **`exact_match`** — used for calculator questions; checks the expected number appears in the answer (normalizing comma thousand-separators, e.g. "8,100" still matches "8100").
- **`semantic_match`** — used for `easy_rag` questions; keyword-overlap between the answer and an expected-answer summary, with light suffix-stripping so "neurons"/"neuron" or "ensembles"/"ensemble" don't count as mismatches. Threshold: 40% overlap.
- **`refusal_check`** — used for `unanswerable` questions; checks the answer contains one of a curated list of refusal phrases.
- **`tool_only`** — used for `web_search` and `ambiguous` questions where the correct *answer* is non-deterministic or content-dependent; only tool selection is auto-scored.

**Hallucination detection** is a direct application of the RAG confidence gate already built into the agent: if `rag_tool` returned `LOW_CONFIDENCE` (gate triggered) but the agent still gave a confident, non-refusal answer instead of admitting the gap, that's flagged as a hallucination.

## Latest results

27/27 questions completed, **96% accuracy**, **100% tool selection correctness**, **0% hallucination rate**, ~17s average latency per question.

## Findings worth highlighting

Building the eval layer surfaced several real issues — in the agent, in the eval framework itself, and in the scoring logic — that wouldn't have been visible without it:

- **A recency-ranking gap in the agent.** In earlier testing, the agent sometimes returned an outdated result (e.g. an old T20 World Cup winner) for questions needing the *most recent* answer, because it had no explicit recency-ranking logic — it just took the first plausible search result. Worth revisiting the system prompt to explicitly instruct cross-checking dates on time-sensitive queries.
- **Redundant tool-call loops inflate latency.** Several questions (e.g. "latest Python version", "current USD/INR rate", a benchmark-accuracy lookup that took **9 separate web searches and 99 seconds**) show the agent repeatedly re-searching to chase a more precise or more recent number instead of stopping at a reasonably confident answer. Invisible in normal chat use; obvious once per-question latency is logged.
- **The `eval_log` side-channel could have leaked data across questions.** `rag_tool`'s confidence log is a module-level list that persists for the whole process. The eval runner snapshots its length *before* each question and only reads newly-appended entries — without that, a slow multi-tool question could have picked up confidence data from a different question.
- **Two independent safety nets can conflict.** A per-question `try/except` in `rag_tool` (added to surface real error messages instead of LangChain's generic "please fix the error" wrapper) initially *swallowed* quota/rate-limit errors instead of letting them propagate — which silently defeated the eval runner's run-level circuit breaker that's supposed to stop the whole run early once quota is exhausted, instead of burning through every remaining question on guaranteed failures. Fixed by re-raising quota-shaped errors specifically.
- **Keyword-based refusal detection is a whack-a-mole problem.** Across separate runs, two different genuine refusals used phrasings the hardcoded refusal-phrase list didn't cover ("I don't have access to..." and "I don't have any information about..."). Both were patched after discovery, but this is a structural limitation, not a one-off bug — see Known Limitations below.

## Known limitations

**`REFUSAL_PHRASES` uses substring matching, which can misfire on confident answers.** A phrase like `"no such"` was added to catch a real refusal, but could in principle flag a substantive technical answer that happens to contain it non-refusal-sense (e.g. "there is no such universal regularization technique, it depends on the model"). Not observed in the current 27-question set, but worth re-checking as the dataset grows. A production system would use a small LLM-judge call for refusal detection instead of substring matching — cheap, since it's only classifying 3 short answers per run, but adds complexity not justified at this scale.

**`refusal_check` is too strict for nuanced "should I" questions.** The `unk_02` case ("Should I switch careers from data science to product management?") is scored as a failure because the agent gives balanced, hedged advice instead of a flat refusal. Arguably the hedged answer is *better* behavior than declining outright — this reflects a genuine design tension between "I have no way to know this" cases (where refusal is clearly correct) and subjective-judgment questions (where a thoughtful, non-authoritative answer is arguably the right response, but doesn't fit a binary refusal check).

**Semantic-match scoring rewards word overlap, not meaning.** The lightweight suffix-stripping normalizer isn't a real stemmer (e.g. "improve" vs "improving" still don't align, due to the silent-e case), so a correct, well-paraphrased answer can still score lower than a weaker one that happens to reuse more of the expected summary's exact wording.

**No detection for partial hallucination.** The hallucination flag only triggers when the RAG confidence gate explicitly rejected a retrieval. It doesn't catch cases where retrieval succeeded but the LLM still added specific, ungrounded details beyond what the retrieved context supported (e.g. inventing precise parameter counts or memory figures in an otherwise-correct conceptual answer). A more robust check would compare the final answer's specific claims against the tool output text directly.

## Possible next steps (not implemented)

- LLM-judge scoring for `refusal_check` and `semantic_match`, to reduce false negatives from phrase/keyword matching
- Claim-level grounding check against retrieved context, to catch partial hallucination
- Explicit recency-ranking instructions in the agent's system prompt for time-sensitive web searches