# tools/rag_tool.py

from langchain.tools import tool
from generator import generate_answer

# Side-channel for eval/observability — read by the eval runner AFTER
# an agent run completes. Never goes back into the LLM's context, so it
# can't influence the agent's own reasoning or final answer.
eval_log = []

@tool
def rag_tool(question: str) -> str:
    """
    Use this tool to answer questions about the content of the uploaded 
    PDF documents (AI/ML interview prep material, ML fundamentals, 
    deep learning, NLP/LLMs, MLOps topics).
    Do NOT use this for current events, live/real-time information, 
    or general math calculations.
    Input should be the user's question as plain text.
    """
    try:
        result = generate_answer(
            question,
            k=4,
            history=[],           # agent har call ko fresh treat karega
            max_distance=0.8,     # tumhara existing threshold
            use_hybrid=False      # confidence gating ke liye zaroori hai
        )
    except Exception as e:
        # Surface the REAL error instead of letting LangChain's generic
        # "please fix the error and try again" message hide it.
        error_text = f"{type(e).__name__}: {e}"
        eval_log.append({
            "question": question,
            "rag_confidence_distance": None,
            "gated": None,
            "error": error_text,
        })

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            # Don't swallow this here — re-raise so it propagates up through
            # run_agent() to the eval runner's run-level circuit breaker.
            # Catching it here instead would let the agent quietly keep
            # going (e.g. falling back to web_search_tool), which defeats
            # the whole point of stopping the run early: every remaining
            # question that needs rag_tool would still burn a full agent
            # invocation (several LLM calls) just to hit the same dead end.
            raise

        return f"RAG_TOOL_ERROR: {error_text}"

    # Log confidence separately for the eval dashboard — never appended
    # to the string that goes back to the LLM.
    eval_log.append({
        "question": question,
        "rag_confidence_distance": result.get("best_distance"),
        "gated": result["gated"],
    })

    if result["gated"]:
        return (
            "LOW_CONFIDENCE: No relevant information found in the PDF documents "
            "for this question. Consider trying a different tool or informing "
            "the user that this information isn't available in the documents."
        )

    return result["answer"]