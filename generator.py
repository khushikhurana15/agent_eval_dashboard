import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retriever import retrieve_with_scores, hybrid_search

load_dotenv()

NOT_FOUND_ANSWER = (
    "I couldn't find anything relevant to that in the document, "
    "so I didn't generate an answer."
)

PROMPT_TEMPLATE = """You are a helpful study assistant answering questions
based ONLY on the provided context from an AI/ML interview prep document.

Rules:
- Answer using only the information in the context below.
- If the context doesn't contain enough information to answer,
  say "I don't have enough information in the document to answer that."
- Keep answers clear and concise, as if explaining to someone studying for an interview.
- Formatting: if the answer naturally involves a comparison, a list of
  types/categories, or multiple structured attributes (e.g. "X vs Y",
  pros/cons, steps), format it as a Markdown table if that fits cleanly,
  or as clear bullet points otherwise. Do not write dense unbroken
  paragraphs when the content is naturally structured or comparative.
{history_section}
Context:
{context}

Question: {question}

Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Given the conversation below and a follow-up question,
rewrite the follow-up question as a single standalone question that can be
understood without the conversation. Keep it short. If the question is already
standalone, return it unchanged. Return ONLY the rewritten question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

MAX_HISTORY_TURNS = 4


def format_history(history):
    lines = []
    for past_question, past_answer in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"Student: {past_question}")
        lines.append(f"Assistant: {past_answer}")
    return "\n".join(lines)


def condense_question(question: str, history):
    if not history:
        return question

    prompt = CONDENSE_PROMPT_TEMPLATE.format(
        history=format_history(history),
        question=question
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    standalone = response.content.strip()

    return standalone if standalone else question


def get_llm():
    # NOTE: this is intentionally the SAME provider/model as agent_core.py's
    # get_agent(), so the whole pipeline (agent reasoning + RAG answer
    # synthesis) draws from one quota/rate-limit instead of two different
    # ones. Embeddings (embedder.py) stay on Google — the existing Chroma
    # vector store was built with Gemini's embedding model, and switching
    # that would make stored embeddings incompatible with new query
    # embeddings, silently breaking retrieval.
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


def generate_answer(question: str, k: int = 4, history=None, max_distance: float = None, use_hybrid: bool = False):
    history = history or []

    # Step 1: Rewrite follow-up questions into standalone ones so retrieval
    # matches the right chunks ("what about boosting?" retrieves nothing useful)
    standalone_question = condense_question(question, history)

    # Step 2: Retrieve relevant chunks — either pure vector search, or
    # hybrid (vector + BM25 keyword matching, better for exact terms
    # like hyperparameter names, e.g. "reg_lambda")
    if use_hybrid:
        chunks = hybrid_search(standalone_question, k=k)
        # Hybrid search doesn't return distance scores (it's rank-fused),
        # so we use placeholder scores for display consistency.
        scores = [0.0] * len(chunks)
    else:
        scored_chunks = retrieve_with_scores(standalone_question, k=k)
        chunks = [doc for doc, _ in scored_chunks]
        scores = [score for _, score in scored_chunks]

    # best_distance is kept regardless of gating outcome — eval/observability
    # needs the actual number (e.g. "how close did it come") even when the
    # gate rejects the retrieval, not just a pass/fail boolean.
    best_distance = min(scores) if scores else None

    # Step 3: Confidence gate — only applies in vector-search mode, since
    # hybrid mode doesn't produce comparable distance scores.
    if max_distance is not None and (not scores or min(scores) > max_distance):
        return {
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "scores": [],
            "gated": True,
            "best_distance": best_distance,
            "standalone_question": standalone_question
        }

    # Step 4: Combine chunk text into one context block
    context = "\n\n---\n\n".join([doc.page_content for doc in chunks])

    # Step 5: Build the final prompt, including recent conversation so the
    # answer can reference it — facts still must come from the context
    if history:
        history_section = (
            "\nConversation so far (for reference and tone only — "
            "facts must still come from the context):\n"
            f"{format_history(history)}\n"
        )
    else:
        history_section = ""

    prompt = PROMPT_TEMPLATE.format(
        history_section=history_section,
        context=context,
        question=question
    )

    # Step 6: Call the LLM
    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": chunks,
        "scores": scores,
        "gated": False,
        "best_distance": best_distance,
        "standalone_question": standalone_question
    }


# ---- Quick standalone test ----
if __name__ == "__main__":
    test_question = "What is the difference between bagging and boosting?"

    result = generate_answer(test_question)

    print(f"Question: {test_question}\n")
    print(f"Answer:\n{result['answer']}\n")

    print("--- Sources used ---")
    for i, doc in enumerate(result["sources"]):
        print(f"Source {i+1} (page {doc.metadata.get('page_label')}): {doc.page_content[:100]}...")