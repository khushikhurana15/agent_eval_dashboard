import re
from embedder import load_existing_vector_store
from rank_bm25 import BM25Okapi


def retrieve_relevant_chunks(query: str, k: int = 4):
    vector_store = load_existing_vector_store()
    results = vector_store.similarity_search(query, k=k)

    return results


def retrieve_with_scores(query: str, k: int = 4):
    vector_store = load_existing_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


CONFIDENCE_THRESHOLD = 0.8


def retrieve_with_confidence_gate(query: str, k: int = 4, threshold: float = CONFIDENCE_THRESHOLD):

    results = retrieve_with_scores(query, k=k)

    if not results:
        return {"passed": False, "chunks": [], "best_distance": None}

    best_distance = results[0][1]

    if best_distance > threshold:
        return {"passed": False, "chunks": [], "best_distance": best_distance}

    chunks = [doc for doc, score in results]
    return {"passed": True, "chunks": chunks, "best_distance": best_distance}


STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "how", "does", "do",
    "why", "when", "where", "which", "who", "explain", "describe"
}


def _tokenize(text: str):
    """
    Tokenizes text into lowercase words, stripping punctuation and
    common question-words/stop-words. This keeps BM25 focused on the
    meaningful terms (e.g. "reg_lambda") rather than diluting the
    score with filler words like "what" and "is".
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _get_all_documents():
    """
    Pulls every chunk currently stored in ChromaDB, so we can build
    a BM25 keyword index from the full corpus (not just a search result).
    """
    vector_store = load_existing_vector_store()
    raw = vector_store.get()  # returns dict with 'documents' and 'metadatas'

    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return docs


def _build_bm25_index():
    """
    Builds a BM25 index over every chunk in the corpus.
    """
    docs = _get_all_documents()
    tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs


def hybrid_search(query: str, k: int = 4, rrf_constant: int = 60):
    """
    Combines vector similarity search and BM25 keyword search using
    Reciprocal Rank Fusion (RRF) — a chunk's final score is based on
    its RANK in each method, not the raw (incomparable) score values.

    Returns:
        List[Document]: top-k chunks by combined rank.
    """
    # --- Vector search ranking ---
    vector_store = load_existing_vector_store()
    vector_results = vector_store.similarity_search(query, k=50)  # wider net so strong BM25-only matches aren't excluded

    # --- BM25 keyword search ranking ---
    bm25, all_docs = _build_bm25_index()
    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranked_docs = [all_docs[i] for i in bm25_ranked_indices[:50]]

    # --- Reciprocal Rank Fusion ---
    rrf_scores = {}

    for rank, doc in enumerate(vector_results):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + rrf_constant)

    for rank, doc in enumerate(bm25_ranked_docs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + rrf_constant)

    all_candidates = {doc.page_content: doc for doc in vector_results + bm25_ranked_docs}
    ranked = sorted(all_candidates.items(), key=lambda item: rrf_scores.get(item[0], 0), reverse=True)

    return [doc for _, doc in ranked[:k]]


if __name__ == "__main__":
    test_queries = [
        "What is the difference between bagging and boosting?",
        "What is a Support Vector Machine?",
        "What is the capital of France?",
    ]

    for q in test_queries:
        result = retrieve_with_confidence_gate(q)
        status = "PASS" if result["passed"] else "GATED (rejected)"
        print(f"\nQuery: {q}")
        print(f"Status: {status} | Best distance: {result['best_distance']}")