"""
embedder.py
-----------
Converts text chunks into embeddings using Google's Generative AI
embedding model, and stores them in a local ChromaDB collection.
"""

import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from loader import load_and_chunk_pdf

# Load environment variables (GOOGLE_API_KEY) from .env
load_dotenv()

PERSIST_DIRECTORY = "chroma_db"  # folder where ChromaDB will save its data
COLLECTION_NAME = "pdf_study_assistant"


def get_embedding_model():
    """
    Returns the Google embedding model.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def build_vector_store(chunks, persist_directory=PERSIST_DIRECTORY, batch_size=90):
    """
    Embeds the given chunks in batches (to respect free-tier rate limits)
    and stores them in a local ChromaDB collection.
    """
    embedding_model = get_embedding_model()

    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}... (this calls the Gemini API)")

    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                collection_name=COLLECTION_NAME,
                persist_directory=persist_directory
            )
        else:
            vector_store.add_documents(batch)

        if i + batch_size < len(chunks):
            print("  Waiting 60 seconds to respect free-tier rate limit...")
            time.sleep(60)

    print(f"Vector store built and saved to '{persist_directory}/'")

    return vector_store


def load_existing_vector_store(persist_directory=PERSIST_DIRECTORY):
    """
    Loads an already-built ChromaDB collection from disk,
    without re-embedding anything (saves API calls).
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    return vector_store


# ---- Quick standalone test ----
if __name__ == "__main__":
    pdf_paths = [
        "ai_ml_interview_qa.pdf",
        "100_ml_interview_questions.pdf",
    ]

    chunks = load_and_chunk_pdf(pdf_paths)
    vector_store = build_vector_store(chunks)

    results = vector_store.similarity_search(
        "What is the difference between supervised, unsupervised, and reinforcement learning?",
        k=2
    )

    print("\n--- Sanity Check: Top 2 Similar Chunks ---")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (source: {doc.metadata.get('source')}):")
        print(doc.page_content[:200], "...")