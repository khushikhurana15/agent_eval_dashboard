"""
loader.py
---------
Loads one or more PDF files and splits them into overlapping text chunks
ready for embedding.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(pdf_paths, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Loads one or more PDFs and splits them into chunks.

    Args:
        pdf_paths: A single PDF path (str), or a list of PDF paths.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Characters shared between consecutive chunks.

    Returns:
        List[Document]: A list of LangChain Document objects,
                         each containing a chunk of text + metadata
                         (including which source PDF it came from).
    """

    # Allow passing either a single path (str) or a list of paths
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_pages = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # one Document per page, with 'source' in metadata
        print(f"Loaded '{pdf_path}' with {len(pages)} pages.")
        all_pages.extend(pages)

    # Split all pages (from all PDFs) into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(all_pages)

    print(f"Split into {len(chunks)} total chunks across {len(pdf_paths)} PDF(s).")

    return chunks


# ---- Quick standalone test ----
if __name__ == "__main__":
    # Test with both PDFs at once
    test_paths = [
        "ai_ml_interview_qa.pdf",
        "100_ml_interview_questions.pdf",  # apna actual doosri PDF ka filename daalna
    ]
    chunks = load_and_chunk_pdf(test_paths)

    print("\n--- Sample Chunk 0 ---")
    print(chunks[0].page_content)
    print("\nMetadata:", chunks[0].metadata)

    print("\n--- Sample Chunk from later in the list (likely second PDF) ---")
    print(chunks[-1].page_content)
    print("\nMetadata:", chunks[-1].metadata)