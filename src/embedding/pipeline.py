from src.ingestion import load_document, load_codebase
from src.chunking import chunk_document, chunk_codebase
from .embedder import Embedder
from .store import VectorStore


def ingest_document(file_path: str, store: VectorStore, embedder: Embedder) -> int:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Returns the number of chunks ingested. Called by the Streamlit UI on file
    upload (Phase 7) and directly by the sanity-check CLI / eval harness.
    """
    pages = load_document(file_path)
    chunks = chunk_document(pages)
    if not chunks:
        return 0
    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    store.add_chunks(chunks, embeddings)
    return len(chunks)


def ingest_codebase(directory_path: str, store: VectorStore, embedder: Embedder) -> int:
    """
    AST-based code ingestion: load → AST-chunk → embed → store.
    Uses the same embed/store steps as ingest_document; only loading and chunking differ.
    Returns the number of code chunks ingested.
    """
    files = load_codebase(directory_path)
    chunks = chunk_codebase(files)
    if not chunks:
        return 0
    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    store.add_chunks(chunks, embeddings)
    return len(chunks)
