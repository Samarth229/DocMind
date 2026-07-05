from src.ingestion import load_document
from src.chunking import chunk_document
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
