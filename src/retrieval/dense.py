from src.embedding.embedder import Embedder
from src.embedding.store import VectorStore


def dense_retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    k: int = 5,
) -> list[dict]:
    """Embed query, return top-k chunks by cosine similarity from ChromaDB."""
    return store.query(embedder.embed_query(query), k=k)
