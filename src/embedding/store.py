import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Persistent ChromaDB-backed vector store.

    ChromaDB is embedded/local with zero infra overhead — the right tradeoff at
    this data volume. Qdrant/Pinecone would be over-engineering here.

    Metadata values must be str | int | float | bool per Chroma's constraint;
    None page values (from .txt files) are stored as -1.
    """

    def __init__(
        self,
        persist_dir: str = "./vectorstore",
        collection_name: str = "docmind",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert chunks so re-ingesting the same document is idempotent."""
        self._col.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"] if c["page"] is not None else -1,
                    "chunk_index": c["chunk_index"],
                }
                for c in chunks
            ],
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def query(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"][0]:
            return []
        return [
            {
                "chunk_id": cid,
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": dist,
            }
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        return self._col.count()

    def clear_all(self) -> None:
        """Delete and recreate the collection, leaving it empty but usable."""
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=self._col.name,
            metadata={"hnsw:space": "cosine"},
        )
