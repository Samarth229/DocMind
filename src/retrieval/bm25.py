from rank_bm25 import BM25Okapi
from src.embedding.store import VectorStore


def _tokenize(text: str) -> list[str]:
    # Simple whitespace + lowercase tokenization.
    # Upgrade point: a proper tokenizer (e.g. nltk word_tokenize with stopword removal)
    # would improve BM25 precision but adds a dependency for marginal gain at this scale.
    return text.lower().split()


class BM25Index:
    """
    In-memory BM25 index over the full chunk corpus.

    Scaling limitation: the index is rebuilt from all chunks at startup (or
    after new documents are ingested). Rebuild cost is O(corpus size) — fine
    for a personal-scale RAG but would need a persistent inverted index (e.g.
    Elasticsearch) for large corpora. Worth flagging in interviews.
    """

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._index: BM25Okapi | None = None
        if chunks:
            self._build(chunks)

    def _build(self, chunks: list[dict]) -> None:
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(tokenized)

    def rebuild_index(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._build(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self._index or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {**chunk, "score": float(score)}
            for score, chunk in ranked[:k]
            if score > 0  # skip chunks with zero overlap
        ]


def build_bm25_from_store(store: VectorStore) -> BM25Index:
    """
    Reconstruct a BM25Index by pulling all documents out of the ChromaDB collection.

    Called once at retrieval-service startup and again after new docs are ingested.
    """
    col = store._col
    total = col.count()
    if total == 0:
        return BM25Index([])

    raw = col.get(include=["documents", "metadatas"])
    chunks = [
        {
            "chunk_id": cid,
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "chunk_index": meta["chunk_index"],
        }
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
    ]
    return BM25Index(chunks)
