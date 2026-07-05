from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps sentence-transformers for chunk and query embedding.

    Model: all-MiniLM-L6-v2 — ~80MB, CPU-friendly, strong semantic similarity
    baseline. Right-sized for this project; swap to a larger model (e.g.
    all-mpnet-base-v2) if retrieval quality becomes the bottleneck.

    The model is loaded once at construction time (not per-call) — loading a
    transformer takes ~1-2s and should not happen inside a hot path.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
