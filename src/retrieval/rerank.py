from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate chunks by jointly scoring (query, chunk_text) pairs.

    Architecture note for interviews:
    - Bi-encoder (Phase 3 embeddings): encodes query and document independently,
      then compares vectors. Fast over large corpora — good for recall.
    - Cross-encoder: feeds the full (query, document) pair through the model at
      once, allowing full attention across both. Much slower but far more precise
      — good for re-ranking a small candidate set. Classic recall-then-precision
      funnel: cheap+broad first, expensive+narrow last.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(
            [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)],
            key=lambda d: d["rerank_score"],
            reverse=True,
        )
        return ranked[:top_n]
