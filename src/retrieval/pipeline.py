from src.embedding.embedder import Embedder
from src.embedding.store import VectorStore
from .dense import dense_retrieve
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker


def retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    bm25_index: BM25Index,
    reranker: CrossEncoderReranker,
    k: int = 5,
    fusion_candidates: int = 20,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline: dense + BM25 -> RRF fusion -> cross-encoder rerank.

    This is the single retrieval entry point for Phase 5 (generation) and
    Phase 6 (citation). Signature is stable regardless of what changes inside.

    fusion_candidates: how many candidates each retriever fetches before fusion.
    A wider net here gives RRF more material to work with; final output is top k.
    """
    dense_results = dense_retrieve(query, store, embedder, k=fusion_candidates)
    bm25_results  = bm25_index.search(query, k=fusion_candidates)

    merged = reciprocal_rank_fusion([dense_results, bm25_results])
    top_candidates = merged[:fusion_candidates]

    return reranker.rerank(query, top_candidates, top_n=k)
