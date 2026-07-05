import json
from pathlib import Path

from src.embedding.embedder import Embedder
from src.embedding.store import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.dense import dense_retrieve
from src.retrieval.pipeline import retrieve
from .metrics import precision_at_k, recall_at_k, reciprocal_rank


def _eval_one(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict:
    return {
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids),
    }


def _aggregate(per_query: list[dict]) -> dict:
    n = len(per_query)
    if n == 0:
        return {"mean_precision_at_k": 0.0, "mean_recall_at_k": 0.0, "mrr": 0.0}
    return {
        "mean_precision_at_k": sum(r["precision"] for r in per_query) / n,
        "mean_recall_at_k":    sum(r["recall"]    for r in per_query) / n,
        "mrr":                 sum(r["rr"]         for r in per_query) / n,
    }


def run_evaluation(
    queries_path: str,
    store: VectorStore,
    embedder: Embedder,
    bm25_index: BM25Index,
    reranker: CrossEncoderReranker,
    k: int = 5,
) -> dict:
    """
    Run both dense-only and hybrid+rerank retrieval against the labeled query set
    and return side-by-side metrics.

    Evaluates retrieval only — not generation quality. Generation eval (faithfulness,
    answer relevance) is documented future work via RAGAS/LLM-as-judge.
    """
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))

    dense_results, hybrid_results = [], []

    for entry in queries:
        query        = entry["query"]
        relevant_ids = set(entry["relevant_chunk_ids"])

        # Dense-only (Phase 3 baseline)
        dense_hits = dense_retrieve(query, store, embedder, k=k)
        dense_ids  = [r["chunk_id"] for r in dense_hits]
        dense_row  = {"query": query, "category": entry.get("_category", ""), **_eval_one(dense_ids, relevant_ids, k)}
        dense_results.append(dense_row)

        # Hybrid + rerank (Phase 4)
        hybrid_hits = retrieve(query, store, embedder, bm25_index, reranker, k=k)
        hybrid_ids  = [r["chunk_id"] for r in hybrid_hits]
        hybrid_row  = {"query": query, "category": entry.get("_category", ""), **_eval_one(hybrid_ids, relevant_ids, k)}
        hybrid_results.append(hybrid_row)

    return {
        "k": k,
        "n_queries": len(queries),
        "dense": {**_aggregate(dense_results),  "per_query": dense_results},
        "hybrid": {**_aggregate(hybrid_results), "per_query": hybrid_results},
    }
