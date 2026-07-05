"""
CLI sanity-check: before/after comparison of dense-only vs hybrid retrieval.

Usage:
    python -m src.retrieval.inspect <file_path> "<query1>" ["<query2>" ...]

Runs each query through both pipelines side-by-side so retrieval improvement
from Phase 3 -> Phase 4 is visible at a glance.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import (
    dense_retrieve,
    build_bm25_from_store,
    CrossEncoderReranker,
    retrieve,
)

PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"
TOP_K       = 5
PREVIEW_LEN = 220


def _print_results(results: list[dict], score_field: str) -> None:
    for i, r in enumerate(results, 1):
        preview = r["text"][:PREVIEW_LEN].replace("\n", " ")
        page_label = r["page"] if r["page"] != -1 else "N/A"
        score = r.get(score_field, r.get("score", 0))
        print(f"  #{i}  [p.{page_label}]  {score_field}={score:.4f}")
        print(f"       {preview}{'...' if len(r['text']) > PREVIEW_LEN else ''}")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python -m src.retrieval.inspect <file_path> \"<query>\" [...]")
        sys.exit(1)

    file_path = args[0]
    queries   = args[1:]

    print("\nLoading models (embedder + cross-encoder)...")
    embedder = Embedder()
    reranker = CrossEncoderReranker()
    store    = VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)

    print(f"Ingesting (upsert): {file_path}")
    n = ingest_document(file_path, store, embedder)
    print(f"Chunks: {n} ingested  |  {store.count()} total in store")

    print("Building BM25 index from store...")
    bm25 = build_bm25_from_store(store)
    print("Ready.\n")

    for query in queries:
        sep = "═" * 70
        print(sep)
        print(f'Query: "{query}"')
        print(sep)

        print("\n[ PHASE 3 — Dense only (cosine similarity) ]")
        dense = dense_retrieve(query, store, embedder, k=TOP_K)
        _print_results(dense, "score")

        print("\n[ PHASE 4 — Hybrid BM25 + Dense → RRF → Cross-encoder rerank ]")
        hybrid = retrieve(query, store, embedder, bm25, reranker, k=TOP_K)
        _print_results(hybrid, "rerank_score")
        print()


if __name__ == "__main__":
    main()
