"""
Run the retrieval evaluation harness and print a comparison table.

Usage:
    python scripts/run_eval.py [--k 5] [--queries data/eval/queries.json]
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.embedding import Embedder, VectorStore
from src.retrieval import CrossEncoderReranker, build_bm25_from_store
from src.evaluation import run_evaluation

PERSIST_DIR  = "./vectorstore"
COLLECTION   = "docmind"
QUERIES_PATH = "data/eval/queries.json"
RESULTS_PATH = "data/eval/results.json"


def fmt(v: float) -> str:
    return f"{v:.4f}"


def print_table(results: dict) -> None:
    k = results["k"]
    n = results["n_queries"]
    d = results["dense"]
    h = results["hybrid"]

    header  = f"{'Metric':<25}  {'Dense-only':>12}  {'Hybrid+rerank':>14}  {'Δ':>8}"
    divider = "─" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  Retrieval Evaluation  |  k={k}  |  n={n} queries")
    print(f"{'=' * len(header)}")
    print(header)
    print(divider)

    metrics = [
        (f"Precision@{k}", "mean_precision_at_k"),
        (f"Recall@{k}",    "mean_recall_at_k"),
        ("MRR",            "mrr"),
    ]
    for label, key in metrics:
        dv = d[key]
        hv = h[key]
        delta = hv - dv
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<23}  {fmt(dv):>12}  {fmt(hv):>14}  {sign}{fmt(delta):>7}")

    print(divider)

    # Per-category breakdown
    categories = sorted({r["category"] for r in d["per_query"] if r["category"]})
    if categories:
        print(f"\n  Per-category MRR (hybrid):")
        for cat in categories:
            rows = [r for r in h["per_query"] if r["category"] == cat]
            avg_mrr = sum(r["rr"] for r in rows) / len(rows) if rows else 0.0
            print(f"    {cat:<20}  MRR={fmt(avg_mrr)}  (n={len(rows)})")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k",       type=int, default=5)
    ap.add_argument("--queries", default=QUERIES_PATH)
    args = ap.parse_args()

    print("Loading models…")
    embedder = Embedder()
    reranker = CrossEncoderReranker()
    store    = VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)

    print("Building BM25 index…")
    bm25 = build_bm25_from_store(store)

    print(f"Running evaluation  (k={args.k}, {args.queries})…")
    results = run_evaluation(args.queries, store, embedder, bm25, reranker, k=args.k)

    print_table(results)

    out = RESULTS_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full results saved → {out}\n")


if __name__ == "__main__":
    main()
