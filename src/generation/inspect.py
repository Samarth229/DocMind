"""
CLI: run a full RAG query against an ingested document and display results.

Usage:
    python -m src.generation.inspect <file_path> "<query>" [--verbose]

Flags:
    --verbose    also print the full prompt sent to the LLM

Example:
    python -m src.generation.inspect data/raw/ACA_Report.pdf "what is the model adapter layer"
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import build_bm25_from_store, CrossEncoderReranker
from src.generation import OllamaProvider, answer_query

PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"
PREVIEW_LEN = 200


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args
    args = [a for a in args if not a.startswith("--")]

    if len(args) < 2:
        print("Usage: python -m src.generation.inspect <file_path> \"<query>\" [--verbose]")
        sys.exit(1)

    file_path, query = args[0], args[1]

    print("\nLoading models...")
    embedder = Embedder()
    reranker = CrossEncoderReranker()
    provider = OllamaProvider()
    store    = VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)

    print(f"Ingesting (upsert): {file_path}")
    n = ingest_document(file_path, store, embedder)
    print(f"Chunks in store: {store.count()}  ({n} upserted this run)")

    print("Building BM25 index...")
    bm25 = build_bm25_from_store(store)

    print(f"\nQuery: \"{query}\"\n")

    result = answer_query(query, store, embedder, bm25, reranker, provider)

    print("── Retrieved chunks ─────────────────────────────────────────")
    for i, c in enumerate(result["chunks_used"], 1):
        preview = c["text"][:PREVIEW_LEN].replace("\n", " ")
        page = c["page"] if c["page"] not in (None, -1) else "N/A"
        score = c.get("rerank_score", c.get("rrf_score", ""))
        score_str = f"  rerank={score:.3f}" if isinstance(score, float) else ""
        print(f"  #{i} [p.{page}]{score_str}  {preview}...")

    if verbose:
        print("\n── Prompt ───────────────────────────────────────────────────")
        print(result["prompt"])

    print("\n── Answer ───────────────────────────────────────────────────")
    print(result["answer"])

    check = result["citation_check"]
    status = "✓ ALL VALID" if check["all_valid"] else "✗ INVALID CITATIONS FOUND"
    print(f"\n── Citation validation  [{status}] ──────────────────────────")
    print(f"  Total citations : {check['total_citations']}")
    print(f"  Valid           : {check['valid_citations']}")
    if check["invalid_citations"]:
        print("  Invalid:")
        for c in check["invalid_citations"]:
            print(f"    ✗  [Source: {c['source']}, Page {c['page']}] — not in retrieved context")
    print()


if __name__ == "__main__":
    main()
