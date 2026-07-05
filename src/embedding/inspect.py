"""
CLI sanity-check: ingest a document and run queries against it.

Usage:
    python -m src.embedding.inspect <file_path> "<query1>" ["<query2>" ...]

Example:
    python -m src.embedding.inspect data/raw/ACA_Report.pdf "what is the model adapter layer"
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.embedding import Embedder, VectorStore, ingest_document

PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"
TOP_K       = 5
PREVIEW_LEN = 300


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python -m src.embedding.inspect <file_path> \"<query>\" [...]")
        sys.exit(1)

    file_path = args[0]
    queries   = args[1:]

    print("\nLoading embedding model...")
    embedder = Embedder()
    store    = VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)

    print(f"Ingesting: {file_path}")
    n = ingest_document(file_path, store, embedder)
    print(f"Chunks ingested (upserted): {n}  |  Total in store: {store.count()}\n")

    for query in queries:
        print(f"Query: \"{query}\"")
        print("─" * 60)
        results = store.query(embedder.embed_query(query), k=TOP_K)
        for i, r in enumerate(results, 1):
            preview = r["text"][:PREVIEW_LEN].replace("\n", " ")
            page_label = r["page"] if r["page"] != -1 else "N/A"
            print(f"  #{i}  [{r['source']}  p.{page_label}]  score={r['score']:.4f}")
            print(f"       {preview}{'...' if len(r['text']) > PREVIEW_LEN else ''}")
            print()
        print()


if __name__ == "__main__":
    main()
