"""
CLI sanity-check: load and optionally chunk a document, then preview output.

Usage:
    python -m src.ingestion.inspect <file_path>           # ingestion only
    python -m src.ingestion.inspect <file_path> --chunk   # ingestion + chunking
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from src.ingestion import load_document

PREVIEW_LEN = 200
CHUNK_PREVIEW_COUNT = 5


def main():
    args = sys.argv[1:]
    do_chunk = "--chunk" in args
    paths = [a for a in args if not a.startswith("--")]

    if not paths:
        print("Usage: python -m src.ingestion.inspect <file_path> [--chunk]")
        sys.exit(1)

    path = paths[0]
    try:
        pages = load_document(path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nFile : {path}")
    print(f"Pages extracted: {len(pages)}\n")
    for r in pages:
        label = f"page {r['page']}" if r["page"] is not None else "full text"
        preview = r["text"][:PREVIEW_LEN].replace("\n", " ")
        print(f"  [{label}] {preview}{'...' if len(r['text']) > PREVIEW_LEN else ''}")

    if do_chunk:
        from src.chunking import chunk_document
        chunks = chunk_document(pages)
        print(f"\n── Chunking ─────────────────────────────────────")
        print(f"Total chunks: {len(chunks)}")
        print(f"Showing first {min(CHUNK_PREVIEW_COUNT, len(chunks))} chunks:\n")
        for c in chunks[:CHUNK_PREVIEW_COUNT]:
            preview = c["text"][:PREVIEW_LEN].replace("\n", " ")
            print(f"  [{c['chunk_id']}]  (page {c['page']}, chunk {c['chunk_index']})")
            print(f"  {preview}{'...' if len(c['text']) > PREVIEW_LEN else ''}\n")

    print()


if __name__ == "__main__":
    main()
