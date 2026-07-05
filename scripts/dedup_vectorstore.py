"""
One-time cleanup: detect and remove duplicate chunks caused by inconsistent
source filenames (e.g. "ACA Report.pdf" vs "ACA_Report.pdf") in the vectorstore.

Dedup strategy: within each normalized-name group, keep the source variant that
has the most chunks (assumes the larger set is the most complete ingestion).
Ties are broken by keeping the lexicographically earlier source string.

Usage:
    python scripts/dedup_vectorstore.py [--dry-run]
"""
import sys
import os

# Run from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from collections import defaultdict
from src.ingestion import normalize_filename

PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"


def main():
    dry_run = "--dry-run" in sys.argv

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col    = client.get_or_create_collection(COLLECTION)

    total_before = col.count()
    print(f"\nVector store: {PERSIST_DIR!r}  collection: {COLLECTION!r}")
    print(f"Total chunks before: {total_before}\n")

    if total_before == 0:
        print("Collection is empty — nothing to deduplicate.")
        return

    # Pull everything (ids + metadatas only — skip embeddings for speed).
    raw = col.get(include=["metadatas"])
    ids      = raw["ids"]
    metas    = raw["metadatas"]

    # Group chunk_ids by normalized source name.
    # norm_name -> { source_variant -> [chunk_id, ...] }
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for cid, meta in zip(ids, metas):
        source = meta.get("source", "")
        norm   = normalize_filename(source)
        groups[norm][source].append(cid)

    print("─── Source inventory ───────────────────────────────────────────")
    duplicates_found = 0
    to_delete: list[str] = []

    for norm, variants in sorted(groups.items()):
        chunk_counts = {v: len(cids) for v, cids in variants.items()}
        print(f"\n  Normalized: {norm!r}")
        for v, count in sorted(chunk_counts.items()):
            print(f"    source={v!r}  chunks={count}")

        if len(variants) == 1:
            continue  # no duplicates for this name

        duplicates_found += 1
        # Prefer the normalized variant (no spaces) so stored metadata stays
        # consistent with future ingestions. Fall back to most-chunks if neither
        # (or both) variants are already normalized.
        def _keeper_key(v: str) -> tuple:
            is_normalized = (v == normalize_filename(v))
            return (is_normalized, chunk_counts[v])

        keeper = max(chunk_counts, key=_keeper_key)
        print(f"    → Keeping: {keeper!r} ({chunk_counts[keeper]} chunks)")
        for v, cids in variants.items():
            if v != keeper:
                print(f"    → Deleting: {v!r} ({len(cids)} chunks)")
                to_delete.extend(cids)

    print("\n─── Summary ────────────────────────────────────────────────────")
    print(f"Duplicate groups found : {duplicates_found}")
    print(f"Chunks to delete       : {len(to_delete)}")

    if not to_delete:
        print("Nothing to delete — vectorstore is clean.")
        return

    if dry_run:
        print("\n[DRY RUN] No changes made. Re-run without --dry-run to apply.")
        return

    # Delete in batches (Chroma has a practical limit per call).
    BATCH = 500
    for i in range(0, len(to_delete), BATCH):
        col.delete(ids=to_delete[i : i + BATCH])

    total_after = col.count()
    print(f"\nChunks before : {total_before}")
    print(f"Chunks deleted: {len(to_delete)}")
    print(f"Chunks after  : {total_after}")
    print("Done.\n")


if __name__ == "__main__":
    main()
