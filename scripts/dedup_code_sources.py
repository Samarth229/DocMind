"""
One-time cleanup: remove duplicate code chunks caused by the old relative-path
source scheme (before the absolute-path fix in load_codebase).

The bug: ingesting monitoring/ then monitoring/pattern_learning_v0_1/ produced
two chunk sets for the same physical file -- one with source="pattern_learning_v0_1/stats_engine.py"
and one with source="stats_engine.py" — because source was relative to whichever
directory was passed to load_codebase.

Strategy: group chunks by file basename. When multiple source strings share the
same basename AND the chunk_ids look like they come from the same file (same
function/method names embedded in the id), keep the longer (more specific) source
path — it's more likely to be the absolute or parent-relative version — and delete
the shorter/orphaned variant.

Usage:
    python scripts/dedup_code_sources.py [--dry-run]
"""
import sys
import io
import os
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb

PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"


def main():
    dry_run = "--dry-run" in sys.argv

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col    = client.get_or_create_collection(COLLECTION)

    total_before = col.count()
    print(f"\nVectorstore: {PERSIST_DIR!r}  collection: {COLLECTION!r}")
    print(f"Total chunks before: {total_before}\n")

    if total_before == 0:
        print("Collection is empty — nothing to deduplicate.")
        return

    raw   = col.get(include=["metadatas"])
    ids   = raw["ids"]
    metas = raw["metadatas"]

    # Group by the "function identity" embedded in each chunk_id.
    # chunk_ids are built as: <source_stem>_<kind>_<name>
    # Two source variants are duplicates of the same file only if their chunk_ids
    # share the same suffix (kind + name) after stripping the source prefix.
    # Strategy: group by (basename, function/method name embedded in chunk_id).
    # If two chunk_ids differ only in their source prefix but share the same
    # trailing _func_name / _class_method portion, they're the same content.

    # Pull document text too so we can compare content, not just chunk_ids.
    raw_with_docs = col.get(include=["metadatas", "documents"])
    docs = raw_with_docs["documents"]

    # Group by (basename, text_hash): same physical file = same basename + identical content.
    import hashlib

    # (basename, text_hash) -> { source -> [chunk_id, ...] }
    content_groups: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for cid, meta, text in zip(ids, metas, docs):
        source   = meta.get("source", "")
        basename = Path(source.replace("\\", "/")).name
        th       = hashlib.md5(text.encode()).hexdigest()
        content_groups[(basename, th)][source].append(cid)

    print("─── Duplicate code chunk detection (content-based) ──────────────────")
    duplicates_found = 0
    to_delete: list[str] = []

    for (basename, _), source_to_cids in sorted(content_groups.items()):
        if len(source_to_cids) == 1:
            continue  # unique content for this basename — no duplicate

        chunk_counts = {v: len(cids) for v, cids in source_to_cids.items()}
        print(f"\n  File: {basename!r}")
        for v, count in sorted(chunk_counts.items()):
            print(f"    source={v!r}  chunks={count}")

        duplicates_found += 1

        # Keep the most specific (longest) path; absolute paths always win.
        def _keeper_key(v: str) -> tuple:
            v_posix = v.replace("\\", "/")
            is_absolute = Path(v_posix).is_absolute()
            return (is_absolute, len(v))

        keeper = max(chunk_counts, key=_keeper_key)
        print(f"    → Keeping: {keeper!r}")
        for v, cids in source_to_cids.items():
            if v != keeper:
                print(f"    → Deleting: {v!r} ({len(cids)} chunks)")
                to_delete.extend(cids)

    print("\n─── Summary ────────────────────────────────────────────────────────")
    print(f"Duplicate groups found : {duplicates_found}")
    print(f"Chunks to delete       : {len(to_delete)}")

    if not to_delete:
        print("Nothing to delete — vectorstore is clean.")
        return

    if dry_run:
        print("\n[DRY RUN] No changes made. Re-run without --dry-run to apply.")
        return

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
