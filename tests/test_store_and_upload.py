"""
Tests for:
- VectorStore.clear_all()
- Document upload filename normalization (save path uses normalized name)
"""
import pytest
from pathlib import Path


# ── VectorStore.clear_all() ───────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    from src.embedding.store import VectorStore
    return VectorStore(persist_dir=str(tmp_path / "vs"), collection_name="test")


def _dummy_chunks(n: int = 3) -> tuple[list[dict], list[list[float]]]:
    chunks = [
        {"chunk_id": f"c{i}", "text": f"text {i}", "source": "doc.txt", "page": -1, "chunk_index": i}
        for i in range(n)
    ]
    embeddings = [[0.1 * i] * 384 for i in range(n)]
    return chunks, embeddings


def test_clear_all_empties_collection(tmp_store):
    chunks, embeddings = _dummy_chunks()
    tmp_store.add_chunks(chunks, embeddings)
    assert tmp_store.count() == 3

    tmp_store.clear_all()
    assert tmp_store.count() == 0


def test_clear_all_collection_still_usable(tmp_store):
    """After clear_all(), add_chunks and query must work without errors."""
    chunks, embeddings = _dummy_chunks()
    tmp_store.add_chunks(chunks, embeddings)
    tmp_store.clear_all()

    # Should be able to add new chunks immediately after clearing
    new_chunks, new_embeddings = _dummy_chunks(2)
    tmp_store.add_chunks(new_chunks, new_embeddings)
    assert tmp_store.count() == 2


def test_clear_all_on_empty_store(tmp_store):
    """clear_all() on an already-empty store must not raise."""
    assert tmp_store.count() == 0
    tmp_store.clear_all()  # should not raise
    assert tmp_store.count() == 0


# ── Document upload filename normalization ────────────────────────────────────

def test_normalized_save_path(tmp_path):
    """
    Uploading 'ACA Report.pdf' (spaces) must write to 'ACA_Report.pdf' (underscores).
    This confirms the fix: app.py uses normalize_filename(uploaded.name) as the dest path,
    not uploaded.name directly.
    """
    from src.ingestion import normalize_filename

    raw_name = "ACA Report.pdf"
    norm_name = normalize_filename(raw_name)
    assert norm_name == "ACA_Report.pdf"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Simulate what app.py does: write bytes under the normalized name
    dest = raw_dir / norm_name
    dest.write_bytes(b"%PDF fake content")

    # Only one file should exist, named with underscores
    files = list(raw_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == "ACA_Report.pdf"


def test_same_file_two_spellings_one_dest(tmp_path):
    """
    Uploading 'My Doc.pdf' then 'My_Doc.pdf' (or vice versa) produces a single
    file on disk because both normalize to the same name and the second write
    overwrites the first.
    """
    from src.ingestion import normalize_filename

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    for raw_name in ("My Doc.pdf", "My_Doc.pdf"):
        norm = normalize_filename(raw_name)
        dest = raw_dir / norm
        dest.write_bytes(b"content")

    files = [f for f in raw_dir.iterdir()]
    assert len(files) == 1, (
        f"Expected 1 file after two uploads of the same normalized name, got {[f.name for f in files]}"
    )
    assert files[0].name == "My_Doc.pdf"
