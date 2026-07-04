import pytest
from pathlib import Path
from src.chunking import RecursiveCharacterSplitter, chunk_document
from src.ingestion import load_document

RAW = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_TXT = RAW / "sample.txt"
SAMPLE_PDF = RAW / "ACA_Report.pdf"


# ── RecursiveCharacterSplitter ────────────────────────────────────────────────

def test_short_text_returns_single_chunk():
    splitter = RecursiveCharacterSplitter(chunk_size=800, chunk_overlap=100)
    result = splitter.split_text("This is a short sentence.")
    assert len(result) == 1

def test_long_text_splits_into_multiple_chunks():
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    # ~300 words — should produce multiple chunks
    long_text = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 1

def test_each_chunk_within_size_limit():
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    long_text = ("word " * 400).strip()
    chunks = splitter.split_text(long_text)
    # Allow a small tolerance: last chunk may be slightly under, recursive
    # merging means individual pieces are bounded by chunk_size.
    for chunk in chunks:
        assert len(chunk.split()) <= 60, f"Chunk too large: {len(chunk.split())} words"

def test_consecutive_chunks_have_overlap():
    splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=10)
    # Build text with clear paragraph breaks so splitter has clean split points.
    para = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    long_text = (para * 8).strip()
    chunks = splitter.split_text(long_text)
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # The last few words of chunk N should appear at the start of chunk N+1.
    tail_words = set(chunks[0].split()[-5:])
    head_words = set(chunks[1].split()[:15])
    assert tail_words & head_words, "Expected overlapping words between consecutive chunks"

def test_empty_text_returns_empty():
    splitter = RecursiveCharacterSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []


# ── chunk_document ────────────────────────────────────────────────────────────

def test_chunk_document_on_txt():
    pages = load_document(str(SAMPLE_TXT))
    chunks = chunk_document(pages)
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "source" in c
        assert "page" in c
        assert "chunk_index" in c
        assert c["source"] == "sample.txt"

def test_chunk_ids_are_unique_on_txt():
    pages = load_document(str(SAMPLE_TXT))
    chunks = chunk_document(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="ACA Report.pdf not found in data/raw/")
def test_chunk_document_on_pdf():
    pages = load_document(str(SAMPLE_PDF))
    chunks = chunk_document(pages)
    assert len(chunks) > 10  # 49-page report should produce many chunks

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="ACA Report.pdf not found in data/raw/")
def test_chunk_ids_unique_on_pdf():
    pages = load_document(str(SAMPLE_PDF))
    chunks = chunk_document(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique across the whole document"

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="ACA Report.pdf not found in data/raw/")
def test_pdf_chunk_metadata_propagation():
    pages = load_document(str(SAMPLE_PDF))
    chunks = chunk_document(pages)
    for c in chunks:
        assert c["source"] == "ACA_Report.pdf"
        assert isinstance(c["page"], int) and c["page"] >= 1
        assert isinstance(c["chunk_index"], int)
