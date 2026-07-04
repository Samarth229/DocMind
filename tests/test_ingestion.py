import pytest
from pathlib import Path
from src.ingestion import load_document, load_text, load_pdf

RAW = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_TXT = RAW / "sample.txt"
SAMPLE_PDF = RAW / "sample.pdf"


# ── load_text ────────────────────────────────────────────────────────────────

def test_load_text_returns_single_record():
    records = load_text(str(SAMPLE_TXT))
    assert len(records) == 1

def test_load_text_fields():
    record = load_text(str(SAMPLE_TXT))[0]
    assert record["source"] == "sample.txt"
    assert record["page"] is None
    assert "DocMind" in record["text"]

def test_load_text_strips_excess_whitespace():
    record = load_text(str(SAMPLE_TXT))[0]
    assert "  " not in record["text"]  # no double spaces


# ── load_document dispatch ────────────────────────────────────────────────────

def test_load_document_dispatches_txt():
    records = load_document(str(SAMPLE_TXT))
    assert records[0]["source"].endswith(".txt")

def test_load_document_unsupported_type(tmp_path):
    fake = tmp_path / "file.docx"
    fake.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(fake))


# ── load_pdf ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Drop a sample.pdf into data/raw/ to run PDF tests")
def test_load_pdf_returns_pages():
    records = load_pdf(str(SAMPLE_PDF))
    assert len(records) >= 1

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Drop a sample.pdf into data/raw/ to run PDF tests")
def test_load_pdf_fields():
    record = load_pdf(str(SAMPLE_PDF))[0]
    assert record["source"] == "sample.pdf"
    assert isinstance(record["page"], int)
    assert record["page"] >= 1
    assert isinstance(record["text"], str)
    assert len(record["text"]) > 0

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Drop a sample.pdf into data/raw/ to run PDF tests")
def test_load_document_dispatches_pdf():
    records = load_document(str(SAMPLE_PDF))
    assert records[0]["source"].endswith(".pdf")
