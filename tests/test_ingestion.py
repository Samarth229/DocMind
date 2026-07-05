import pytest
from pathlib import Path
from src.ingestion import load_document, load_text, load_pdf, normalize_filename


# ── normalize_filename ────────────────────────────────────────────────────────

def test_normalize_spaces_to_underscores():
    assert normalize_filename("ACA Report.pdf") == "ACA_Report.pdf"

def test_normalize_multiple_spaces():
    assert normalize_filename("my  doc  name.pdf") == "my__doc__name.pdf"

def test_normalize_strips_whitespace():
    assert normalize_filename("  file.txt  ") == "file.txt"

def test_normalize_already_clean():
    assert normalize_filename("ACA_Report.pdf") == "ACA_Report.pdf"

def test_normalize_preserves_case():
    assert normalize_filename("MyDoc.PDF") == "MyDoc.PDF"

def test_normalize_applied_in_load_text():
    # source metadata must use normalized name even if file has spaces in name.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".txt", prefix="my doc ", delete=False, mode="w") as f:
        f.write("hello world")
        tmp = f.name
    try:
        records = load_text(tmp)
        assert " " not in records[0]["source"]
    finally:
        os.unlink(tmp)

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
