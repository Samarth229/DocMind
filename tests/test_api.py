import pytest
import io
from pathlib import Path
from fastapi.testclient import TestClient

from src.ingestion import normalize_filename

RAW = Path(__file__).parent.parent / "data" / "raw"
TXT = RAW / "sample.txt"
# Accept either the normalized or original (space) filename.
_pdf_candidates = [RAW / "ACA_Report.pdf", RAW / "ACA Report.pdf"]
PDF = next((p for p in _pdf_candidates if p.exists()), _pdf_candidates[0])

pdf_available = PDF.exists()
txt_available = TXT.exists()


def _ollama_reachable() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def client():
    """TestClient with real models loaded (integration-level tests)."""
    from api.main import app
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "vectorstore_chunks" in body
    assert "ollama_model" in body

def test_health_503_when_ollama_down(client):
    """Health returns 503 when Ollama is unreachable — tested by mocking."""
    import api.main as api_module
    original = api_module._provider
    # Temporarily replace provider to force Ollama check to fail.
    import ollama as _ol
    from unittest.mock import patch
    with patch.object(_ol, "list", side_effect=Exception("connection refused")):
        r = client.get("/health")
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


# ── /documents ────────────────────────────────────────────────────────────────

def test_documents_returns_list(client):
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body
    assert "total_chunks" in body
    assert isinstance(body["documents"], list)


# ── /ingest ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not txt_available, reason="sample.txt not found")
def test_ingest_txt(client):
    with open(TXT, "rb") as f:
        r = client.post("/ingest", files={"file": ("sample.txt", f, "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["filename"] == "sample.txt"
    assert body["chunks_ingested"] >= 1

def test_ingest_unsupported_type(client):
    r = client.post(
        "/ingest",
        files={"file": ("bad.docx", io.BytesIO(b"hello"), "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_ingest_pdf(client):
    with open(PDF, "rb") as f:
        r = client.post("/ingest", files={"file": ("ACA_Report.pdf", f, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_ingested"] == 49

def test_ingest_filename_normalized(client):
    """File with spaces in name is saved and stored under normalized (underscore) name."""
    r = client.post(
        "/ingest",
        files={"file": ("my doc.txt", io.BytesIO(b"hello world test content here"), "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["filename"] == "my_doc.txt"


# ── /query ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_query_response_shape(client):
    # Ensure ACA_Report.pdf is ingested first.
    with open(PDF, "rb") as f:
        client.post("/ingest", files={"file": ("ACA_Report.pdf", f, "application/pdf")})

    r = client.post("/query", json={"question": "what is the model adapter layer"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "chunks_used" in body
    assert "citation_check" in body
    assert len(body["answer"]) > 0
    assert len(body["chunks_used"]) > 0

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_query_chunk_fields(client):
    with open(PDF, "rb") as f:
        client.post("/ingest", files={"file": ("ACA_Report.pdf", f, "application/pdf")})
    r = client.post("/query", json={"question": "what is ATLAS", "k": 3})
    assert r.status_code == 200
    chunk = r.json()["chunks_used"][0]
    for field in ("chunk_id", "text", "source", "page"):
        assert field in chunk

def test_query_empty_question_rejected(client):
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422  # Pydantic min_length validation

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_query_citation_check_present(client):
    with open(PDF, "rb") as f:
        client.post("/ingest", files={"file": ("ACA_Report.pdf", f, "application/pdf")})
    r = client.post("/query", json={"question": "what is the Atlas Engine"})
    check = r.json()["citation_check"]
    assert "total_citations" in check
    assert "all_valid" in check
    assert "invalid_citations" in check
