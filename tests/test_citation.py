import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.generation.citation import extract_citations, validate_citations
from src.generation.pipeline import answer_query
from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import build_bm25_from_store, CrossEncoderReranker

PDF = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
pdf_available = PDF.exists()


# ── extract_citations ─────────────────────────────────────────────────────────

def test_extract_no_citations():
    assert extract_citations("This answer has no citations.") == []

def test_extract_single_citation():
    text = "The layer is described [Source: doc.pdf, Page 5]."
    result = extract_citations(text)
    assert len(result) == 1
    assert result[0] == {"source": "doc.pdf", "page": 5}

def test_extract_multiple_citations():
    text = (
        "First fact [Source: a.pdf, Page 1]. "
        "Second fact [Source: b.pdf, Page 22]."
    )
    result = extract_citations(text)
    assert len(result) == 2
    assert result[0]["page"] == 1
    assert result[1]["source"] == "b.pdf"

def test_extract_tolerates_extra_whitespace():
    text = "[Source:  doc.pdf ,  Page  7 ]"
    result = extract_citations(text)
    assert len(result) == 1
    assert result[0]["page"] == 7

def test_extract_case_insensitive():
    text = "[source: Doc.pdf, page 3]"
    result = extract_citations(text)
    assert len(result) == 1
    assert result[0]["page"] == 3

def test_extract_page_na():
    text = "See [Source: notes.txt, Page N/A] for details."
    result = extract_citations(text)
    assert len(result) == 1
    assert result[0]["page"] is None


# ── validate_citations ────────────────────────────────────────────────────────

CHUNKS = [
    {"chunk_id": "a_p1_c0", "text": "...", "source": "doc.pdf", "page": 1, "chunk_index": 0},
    {"chunk_id": "a_p5_c0", "text": "...", "source": "doc.pdf", "page": 5, "chunk_index": 0},
    {"chunk_id": "b_p3_c0", "text": "...", "source": "other.pdf", "page": 3, "chunk_index": 0},
]

def test_validate_all_valid():
    citations = [{"source": "doc.pdf", "page": 1}, {"source": "other.pdf", "page": 3}]
    report = validate_citations(citations, CHUNKS)
    assert report["all_valid"] is True
    assert report["total_citations"] == 2
    assert report["valid_citations"] == 2
    assert report["invalid_citations"] == []

def test_validate_one_invalid():
    citations = [
        {"source": "doc.pdf", "page": 1},
        {"source": "doc.pdf", "page": 99},  # page 99 was never retrieved
    ]
    report = validate_citations(citations, CHUNKS)
    assert report["all_valid"] is False
    assert report["valid_citations"] == 1
    assert len(report["invalid_citations"]) == 1
    assert report["invalid_citations"][0]["page"] == 99

def test_validate_hallucinated_source():
    citations = [{"source": "invented.pdf", "page": 1}]
    report = validate_citations(citations, CHUNKS)
    assert report["all_valid"] is False
    assert report["invalid_citations"][0]["source"] == "invented.pdf"

def test_validate_no_citations():
    report = validate_citations([], CHUNKS)
    assert report["all_valid"] is True
    assert report["total_citations"] == 0


# ── answer_query integration ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    return Embedder()

@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()

@pytest.fixture(scope="module")
def pdf_store_bm25(embedder, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vs_cite")
    store = VectorStore(persist_dir=str(tmp), collection_name="test_cite")
    ingest_document(str(PDF), store, embedder)
    bm25 = build_bm25_from_store(store)
    return store, bm25

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_answer_query_has_citation_check_key(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "No citations here."
    result = answer_query("what is ATLAS", store, embedder, bm25, reranker, mock_provider)
    assert "citation_check" in result
    check = result["citation_check"]
    assert "total_citations" in check
    assert "valid_citations" in check
    assert "invalid_citations" in check
    assert "all_valid" in check

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_answer_query_citation_check_with_valid_citation(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    # The mock answer cites page 22 — which is in ACA_Report.pdf's retrieved chunks
    # for this query (confirmed in Phase 5 output).
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "The adapter layer is described [Source: ACA_Report.pdf, Page 22]."
    )
    result = answer_query(
        "what is the model adapter layer", store, embedder, bm25, reranker, mock_provider
    )
    check = result["citation_check"]
    assert check["total_citations"] == 1
    # Page 22 should be in the retrieved top-5 for this query
    assert check["all_valid"] is True

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_answer_query_citation_check_catches_hallucinated_page(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "See [Source: ACA_Report.pdf, Page 999] for details."
    )
    result = answer_query(
        "what is ATLAS", store, embedder, bm25, reranker, mock_provider
    )
    check = result["citation_check"]
    assert check["all_valid"] is False
    assert any(c["page"] == 999 for c in check["invalid_citations"])
