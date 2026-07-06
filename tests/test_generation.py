import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.generation import build_prompt, OllamaProvider, answer_query
from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import build_bm25_from_store, CrossEncoderReranker

PDF = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
pdf_available = PDF.exists()


def _ollama_reachable() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


# ── build_prompt ──────────────────────────────────────────────────────────────

FAKE_CHUNKS = [
    {"chunk_id": "a_p1_c0", "text": "The sky is blue.", "source": "doc.pdf", "page": 1, "chunk_index": 0},
    {"chunk_id": "a_p2_c0", "text": "Python is a programming language.", "source": "doc.pdf", "page": 2, "chunk_index": 0},
]

def test_build_prompt_contains_query():
    prompt = build_prompt("what color is the sky?", FAKE_CHUNKS)
    assert "what color is the sky?" in prompt

def test_build_prompt_contains_chunk_texts():
    prompt = build_prompt("test", FAKE_CHUNKS)
    assert "The sky is blue." in prompt
    assert "Python is a programming language." in prompt

def test_build_prompt_contains_source_labels():
    prompt = build_prompt("test", FAKE_CHUNKS)
    assert "[Source: doc.pdf, Page 1]" in prompt
    assert "[Source: doc.pdf, Page 2]" in prompt

def test_build_prompt_none_page_omits_page():
    # Code chunks (page=None) must not show "Page N/A" — page is omitted entirely.
    chunks = [{"chunk_id": "x", "text": "text", "source": "file.txt", "page": None, "chunk_index": 0}]
    prompt = build_prompt("q", chunks)
    assert "Page N/A" not in prompt
    assert "[Source: file.txt]" in prompt

def test_build_prompt_has_grounding_instruction():
    prompt = build_prompt("q", FAKE_CHUNKS)
    assert "ONLY" in prompt
    assert "context" in prompt.lower()


# ── OllamaProvider ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama server not running")
def test_ollama_generate_returns_string():
    provider = OllamaProvider()
    result = provider.generate("Say the word 'hello' and nothing else.")
    assert isinstance(result, str)
    assert len(result) > 0


# ── answer_query ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    return Embedder()

@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()

@pytest.fixture(scope="module")
def pdf_store_bm25(embedder, tmp_path_factory):
    if not pdf_available:
        pytest.skip("ACA_Report.pdf not found")
    tmp = tmp_path_factory.mktemp("vs_gen")
    store = VectorStore(persist_dir=str(tmp), collection_name="test_gen")
    ingest_document(str(PDF), store, embedder)
    bm25 = build_bm25_from_store(store)
    return store, bm25

def test_answer_query_result_keys(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "The model adapter layer does X."

    result = answer_query("what is the model adapter layer", store, embedder, bm25, reranker, mock_provider)
    assert "answer" in result
    assert "chunks_used" in result
    assert "prompt" in result

def test_answer_query_chunks_used_populated(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "answer"
    result = answer_query("what is ATLAS", store, embedder, bm25, reranker, mock_provider, k=3)
    assert len(result["chunks_used"]) == 3

def test_answer_query_prompt_passed_to_provider(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "answer"
    result = answer_query("what is ATLAS", store, embedder, bm25, reranker, mock_provider)
    mock_provider.generate.assert_called_once_with(result["prompt"])

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama server not running")
def test_answer_query_live_ollama(embedder, reranker, pdf_store_bm25):
    store, bm25 = pdf_store_bm25
    provider = OllamaProvider()
    result = answer_query("what is the model adapter layer", store, embedder, bm25, reranker, provider)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 20
