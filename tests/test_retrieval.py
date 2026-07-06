import pytest
from pathlib import Path

from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import (
    dense_retrieve,
    BM25Index,
    build_bm25_from_store,
    reciprocal_rank_fusion,
    CrossEncoderReranker,
    retrieve,
)

PDF = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
pdf_available = PDF.exists()


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


@pytest.fixture(scope="module")
def pdf_store_and_bm25(embedder, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vs_retrieval")
    store = VectorStore(persist_dir=str(tmp), collection_name="test_retrieval")
    ingest_document(str(PDF), store, embedder)
    bm25 = build_bm25_from_store(store)
    return store, bm25


# ── dense_retrieve ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_dense_retrieve_returns_k_results(embedder, pdf_store_and_bm25):
    store, _ = pdf_store_and_bm25
    results = dense_retrieve("model adapter", store, embedder, k=5)
    assert len(results) == 5

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_dense_retrieve_result_shape(embedder, pdf_store_and_bm25):
    store, _ = pdf_store_and_bm25
    r = dense_retrieve("system architecture", store, embedder, k=3)[0]
    for field in ("chunk_id", "text", "source", "page", "score"):
        assert field in r


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_keyword_match_ranks_first():
    chunks = [
        {"chunk_id": "a", "text": "The quick brown fox jumps", "source": "x", "page": 1, "chunk_index": 0},
        {"chunk_id": "b", "text": "intent classification uses keyword matching rules", "source": "x", "page": 2, "chunk_index": 0},
        {"chunk_id": "c", "text": "ChromaDB stores vectors on disk persistently", "source": "x", "page": 3, "chunk_index": 0},
    ]
    index = BM25Index(chunks)
    results = index.search("intent classification", k=3)
    assert results[0]["chunk_id"] == "b"

def test_bm25_search_result_shape():
    # BM25Okapi IDF = log(N-n+0.5) - log(n+0.5); needs n < N/2 to be positive.
    # Use 3 docs with the target term in only 1 so IDF > 0.
    chunks = [
        {"chunk_id": "z", "text": "hello world test", "source": "s", "page": 1, "chunk_index": 0},
        {"chunk_id": "w", "text": "something completely different", "source": "s", "page": 2, "chunk_index": 0},
        {"chunk_id": "v", "text": "another unrelated document here", "source": "s", "page": 3, "chunk_index": 0},
    ]
    index = BM25Index(chunks)
    r = index.search("hello world", k=1)
    assert len(r) == 1
    assert "score" in r[0]

def test_bm25_empty_corpus_returns_empty():
    index = BM25Index([])
    assert index.search("anything", k=5) == []

def test_bm25_zero_overlap_excluded():
    chunks = [{"chunk_id": "q", "text": "cats and dogs", "source": "s", "page": 1, "chunk_index": 0}]
    index = BM25Index(chunks)
    results = index.search("quantum mechanics", k=1)
    assert results == []


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

def test_rrf_merges_and_deduplicates():
    list_a = [
        {"chunk_id": "1", "text": "a", "source": "s", "page": 1, "chunk_index": 0},
        {"chunk_id": "2", "text": "b", "source": "s", "page": 2, "chunk_index": 0},
    ]
    list_b = [
        {"chunk_id": "2", "text": "b", "source": "s", "page": 2, "chunk_index": 0},
        {"chunk_id": "3", "text": "c", "source": "s", "page": 3, "chunk_index": 0},
    ]
    merged = reciprocal_rank_fusion([list_a, list_b])
    ids = [d["chunk_id"] for d in merged]
    assert len(ids) == len(set(ids)), "No duplicates"
    assert len(ids) == 3

def test_rrf_top_ranked_by_formula():
    # chunk "2" ranks #1 in both lists -> should have highest RRF score
    list_a = [
        {"chunk_id": "2", "text": "b", "source": "s", "page": 1, "chunk_index": 0},
        {"chunk_id": "1", "text": "a", "source": "s", "page": 2, "chunk_index": 0},
    ]
    list_b = [
        {"chunk_id": "2", "text": "b", "source": "s", "page": 1, "chunk_index": 0},
        {"chunk_id": "3", "text": "c", "source": "s", "page": 3, "chunk_index": 0},
    ]
    merged = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert merged[0]["chunk_id"] == "2"
    # Verify formula: 1/(60+1) + 1/(60+1) = 2/61
    assert abs(merged[0]["rrf_score"] - 2 / 61) < 1e-9

def test_rrf_result_has_rrf_score_field():
    lst = [{"chunk_id": "x", "text": "t", "source": "s", "page": 1, "chunk_index": 0}]
    merged = reciprocal_rank_fusion([lst])
    assert "rrf_score" in merged[0]


# ── CrossEncoderReranker ──────────────────────────────────────────────────────

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_reranker_returns_top_n(reranker, embedder, pdf_store_and_bm25):
    store, _ = pdf_store_and_bm25
    candidates = dense_retrieve("adapter layer", store, embedder, k=10)
    reranked = reranker.rerank("adapter layer", candidates, top_n=3)
    assert len(reranked) == 3

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_reranker_result_has_rerank_score(reranker, embedder, pdf_store_and_bm25):
    store, _ = pdf_store_and_bm25
    candidates = dense_retrieve("adapter layer", store, embedder, k=5)
    reranked = reranker.rerank("adapter layer", candidates, top_n=5)
    assert all("rerank_score" in r for r in reranked)


# ── full retrieve() pipeline ──────────────────────────────────────────────────

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_retrieve_returns_k_results(embedder, reranker, pdf_store_and_bm25):
    store, bm25 = pdf_store_and_bm25
    results = retrieve("user intent classification", store, embedder, bm25, reranker, k=5)
    assert len(results) == 5

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_retrieve_intent_query_top_result(embedder, reranker, pdf_store_and_bm25):
    store, bm25 = pdf_store_and_bm25
    results = retrieve("how does ATLAS classify user intent", store, embedder, bm25, reranker, k=5)
    # Top results should contain intent-related content.
    # Exact page numbers vary by PDF version, so we check semantic relevance.
    assert len(results) >= 1
    top_text = " ".join(r["text"].lower() for r in results[:2])
    assert "intent" in top_text or "classif" in top_text, (
        f"Expected intent-related content in top-2, got: {top_text[:200]}"
    )
