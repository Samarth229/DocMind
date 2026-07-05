import json
import pytest
import tempfile
from pathlib import Path

from src.evaluation import precision_at_k, recall_at_k, reciprocal_rank, run_evaluation
from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import CrossEncoderReranker, build_bm25_from_store

PDF = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
pdf_available = PDF.exists()


# ── precision_at_k ────────────────────────────────────────────────────────────

def test_precision_perfect():
    assert precision_at_k(["A", "B", "C"], {"A", "B", "C"}, k=3) == 1.0

def test_precision_none():
    assert precision_at_k(["X", "Y", "Z"], {"A", "B"}, k=3) == 0.0

def test_precision_partial():
    # 2 hits out of 3 retrieved
    assert abs(precision_at_k(["A", "X", "B"], {"A", "B", "C"}, k=3) - 2/3) < 1e-9

def test_precision_truncates_to_k():
    # Only look at first k=2, both relevant
    assert precision_at_k(["A", "B", "C"], {"A", "B"}, k=2) == 1.0

def test_precision_k_zero():
    assert precision_at_k(["A"], {"A"}, k=0) == 0.0


# ── recall_at_k ───────────────────────────────────────────────────────────────

def test_recall_all_found():
    assert recall_at_k(["A", "B"], {"A", "B"}, k=2) == 1.0

def test_recall_partial():
    # 1 of 2 relevant chunks found in top-3
    assert abs(recall_at_k(["A", "X", "Y"], {"A", "B"}, k=3) - 0.5) < 1e-9

def test_recall_none_found():
    assert recall_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0

def test_recall_no_relevant_returns_one():
    # No-answer query — should not penalize the retriever
    assert recall_at_k(["A", "B"], set(), k=5) == 1.0


# ── reciprocal_rank ───────────────────────────────────────────────────────────

def test_rr_first_hit_rank_1():
    assert reciprocal_rank(["A", "B", "C"], {"A"}) == 1.0

def test_rr_first_hit_rank_2():
    assert abs(reciprocal_rank(["X", "A", "B"], {"A"}) - 0.5) < 1e-9

def test_rr_first_hit_rank_3():
    assert abs(reciprocal_rank(["X", "Y", "A"], {"A"}) - 1/3) < 1e-9

def test_rr_no_hit():
    assert reciprocal_rank(["X", "Y"], {"A"}) == 0.0

def test_rr_no_relevant_returns_one():
    assert reciprocal_rank(["A", "B"], set()) == 1.0


# ── run_evaluation ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    return Embedder()

@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()

@pytest.fixture(scope="module")
def eval_store_bm25(embedder, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vs_eval")
    store = VectorStore(persist_dir=str(tmp), collection_name="test_eval")
    ingest_document(str(PDF), store, embedder)
    bm25 = build_bm25_from_store(store)
    return store, bm25

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_run_evaluation_output_shape(embedder, reranker, eval_store_bm25, tmp_path):
    store, bm25 = eval_store_bm25
    # Minimal 3-query eval file
    mini_queries = [
        {"_category": "easy", "query": "what is ATLAS", "relevant_chunk_ids": ["ACA_Report_p6_c0"]},
        {"_category": "keyword", "query": "atlas_engine_v0_2", "relevant_chunk_ids": ["ACA_Report_p19_c0"]},
        {"_category": "no-answer", "query": "what testing framework does ATLAS use", "relevant_chunk_ids": []},
    ]
    qfile = tmp_path / "mini_queries.json"
    qfile.write_text(json.dumps(mini_queries), encoding="utf-8")

    results = run_evaluation(str(qfile), store, embedder, bm25, reranker, k=5)

    assert "k" in results
    assert "n_queries" in results and results["n_queries"] == 3
    assert "dense" in results and "hybrid" in results
    for mode in ("dense", "hybrid"):
        assert "mean_precision_at_k" in results[mode]
        assert "mean_recall_at_k" in results[mode]
        assert "mrr" in results[mode]
        assert len(results[mode]["per_query"]) == 3

@pytest.mark.skipif(not pdf_available, reason="ACA_Report.pdf not found")
def test_run_evaluation_metrics_in_range(embedder, reranker, eval_store_bm25, tmp_path):
    store, bm25 = eval_store_bm25
    mini_queries = [
        {"_category": "easy", "query": "what is the model adapter layer", "relevant_chunk_ids": ["ACA_Report_p22_c0"]},
    ]
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(mini_queries), encoding="utf-8")
    results = run_evaluation(str(qfile), store, embedder, bm25, reranker, k=5)
    for mode in ("dense", "hybrid"):
        assert 0.0 <= results[mode]["mean_precision_at_k"] <= 1.0
        assert 0.0 <= results[mode]["mean_recall_at_k"]    <= 1.0
        assert 0.0 <= results[mode]["mrr"]                 <= 1.0
