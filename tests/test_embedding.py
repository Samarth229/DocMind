import pytest
import tempfile
import shutil
from pathlib import Path

from src.embedding import Embedder, VectorStore, ingest_document

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture()
def tmp_store(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "vs"), collection_name="test")
    yield store


# ── Embedder ──────────────────────────────────────────────────────────────────

def test_embed_texts_returns_correct_count(embedder):
    texts = ["hello world", "foo bar baz"]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 2

def test_embed_texts_correct_dimensionality(embedder):
    vecs = embedder.embed_texts(["test sentence"])
    assert len(vecs[0]) == VECTOR_DIM

def test_embed_query_returns_single_vector(embedder):
    vec = embedder.embed_query("what is ATLAS?")
    assert len(vec) == VECTOR_DIM
    assert isinstance(vec[0], float)


# ── VectorStore ───────────────────────────────────────────────────────────────

def test_add_and_query_roundtrip(embedder, tmp_store):
    chunks = [
        {"chunk_id": "doc_p1_c0", "text": "The sky is blue and full of clouds.", "source": "doc.txt", "page": 1, "chunk_index": 0},
        {"chunk_id": "doc_p1_c1", "text": "Python is a popular programming language.", "source": "doc.txt", "page": 1, "chunk_index": 1},
        {"chunk_id": "doc_p1_c2", "text": "Machine learning models learn from data.", "source": "doc.txt", "page": 1, "chunk_index": 2},
    ]
    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    tmp_store.add_chunks(chunks, embeddings)

    results = tmp_store.query(embedder.embed_query("programming language syntax"), k=3)
    assert len(results) > 0
    top_ids = [r["chunk_id"] for r in results]
    assert "doc_p1_c1" in top_ids[:2]  # Python chunk should rank high

def test_query_result_fields(embedder, tmp_store):
    chunks = [{"chunk_id": "x_p1_c0", "text": "Vector databases store embeddings.", "source": "x.txt", "page": 1, "chunk_index": 0}]
    tmp_store.add_chunks(chunks, embedder.embed_texts([c["text"] for c in chunks]))
    results = tmp_store.query(embedder.embed_query("embeddings storage"), k=1)
    r = results[0]
    assert "chunk_id" in r
    assert "text" in r
    assert "source" in r
    assert "page" in r
    assert "score" in r

def test_upsert_no_duplicates(embedder, tmp_store):
    chunk = {"chunk_id": "dup_p1_c0", "text": "Deduplication test chunk.", "source": "dup.txt", "page": 1, "chunk_index": 0}
    emb = embedder.embed_texts([chunk["text"]])
    tmp_store.add_chunks([chunk], emb)
    count_before = tmp_store.count()
    tmp_store.add_chunks([chunk], emb)  # same chunk_id — should upsert, not duplicate
    assert tmp_store.count() == count_before

def test_none_page_stored_as_minus_one(embedder, tmp_store):
    chunk = {"chunk_id": "txt_p0_c0", "text": "Text file with no page number.", "source": "file.txt", "page": None, "chunk_index": 0}
    tmp_store.add_chunks([chunk], embedder.embed_texts([chunk["text"]]))
    results = tmp_store.query(embedder.embed_query("text file page"), k=1)
    assert results[0]["page"] == -1


# ── ingest_document pipeline ──────────────────────────────────────────────────

def test_ingest_txt_pipeline(embedder, tmp_store):
    txt = Path(__file__).parent.parent / "data" / "raw" / "sample.txt"
    n = ingest_document(str(txt), tmp_store, embedder)
    assert n >= 1
    assert tmp_store.count() >= 1

@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf").exists(),
    reason="ACA_Report.pdf not found in data/raw/",
)
def test_ingest_pdf_pipeline(embedder, tmp_store):
    pdf = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
    n = ingest_document(str(pdf), tmp_store, embedder)
    assert n == 49  # known chunk count from Phase 2

@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf").exists(),
    reason="ACA_Report.pdf not found in data/raw/",
)
def test_ingest_pdf_idempotent(embedder, tmp_store):
    pdf = Path(__file__).parent.parent / "data" / "raw" / "ACA_Report.pdf"
    ingest_document(str(pdf), tmp_store, embedder)
    count_after_first = tmp_store.count()
    ingest_document(str(pdf), tmp_store, embedder)
    assert tmp_store.count() == count_after_first
