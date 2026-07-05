"""
DocMind REST API — thin wrapper around the same pipeline functions Streamlit uses.
Both entrypoints (app.py and api/main.py) call src/* directly; no logic is duplicated.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from src.ingestion import normalize_filename
from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import CrossEncoderReranker, build_bm25_from_store
from src.generation import OllamaProvider, answer_query
from .schemas import (
    QueryRequest, QueryResponse, ChunkResult, CitationCheck,
    IngestResponse, DocumentsResponse, HealthResponse,
)

RAW_DIR     = Path("data/raw")
PERSIST_DIR = "./vectorstore"
COLLECTION  = "docmind"
ALLOWED_EXT = {".pdf", ".txt"}

# ── module-level singletons (loaded once at startup) ─────────────────────────
# Equivalent to Streamlit's @st.cache_resource — the business logic lives in
# src/, this is just the front door. Both Streamlit and FastAPI share the same
# ChromaDB vectorstore on disk.

_embedder: Embedder | None = None
_store:    VectorStore | None = None
_reranker: CrossEncoderReranker | None = None
_provider: OllamaProvider | None = None
_bm25      = None   # BM25Index — rebuilt after each ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _reranker, _provider, _bm25
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _embedder = Embedder()
    _store    = VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)
    _reranker = CrossEncoderReranker()
    _provider = OllamaProvider()
    _bm25     = build_bm25_from_store(_store)
    yield
    # No teardown needed for these resources.


app = FastAPI(
    title="DocMind API",
    description="Local RAG assistant — ingest documents, ask questions, get grounded answers.",
    version="1.0.0",
    lifespan=lifespan,
)


def _rebuild_bm25():
    global _bm25
    _bm25 = build_bm25_from_store(_store)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness check — confirms vectorstore and Ollama are reachable."""
    try:
        import ollama as _ol
        _ol.list()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama is not reachable. Start it with `ollama serve`.",
        )
    return HealthResponse(
        status="ok",
        vectorstore_chunks=_store.count(),
        ollama_model=_provider.model,
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents():
    """Return all unique source documents currently in the vectorstore."""
    raw = _store._col.get(include=["metadatas"])
    sources = sorted({m["source"] for m in raw["metadatas"]})
    return DocumentsResponse(documents=sources, total_chunks=_store.count())


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF or .txt file and ingest it into the vectorstore.

    Known limitation: large file ingestion blocks this request synchronously.
    A background task queue (e.g. Celery + Redis) would be the production fix.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported: .pdf, .txt",
        )

    norm_name = normalize_filename(file.filename)
    dest = RAW_DIR / norm_name
    dest.write_bytes(file.file.read())

    try:
        n = ingest_document(str(dest), _store, _embedder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    _rebuild_bm25()
    return IngestResponse(filename=norm_name, chunks_ingested=n, status="success")


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest):
    """
    Ask a question over all ingested documents.
    Returns a grounded answer with source citations and citation validation.
    """
    if _store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents ingested yet. POST a file to /ingest first.",
        )

    try:
        result = answer_query(
            req.question,
            _store,
            _embedder,
            _bm25,
            _reranker,
            _provider,
            k=req.k,
        )
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower() or "not found" in err.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama error: {e}. Ensure `ollama serve` is running and the model is pulled.",
            )
        raise HTTPException(status_code=500, detail=str(e))

    chunks = [
        ChunkResult(
            chunk_id    = c["chunk_id"],
            text        = c["text"],
            source      = c["source"],
            page        = c["page"] if c["page"] != -1 else None,
            rerank_score= c.get("rerank_score"),
            rrf_score   = c.get("rrf_score"),
        )
        for c in result["chunks_used"]
    ]
    check = result["citation_check"]
    return QueryResponse(
        answer       = result["answer"],
        chunks_used  = chunks,
        citation_check = CitationCheck(**check),
    )
