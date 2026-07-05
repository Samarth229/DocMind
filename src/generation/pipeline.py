from src.embedding.embedder import Embedder
from src.embedding.store import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.pipeline import retrieve
from .prompt import build_prompt
from .provider import LLMProvider
from .citation import extract_citations, validate_citations


def answer_query(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    bm25_index: BM25Index,
    reranker: CrossEncoderReranker,
    provider: LLMProvider,
    k: int = 5,
) -> dict:
    """
    Full RAG pipeline: retrieve -> prompt -> generate -> validate citations.

    citation_check is always present so the Streamlit UI (Phase 7) can surface
    a warning badge when all_valid is False without additional logic.
    """
    chunks = retrieve(query, store, embedder, bm25_index, reranker, k=k)
    prompt = build_prompt(query, chunks)
    answer = provider.generate(prompt)
    citations = extract_citations(answer)
    citation_check = validate_citations(citations, chunks)
    return {
        "answer": answer,
        "chunks_used": chunks,
        "prompt": prompt,
        "citation_check": citation_check,
    }
