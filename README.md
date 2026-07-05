# DocMind

A local RAG (Retrieval-Augmented Generation) assistant that ingests PDFs and text documents and answers questions over them with grounded, cited answers.

Built from scratch as a resume/interview project — every pipeline stage is implemented directly (no LangChain wrappers) so the internals can be explained in depth.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to set OLLAMA_MODEL if needed (default: llama3.1:8b)
```

## Running the app

```bash
# 1. Start Ollama (must be running before the app)
ollama serve
ollama pull llama3.1:8b   # or whichever model is set in .env

# 2. Launch the Streamlit UI
streamlit run app.py
```

Then open http://localhost:8501, upload a PDF or .txt file, and ask questions.

## Architecture

The pipeline runs entirely locally:

```
PDF / TXT
   │
   ▼
[Ingestion]  pypdf extracts text per page with source+page metadata
   │
   ▼
[Chunking]   RecursiveCharacterSplitter (built from scratch):
             paragraph → sentence → word fallback, ~800-token chunks, 100-token overlap
   │
   ▼
[Embedding]  sentence-transformers all-MiniLM-L6-v2 (~80MB, CPU-friendly)
   │
   ▼
[Vector store]  ChromaDB persisted to ./vectorstore (cosine similarity, upsert-safe)
   │
   ▼
[Retrieval]  Hybrid pipeline:
             ├─ Dense (ChromaDB cosine)  ─┐
             └─ BM25 (rank_bm25)         ├─ RRF fusion → cross-encoder rerank
                                         │  (cross-encoder/ms-marco-MiniLM-L-6-v2)
                                         └─ top-k chunks
   │
   ▼
[Generation]  Grounded prompt → Ollama (local LLM) → answer with inline citations
   │
   ▼
[Citation validation]  Regex-parses [Source: file, Page N] from the answer,
                       verifies each cited chunk was actually in the retrieved context
   │
   ▼
[Streamlit UI]  Upload → Ask → Answer + sources + citation badge
```

**Key design decisions:**
- **Hybrid retrieval (BM25 + dense + RRF):** pure dense retrieval misses exact keyword matches; BM25 catches them; RRF merges both rankings without tuning weights.
- **Cross-encoder reranking:** bi-encoder (fast, approximate) for recall; cross-encoder (slow, precise) for final precision — classic recall-then-precision funnel.
- **Citation validation:** LLMs can hallucinate citations; every `[Source: ...]` in the answer is checked against the actual retrieved set.
- **No LangChain:** every stage is implemented directly to be fully explainable.

## Running tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

PDF tests require `data/raw/ACA_Report.pdf` (or any PDF). Ollama tests require `ollama serve` to be running.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with your chosen model pulled
