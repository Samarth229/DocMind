# DocMind

**A locally-running RAG assistant that ingests PDFs, text documents, and Python codebases, and answers questions with grounded, cited answers. No API keys, no cloud, built entirely from scratch.**

---

### 🧠 What it is

DocMind reads your documents and code, finds the right information using hybrid search (keyword + semantic), and answers **only** from what it actually found — no answer without evidence. Every response cites exactly where it came from, and DocMind independently verifies its own citations rather than trusting the LLM to get them right.

> **Headline result:** Hybrid retrieval (BM25 + dense + RRF + cross-encoder reranking) improved MRR from **0.62** (dense-only) to **0.89** on a 25-query hand-labeled evaluation set.

---

### ⚙️ Features

**Ingestion**
- PDF and text ingestion via `pypdf`, with page-level metadata preserved
- Python codebase ingestion with recursive directory walking

**Chunking**
- Documents: a from-scratch recursive character splitter (paragraph → sentence → word fallback, ~800 tokens, with overlap)
- Code: AST-based chunking via `tree-sitter`, splitting by function/class boundaries so long functions are never cut mid-body

**Retrieval**
- Hybrid dense retrieval (ChromaDB, cosine similarity) + BM25 keyword search
- Merged via Reciprocal Rank Fusion (RRF), then refined with cross-encoder reranking

**Generation**
- Local LLM via Ollama, with a grounded prompt that correctly declines to answer when the retrieved context doesn't contain the answer

**Citation Validation**
- Every cited source is checked against the chunks that were actually retrieved
- Caught a real hallucinated citation during testing — the model cited textbook pages that were never retrieved

**Interfaces**
- Streamlit UI for interactive use
- FastAPI REST API — `/ingest`, `/query`, `/documents`, `/health`
- Two-click "Clear all data" reset

---

### 🏗️ Architecture

```
Ingestion
   ↓
Chunking
   ↓
Embedding (sentence-transformers · all-MiniLM-L6-v2)
   ↓
Vector Store (ChromaDB)
   ↓
Retrieval (hybrid dense + BM25 → RRF → cross-encoder rerank)
   ↓
Generation (Ollama)
   ↓
Citation Validation
   ↓
Streamlit UI / FastAPI
```

---

### 🎯 Key Design Decisions

| Decision | Why |
|---|---|
| Hybrid retrieval | Catches keyword matches that dense-only search misses |
| Cross-encoder reranking | Recall-then-precision funnel — cast a wide net, then refine |
| AST-based code chunking | Keeps functions and classes whole, never split mid-body |
| Citation validation | Catches hallucinated sources before they reach the user |
| Stable source identity | Normalized paths prevent duplicate chunks on re-ingestion |
| No LangChain | Every component is hand-built and fully explainable |

---

### 📊 Evaluation

Measured on a 25-query hand-labeled evaluation set, comparing dense-only retrieval against the full hybrid + reranking pipeline:

| Metric | Dense-only | Hybrid + Rerank |
|---|---|---|
| Precision@5 | 0.20 | 0.272 |
| Recall@5 | 0.655 | 0.857 |
| MRR | 0.625 | 0.887 |

**Per-category MRR (hybrid + rerank):**

| Category | MRR |
|---|---|
| Easy | 1.00 |
| Keyword | 0.93 |
| No-answer | 1.00 |
| Semantic | 0.71 |

---

### 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama — `llama3.1:8b` |
| Embeddings | sentence-transformers — `all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Keyword search | `rank_bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Code parsing | `tree-sitter` |
| PDF parsing | `pypdf` |
| UI | Streamlit |
| API | FastAPI |
| Language | Python 3.10+ |

---

### 🚀 Setup & Running

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start Ollama and pull the model
ollama serve
ollama pull llama3.1:8b

# 5. Run the Streamlit UI
streamlit run app.py
# → http://localhost:8501

# 6. (Optional) Run the FastAPI server
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

---

### ⚠️ Known Limitations

- Citation *completeness* isn't enforced — the system catches false citations, but not missing ones
- Per-page chunking means page-boundary sentences can occasionally be split
- No cross-file reasoning for code (each file is chunked independently)
- AST-based chunking currently supports Python only
- No OCR — scanned/image-only PDFs aren't supported
- Single-user, local-only — not designed for concurrent multi-user deployment

---

### 👤 Built By

**Samarth Kadam** — B.Tech CS, VIT Bhopal
Built July 2026.

### 📄 License

MIT License
