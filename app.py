import streamlit as st
from pathlib import Path

from src.ingestion import normalize_filename
from src.embedding import Embedder, VectorStore, ingest_document
from src.retrieval import CrossEncoderReranker, build_bm25_from_store
from src.generation import OllamaProvider, answer_query

RAW_DIR      = Path("data/raw")
PERSIST_DIR  = "./vectorstore"
COLLECTION   = "docmind"

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

# ── cached resources (loaded once per session) ────────────────────────────────

@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_store():
    return VectorStore(persist_dir=PERSIST_DIR, collection_name=COLLECTION)

@st.cache_resource
def get_reranker():
    return CrossEncoderReranker()

@st.cache_resource
def get_provider():
    return OllamaProvider()

# ── session state defaults ────────────────────────────────────────────────────

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "history" not in st.session_state:
    st.session_state.history = []


def _already_in_store(norm_name: str) -> bool:
    """Check persistent vectorstore for any chunk with this normalized source name."""
    store = get_store()
    results = store._col.get(where={"source": norm_name}, limit=1)
    return len(results["ids"]) > 0


# ── sidebar: upload & ingest ──────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 DocMind")
    st.caption("Local RAG assistant — answers grounded in your documents.")
    st.divider()

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        help="PDF or plain-text files. The document is chunked, embedded, and stored locally.",
    )

    if uploaded is not None:
        norm_name = normalize_filename(uploaded.name)
        dest = RAW_DIR / norm_name  # save under normalized name

        if norm_name in st.session_state.ingested_docs:
            st.info(f"{norm_name} already ingested this session.")
        elif _already_in_store(norm_name):
            # Document persisted from a prior session — register without re-ingesting.
            st.session_state.ingested_docs.append(norm_name)
            if st.session_state.bm25_index is None:
                st.session_state.bm25_index = build_bm25_from_store(get_store())
            st.info(f"{norm_name} already exists in the vector store (loaded from disk).")
        else:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getvalue())
            with st.spinner(f"Ingesting {norm_name}…"):
                try:
                    n = ingest_document(str(dest), get_store(), get_embedder())
                    st.session_state.bm25_index = build_bm25_from_store(get_store())
                    st.session_state.ingested_docs.append(norm_name)
                    st.success(f"✓ {norm_name} — {n} chunks indexed")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested_docs:
        st.divider()
        st.markdown("**Ingested documents**")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"- {doc}")

# ── main area ─────────────────────────────────────────────────────────────────

st.header("Ask a question")

if not st.session_state.ingested_docs:
    st.info("Upload a PDF or text file in the sidebar to get started.")
    st.stop()

query = st.chat_input("Ask something about your documents…")

if query:
    if st.session_state.bm25_index is None:
        with st.spinner("Building search index…"):
            st.session_state.bm25_index = build_bm25_from_store(get_store())

    with st.spinner("Thinking…"):
        try:
            result = answer_query(
                query,
                get_store(),
                get_embedder(),
                st.session_state.bm25_index,
                get_reranker(),
                get_provider(),
            )
            st.session_state.history.append({"query": query, **result})
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                st.error("Ollama isn't running. Start it with `ollama serve` and try again.")
            else:
                st.error(f"Error: {e}")

# ── render history ────────────────────────────────────────────────────────────

for item in reversed(st.session_state.history):
    st.divider()
    st.markdown(f"**Q: {item['query']}**")
    st.markdown(item["answer"])

    check = item["citation_check"]
    if check["total_citations"] == 0:
        st.caption("No inline citations in this answer.")
    elif check["all_valid"]:
        st.success(f"✓ All {check['total_citations']} citation(s) verified against retrieved context.")
    else:
        st.warning(
            f"⚠ {len(check['invalid_citations'])} citation(s) not found in retrieved context: "
            + ", ".join(
                f"[{c['source']}, p.{c['page']}]" for c in check["invalid_citations"]
            )
        )

    with st.expander(f"📚 Retrieved chunks ({len(item['chunks_used'])})"):
        for i, chunk in enumerate(item["chunks_used"], 1):
            page = chunk["page"] if chunk["page"] not in (None, -1) else "N/A"
            score = chunk.get("rerank_score", chunk.get("rrf_score", ""))
            score_str = f"  `rerank={score:.2f}`" if isinstance(score, float) else ""
            st.markdown(f"**#{i} — {chunk['source']}, page {page}**{score_str}")
            st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
