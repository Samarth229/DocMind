"""
Tests for AST-based code chunking (Phase 11).

Key test: a function long enough that RecursiveCharacterSplitter would have
split it mid-body stays as ONE complete chunk under AST chunking — the clearest
before/after demonstration of why AST chunking matters.
"""
import pytest
from pathlib import Path

from src.chunking.code_splitter import ASTCodeSplitter
from src.chunking.code_chunker import chunk_codebase
from src.ingestion.code_loader import load_codebase

SPLITTER = ASTCodeSplitter()

# ── fixture source ─────────────────────────────────────────────────────────────

SAMPLE_SOURCE = '''\
"""Module docstring."""
import os
import sys

TOP_CONST = 42


def standalone(x: int) -> int:
    """Add one."""
    return x + 1


class MyClass:
    """A sample class."""

    def __init__(self, value):
        """Init."""
        self.value = value

    def compute(self):
        """Do something."""
        return self.value * 2

if __name__ == "__main__":
    print(standalone(1))
'''


# ── basic structure tests ──────────────────────────────────────────────────────

def test_top_level_function_detected():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    funcs = [c for c in chunks if c["chunk_type"] == "function"]
    assert any(c["function_name"] == "standalone" for c in funcs)


def test_methods_chunked_separately():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    methods = [c for c in chunks if c["chunk_type"] == "method"]
    names = {c["function_name"] for c in methods}
    assert "__init__" in names
    assert "compute" in names


def test_method_carries_class_name():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    init = next(c for c in chunks if c["function_name"] == "__init__")
    assert init["class_name"] == "MyClass"


def test_docstring_extracted():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    standalone = next(c for c in chunks if c["function_name"] == "standalone")
    assert standalone["docstring"] == "Add one."


def test_module_level_chunk_captured():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    mod = [c for c in chunks if c["chunk_type"] == "module_level"]
    assert len(mod) == 1
    # imports and TOP_CONST should be in the module-level text
    assert "import" in mod[0]["text"]


def test_chunk_shape_has_required_fields():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    for c in chunks:
        for field in ("chunk_id", "text", "source", "page", "chunk_type",
                      "function_name", "class_name", "start_line", "end_line"):
            assert field in c, f"Missing field '{field}' in chunk {c.get('chunk_id')}"
        assert c["page"] is None  # code chunks always have page=None


def test_chunk_count():
    chunks = SPLITTER.split_code(SAMPLE_SOURCE, "sample.py")
    # standalone(func) + __init__(method) + compute(method) + module_level = 4
    assert len(chunks) == 4


# ── the key AST vs character-split demonstration ───────────────────────────────

def test_long_function_stays_as_one_chunk():
    """
    A function with ~1 200 words of body would be split into 2 chunks by
    RecursiveCharacterSplitter (chunk_size=800 tokens). AST chunking keeps it
    as exactly ONE chunk regardless of length, because the function boundary
    is the semantically correct split point.
    """
    # Build a function whose body is well over 800 words
    body_lines = [f"    result_{i} = i * {i}  # step {i}" for i in range(200)]
    long_source = "def long_computation():\n" + "\n".join(body_lines) + "\n    return result_199\n"
    word_count = len(long_source.split())

    chunks = SPLITTER.split_code(long_source, "long.py")
    func_chunks = [c for c in chunks if c["chunk_type"] == "function"]

    assert len(func_chunks) == 1, (
        f"Expected 1 chunk for long_computation, got {len(func_chunks)}. "
        f"Function had {word_count} words — would have been split by character chunking."
    )
    assert func_chunks[0]["function_name"] == "long_computation"
    assert "result_199" in func_chunks[0]["text"]
    # Report the word count so it's visible in the test output
    print(f"\n  long_computation: {word_count} words → 1 AST chunk (char splitter would have made 2+)")


# ── graceful error handling ────────────────────────────────────────────────────

def test_syntax_error_does_not_crash_chunk_codebase():
    """chunk_codebase must not raise even when a file has a syntax error."""
    bad_files = [
        {"text": "def broken(:\n    pass\n", "source": "broken.py", "page": None},
        {"text": "def good():\n    return 1\n", "source": "good.py", "page": None},
    ]
    # Should not raise — bad file is skipped or parsed best-effort
    chunks = chunk_codebase(bad_files)
    # At minimum the good file should produce a chunk
    good = [c for c in chunks if c["source"] == "good.py"]
    assert len(good) >= 1


# ── end-to-end on DocMind's own src/ ──────────────────────────────────────────

SRC_DIR = Path(__file__).parent.parent / "src"


def test_load_codebase_finds_py_files():
    files = load_codebase(str(SRC_DIR))
    assert len(files) >= 5, "Expected at least 5 .py files in src/"
    sources = [f["source"] for f in files]
    # Known files that must be present
    assert any("embedder" in s for s in sources)
    assert any("splitter" in s for s in sources)


def test_chunk_codebase_on_docmind_src():
    files = load_codebase(str(SRC_DIR))
    chunks = chunk_codebase(files)
    assert len(chunks) >= 20, f"Expected ≥20 chunks from src/, got {len(chunks)}"


def test_known_function_present_in_chunks():
    files = load_codebase(str(SRC_DIR))
    chunks = chunk_codebase(files)
    # reciprocal_rank_fusion is a known top-level function in src/retrieval/fusion.py
    rrfs = [c for c in chunks if c.get("function_name") == "reciprocal_rank_fusion"]
    assert len(rrfs) >= 1, "reciprocal_rank_fusion function not found in chunked src/"
    # Confirm the complete body is present (not truncated mid-function)
    assert "rrf_score" in rrfs[0]["text"]


def test_known_class_method_present():
    files = load_codebase(str(SRC_DIR))
    chunks = chunk_codebase(files)
    # embed_query is a method on Embedder in src/embedding/embedder.py
    eq = [c for c in chunks if c.get("function_name") == "embed_query"]
    assert len(eq) >= 1
    assert eq[0]["class_name"] == "Embedder"


def test_source_is_absolute_path():
    """load_codebase must use absolute paths so the same file always gets the same source."""
    files = load_codebase(str(SRC_DIR))
    for f in files:
        p = Path(f["source"])
        assert p.is_absolute(), (
            f"Expected absolute path, got '{f['source']}'. "
            "Relative paths cause duplicate chunk entries when the same file is ingested "
            "from two different root directories."
        )


def test_no_duplicate_source_from_different_roots(tmp_path):
    """
    Ingesting a parent dir then one of its subdirs must NOT produce two source
    strings for the same physical file — absolute paths prevent this.
    """
    # Build: tmp/pkg/__init__.py  and  tmp/pkg/sub/__init__.py
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("def top(): pass\n")
    (sub / "__init__.py").write_text("def nested(): pass\n")

    files_from_parent = load_codebase(str(tmp_path))
    files_from_sub    = load_codebase(str(sub))

    all_sources = [f["source"] for f in files_from_parent + files_from_sub]

    # sub/__init__.py appears in both ingest calls; its source must be identical
    sub_init_abs = (sub / "__init__.py").resolve().as_posix()
    sub_occurrences = [s for s in all_sources if s == sub_init_abs]
    assert len(sub_occurrences) == 2, (
        "Expected exactly 2 entries (one per load_codebase call) with the same "
        f"absolute path '{sub_init_abs}', got: {all_sources}"
    )

    # Because chunk_id is derived from source + function name, identical chunk_ids
    # mean ChromaDB upsert will overwrite rather than duplicate — verify:
    chunks_parent = chunk_codebase(files_from_parent)
    chunks_sub    = chunk_codebase(files_from_sub)

    ids_parent = {c["chunk_id"] for c in chunks_parent if "nested" in c.get("function_name", "")}
    ids_sub    = {c["chunk_id"] for c in chunks_sub    if "nested" in c.get("function_name", "")}

    assert ids_parent == ids_sub, (
        "chunk_ids for the same function differ between parent and subdirectory ingestion — "
        "ChromaDB upsert would create duplicates instead of overwriting.\n"
        f"  From parent: {ids_parent}\n"
        f"  From sub:    {ids_sub}"
    )
