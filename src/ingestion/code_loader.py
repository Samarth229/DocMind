"""
Code ingestion: read raw Python source files from disk.
Parsing and AST chunking happen downstream in src/chunking/code_splitter.py.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_DIRS = {"__pycache__", ".venv", ".git", ".mypy_cache", ".pytest_cache", "node_modules"}


def load_python_file(file_path: str) -> str:
    """Read a .py file as raw text."""
    return Path(file_path).read_text(encoding="utf-8", errors="replace")


def load_codebase(directory_path: str) -> list[dict]:
    """
    Walk a directory recursively and return one entry per .py file.

    Returns:
        [{"text": raw_source, "source": abs_path_forward_slash, "page": None}, ...]

    source is the *absolute* path (forward-slash-normalized) of each file, not
    a path relative to whatever directory was passed in. Relative paths change
    depending on which directory the caller points at, so the same physical file
    gets different source values on different ingestion calls — an absolute path
    is a stable, canonical identity regardless of how or when the file is ingested.

    page=None keeps the dict shape consistent with the document pipeline so
    retrieval, generation, and citation work unchanged downstream.

    Known simplification: absolute paths are meaningful only on this machine; if
    the project is moved, stored source values would need to be updated. That's
    an acceptable trade-off for this single-user local tool.
    """
    root = Path(directory_path).resolve()
    results = []
    for py_file in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        # Absolute path, forward slashes for consistency across the codebase.
        abs_source = py_file.resolve().as_posix()
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read '%s': %s — skipping.", py_file, e)
            continue
        results.append({"text": text, "source": abs_source, "page": None})
    return results
