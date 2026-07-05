"""
Pipeline-facing wrapper: takes load_codebase output, returns flat list of code chunks.
Mirrors chunk_document's role for the document pipeline.
"""
import logging
from .code_splitter import ASTCodeSplitter

logger = logging.getLogger(__name__)


def chunk_codebase(files: list[dict]) -> list[dict]:
    """
    Chunk a list of Python source files using AST parsing.

    Args:
        files: Output of load_codebase — [{"text": src, "source": rel_path, "page": None}, ...]

    Returns:
        Flat list of code chunk dicts ready for embedding.
    """
    splitter = ASTCodeSplitter()
    all_chunks: list[dict] = []
    for file_entry in files:
        try:
            chunks = splitter.split_code(file_entry["text"], file_entry["source"])
        except Exception as e:
            logger.warning("Failed to chunk '%s': %s — skipping.", file_entry["source"], e)
            continue
        all_chunks.extend(chunks)
    return all_chunks
