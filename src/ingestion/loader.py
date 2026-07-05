import re
import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def normalize_filename(filename: str) -> str:
    """
    Canonical source name used in all chunk metadata and citation matching.
    Spaces → underscores, strip leading/trailing whitespace.
    Case is preserved — filename casing is kept consistent to avoid silent
    mismatches between OS-level files and stored metadata.
    """
    return filename.strip().replace(" ", "_")


def _clean(text: str) -> str:
    # Collapse runs of whitespace/newlines; trim edges.
    # Known limitation: does not strip page headers/footers — those vary
    # by document and would require heuristics (Phase 3+).
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)
    source = normalize_filename(path.name)
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _clean(raw)
        if not text:
            # Likely a scanned/image-only page — no OCR support yet.
            logger.warning("Page %d of '%s' yielded no text (image-only?); skipping.", i, source)
            continue
        pages.append({"text": text, "source": source, "page": i})
    return pages


def load_text(file_path: str) -> list[dict]:
    path = Path(file_path)
    source = normalize_filename(path.name)
    text = _clean(path.read_text(encoding="utf-8"))
    return [{"text": text, "source": source, "page": None}]


def load_document(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".txt":
        return load_text(file_path)
    raise ValueError(f"Unsupported file type '{ext}'. Supported: .pdf, .txt")
