import re
from .splitter import RecursiveCharacterSplitter

_splitter = RecursiveCharacterSplitter()


def _make_chunk_id(source: str, page: int | None, index: int) -> str:
    stem = re.sub(r"\W+", "_", source.rsplit(".", 1)[0])
    page_label = f"p{page}" if page is not None else "p0"
    return f"{stem}_{page_label}_c{index}"


def chunk_document(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split a list of page dicts (from load_document) into retrieval-sized chunks.

    Each output dict carries the metadata contract that Phase 3 (embedding) and
    Phase 6 (citation) depend on — keep field names stable.
    """
    splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict] = []

    for page in pages:
        texts = splitter.split_text(page["text"])
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(page["source"], page["page"], i),
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": i,
                }
            )

    return chunks
