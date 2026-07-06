from .citation import display_source

# Prompt wording is a real tuning lever — small changes to the instruction
# phrasing can meaningfully shift answer quality and grounding compliance.
# Keep this as a top-level constant so it's easy to iterate on.
_TEMPLATE = """\
Answer the question using ONLY the context provided below. \
If the context does not contain enough information to answer, say so explicitly \
rather than guessing or drawing on outside knowledge. \
When you use information from the context, cite it inline. \
For document sources use [Source: filename, Page X]. \
For code sources (no page number) use [Source: path/to/file.py].

Context:
{context}

Question: {query}

Answer:"""


def _format_chunk(chunk: dict) -> str:
    src = display_source(chunk["source"])
    page = chunk["page"]
    if page in (None, -1):
        header = f"[Source: {src}]"
    else:
        header = f"[Source: {src}, Page {page}]"
    return f"{header}\n{chunk['text']}"


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(_format_chunk(c) for c in chunks)
    return _TEMPLATE.format(context=context_block, query=query)
