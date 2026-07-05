# Prompt wording is a real tuning lever — small changes to the instruction
# phrasing can meaningfully shift answer quality and grounding compliance.
# Keep this as a top-level constant so it's easy to iterate on.
_TEMPLATE = """\
Answer the question using ONLY the context provided below. \
If the context does not contain enough information to answer, say so explicitly \
rather than guessing or drawing on outside knowledge. \
When you use information from the context, cite it inline like [Source: filename, Page X].

Context:
{context}

Question: {query}

Answer:"""


def _format_chunk(chunk: dict) -> str:
    page = chunk["page"] if chunk["page"] not in (None, -1) else "N/A"
    return f"[Source: {chunk['source']}, Page {page}]\n{chunk['text']}"


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(_format_chunk(c) for c in chunks)
    return _TEMPLATE.format(context=context_block, query=query)
