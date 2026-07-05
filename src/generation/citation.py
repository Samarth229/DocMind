import re

# Matches: [Source: <filename>, Page <number>]
# Tolerates extra whitespace around commas and "Page".
# Why validate at all: LLMs can hallucinate citations (a model can output
# [Source: file.pdf, Page 12] even if that page was never in the retrieved
# context). Checking against the actual retrieved set catches this silently
# failing failure mode that most RAG implementations skip.
_CITATION_RE = re.compile(
    r"\[Source:\s*(.+?)\s*,\s*Page\s*(\d+|N/A)\s*\]",
    re.IGNORECASE,
)


def extract_citations(answer_text: str) -> list[dict]:
    """Parse all [Source: ..., Page ...] citations from an answer string."""
    matches = _CITATION_RE.findall(answer_text)
    results = []
    for source, page in matches:
        results.append({
            "source": source.strip(),
            "page": int(page) if page.upper() != "N/A" else None,
        })
    return results


def validate_citations(
    citations: list[dict],
    retrieved_chunks: list[dict],
) -> dict:
    """
    Verify each citation against the chunks that were actually in the prompt.

    A citation is valid iff a retrieved chunk exists with the same source filename
    and page number. Invalid citations indicate the model cited a source/page that
    was not part of its input context — a hallucinated reference.
    """
    retrieved_keys = {
        (c["source"], c["page"] if c["page"] not in (None, -1) else None)
        for c in retrieved_chunks
    }

    invalid: list[dict] = []
    for cit in citations:
        key = (cit["source"], cit["page"])
        if key not in retrieved_keys:
            invalid.append(cit)

    return {
        "total_citations": len(citations),
        "valid_citations": len(citations) - len(invalid),
        "invalid_citations": invalid,
        "all_valid": len(invalid) == 0,
    }
