import re
from pathlib import Path

# Project root is two levels above this file: src/generation/citation.py → project/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Matches: [Source: <source>] or [Source: <source>, Page <n|N/A>]
# Page portion is optional — code chunks have no page concept so the model is
# instructed to omit it entirely rather than write "Page N/A".
_CITATION_RE = re.compile(
    r"\[Source:\s*(.+?)\s*(?:,\s*Page\s*(\d+|N/A)\s*)?\]",
    re.IGNORECASE,
)

# Why validate at all: LLMs can hallucinate citations (a model can output
# [Source: file.pdf, Page 12] even if that page was never in the retrieved
# context). Checking against the actual retrieved set catches this silently
# failing failure mode that most RAG implementations skip.


def display_source(source: str) -> str:
    """
    Shorten a source path for display — in the LLM prompt and in the UI.
    The absolute path stored internally is never changed; this is purely cosmetic.

    Rules:
    - Absolute path inside the project root → relative path (e.g. src/retrieval/fusion.py)
    - Absolute path outside the project root → parent_dir/filename (e.g. monitoring/stats_engine.py)
    - Already a simple filename → unchanged (PDF/text docs with no path prefix)
    """
    p = Path(source.replace("\\", "/"))
    if not p.is_absolute():
        return source  # already a short name (PDF/txt doc sources)
    try:
        rel = p.relative_to(_PROJECT_ROOT)
        return rel.as_posix()
    except ValueError:
        # Outside the project root — show parent_dir/filename for context
        return f"{p.parent.name}/{p.name}"


def extract_citations(answer_text: str) -> list[dict]:
    """Parse all [Source: ...] and [Source: ..., Page ...] citations from an answer string."""
    results = []
    for source, page in _CITATION_RE.findall(answer_text):
        results.append({
            "source": source.strip(),
            "page": int(page) if page and page.upper() != "N/A" else None,
        })
    return results


def validate_citations(
    citations: list[dict],
    retrieved_chunks: list[dict],
) -> dict:
    """
    Verify each citation against the chunks that were actually in the prompt.

    Both sides go through display_source so they share the same shortened form —
    the model sees and generates display_source strings (set by build_prompt),
    and retrieved_chunks have the internal absolute paths that need the same
    transformation before comparison.

    A citation is valid iff a retrieved chunk exists with the same display_source
    and page. Invalid citations indicate hallucinated references.
    """
    retrieved_keys = {
        (display_source(c["source"]), c["page"] if c["page"] not in (None, -1) else None)
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
