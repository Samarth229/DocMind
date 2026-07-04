class RecursiveCharacterSplitter:
    """
    Splits text using a hierarchy of separators, falling back to finer-grained
    splits only when a piece still exceeds chunk_size.

    Separators are tried in priority order: paragraph breaks → sentence breaks →
    word breaks → character-level. This preserves semantic units (paragraphs,
    sentences) without the latency/cost of embedding-based semantic chunking —
    the industry-default tradeoff for retrieval pipelines.

    Length is measured in approximate tokens via word count (len(text.split())).
    A real tokenizer (e.g. tiktoken) would be a straightforward drop-in upgrade
    for exact token counting, but adds a hard dependency for marginal accuracy gain.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    # ── length proxy ──────────────────────────────────────────────────────────

    @staticmethod
    def _length(text: str) -> int:
        return len(text.split())

    # ── core split ────────────────────────────────────────────────────────────

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """Merge a flat list of small splits into chunks, adding overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in splits:
            piece_len = self._length(piece)

            if current_len + piece_len > self.chunk_size and current:
                # Emit the current chunk.
                chunks.append(" ".join(current).strip())

                # Build overlap: walk back from the end of current until we've
                # accumulated chunk_overlap tokens' worth of content.
                overlap: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    s_len = self._length(s)
                    if overlap_len + s_len > self.chunk_overlap:
                        break
                    overlap.insert(0, s)
                    overlap_len += s_len
                current = overlap
                current_len = overlap_len

            current.append(piece)
            current_len += piece_len

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text until all pieces fit within chunk_size."""
        if not text.strip():
            return []

        # If the text already fits, no split needed.
        if self._length(text) <= self.chunk_size:
            return [text.strip()]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Last resort: hard split by words.
            words = text.split()
            pieces = [
                " ".join(words[i : i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size - self.chunk_overlap)
            ]
            return pieces

        raw_splits = text.split(sep)

        # Re-attach the separator to maintain readability (except for whitespace seps).
        if sep not in ("\n\n", "\n", " "):
            raw_splits = [s + sep for s in raw_splits[:-1]] + [raw_splits[-1]]

        # Recursively break any piece that's still too big.
        fine_splits: list[str] = []
        for piece in raw_splits:
            piece = piece.strip()
            if not piece:
                continue
            if self._length(piece) > self.chunk_size and remaining_seps:
                fine_splits.extend(self._split_recursive(piece, remaining_seps))
            else:
                fine_splits.append(piece)

        return self._merge_with_overlap(fine_splits)

    def split_text(self, text: str) -> list[str]:
        return self._split_recursive(text, self.separators)
