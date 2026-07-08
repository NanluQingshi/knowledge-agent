"""Recursive chunking strategy — LangChain-style hierarchical splitting."""

from knowledge_agent.chunkers.base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Recursive chunker that tries progressively finer separators.

    Separator hierarchy (tried in order):

        ``\\n\\n`` → ``\\n`` → ``. `` → `` `` → ``""`` (character)

    For each separator the text is split; pieces that are still larger
    than ``chunk_size`` are passed down to the next separator in the
    hierarchy.  Adjacent small pieces are merged back into chunks that
    respect ``chunk_size`` and ``chunk_overlap`` (measured in tokens).
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)

    # ------------------------------------------------------------------
    # Token counting helper
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return self.count_tokens(text)

    # ------------------------------------------------------------------
    # Recursive split logic
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the given *separators*."""
        if not text:
            return []

        if not separators:
            # Character-level: no more separators to try.
            return self._split_by_char(text)

        separator = separators[0]
        remaining = separators[1:]

        splits = text.split(separator) if separator else list(text)

        good: list[str] = []
        result: list[str] = []

        for piece in splits:
            if not piece:
                continue

            if self._count_tokens(piece) < self.chunk_size:
                good.append(piece)
            else:
                # Flush accumulated good pieces.
                if good:
                    result.extend(self._merge_splits(good, separator))
                    good = []

                # Recurse on this oversized piece with finer separators.
                if remaining:
                    result.extend(self._split_text(piece, remaining))
                else:
                    result.extend(self._split_text(piece, []))

        # Flush any remaining good pieces.
        if good:
            result.extend(self._merge_splits(good, separator))

        return result

    def _split_by_char(self, text: str) -> list[str]:
        """Fallback: split text into chunks of at most chunk_size characters."""
        results = []
        for i in range(0, len(text), self.chunk_size):
            results.append(text[i : i + self.chunk_size])
        return results

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge a list of text pieces into chunks respecting size & overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        sep_tokens = self._count_tokens(separator) if separator else 0

        for piece in splits:
            piece_tokens = self._count_tokens(piece)
            add_sep = sep_tokens if current else 0

            if current and current_tokens + add_sep + piece_tokens > self.chunk_size:
                # Finalise the current chunk.
                chunks.append(separator.join(current))

                # Build overlap from the tail of the exhausted chunk.
                overlap_pieces: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    p_tokens = self._count_tokens(p)
                    s_cost = sep_tokens if overlap_pieces else 0
                    if overlap_tokens + s_cost + p_tokens <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_tokens += s_cost + p_tokens
                    else:
                        break

                current = overlap_pieces
                current_tokens = overlap_tokens

            current.append(piece)
            current_tokens += piece_tokens + (sep_tokens if len(current) > 1 else 0)

        if current:
            chunks.append(separator.join(current))

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* using recursive separator hierarchy."""
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text, list(self.SEPARATORS))
        meta = metadata or {}

        return [
            Chunk(
                text=chunk,
                metadata={**meta, "chunk_index": idx},
                chunk_index=idx,
            )
            for idx, chunk in enumerate(raw_chunks)
        ]
