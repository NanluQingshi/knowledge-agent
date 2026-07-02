"""Semantic chunking strategy — split text by sentence boundaries."""

import re

import tiktoken

from knowledge_agent.chunkers.base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Semantic chunker that groups sentences together.

    Uses ``nltk.sent_tokenize`` when available; falls back to a simple
    regex split on sentence-ending punctuation (``.``, ``!``, ``?``,
    as well as Chinese ``。``, ``！``, ``？``).

    Sentences are accumulated until the combined token count reaches
    ``chunk_size``.  The ``chunk_overlap`` controls how many sentences
    from the end of the previous chunk are repeated at the beginning
    of the next chunk (measured in **sentences**, not tokens).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 1) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, chunk_overlap)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Return a list of sentences extracted from *text*."""
        try:
            import nltk  # noqa: F811

            return nltk.sent_tokenize(text)
        except (ImportError, LookupError):
            # Graceful fallback — no NLTK punkt data available.
            # Keep the sentence-ending punctuation on the left so that
            # ``" ".join(sentences)`` is still readable.
            parts = re.split(
                r"(?<=[.!?])\s+|(?<=[。！？])",
                text,
            )
            return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Token counting helper
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* along sentence boundaries, grouping into chunks."""
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Pre-compute token counts for each sentence.
        sent_tokens = [self._count_tokens(s) for s in sentences]

        meta = metadata or {}
        chunks: list[Chunk] = []
        i = 0
        idx = 0

        while i < len(sentences):
            end = i
            total = 0

            # Add sentences until we hit the token limit.
            while end < len(sentences) and total + sent_tokens[end] <= self.chunk_size:
                total += sent_tokens[end]
                end += 1

            # If a single sentence is already over chunk_size, take it anyway.
            if end == i:
                end = i + 1

            chunk_text = " ".join(sentences[i:end])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={**meta, "chunk_index": idx},
                    chunk_index=idx,
                )
            )
            idx += 1

            # Advance *i* respecting overlap.
            # ``max(i + 1, end - overlap)`` guarantees forward progress so
            # we never get stuck on a single oversized sentence.
            next_i = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            i = max(i + 1, next_i)

        return chunks
