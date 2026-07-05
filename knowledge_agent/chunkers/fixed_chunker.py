"""Fixed-size chunking strategy — split text by token count."""

import tiktoken

from knowledge_agent.chunkers.base import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """Fixed-size chunker using tiktoken tokenizer.

    Encodes the full text into tokens, then slides a window of
    ``chunk_size`` tokens across the encoded sequence with an
    optional ``chunk_overlap`` of tokens between consecutive chunks.
    Each window is decoded back into text.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* into fixed-size token windows."""
        if not text or not text.strip():
            return []

        tokens = self._encoding.encode(text)
        meta = metadata or {}
        chunks: list[Chunk] = []

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = start + self.chunk_size
            segment = self._encoding.decode(tokens[start:end])
            chunks.append(
                Chunk(
                    text=segment,
                    metadata={**meta, "chunk_index": chunk_idx},
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

            if end >= len(tokens):
                break

            start += self.chunk_size - self.chunk_overlap

        return chunks
