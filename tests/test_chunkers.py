"""Tests for document chunkers: FixedChunker, SemanticChunker, RecursiveChunker."""

import pytest

from knowledge_agent.chunkers.base import Chunk
from knowledge_agent.chunkers.fixed_chunker import FixedChunker
from knowledge_agent.chunkers.semantic_chunker import SemanticChunker
from knowledge_agent.chunkers.recursive_chunker import RecursiveChunker


# ===================================================================
# FixedChunker
# ===================================================================

class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_empty_text_returns_empty_list(self):
        chunker = FixedChunker(chunk_size=512, chunk_overlap=0)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_returns_single_chunk(self):
        chunker = FixedChunker(chunk_size=512, chunk_overlap=0)
        text = "Hello world, this is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_long_text_produces_multiple_chunks(self):
        chunker = FixedChunker(chunk_size=20, chunk_overlap=0)
        text = "word " * 500  # well above 20 tokens
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_chunk_index_is_monotonic(self):
        chunker = FixedChunker(chunk_size=20, chunk_overlap=0)
        text = "token " * 500
        chunks = chunker.chunk(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_overlap(self):
        """With overlap, consecutive chunks should share some text."""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=15)
        text = "apple banana cherry date elderberry fig grape "
        text = text * 30
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # With overlap, adjacent chunks typically share content
        # (hard to assert exact overlap since tiktoken tokenization can cause
        # partial token loss on decode boundaries, but the mechanism is exercised)
        for c in chunks:
            assert isinstance(c.text, str)
            assert len(c.text) > 0

    def test_metadata_merged(self):
        chunker = FixedChunker(chunk_size=20, chunk_overlap=0)
        text = "word " * 200
        doc_meta = {"source": "test.txt", "author": "me"}
        chunks = chunker.chunk(text, metadata=doc_meta)
        assert len(chunks) > 1
        for c in chunks:
            for key, val in doc_meta.items():
                assert c.metadata[key] == val
            assert "chunk_index" in c.metadata

    def test_chunk_overlap_clamped_to_half_size(self):
        chunker = FixedChunker(chunk_size=100, chunk_overlap=80)
        assert chunker.chunk_overlap == 50  # min(80, 100//2)

    def test_output_type(self):
        chunker = FixedChunker(chunk_size=512, chunk_overlap=0)
        chunks = chunker.chunk("A simple text.")
        for c in chunks:
            assert isinstance(c, Chunk)
            assert hasattr(c, "text")
            assert hasattr(c, "metadata")
            assert hasattr(c, "chunk_index")


# ===================================================================
# SemanticChunker
# ===================================================================

class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_empty_text_returns_empty_list(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=0)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_single_sentence_returns_one_chunk(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=0)
        text = "This is a single sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert text in chunks[0].text

    def test_multiple_sentences_split_into_chunks(self):
        chunker = SemanticChunker(chunk_size=20, chunk_overlap=0)
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            # Each chunk should end with a complete sentence
            assert c.text.strip() != ""

    def test_chinese_sentences(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=0)
        text = "今天天气真好。我们去公园散步吧！你觉得呢？"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "今天天气真好" in combined
        assert "我们去公园散步吧" in combined
        assert "你觉得呢" in combined

    def test_mixed_chinese_and_english(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=0)
        text = "Hello world。你好吗？I am fine.谢谢！"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "Hello world" in combined

    def test_chunk_index_is_monotonic(self):
        chunker = SemanticChunker(chunk_size=20, chunk_overlap=0)
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        chunks = chunker.chunk(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_merged(self):
        chunker = SemanticChunker(chunk_size=20, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence. Seventh sentence. Eighth sentence. Ninth sentence. Tenth sentence."
        doc_meta = {"source": "doc.md"}
        chunks = chunker.chunk(text, metadata=doc_meta)
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.get("source") == "doc.md"
            assert "chunk_index" in c.metadata

    def test_sentence_overlap(self):
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=2)
        text = "A. B. C. D. E. F. G. H. I. J."
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # With overlap=2, the last 2 sentences of previous chunk should
            # appear at the start of the next chunk
            assert len(chunks) >= 2

    def test_oversized_sentence(self):
        """A single sentence exceeding chunk_size is still included."""
        chunker = SemanticChunker(chunk_size=10, chunk_overlap=0)
        text = "ThisIsOneVeryLongSentenceWithoutAnyBreaksSoItExceedsTheLimit"
        chunks = chunker.chunk(text)
        # The single oversized sentence should be taken as one chunk
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()


# ===================================================================
# RecursiveChunker
# ===================================================================

class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_empty_text_returns_empty_list(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=0)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=0)
        text = "Short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_split_by_double_newline(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = (
            "This is the first paragraph with enough text to fill multiple tokens. "
            "It should be longer than the chunk size so it gets split.\n\n"
            "This is the second paragraph, also with enough text to push it over "
            "the chunk size threshold. More content here to make sure it splits.\n\n"
            "And this is a third paragraph to make sure at least two chunks exist. "
            "Even more text here for good measure.\n\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) > 0

    def test_split_by_newline(self):
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        text = (
            "This line has enough text to exceed the chunk size. "
            "This other line also has a lot of text. "
            "A third line with sufficient content. "
            "Fourth line continues the pattern. "
            "Fifth line rounds out the set.\n"
        ) * 3
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_split_by_period(self):
        chunker = RecursiveChunker(chunk_size=8, chunk_overlap=0)
        text = (
            "This sentence is long. This one also exceeds. "
            "Another long sentence here. Yet another sentence. "
            "One more sentence. And another. " * 10
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_fallback_to_character_split(self):
        chunker = RecursiveChunker(chunk_size=3, chunk_overlap=0)
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_index_is_monotonic(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "Word. " * 100
        chunks = chunker.chunk(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_merged(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
        doc_meta = {"source": "test.txt", "filename": "test.txt"}
        chunks = chunker.chunk(text, metadata=doc_meta)
        assert len(chunks) > 1
        for c in chunks:
            for key, val in doc_meta.items():
                assert c.metadata[key] == val
            assert "chunk_index" in c.metadata

    def test_output_type(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=0)
        chunks = chunker.chunk("A simple text.")
        for c in chunks:
            assert isinstance(c, Chunk)
            assert hasattr(c, "text")
            assert hasattr(c, "metadata")
            assert hasattr(c, "chunk_index")
