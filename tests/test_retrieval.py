"""Tests for retrieval: BM25Retriever and HybridRetriever."""

from unittest.mock import MagicMock

import pytest

from knowledge_agent.retrieval.bm25_retriever import BM25Retriever
from knowledge_agent.retrieval.hybrid_retriever import HybridRetriever


# ===================================================================
# BM25Retriever
# ===================================================================


class TestBM25Retriever:
    """Tests for BM25-based sparse retriever."""

    @pytest.fixture
    def corpus(self):
        return [
            {"id": "1", "text": "The quick brown fox jumps over the lazy dog"},
            {"id": "2", "text": "A quick brown fox is very fast"},
            {"id": "3", "text": "The lazy dog sleeps all day long"},
            {"id": "4", "text": "Machine learning is transforming artificial intelligence"},
        ]

    def test_index_and_retrieve(self, corpus):
        retriever = BM25Retriever()
        retriever.index(corpus)

        assert retriever.is_indexed is True
        assert retriever.corpus_size == 4

        results = retriever.retrieve("quick fox", top_k=2)
        assert len(results) == 2
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert r["score"] >= 0  # BM25 scores are non-negative

    def test_retrieve_before_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="index has not been built"):
            retriever.retrieve("test query")

    def test_empty_corpus_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="Corpus must not be empty"):
            retriever.index([])

    def test_empty_query_returns_empty_list(self, corpus):
        retriever = BM25Retriever()
        retriever.index(corpus)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_top_k_limits_results(self, corpus):
        retriever = BM25Retriever()
        retriever.index(corpus)
        results = retriever.retrieve("fox", top_k=1)
        assert len(results) == 1

    def test_is_indexed_property(self):
        retriever = BM25Retriever()
        assert retriever.is_indexed is False
        assert retriever.corpus_size == 0

    def test_results_ordered_by_score_desc(self, corpus):
        retriever = BM25Retriever()
        retriever.index(corpus)
        results = retriever.retrieve("quick brown fox", top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_preserved(self, corpus):
        corpus_with_meta = [
            {"id": "1", "text": "Hello world", "metadata": {"source": "doc1"}},
            {"id": "2", "text": "Goodbye world", "metadata": {"source": "doc2"}},
        ]
        retriever = BM25Retriever()
        retriever.index(corpus_with_meta)
        results = retriever.retrieve("world", top_k=2)
        for r in results:
            assert "metadata" in r
            assert "source" in r["metadata"]

    def test_chinese_text(self):
        corpus = [
            {"id": "1", "text": "今天天气真好"},
            {"id": "2", "text": "我们去公园散步"},
        ]
        retriever = BM25Retriever()
        retriever.index(corpus)
        results = retriever.retrieve("天气", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "1"

    def test_score_is_float(self, corpus):
        retriever = BM25Retriever()
        retriever.index(corpus)
        results = retriever.retrieve("fox", top_k=1)
        assert isinstance(results[0]["score"], float)


# ===================================================================
# HybridRetriever
# ===================================================================


class TestHybridRetriever:
    """Tests for HybridRetriever with mocked sub-retrievers."""

    @pytest.fixture
    def mock_vector_retriever(self):
        vr = MagicMock()
        vr.retrieve.return_value = [
            {"id": "A", "text": "Document A", "metadata": {"source": "v1"}, "distance": 0.1},
            {"id": "B", "text": "Document B", "metadata": {"source": "v1"}, "distance": 0.2},
            {"id": "C", "text": "Document C", "metadata": {"source": "v1"}, "distance": 0.3},
        ]
        return vr

    @pytest.fixture
    def mock_bm25_retriever(self):
        bm25 = MagicMock()
        bm25.retrieve.return_value = [
            {"id": "B", "text": "Document B", "metadata": {"source": "bm25"}, "score": 5.0},
            {"id": "D", "text": "Document D", "metadata": {"source": "bm25"}, "score": 4.0},
            {"id": "E", "text": "Document E", "metadata": {"source": "bm25"}, "score": 3.0},
        ]
        return bm25

    @pytest.fixture
    def hybrid(self, mock_vector_retriever, mock_bm25_retriever):
        return HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever,
        )

    def test_rrf_fusion_includes_all_docs(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        result_ids = {r["id"] for r in results}
        # All docs from both retrievers should be present
        assert result_ids == {"A", "B", "C", "D", "E"}
        assert len(results) == 5

    def test_rrf_fusion_sorted_by_score_desc(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_doc_in_both_retrievers_has_both_sources(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        doc_b = next(r for r in results if r["id"] == "B")
        assert doc_b["sources"] == ["vector", "bm25"]

    def test_doc_only_in_vector_has_single_source(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        doc_a = next(r for r in results if r["id"] == "A")
        assert doc_a["sources"] == ["vector"]

    def test_doc_only_in_bm25_has_single_source(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        doc_e = next(r for r in results if r["id"] == "E")
        assert doc_e["sources"] == ["bm25"]

    def test_top_k_limits_results(self, hybrid):
        results = hybrid.retrieve("test query", top_k=3)
        assert len(results) == 3

    def test_empty_vector_results(self, mock_bm25_retriever):
        vr = MagicMock()
        vr.retrieve.return_value = []
        bm25 = mock_bm25_retriever
        hybrid = HybridRetriever(vector_retriever=vr, bm25_retriever=bm25)
        results = hybrid.retrieve("query", top_k=10)
        assert len(results) == 3  # only BM25 results
        for r in results:
            assert r["sources"] == ["bm25"]

    def test_empty_bm25_results(self, mock_vector_retriever):
        vr = mock_vector_retriever
        bm25 = MagicMock()
        bm25.retrieve.return_value = []
        hybrid = HybridRetriever(vector_retriever=vr, bm25_retriever=bm25)
        results = hybrid.retrieve("query", top_k=10)
        assert len(results) == 3  # only vector results
        for r in results:
            assert r["sources"] == ["vector"]

    def test_both_empty(self):
        vr = MagicMock()
        vr.retrieve.return_value = []
        bm25 = MagicMock()
        bm25.retrieve.return_value = []
        hybrid = HybridRetriever(vector_retriever=vr, bm25_retriever=bm25)
        results = hybrid.retrieve("query", top_k=10)
        assert results == []

    def test_rrf_scoring_math(self, mock_vector_retriever, mock_bm25_retriever):
        """Verify RRF scoring: doc in both retrievers should have higher score."""
        hybrid = HybridRetriever(
            vector_retriever=mock_vector_retriever, bm25_retriever=mock_bm25_retriever
        )
        results = hybrid.retrieve("query", top_k=10)
        # Doc B is present in both (rank_v=2, rank_b=1), so should have highest RRF score
        doc_b = next(r for r in results if r["id"] == "B")
        assert doc_b["score"] > 0
        # Verify it's the top result
        assert results[0]["id"] == "B"

    def test_retrieve_returns_required_fields(self, hybrid):
        results = hybrid.retrieve("test query", top_k=10)
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert "sources" in r


# ===================================================================
# MultiQueryFusion
# ===================================================================


class TestMultiQueryFusion:
    """Tests for multi-query fusion."""

    def test_fuse_results_empty(self):
        from knowledge_agent.retrieval.enhancer import MultiQueryFusion

        result = MultiQueryFusion.fuse_results([], top_k=5)
        assert result == []

    def test_fuse_results_single_list(self):
        from knowledge_agent.retrieval.enhancer import MultiQueryFusion

        results = [
            [{"id": "a", "text": "doc a"}, {"id": "b", "text": "doc b"}],
        ]
        fused = MultiQueryFusion.fuse_results(results, top_k=5)
        assert len(fused) == 2
        assert fused[0]["id"] == "a"

    def test_fuse_results_deduplicates(self):
        from knowledge_agent.retrieval.enhancer import MultiQueryFusion

        results = [
            [{"id": "a", "text": "doc a"}, {"id": "b", "text": "doc b"}],
            [{"id": "a", "text": "doc a again"}, {"id": "c", "text": "doc c"}],
        ]
        fused = MultiQueryFusion.fuse_results(results, top_k=5)
        ids = {r["id"] for r in fused}
        assert ids == {"a", "b", "c"}
        # 'a' appears in both lists, should get higher RRF score
        assert fused[0]["id"] == "a"

    def test_fuse_results_respects_top_k(self):
        from knowledge_agent.retrieval.enhancer import MultiQueryFusion

        results = [
            [{"id": str(i), "text": f"doc {i}"} for i in range(10)],
        ]
        fused = MultiQueryFusion.fuse_results(results, top_k=3)
        assert len(fused) == 3

    def test_expand_queries_returns_original_when_no_llm(self):
        from knowledge_agent.retrieval.enhancer import MultiQueryFusion

        fusion = MultiQueryFusion()
        queries = fusion.expand_queries("hello world")
        # Without API key, should at least contain the original question
        assert len(queries) >= 1
        assert "hello world" in queries


# ===================================================================
# QueryRewriter & HyDEGenerator (edge cases)
# ===================================================================


class TestQueryRewriter:
    """Tests for QueryRewriter edge cases."""

    def test_empty_question(self):
        from knowledge_agent.retrieval.enhancer import QueryRewriter

        rewriter = QueryRewriter()
        assert rewriter.rewrite("") == []
        assert rewriter.rewrite("   ") == []

    def test_no_api_key_regression(self):
        """Without API key, rewrite should not crash."""
        from knowledge_agent.retrieval.enhancer import QueryRewriter

        rewriter = QueryRewriter()
        # Should handle API error gracefully and return at least original question
        result = rewriter.rewrite("hello world", num_variations=2)
        assert len(result) >= 1


class TestHyDEGenerator:
    """Tests for HyDEGenerator edge cases."""

    def test_empty_question(self):
        from knowledge_agent.retrieval.enhancer import HyDEGenerator

        hyde = HyDEGenerator()
        assert hyde.generate("") == ""
        assert hyde.generate("   ") == ""

    def test_no_api_key_fallback(self):
        """Without API key, should return original question."""
        from knowledge_agent.retrieval.enhancer import HyDEGenerator

        hyde = HyDEGenerator()
        result = hyde.generate("What is AI?")
        # Should return original question or something non-empty
        assert result
