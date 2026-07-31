"""Tests for Cross-Encoder reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderRerankerLocal:
    """Tests for CrossEncoderReranker in 'local' mode (model fallback)."""

    @pytest.fixture
    def reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(model_name="test-model", mode="local")

    @pytest.fixture
    def sample_results(self) -> list[dict]:
        return [
            {"id": "a", "text": "Apple is a fruit.", "score": 2.0},
            {"id": "b", "text": "Banana is yellow.", "score": 1.0},
            {"id": "c", "text": "Cherry is red.", "score": 0.5},
        ]

    def test_empty_results(self, reranker: CrossEncoderReranker):
        assert reranker.rerank("query", []) == []

    def test_results_without_text_coerced(self, reranker: CrossEncoderReranker):
        results = [{"id": "x", "score": 1.0}]
        out = reranker.rerank("query", results, top_k=5)
        assert len(out) == 1
        assert "rerank_score" in out[0]

    def test_top_k_limits_output(self, reranker: CrossEncoderReranker, sample_results):
        out = reranker.rerank("test", sample_results, top_k=1)
        assert len(out) == 1

    def test_top_k_defaults_to_input_length(self, reranker: CrossEncoderReranker, sample_results):
        out = reranker.rerank("test", sample_results)
        assert len(out) == 3

    def test_rerank_score_added(self, reranker: CrossEncoderReranker, sample_results):
        out = reranker.rerank("test", sample_results, top_k=3)
        for item in out:
            assert "rerank_score" in item

    def test_fallback_on_model_failure(self, reranker: CrossEncoderReranker, sample_results):
        """When model fails, scores should be based on original order."""
        out = reranker.rerank("test", sample_results, top_k=3)
        # Original order should be preserved when model unavailable
        assert out[0]["id"] == "a"
        assert out[1]["id"] == "b"


class TestCrossEncoderRerankerAPIMode:
    """Tests for CrossEncoderReranker in 'api' mode."""

    @pytest.fixture
    def reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(mode="api")

    def test_api_mode_without_key(self, reranker: CrossEncoderReranker):
        results = [{"id": "a", "text": "Some text.", "score": 1.0}]
        with patch("knowledge_agent.retrieval.reranker.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            out = reranker.rerank("query", results, top_k=1)
            assert len(out) == 1
            assert "rerank_score" in out[0]

    @patch("openai.OpenAI")
    def test_api_mode_with_key(self, mock_openai, reranker: CrossEncoderReranker):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "8"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        results = [{"id": "a", "text": "Relevant doc.", "score": 1.0}]
        with patch("knowledge_agent.retrieval.reranker.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.openai_base_url = "https://test.com/v1"
            mock_settings.llm_model = "gpt-4o-mini"
            out = reranker.rerank("test query", results, top_k=1)
            assert len(out) == 1
            # Score should be 8/10 = 0.8
            assert abs(out[0]["rerank_score"] - 0.8) < 0.001
