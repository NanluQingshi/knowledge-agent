"""Tests for Config — default values and environment variable overrides."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for Settings configuration."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        from knowledge_agent.config import Settings
        settings = Settings()
        assert settings.openai_api_key == ""
        assert settings.openai_base_url == "https://api.openai.com/v1"
        assert settings.llm_model == "gpt-4o"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dim == 1536
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.retrieval_top_k == 5
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_env_prefix(self):
        """Verify env_prefix is KA_."""
        from knowledge_agent.config import Settings
        assert Settings.model_config["env_prefix"] == "KA_"

    def test_openai_api_key_override(self):
        """Verify OPENAI_API_KEY can be overridden via env."""
        with patch.dict(os.environ, {"KA_OPENAI_API_KEY": "sk-test-key-123"}):
            from knowledge_agent.config import Settings
            settings = Settings()
            assert settings.openai_api_key == "sk-test-key-123"

    def test_llm_model_override(self):
        """Verify LLM_MODEL can be overridden via env."""
        with patch.dict(os.environ, {"KA_LLM_MODEL": "gpt-4o-mini"}):
            from knowledge_agent.config import Settings
            settings = Settings()
            assert settings.llm_model == "gpt-4o-mini"

    def test_chunk_size_override(self):
        """Verify CHUNK_SIZE can be overridden via env."""
        with patch.dict(os.environ, {"KA_CHUNK_SIZE": "1024"}):
            from knowledge_agent.config import Settings
            settings = Settings()
            assert settings.chunk_size == 1024

    def test_retrieval_top_k_override(self):
        """Verify RETRIEVAL_TOP_K can be overridden via env."""
        with patch.dict(os.environ, {"KA_RETRIEVAL_TOP_K": "10"}):
            from knowledge_agent.config import Settings
            settings = Settings()
            assert settings.retrieval_top_k == 10

    def test_multiple_overrides(self):
        """Verify multiple env vars work together."""
        with patch.dict(os.environ, {
            "KA_OPENAI_API_KEY": "sk-key",
            "KA_LLM_MODEL": "claude-sonnet-4",
            "KA_EMBEDDING_MODEL": "text-embedding-ada-002",
        }):
            from knowledge_agent.config import Settings
            settings = Settings()
            assert settings.openai_api_key == "sk-key"
            assert settings.llm_model == "claude-sonnet-4"
            assert settings.embedding_model == "text-embedding-ada-002"
