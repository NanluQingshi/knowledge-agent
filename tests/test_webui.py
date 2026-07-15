"""Smoke tests for Web UI module."""

from __future__ import annotations

import pytest


class TestWebUI:
    """Basic smoke tests for Gradio Web UI creation."""

    def test_create_ui_imports(self):
        """Verify the webui module can be imported and create_ui exists."""
        from knowledge_agent.webui import create_ui, main, _render_graph, _clear_memories, _ingest_url
        assert callable(create_ui)
        assert callable(main)
        assert callable(_render_graph)
        assert callable(_clear_memories)
        assert callable(_ingest_url)

    def test_render_graph_empty(self):
        """_render_graph should return a message when graph is empty."""
        from knowledge_agent.webui import _render_graph
        result = _render_graph()
        assert "知识图谱为空" in result

    def test_clear_memories_smoke(self):
        """_clear_memories should not crash."""
        from knowledge_agent.webui import _clear_memories
        result = _clear_memories()
        assert result is not None

    def test_ingest_url_empty(self):
        """_ingest_url should return error for empty input."""
        from knowledge_agent.webui import _ingest_url
        result = _ingest_url("")
        assert "请输入 URL" in result
        result = _ingest_url("   ")
        assert "请输入 URL" in result
