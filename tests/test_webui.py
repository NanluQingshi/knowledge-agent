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


class TestWebUINewFeatures:
    """Tests for newly added Web UI features."""

    def test_cache_stats_imports(self):
        from knowledge_agent.webui import _cache_stats
        assert callable(_cache_stats)

    def test_export_functions_import(self):
        from knowledge_agent.webui import _export_knowledge_base
        assert callable(_export_knowledge_base)

    def test_search_docs_import(self):
        from knowledge_agent.webui import _search_docs
        assert callable(_search_docs)

    def test_add_tag_import(self):
        from knowledge_agent.webui import _add_tag_to_doc
        assert callable(_add_tag_to_doc)

    def test_search_docs_empty(self):
        from knowledge_agent.webui import _search_docs
        result = _search_docs("")
        assert "请输入搜索关键词" in result

    def test_add_tag_empty(self):
        from knowledge_agent.webui import _add_tag_to_doc
        result = _add_tag_to_doc("", "")
        assert "请输入" in result
        result = _add_tag_to_doc("doc123", "")
        assert "请输入" in result
