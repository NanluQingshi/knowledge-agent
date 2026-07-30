"""Smoke tests for Web UI module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestWebUI:
    """Basic smoke tests for Gradio Web UI creation."""

    def test_create_ui_imports(self):
        """Verify the webui module can be imported and create_ui exists."""
        from knowledge_agent.webui import (
            _clear_memories,
            _ingest_url,
            _monitoring_dashboard,
            _render_graph,
            _reset_metrics,
            create_ui,
            main,
        )

        assert callable(create_ui)
        assert callable(main)
        assert callable(_render_graph)
        assert callable(_clear_memories)
        assert callable(_ingest_url)
        assert callable(_monitoring_dashboard)
        assert callable(_reset_metrics)

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

    def test_monitoring_dashboard_uses_active_orchestrator(self):
        from knowledge_agent.webui import _monitoring_dashboard

        orchestrator = MagicMock()
        orchestrator.get_monitoring_report.return_value = {
            "timings": {
                "query": {
                    "count": 1,
                    "mean": 12.5,
                    "p50": 12.5,
                    "p95": 12.5,
                    "p99": 12.5,
                    "max": 12.5,
                }
            },
            "counters": {"query.count": 1},
        }

        with patch("knowledge_agent.webui._get_orchestrator", return_value=orchestrator):
            result = _monitoring_dashboard()

        assert "| query | 1 | 12.5 |" in result
        assert "| query.count | 1 |" in result

    def test_reset_metrics_uses_active_orchestrator(self):
        from knowledge_agent.webui import _reset_metrics

        orchestrator = MagicMock()
        with patch("knowledge_agent.webui._get_orchestrator", return_value=orchestrator):
            result = _reset_metrics()

        orchestrator.metrics.reset.assert_called_once_with()
        assert "已重置" in result


class TestWebUINewFeatures:
    """Tests for newly added Web UI features."""

    def test_cache_stats_imports(self):
        from knowledge_agent.webui import _cache_stats

        assert callable(_cache_stats)

    def test_cache_stats_uses_active_orchestrator(self):
        from knowledge_agent.webui import _cache_stats

        orchestrator = MagicMock()
        orchestrator.get_cache_stats.return_value = {
            "size": 2,
            "ttl": 300,
            "max_size": 100,
        }
        with patch("knowledge_agent.webui._get_orchestrator", return_value=orchestrator):
            result = _cache_stats()

        assert "2 条" in result
        orchestrator.get_cache_stats.assert_called_once_with()

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
