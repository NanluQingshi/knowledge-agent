"""Tests for the FastAPI application factory and route registration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from knowledge_agent.api.routes import create_app


def test_create_app_registers_expected_routes():
    app = create_app()
    routes = {
        (method, route.path) for route in app.routes for method in getattr(route, "methods", set())
    }

    assert routes >= {
        ("POST", "/ingest"),
        ("POST", "/query"),
        ("POST", "/query/stream"),
        ("GET", "/documents"),
        ("DELETE", "/documents/{doc_id}"),
        ("GET", "/health"),
        ("POST", "/evaluate/retrieval"),
        ("POST", "/evaluate/answer"),
    }


def test_create_app_can_be_called_more_than_once():
    first = create_app()
    second = create_app()

    assert first is not second
    assert len(first.routes) == len(second.routes)


# ===================================================================
# 路由行为测试 (TC-06)
# ===================================================================


class TestRouteBehavior:
    """Behavior tests for API routes using TestClient."""

    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    # ------------------------------------------------------------------
    # /query — 空库
    # ------------------------------------------------------------------

    def test_query_empty_knowledge_base_returns_404(self, client):
        with patch(
            "knowledge_agent.storage.vector_store.VectorStore.count",
            return_value=0,
        ):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 404

    def test_query_invalid_payload_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422
        # 统一错误格式
        assert "error" in resp.json()
        assert resp.json()["error"]["code"] == "validation_error"

    # ------------------------------------------------------------------
    # /documents — 空库
    # ------------------------------------------------------------------

    def test_documents_empty_returns_empty_list(self, client):
        with patch(
            "knowledge_agent.storage.doc_store.DocStore.list_documents",
            return_value=[],
        ):
            resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total_chunks"] == 0

    # ------------------------------------------------------------------
    # 统一错误响应格式
    # ------------------------------------------------------------------

    def test_error_response_format_is_uniform(self, client):
        """4xx 错误响应格式应统一为 {error: {code, message, detail}}."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert set(body.keys()) == {"error"}
        assert set(body["error"].keys()) == {"code", "message", "detail"}

    def test_not_found_error_format(self, client):
        with patch(
            "knowledge_agent.storage.vector_store.VectorStore.count",
            return_value=0,
        ):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "http_404"
