"""Tests for the FastAPI application factory and route registration."""

from __future__ import annotations

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
