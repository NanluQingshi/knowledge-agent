"""Tests for Exporter — JSON and Markdown export."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from knowledge_agent.exporter import Exporter


class TestExporter:
    """Tests for knowledge base export."""

    @pytest.fixture
    def mock_stores(self):
        """Create mocked DocStore and VectorStore with sample data."""
        doc_store = MagicMock()
        doc_store.list_documents.return_value = [
            {
                "id": "doc1",
                "source": "/path/to/file1.md",
                "filename": "file1.md",
                "file_type": "md",
                "chunk_count": 2,
                "version": 1,
                "ingested_at": "2026-01-01T00:00:00",
                "metadata": {"tags": ["important"]},
            },
        ]

        vector_store = MagicMock()
        vector_store.collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "documents": ["Content of chunk 1", "Content of chunk 2"],
            "metadatas": [
                {"doc_id": "doc1", "source": "/path/to/file1.md"},
                {"doc_id": "doc1", "source": "/path/to/file1.md"},
            ],
        }

        return doc_store, vector_store

    def test_export_json_creates_file(self, mock_stores, tmp_path):
        doc_store, vector_store = mock_stores
        exporter = Exporter(doc_store=doc_store, vector_store=vector_store)
        out_path = tmp_path / "export.json"
        result = exporter.export_json(str(out_path))
        assert result.exists()
        data = json.loads(result.read_text(encoding="utf-8"))
        assert "documents" in data
        assert "chunks" in data
        assert data["stats"]["documents"] == 1
        assert data["stats"]["chunks"] == 2

    def test_export_markdown_creates_files(self, mock_stores, tmp_path):
        doc_store, vector_store = mock_stores
        exporter = Exporter(doc_store=doc_store, vector_store=vector_store)
        out_dir = exporter.export_markdown(str(tmp_path))
        assert out_dir.exists()
        # Should create index and a file for each source
        files = list(out_dir.iterdir())
        assert len(files) >= 1
        assert any(f.name == "_index.md" for f in files)

    def test_export_empty_knowledge_base(self, tmp_path):
        doc_store = MagicMock()
        doc_store.list_documents.return_value = []
        vector_store = MagicMock()
        vector_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        exporter = Exporter(doc_store=doc_store, vector_store=vector_store)
        out_dir = exporter.export_markdown(str(tmp_path))
        index_file = out_dir / "_index.md"
        assert index_file.exists()
        assert "知识库为空" in index_file.read_text(encoding="utf-8")

    def test_export_json_empty(self, tmp_path):
        doc_store = MagicMock()
        doc_store.list_documents.return_value = []
        vector_store = MagicMock()
        vector_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        exporter = Exporter(doc_store=doc_store, vector_store=vector_store)
        out_path = tmp_path / "empty.json"
        result = exporter.export_json(str(out_path))
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["stats"]["documents"] == 0
        assert data["stats"]["chunks"] == 0
