"""Tests for storage layer: DocStore (SQLite) and VectorStore (ChromaDB mock)."""

from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.chunkers.base import Chunk
from knowledge_agent.storage.doc_store import DocStore
from knowledge_agent.storage.vector_store import VectorStore


# ===================================================================
# DocStore
# ===================================================================


class TestDocStore:
    """Tests for SQLite-backed DocStore."""

    @pytest.fixture
    def doc_store(self, tmp_path):
        db_path = str(tmp_path / "test_docs.db")
        return DocStore(db_path=db_path)

    def test_add_and_get_document(self, doc_store):
        doc_store.add_document(
            doc_id="doc1",
            source="/path/to/file.txt",
            filename="file.txt",
            file_type="txt",
            chunk_count=5,
            metadata={"author": "test"},
        )
        doc = doc_store.get_document("doc1")
        assert doc is not None
        assert doc["id"] == "doc1"
        assert doc["source"] == "/path/to/file.txt"
        assert doc["filename"] == "file.txt"
        assert doc["file_type"] == "txt"
        assert doc["chunk_count"] == 5
        assert doc["metadata"] == {"author": "test"}
        assert "ingested_at" in doc

    def test_get_nonexistent_document(self, doc_store):
        doc = doc_store.get_document("nonexistent")
        assert doc is None

    def test_list_documents(self, doc_store):
        doc_store.add_document("a", "/a.txt", "a.txt", "txt", 3)
        doc_store.add_document("b", "/b.txt", "b.txt", "txt", 5)
        docs = doc_store.list_documents()
        assert len(docs) == 2
        # Results are ordered by ingested_at DESC, both docs likely have the same timestamp
        # so ordering is not strictly guaranteed. Just verify both exist.
        doc_ids = {d["id"] for d in docs}
        assert doc_ids == {"a", "b"}

    def test_delete_document(self, doc_store):
        doc_store.add_document("del_me", "/d.txt", "d.txt", "txt", 1)
        assert doc_store.get_document("del_me") is not None
        doc_store.delete_document("del_me")
        assert doc_store.get_document("del_me") is None

    def test_get_total_chunks(self, doc_store):
        assert doc_store.get_total_chunks() == 0
        doc_store.add_document("a", "/a.txt", "a.txt", "txt", 3)
        doc_store.add_document("b", "/b.txt", "b.txt", "txt", 7)
        assert doc_store.get_total_chunks() == 10

    def test_duplicate_id_raises(self, doc_store):
        doc_store.add_document("dup", "/d.txt", "d.txt", "txt", 1)
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            doc_store.add_document("dup", "/d2.txt", "d2.txt", "txt", 2)

    def test_metadata_defaults_to_empty(self, doc_store):
        doc_store.add_document("no_meta", "/n.txt", "n.txt", "txt", 1)
        doc = doc_store.get_document("no_meta")
        assert doc is not None
        assert doc["metadata"] == {}

    def test_get_total_chunks_empty_db(self, doc_store):
        assert doc_store.get_total_chunks() == 0

    def test_custom_db_path_is_used(self, tmp_path):
        db_path = str(tmp_path / "custom" / "docs.db")
        store = DocStore(db_path=db_path)
        store.add_document("x", "/x.txt", "x.txt", "txt", 2)
        assert store.get_document("x") is not None

    def test_add_document_with_content_hash(self, doc_store):
        doc_store.add_document(
            doc_id="hashed_doc",
            source="/h.txt",
            filename="h.txt",
            file_type="txt",
            chunk_count=3,
            content_hash="abc123def456",
        )
        doc = doc_store.get_document("hashed_doc")
        assert doc is not None
        assert doc["content_hash"] == "abc123def456"

    def test_find_by_hash_finds_matching_docs(self, doc_store):
        doc_store.add_document("a", "/a.txt", "a.txt", "txt", 1, content_hash="hash1")
        doc_store.add_document("b", "/b.txt", "b.txt", "txt", 2, content_hash="hash2")
        doc_store.add_document("c", "/c.txt", "c.txt", "txt", 3, content_hash="hash1")

        results = doc_store.find_by_hash("hash1")
        assert len(results) == 2
        doc_ids = {r["id"] for r in results}
        assert doc_ids == {"a", "c"}

    def test_find_by_hash_no_match(self, doc_store):
        results = doc_store.find_by_hash("nonexistent_hash")
        assert results == []

    def test_find_by_hash_empty_string(self, doc_store):
        doc_store.add_document("d", "/d.txt", "d.txt", "txt", 1)
        results = doc_store.find_by_hash("")
        assert results == []


# ===================================================================
# VectorStore
# ===================================================================


class TestVectorStore:
    """Tests for ChromaDB-backed VectorStore with mocked client."""

    @pytest.fixture
    def mock_collection(self):
        return MagicMock()

    @pytest.fixture
    def mock_client(self, mock_collection):
        client = MagicMock()
        client.get_or_create_collection.return_value = mock_collection
        return client

    @pytest.fixture
    def vector_store(self, mock_client, mock_collection):
        with patch(
            "knowledge_agent.storage.vector_store.chromadb.PersistentClient",
            return_value=mock_client,
        ):
            store = VectorStore(persist_dir="/tmp/fake_chroma")
            return store

    def test_add_calls_collection_add(self, vector_store, mock_collection):
        chunks = [
            Chunk(text="Hello world", metadata={"chunk_index": 0}, chunk_index=0),
            Chunk(text="Second chunk", metadata={"chunk_index": 1}, chunk_index=1),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        ids = ["chunk_0", "chunk_1"]

        vector_store.add(chunks, embeddings, metadatas, ids)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["documents"] == ["Hello world", "Second chunk"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert call_kwargs["ids"] == ["chunk_0", "chunk_1"]
        # Chunk metadata should be merged into the passed metadatas
        for meta in call_kwargs["metadatas"]:
            assert "chunk_index" in meta

    def test_add_raises_on_length_mismatch(self, vector_store):
        chunks = [Chunk(text="Hi", metadata={})]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{}]
        ids = ["id1"]
        with pytest.raises(ValueError, match="All arguments must have the same length"):
            vector_store.add(chunks, embeddings, metadatas, ids)

    def test_search_parses_results(self, vector_store, mock_collection):
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = vector_store.search(query_embedding=[0.5, 0.5], top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["text"] == "text1"
        assert results[0]["metadata"] == {"key": "val1"}
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == "id2"

    def test_search_empty_results(self, vector_store, mock_collection):
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        results = vector_store.search(query_embedding=[0.5, 0.5], top_k=5)
        assert results == []

    def test_search_no_ids(self, vector_store, mock_collection):
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        results = vector_store.search(query_embedding=[0.5, 0.5], top_k=5)
        assert results == []

    def test_delete_passes_ids_through(self, vector_store, mock_collection):
        vector_store.delete(["id1", "id2"])
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_delete_with_empty_list(self, vector_store, mock_collection):
        vector_store.delete([])
        mock_collection.delete.assert_not_called()

    def test_count_delegates(self, vector_store, mock_collection):
        mock_collection.count.return_value = 42
        assert vector_store.count() == 42
        mock_collection.count.assert_called_once()

    def test_collection_property(self, vector_store, mock_collection):
        assert vector_store.collection is mock_collection

    def test_search_without_metadatas(self, vector_store, mock_collection):
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["text1"]],
            "metadatas": [[]],
            "distances": [[0.5]],
        }
        results = vector_store.search(query_embedding=[0.5, 0.5], top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"] == {}
