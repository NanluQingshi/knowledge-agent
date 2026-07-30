"""Tests for knowledge graph: GraphStore and CommunityDetector."""

from pathlib import Path
from unittest.mock import patch

import pytest

from knowledge_agent.graph.graph_store import GraphStore
from knowledge_agent.graph.community_detector import CommunityDetector


# ===================================================================
# GraphStore
# ===================================================================


class TestGraphStore:
    """Tests for NetworkX-backed GraphStore."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        path = str(tmp_path / "graph.json")
        return GraphStore(path=path)

    def test_add_and_get_entity(self, graph_store):
        graph_store.add_entity("e1", "Alice", "person", {"age": 30})
        entity = graph_store.get_entity("e1")
        assert entity is not None
        assert entity["id"] == "e1"
        assert entity["name"] == "Alice"
        assert entity["type"] == "person"
        assert entity["properties"] == {"age": 30}

    def test_get_nonexistent_entity(self, graph_store):
        assert graph_store.get_entity("nonexistent") is None

    def test_get_all_entities(self, graph_store):
        graph_store.add_entity("e1", "Alice", "person")
        graph_store.add_entity("e2", "Bob", "person")
        entities = graph_store.get_all_entities()
        assert len(entities) == 2
        names = {e["name"] for e in entities}
        assert names == {"Alice", "Bob"}

    def test_add_relation_creates_nodes_if_missing(self, graph_store):
        graph_store.add_relation("alice", "works_at", "acme_corp")
        assert graph_store.node_count == 2
        alice = graph_store.get_entity("alice")
        assert alice is not None
        assert alice["name"] == "alice"  # auto-named from id
        assert alice["type"] == "unknown"

    def test_get_relations_between(self, graph_store):
        graph_store.add_entity("alice", "Alice", "person")
        graph_store.add_entity("bob", "Bob", "person")
        graph_store.add_relation("alice", "knows", "bob", weight=0.9, evidence="common friends")

        relations = graph_store.get_relations_between("alice", "bob")
        assert len(relations) == 1
        assert relations[0]["predicate"] == "knows"
        assert relations[0]["subject_id"] == "alice"
        assert relations[0]["object_id"] == "bob"
        assert relations[0]["weight"] == 0.9
        assert relations[0]["evidence"] == "common friends"
        assert relations[0]["direction"] == "outgoing"

    def test_get_relations_between_reverse(self, graph_store):
        graph_store.add_entity("alice", "Alice", "person")
        graph_store.add_entity("bob", "Bob", "person")
        # Add relation: bob -> alice
        graph_store.add_relation("bob", "reports_to", "alice")
        # Query alice -> bob should return incoming relation
        relations = graph_store.get_relations_between("alice", "bob")
        assert len(relations) == 1
        assert relations[0]["direction"] == "incoming"

    def test_get_neighbors_direct(self, graph_store):
        graph_store.add_entity("alice", "Alice", "person")
        graph_store.add_entity("bob", "Bob", "person")
        graph_store.add_entity("carol", "Carol", "person")
        graph_store.add_relation("alice", "knows", "bob")
        graph_store.add_relation("alice", "knows", "carol")

        neighbors = graph_store.get_neighbors("alice", depth=1)
        assert len(neighbors) == 2
        neighbor_ids = {n["id"] for n in neighbors}
        assert neighbor_ids == {"bob", "carol"}

    def test_get_neighbors_depth(self, graph_store):
        graph_store.add_entity("a", "A", "entity")
        graph_store.add_entity("b", "B", "entity")
        graph_store.add_entity("c", "C", "entity")
        graph_store.add_relation("a", "links_to", "b")
        graph_store.add_relation("b", "links_to", "c")

        # Depth 1: only direct neighbors
        neighbors_d1 = graph_store.get_neighbors("a", depth=1)
        assert len(neighbors_d1) == 1
        assert neighbors_d1[0]["id"] == "b"

        # Depth 2: includes indirect neighbors
        neighbors_d2 = graph_store.get_neighbors("a", depth=2)
        assert len(neighbors_d2) == 2

    def test_get_neighbors_unknown_entity(self, graph_store):
        assert graph_store.get_neighbors("ghost") == []

    def test_search_entities_by_substring(self, graph_store):
        graph_store.add_entity("e1", "Alice Smith", "person")
        graph_store.add_entity("e2", "Bob Johnson", "person")
        graph_store.add_entity("e3", "Acme Corporation", "organization")

        results = graph_store.search_entities("alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice Smith"

        results = graph_store.search_entities("corp")
        assert len(results) == 1
        assert results[0]["name"] == "Acme Corporation"

        results = graph_store.search_entities("nonexistent")
        assert len(results) == 0

    def test_search_entities_empty_query(self, graph_store):
        graph_store.add_entity("e1", "Test", "entity")
        assert graph_store.search_entities("") == []
        assert graph_store.search_entities("   ") == []

    def test_search_entities_case_insensitive(self, graph_store):
        graph_store.add_entity("e1", "Hello World", "concept")
        results = graph_store.search_entities("hello")
        assert len(results) == 1
        results = graph_store.search_entities("WORLD")
        assert len(results) == 1

    def test_get_all_relations(self, graph_store):
        graph_store.add_relation("a", "knows", "b")
        graph_store.add_relation("b", "likes", "c")
        relations = graph_store.get_all_relations()
        assert len(relations) == 2

    def test_save_and_load_round_trip(self, tmp_path):
        path = str(tmp_path / "graph.json")
        store = GraphStore(path=path)
        store.add_entity("e1", "Alice", "person")
        store.add_entity("e2", "Bob", "person")
        store.add_relation("e1", "knows", "e2", weight=0.8)
        store.save()

        # Create a new store with the same path - should load automatically
        store2 = GraphStore(path=path)
        assert store2.node_count == 2
        assert store2.edge_count == 1
        alice = store2.get_entity("e1")
        assert alice is not None
        assert alice["name"] == "Alice"

    def test_save_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "nested" / "subdir" / "graph.json")
        store = GraphStore(path=path)
        store.add_entity("e1", "Test", "entity")
        store.save()
        assert Path(path).exists()

    def test_node_count(self, graph_store):
        assert graph_store.node_count == 0
        graph_store.add_entity("e1", "A", "type")
        assert graph_store.node_count == 1
        graph_store.add_entity("e2", "B", "type")
        assert graph_store.node_count == 2

    def test_edge_count(self, graph_store):
        graph_store.add_relation("a", "knows", "b")
        assert graph_store.edge_count == 1
        graph_store.add_relation("a", "likes", "c")
        assert graph_store.edge_count == 2

    def test_load_corrupted_json_creates_empty_graph(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text("{corrupted json", encoding="utf-8")
        store = GraphStore(path=str(path))
        assert store.node_count == 0
        assert store.edge_count == 0

    def test_graph_property(self, graph_store):
        import networkx as nx

        assert isinstance(graph_store.graph, nx.DiGraph)
        graph_store.add_entity("e1", "Test", "type")
        assert "e1" in graph_store.graph


# ===================================================================
# CommunityDetector
# ===================================================================


class TestCommunityDetector:
    """Tests for Louvain-based CommunityDetector."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        path = str(tmp_path / "comm_graph.json")
        gs = GraphStore(path=path)
        gs.add_entity("a", "Alpha", "person")
        gs.add_entity("b", "Beta", "person")
        gs.add_entity("c", "Gamma", "person")
        gs.add_entity("d", "Delta", "organization")
        gs.add_entity("e", "Epsilon", "organization")
        gs.add_relation("a", "knows", "b")
        gs.add_relation("a", "knows", "c")
        gs.add_relation("b", "knows", "c")
        gs.add_relation("d", "partner", "e")
        return gs

    @patch("knowledge_agent.graph.community_detector.community_louvain")
    def test_detect_returns_communities(self, mock_louvain, graph_store):
        # Mock Louvain partition: two communities
        mock_louvain.best_partition.return_value = {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 1,
            "e": 1,
        }

        detector = CommunityDetector()
        communities = detector.detect(graph_store)

        assert len(communities) == 2

        # Community 0 should contain a, b, c
        assert set(communities[0]["entity_ids"]) == {"a", "b", "c"}
        assert communities[0]["size"] == 3
        assert communities[0]["summary"] == ""

        # Community 1 should contain d, e
        assert set(communities[1]["entity_ids"]) == {"d", "e"}
        assert communities[1]["size"] == 2

    def test_detect_empty_graph_raises(self):
        gs = GraphStore(path="/tmp/empty_graph.json")
        detector = CommunityDetector()
        with pytest.raises(ValueError, match="Cannot detect communities in an empty graph"):
            detector.detect(gs)

    @patch("knowledge_agent.graph.community_detector.community_louvain")
    def test_generate_summaries_without_llm(self, mock_louvain, graph_store):
        mock_louvain.best_partition.return_value = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}

        detector = CommunityDetector()
        communities = detector.detect(graph_store)
        detector.generate_summaries(graph_store, communities, llm_client=None)

        # Summaries should be auto-generated from entity types
        assert communities[0]["summary"] != ""
        assert "person" in communities[0]["summary"]

    @patch("knowledge_agent.graph.community_detector.community_louvain")
    def test_get_hierarchy(self, mock_louvain, graph_store):
        mock_louvain.best_partition.return_value = {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 1,
            "e": 1,
        }

        detector = CommunityDetector()
        hierarchy = detector.get_hierarchy(graph_store)

        assert len(hierarchy) == 2
        for entry in hierarchy:
            assert "community_id" in entry
            assert "entities" in entry
            assert "inter_community_edges" in entry
            assert "size" in entry
            assert "summary" in entry

    def test_detect_single_entity(self, tmp_path):
        gs = GraphStore(path=str(tmp_path / "single.json"))
        gs.add_entity("a", "Alone", "entity")

        detector = CommunityDetector()
        with patch("knowledge_agent.graph.community_detector.community_louvain") as mock_louv:
            mock_louv.best_partition.return_value = {"a": 0}
            communities = detector.detect(gs)
            assert len(communities) == 1
            assert communities[0]["entity_ids"] == ["a"]
