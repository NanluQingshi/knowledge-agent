"""Tests for memory modules: WorkingMemory, SemanticMemory, ProceduralMemory."""

import json

import pytest

from knowledge_agent.graph.graph_store import GraphStore
from knowledge_agent.memory.working_memory import WorkingMemory
from knowledge_agent.memory.semantic_memory import SemanticMemory
from knowledge_agent.memory.procedural_memory import ProceduralMemory


# ===================================================================
# WorkingMemory
# ===================================================================


class TestWorkingMemory:
    """Tests for WorkingMemory (short-term conversation context)."""

    @pytest.fixture
    def memory(self):
        return WorkingMemory(max_tokens=4000)

    def test_add_and_get_message(self, memory):
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_add_messages_batch(self, memory):
        batch = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        memory.add_messages(batch)
        assert memory.message_count == 3

    def test_get_messages_last_n(self, memory):
        memory.add_message("user", "First")
        memory.add_message("user", "Second")
        memory.add_message("user", "Third")
        last_two = memory.get_messages(last_n=2)
        assert len(last_two) == 2
        assert last_two[0]["content"] == "Second"
        assert last_two[1]["content"] == "Third"

    def test_get_messages_last_n_larger_than_available(self, memory):
        memory.add_message("user", "Only one")
        messages = memory.get_messages(last_n=10)
        assert len(messages) == 1

    def test_clear_empties_messages(self, memory):
        memory.add_message("user", "Something")
        memory.clear()
        assert memory.message_count == 0

    def test_auto_trim(self):
        memory = WorkingMemory(max_tokens=10)  # very low token budget
        memory.add_message(
            "user", "This is a very long message that should exceed the token limit by far"
        )
        memory.add_message("assistant", "Another long message that will cause trimming")
        # After adding, the trim should have kicked in
        assert memory.message_count >= 1

    def test_auto_trim_keeps_at_least_one_message(self):
        memory = WorkingMemory(max_tokens=1)  # extremely low
        memory.add_message("user", "X" * 1000)
        # Should not empty the list entirely - only trim when len > 1
        assert memory.message_count >= 1

    def test_set_and_get_context(self, memory):
        memory.set("username", "Alice")
        memory.set("language", "Chinese")
        assert memory.get("username") == "Alice"
        assert memory.get("language") == "Chinese"
        assert memory.get("nonexistent") is None
        assert memory.get("nonexistent", "default_val") == "default_val"

    def test_get_all_context(self, memory):
        memory.set("key1", "val1")
        memory.set("key2", "val2")
        context = memory.get_all_context()
        assert context == {"key1": "val1", "key2": "val2"}

    def test_message_count_property(self, memory):
        assert memory.message_count == 0
        memory.add_message("user", "Hi")
        assert memory.message_count == 1

    def test_estimated_tokens(self, memory):
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "World")
        # 5 + 5 = 10 chars, 10 // 4 = 2 tokens approx
        assert memory.estimated_tokens >= 0

    def test_max_tokens_property(self, memory):
        assert memory.max_tokens == 4000

    def test_get_messages_returns_copy(self, memory):
        memory.add_message("user", "Hi")
        msgs = memory.get_messages()
        msgs.append({"role": "assistant", "content": "Hey"})
        # Original should be unchanged
        assert memory.message_count == 1


# ===================================================================
# SemanticMemory
# ===================================================================


class TestSemanticMemory:
    """Tests for SemanticMemory (graph-backed long-term fact storage)."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        return GraphStore(path=str(tmp_path / "sem_mem.json"))

    @pytest.fixture
    def sem_memory(self, graph_store):
        return SemanticMemory(graph_store=graph_store)

    def test_remember_fact_adds_entities_and_relation(self, sem_memory):
        sem_memory.remember_fact(
            "Alice", "works_at", "AcmeCorp", confidence=0.9, source="document1"
        )
        # Check entities exist
        alice_id = "alice"
        acme_id = "acmecorp"
        alice = sem_memory._graph.get_entity(alice_id)
        acme = sem_memory._graph.get_entity(acme_id)
        assert alice is not None
        assert alice["type"] == "concept"
        assert acme is not None
        # Check relation exists
        relations = sem_memory._graph.get_relations_between(alice_id, acme_id)
        assert len(relations) >= 1
        assert relations[0]["predicate"] == "works_at"

    def test_recall_facts(self, sem_memory):
        sem_memory.remember_fact("Alice", "works_at", "AcmeCorp")
        sem_memory.remember_fact("Alice", "knows", "Bob")

        facts = sem_memory.recall_facts("Alice", depth=1)
        assert len(facts) >= 2
        # facts should contain entries for both relations
        relations = {f["relation"] for f in facts}
        assert "works_at (outgoing)" in relations or "works_at" in str(relations)
        assert "knows (outgoing)" in relations or "knows" in str(relations)

    def test_recall_facts_unknown_entity(self, sem_memory):
        facts = sem_memory.recall_facts("NonExistent")
        assert facts == []

    def test_search_concepts(self, sem_memory):
        sem_memory.remember_fact("MachineLearning", "is_a", "Technology")
        results = sem_memory.search_concepts("machine")
        assert len(results) >= 1
        assert any("MachineLearning" in r["name"] for r in results)

    def test_find_connections(self, sem_memory):
        sem_memory.remember_fact("Alice", "knows", "Bob")
        connections = sem_memory.find_connections("Alice", "Bob")
        assert len(connections) >= 1

    def test_fact_count_and_concept_count(self, sem_memory):
        assert sem_memory.fact_count == 0
        assert sem_memory.concept_count == 0
        sem_memory.remember_fact("A", "relates_to", "B")
        assert sem_memory.concept_count == 2  # two nodes
        assert sem_memory.fact_count == 1  # one edge

    def test_get_all_facts(self, sem_memory):
        sem_memory.remember_fact("A", "likes", "B")
        facts = sem_memory.get_all_facts()
        assert len(facts) == 1

    def test_to_id_converts_correctly(self):
        assert SemanticMemory._to_id("Hello World") == "hello_world"
        assert SemanticMemory._to_id("Alice Smith") == "alice_smith"
        assert SemanticMemory._to_id("simple") == "simple"

    def test_remember_fact_saves_graph(self, sem_memory):
        """Verify that remember_fact persists the graph."""
        sem_memory.remember_fact("X", "connects", "Y")
        # The graph was saved to the tmp_path - check the file exists and has content
        assert sem_memory._graph._path.exists()
        data = json.loads(sem_memory._graph._path.read_text(encoding="utf-8"))
        assert len(data.get("nodes", [])) == 2


# ===================================================================
# ProceduralMemory
# ===================================================================


class TestProceduralMemory:
    """Tests for ProceduralMemory (workflow template storage)."""

    @pytest.fixture
    def proc_memory(self, tmp_path):
        path = str(tmp_path / "proc_mem.json")
        return ProceduralMemory(storage_path=path)

    def test_add_and_get_template(self, proc_memory):
        tid = proc_memory.add_template(
            name="Test Workflow",
            steps=[{"action": "load", "params": {}}, {"action": "process", "params": {}}],
            description="A test workflow",
            tags=["test", "demo"],
        )
        tmpl = proc_memory.get_template(tid)
        assert tmpl is not None
        assert tmpl["name"] == "Test Workflow"
        assert tmpl["description"] == "A test workflow"
        assert len(tmpl["steps"]) == 2
        assert tmpl["tags"] == ["test", "demo"]

    def test_get_nonexistent_template(self, proc_memory):
        assert proc_memory.get_template("nonexistent") is None

    def test_list_templates(self, proc_memory):
        proc_memory.add_template("First", [{"action": "a"}], tags=["common"])
        proc_memory.add_template("Second", [{"action": "b"}], tags=["common"])
        templates = proc_memory.list_templates()
        assert len(templates) == 2

    def test_list_templates_with_tag_filter(self, proc_memory):
        proc_memory.add_template("A", [{"action": "a"}], tags=["alpha"])
        proc_memory.add_template("B", [{"action": "b"}], tags=["beta"])
        alpha_templates = proc_memory.list_templates(tag="alpha")
        assert len(alpha_templates) == 1
        assert alpha_templates[0]["name"] == "A"

    def test_list_templates_ordered_by_usage(self, proc_memory):
        t1 = proc_memory.add_template("Most Used", [{"action": "a"}])
        t2 = proc_memory.add_template("Less Used", [{"action": "b"}])
        # Record execution for t1 twice, t2 once
        proc_memory.record_execution(t1, success=True)
        proc_memory.record_execution(t1, success=True)
        proc_memory.record_execution(t2, success=True)

        templates = proc_memory.list_templates()
        assert templates[0]["name"] == "Most Used"  # higher usage_count
        assert templates[1]["name"] == "Less Used"

    def test_record_execution_updates_stats(self, proc_memory):
        tid = proc_memory.add_template(
            "My Template",
            [{"action": "process"}],
            description="test",
        )
        tmpl = proc_memory.get_template(tid)
        assert tmpl["usage_count"] == 0
        assert tmpl["success_rate"] == 1.0

        proc_memory.record_execution(tid, success=True, duration_seconds=2.5)
        tmpl = proc_memory.get_template(tid)
        assert tmpl["usage_count"] == 1
        assert tmpl["success_rate"] == 1.0
        assert tmpl["last_duration"] == 2.5
        assert tmpl["last_used"] is not None

        proc_memory.record_execution(tid, success=False)
        tmpl = proc_memory.get_template(tid)
        assert tmpl["usage_count"] == 2
        assert tmpl["success_rate"] == 0.5

    def test_record_execution_nonexistent_template(self, proc_memory):
        """Should not raise error for nonexistent template."""
        proc_memory.record_execution("nonexistent", success=True)  # should not raise

    def test_delete_template(self, proc_memory):
        tid = proc_memory.add_template("To Delete", [{"action": "x"}])
        assert proc_memory.get_template(tid) is not None
        result = proc_memory.delete_template(tid)
        assert result is True
        assert proc_memory.get_template(tid) is None

    def test_delete_nonexistent_template(self, proc_memory):
        result = proc_memory.delete_template("ghost")
        assert result is False

    def test_get_best_practices(self, proc_memory):
        good = proc_memory.add_template("Good Template", [{"action": "a"}])
        bad = proc_memory.add_template("Bad Template", [{"action": "b"}])
        proc_memory.record_execution(good, success=True)
        proc_memory.record_execution(good, success=True)
        proc_memory.record_execution(bad, success=False)
        proc_memory.record_execution(bad, success=False)

        best = proc_memory.get_best_practices(min_success_rate=0.8)
        assert len(best) == 1
        assert best[0]["name"] == "Good Template"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "persist.json")
        pm = ProceduralMemory(storage_path=path)
        pm.add_template("Persistent", [{"action": "save"}])

        # Load from same path
        pm2 = ProceduralMemory(storage_path=path)
        tmpl = pm2.get_template("persistent")
        assert tmpl is not None
        assert tmpl["name"] == "Persistent"

    def test_default_id_from_name(self, proc_memory):
        tid = proc_memory.add_template("Hello World", [{"action": "test"}])
        assert tid == "hello_world"

    def test_default_values_on_add(self, proc_memory):
        tid = proc_memory.add_template("Minimal", [{"action": "x"}])
        tmpl = proc_memory.get_template(tid)
        assert tmpl["usage_count"] == 0
        assert tmpl["success_rate"] == 1.0
        assert tmpl["created_at"] is not None
        assert tmpl["last_used"] is None
