"""Tests for evaluation module: dataset, metrics, and runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.evaluation.dataset import EvaluationDataset
from knowledge_agent.evaluation.metrics import RetrievalMetrics
from knowledge_agent.evaluation.runner import EvaluationRunner


# ===================================================================
# EvaluationDataset
# ===================================================================

class TestEvaluationDataset:
    """Tests for EvaluationDataset (JSON-backed storage)."""

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> EvaluationDataset:
        path = str(tmp_path / "eval.json")
        return EvaluationDataset(path=path)

    def test_add_and_list(self, dataset: EvaluationDataset):
        qid = dataset.add_item(
            query="What is GraphRAG?",
            expected_doc_ids=["doc1", "doc2"],
            expected_answer="GraphRAG is...",
            category="tech",
            difficulty="medium",
        )
        assert qid is not None
        items = dataset.list_items()
        assert len(items) == 1
        assert items[0]["query"] == "What is GraphRAG?"
        assert items[0]["expected_doc_ids"] == ["doc1", "doc2"]

    def test_list_with_category_filter(self, dataset: EvaluationDataset):
        dataset.add_item(query="Q1", category="tech")
        dataset.add_item(query="Q2", category="science")
        dataset.add_item(query="Q3", category="tech")
        items = dataset.list_items(category="tech")
        assert len(items) == 2
        assert all(i["category"] == "tech" for i in items)

    def test_list_with_difficulty_filter(self, dataset: EvaluationDataset):
        dataset.add_item(query="Q1", difficulty="easy")
        dataset.add_item(query="Q2", difficulty="hard")
        items = dataset.list_items(difficulty="easy")
        assert len(items) == 1
        assert items[0]["query"] == "Q1"

    def test_remove_item(self, dataset: EvaluationDataset):
        qid = dataset.add_item(query="Remove me")
        assert dataset.size == 1
        assert dataset.remove_item(qid) is True
        assert dataset.size == 0

    def test_remove_nonexistent(self, dataset: EvaluationDataset):
        assert dataset.remove_item("nonexistent") is False

    def test_clear(self, dataset: EvaluationDataset):
        dataset.add_item(query="A")
        dataset.add_item(query="B")
        assert dataset.size == 2
        dataset.clear()
        assert dataset.size == 0

    def test_export_and_reload(self, dataset: EvaluationDataset, tmp_path: Path):
        dataset.add_item(query="Export test", category="test")
        export_path = str(tmp_path / "exported.json")
        dataset.export_to(export_path)

        loaded = EvaluationDataset.from_file(export_path)
        assert loaded.size == 1
        assert loaded.list_items()[0]["query"] == "Export test"

    def test_empty_dataset(self, dataset: EvaluationDataset):
        assert dataset.size == 0
        assert dataset.list_items() == []

    def test_metadata_defaults(self, dataset: EvaluationDataset):
        dataset.add_item(query="Meta test")
        item = dataset.list_items()[0]
        assert item["metadata"] == {}
        assert item["category"] == "general"
        assert item["difficulty"] == "medium"

    def test_custom_metadata(self, dataset: EvaluationDataset):
        dataset.add_item(query="Custom", metadata={"source": "test"})
        item = dataset.list_items()[0]
        assert item["metadata"]["source"] == "test"


# ===================================================================
# RetrievalMetrics
# ===================================================================

class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def test_mrr_perfect_rank(self):
        """MRR = 1.0 when first result is relevant."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a"],
            k=3,
        )
        assert metrics["mrr"] == 1.0

    def test_mrr_second_rank(self):
        """MRR = 0.5 when second result is relevant."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["b"],
            k=3,
        )
        assert metrics["mrr"] == 0.5

    def test_mrr_no_relevant(self):
        """MRR = 0.0 when no relevant doc is retrieved."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b"],
            relevant_ids=["c"],
            k=2,
        )
        assert metrics["mrr"] == 0.0

    def test_recall_at_k(self):
        """Recall@3 = 2/3 = 0.6667."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b", "c", "d"],
            relevant_ids=["a", "b", "e"],
            k=3,
        )
        assert abs(metrics["recall@3"] - 2.0 / 3.0) < 0.001

    def test_recall_all_relevant_retrieved(self):
        """Recall@k = 1.0 when all relevant docs are in top-k."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a", "b"],
            k=3,
        )
        assert metrics["recall@3"] == 1.0

    def test_precision_at_k(self):
        """Precision@3 = 1/3 when only first of 3 is relevant."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "x", "y"],
            relevant_ids=["a"],
            k=3,
        )
        assert abs(metrics["precision@3"] - 1.0 / 3.0) < 0.001

    def test_precision_perfect(self):
        """Precision@k = 1.0 when all top-k are relevant."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b"],
            relevant_ids=["a", "b", "c"],
            k=2,
        )
        assert metrics["precision@2"] == 1.0

    def test_ndcg_perfect(self):
        """NDCG@k = 1.0 when relevant docs are perfectly ranked."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a", "b"],
            k=3,
        )
        assert abs(metrics["ndcg@3"] - 1.0) < 0.001

    def test_ndcg_imperfect(self):
        """NDCG is less than 1.0 when ranking is suboptimal."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["x", "a", "y"],
            relevant_ids=["a"],
            k=3,
        )
        assert 0 < metrics["ndcg@3"] < 1.0

    def test_empty_retrieved(self):
        """All metrics are 0 when nothing is retrieved."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=[],
            relevant_ids=["a", "b"],
            k=5,
        )
        assert metrics["mrr"] == 0.0

    def test_empty_relevant(self):
        """All metrics are 0 when there are no relevant docs."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["a", "b"],
            relevant_ids=[],
            k=5,
        )
        assert metrics["mrr"] == 0.0

    def test_evaluate_batch(self):
        """Batch evaluation averages across queries."""
        results = [
            {"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},
            {"retrieved_ids": ["x", "y"], "relevant_ids": ["y"]},
        ]
        agg = RetrievalMetrics.evaluate_batch(results, k=2)
        assert "mrr" in agg
        assert "recall" in agg
        assert "precision" in agg
        assert "ndcg" in agg
        assert agg["num_queries"] == 2
        # MRR: (1.0 + 0.5) / 2 = 0.75
        assert abs(agg["mrr"] - 0.75) < 0.001

    def test_evaluate_batch_empty(self):
        """Batch with no valid queries returns zeros."""
        agg = RetrievalMetrics.evaluate_batch([], k=5)
        assert agg["mrr"] == 0.0
        assert "num_queries" not in agg

    def test_ndcg_no_gain(self):
        """NDCG = 0 when no relevant docs in any position."""
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids=["x", "y", "z"],
            relevant_ids=[],
            k=3,
        )
        # NDCG returns 0.0 when idcg is 0 (no relevant docs)
        assert "ndcg@3" in metrics


# ===================================================================
# EvaluationRunner
# ===================================================================

class TestEvaluationRunner:
    """Tests for EvaluationRunner with mocked Orchestrator."""

    @pytest.fixture
    def mock_orchestrator(self):
        orch = MagicMock()
        qa_agent = MagicMock()
        hybrid_retriever = MagicMock()
        hybrid_retriever.retrieve.return_value = [
            {"id": "doc1", "text": "Doc 1 content", "metadata": {"source": "s1"}},
            {"id": "doc2", "text": "Doc 2 content", "metadata": {"source": "s2"}},
        ]
        qa_agent._retriever = hybrid_retriever
        orch._get_qa_agent.return_value = qa_agent
        orch.run_query.return_value = {
            "answer": "Test answer from LLM",
            "sources": [{"text": "Source text", "metadata": {"source": "doc1"}}],
        }
        return orch

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> EvaluationDataset:
        path = str(tmp_path / "runner_eval.json")
        ds = EvaluationDataset(path=path)
        ds.add_item(
            query="Test query",
            expected_doc_ids=["doc1", "doc2"],
            category="general",
        )
        return ds

    def test_evaluate_retrieval_returns_metrics(
        self, mock_orchestrator, dataset: EvaluationDataset
    ):
        runner = EvaluationRunner(
            orchestrator=mock_orchestrator,
            dataset=dataset,
        )
        result = runner.evaluate_retrieval(top_k=5)
        assert result["status"] == "ok"
        assert result["num_queries"] == 1
        assert "metrics" in result
        assert "summary" in result
        assert "details" in result

    def test_evaluate_retrieval_no_data(self, mock_orchestrator, tmp_path: Path):
        empty_ds = EvaluationDataset(path=str(tmp_path / "empty_eval.json"))
        runner = EvaluationRunner(
            orchestrator=mock_orchestrator,
            dataset=empty_ds,
        )
        result = runner.evaluate_retrieval(top_k=5)
        assert result["status"] == "no_data"

    def test_evaluate_answer_quality(
        self, mock_orchestrator, dataset: EvaluationDataset
    ):
        runner = EvaluationRunner(
            orchestrator=mock_orchestrator,
            dataset=dataset,
        )
        result = runner.evaluate_answer_quality(top_k=5)
        assert result["status"] == "ok"
        assert result["num_queries"] >= 1
        assert "summary" in result
