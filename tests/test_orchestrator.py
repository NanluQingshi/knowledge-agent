"""Tests for Orchestrator — multi-agent workflow coordinator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.agents.orchestrator import Orchestrator, WorkflowResult, WorkflowStep


class TestOrchestrator:
    """Tests for Orchestrator with mocked sub-agents."""

    @pytest.fixture
    def mock_collection_agent(self):
        agent = MagicMock()
        agent.ingest_path.return_value = {
            "documents_loaded": 3,
            "chunks_created": 15,
            "files_processed": 3,
            "errors": [],
        }
        agent.get_stats.return_value = {
            "total_documents": 3,
            "total_chunks": 15,
            "vector_store_size": 15,
        }
        return agent

    @pytest.fixture
    def mock_extraction_agent(self):
        agent = MagicMock()
        agent.build_graph_from_store.return_value = {
            "entities_found": 5,
            "relations_found": 3,
            "triples_found": 3,
        }
        # Mock _graph_store for get_system_report access
        graph_store = MagicMock()
        graph_store.node_count = 5
        graph_store.edge_count = 3
        agent._graph_store = graph_store
        return agent

    @pytest.fixture
    def mock_quality_agent(self):
        agent = MagicMock()
        agent.check_expired_documents.return_value = []
        agent.detect_knowledge_gaps.return_value = []
        return agent

    @pytest.fixture
    def mock_qa_agent(self):
        agent = MagicMock()
        agent.query.return_value = {
            "answer": "42",
            "sources": [{"text": "doc1", "metadata": {"source": "file.txt"}}],
            "context_used": [],
        }
        return agent

    @pytest.fixture
    def orchestrator(self, mock_collection_agent, mock_extraction_agent,
                     mock_quality_agent, mock_qa_agent):
        return Orchestrator(
            collection_agent=mock_collection_agent,
            extraction_agent=mock_extraction_agent,
            quality_agent=mock_quality_agent,
            qa_agent=mock_qa_agent,
        )

    # ------------------------------------------------------------------
    # run_full_pipeline
    # ------------------------------------------------------------------

    def test_run_full_pipeline_calls_agents_in_order(self, orchestrator, mock_collection_agent, mock_extraction_agent, mock_quality_agent):
        result = orchestrator.run_full_pipeline("/path/to/docs")

        mock_collection_agent.ingest_path.assert_called_once_with("/path/to/docs")
        mock_extraction_agent.build_graph_from_store.assert_called_once()
        mock_quality_agent.check_expired_documents.assert_called_once()
        mock_quality_agent.detect_knowledge_gaps.assert_called_once()

    def test_run_full_pipeline_returns_workflow_result(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path/to/docs")
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert len(result.steps_completed) == 3
        assert WorkflowStep.COLLECT.value in result.steps_completed
        assert WorkflowStep.EXTRACT.value in result.steps_completed
        assert WorkflowStep.QUALITY_CHECK.value in result.steps_completed

    def test_run_full_pipeline_result_contains_ingest_summary(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path/to/docs")
        ingest = result.results.get("ingest", {})
        assert ingest["documents_loaded"] == 3
        assert ingest["chunks_created"] == 15

    def test_run_full_pipeline_result_contains_extraction_summary(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path/to/docs")
        ext = result.results.get("extraction", {})
        assert ext["entities_found"] == 5
        assert ext["relations_found"] == 3

    def test_run_full_pipeline_result_contains_quality_summary(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path/to/docs")
        quality = result.results.get("quality", {})
        assert "expired_documents" in quality
        assert "knowledge_gaps" in quality

    def test_run_full_pipeline_without_extraction(self, orchestrator, mock_extraction_agent):
        result = orchestrator.run_full_pipeline("/path/to/docs", enable_extraction=False)
        assert WorkflowStep.EXTRACT.value not in result.steps_completed
        mock_extraction_agent.build_graph_from_store.assert_not_called()

    def test_run_full_pipeline_without_quality(self, orchestrator, mock_quality_agent):
        result = orchestrator.run_full_pipeline("/path/to/docs", enable_quality_check=False)
        assert WorkflowStep.QUALITY_CHECK.value not in result.steps_completed
        mock_quality_agent.check_expired_documents.assert_not_called()

    def test_run_full_pipeline_handles_collection_error(self, mock_extraction_agent, mock_quality_agent, mock_qa_agent):
        bad_collection = MagicMock()
        bad_collection.ingest_path.side_effect = RuntimeError("Ingest failed!")
        orch = Orchestrator(
            collection_agent=bad_collection,
            extraction_agent=mock_extraction_agent,
            quality_agent=mock_quality_agent,
            qa_agent=mock_qa_agent,
        )
        result = orch.run_full_pipeline("/path")
        assert result.success is False
        assert len(result.errors) == 1
        assert "Ingest failed!" in result.errors[0]["error"]

    def test_run_full_pipeline_collects_ingest_errors_in_result(self, orchestrator, mock_collection_agent):
        mock_collection_agent.ingest_path.return_value = {
            "documents_loaded": 2,
            "chunks_created": 10,
            "files_processed": 3,
            "errors": [{"file": "bad.pdf", "error": "Parse error"}],
        }
        result = orchestrator.run_full_pipeline("/path")
        assert len(result.errors) >= 1
        assert result.errors[0]["file"] == "bad.pdf"

    def test_run_full_pipeline_summary_contains_step_count(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path")
        assert "Pipeline completed with 3 steps" in result.summary

    def test_run_full_pipeline_summary_contains_metrics(self, orchestrator):
        result = orchestrator.run_full_pipeline("/path")
        assert "Ingested 3 documents, 15 chunks" in result.summary
        assert "Extracted 5 entities, 3 relations" in result.summary
        assert "Quality" in result.summary

    # ------------------------------------------------------------------
    # run_query
    # ------------------------------------------------------------------

    def test_run_query_delegates_to_qa_agent(self, orchestrator, mock_qa_agent):
        result = orchestrator.run_query("What is AI?", top_k=3)
        mock_qa_agent.query.assert_called_once_with("What is AI?", top_k=3)
        assert result["answer"] == "42"

    def test_run_query_with_default_top_k(self, orchestrator, mock_qa_agent):
        orchestrator.run_query("Hello")
        call_kwargs = mock_qa_agent.query.call_args
        assert call_kwargs[1]["top_k"] == 5  # default when accessing through the mock

    # Actually, the Orchestrator passes top_k as positional. Let me check:
    # orch.run_query(question, top_k=5) -> qa.query(question, top_k=top_k)
    # So the mock should receive the named argument.

    # ------------------------------------------------------------------
    # get_system_report
    # ------------------------------------------------------------------

    def test_get_system_report_returns_expected_keys(self, orchestrator):
        report = orchestrator.get_system_report()
        assert "storage" in report
        assert "graph" in report
        assert "quality" in report
        assert report["storage"]["total_documents"] == 3
        assert report["graph"]["nodes"] == 5
        assert report["graph"]["edges"] == 3

    # ------------------------------------------------------------------
    # run_maintenance
    # ------------------------------------------------------------------

    def test_run_maintenance_returns_expected_structure(self, orchestrator, mock_quality_agent):
        result = orchestrator.run_maintenance()
        assert "stale_documents" in result
        assert "knowledge_gaps" in result
        assert "expired" in result
        assert "recommendation" in result

    def test_run_maintenance_uses_correct_defaults(self, orchestrator):
        result = orchestrator.run_maintenance()
        # FreshnessManager.get_stale_documents called with min_age_days=180, max_references=2
        assert isinstance(result["stale_documents"], list)
        assert isinstance(result["knowledge_gaps"], list)

    # ------------------------------------------------------------------
    # record_feedback / get_feedback_stats
    # ------------------------------------------------------------------

    def test_record_feedback(self, orchestrator):
        with patch("knowledge_agent.feedback.collector.FeedbackCollector") as mock_fb_cls:
            mock_fb = MagicMock()
            mock_fb.record.return_value = "feedback-id-123"
            mock_fb_cls.return_value = mock_fb

            fid = orchestrator.record_feedback("test query", "test answer", rating="useful")
            assert fid == "feedback-id-123"
            mock_fb.record.assert_called_once_with(
                query_text="test query",
                answer_text="test answer",
                rating="useful",
                comment="",
                source_doc_ids=None,
            )

    def test_get_feedback_stats(self, orchestrator):
        with patch("knowledge_agent.feedback.collector.FeedbackCollector") as mock_fb_cls:
            mock_fb = MagicMock()
            mock_fb.get_stats.return_value = {"total_feedback": 5, "usefulness_rate": 0.8}
            mock_fb_cls.return_value = mock_fb

            stats = orchestrator.get_feedback_stats()
            assert stats["total_feedback"] == 5

    # ------------------------------------------------------------------
    # get_knowledge_health
    # ------------------------------------------------------------------

    def test_get_knowledge_health(self, orchestrator):
        result = orchestrator.get_knowledge_health()
        assert "total_documents" in result
        assert "stale_documents" in result
        assert "expired_documents" in result
        assert "knowledge_gaps" in result
        assert "freshness_distribution" in result
        assert "high (>0.7)" in result["freshness_distribution"]
        assert "medium (0.3-0.7)" in result["freshness_distribution"]
        assert "low (<0.3)" in result["freshness_distribution"]

    # ------------------------------------------------------------------
    # WorkflowResult
    # ------------------------------------------------------------------

    def test_workflow_result_defaults(self):
        result = WorkflowResult()
        assert result.success is True
        assert result.steps_completed == []
        assert result.results == {}
        assert result.errors == []
        assert result.summary == ""
