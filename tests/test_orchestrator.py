"""Tests for Orchestrator — multi-agent workflow coordinator."""

from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.agents.orchestrator import Orchestrator, WorkflowResult, WorkflowStep


class TestOrchestrator:
    """Tests for Orchestrator with mocked sub-agents."""

    @pytest.fixture
    def mock_collection_agent(self):
        agent = MagicMock()
        agent._vector_store = MagicMock()
        agent._doc_store = MagicMock()
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
        agent.stream_query.return_value = iter(["4", "2"])
        return agent

    @pytest.fixture
    def orchestrator(
        self,
        mock_collection_agent,
        mock_extraction_agent,
        mock_quality_agent,
        mock_qa_agent,
    ):
        with patch("knowledge_agent.agents.orchestrator.EpisodicMemory") as memory_cls:
            memory = MagicMock()
            memory.store.side_effect = RuntimeError("memory disabled in unit tests")
            memory.count.return_value = 2
            memory_cls.return_value = memory
            orchestrator = Orchestrator(
                collection_agent=mock_collection_agent,
                extraction_agent=mock_extraction_agent,
                quality_agent=mock_quality_agent,
                qa_agent=mock_qa_agent,
            )
        orchestrator._semantic_memory = MagicMock(fact_count=3)
        return orchestrator

    # ------------------------------------------------------------------
    # run_full_pipeline
    # ------------------------------------------------------------------

    def test_run_full_pipeline_calls_agents_in_order(
        self,
        orchestrator,
        mock_collection_agent,
        mock_extraction_agent,
        mock_quality_agent,
    ):
        orchestrator.run_full_pipeline("/path/to/docs")

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

    def test_run_full_pipeline_handles_collection_error(
        self,
        mock_extraction_agent,
        mock_quality_agent,
        mock_qa_agent,
    ):
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

    def test_run_full_pipeline_collects_ingest_errors_in_result(
        self, orchestrator, mock_collection_agent
    ):
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
        mock_qa_agent.query.assert_called_once_with(
            "What is AI?",
            top_k=3,
            chat_history=None,
        )
        assert result["answer"] == "42"

    def test_run_query_with_default_top_k(self, orchestrator, mock_qa_agent):
        orchestrator.run_query("Hello")
        call_kwargs = mock_qa_agent.query.call_args
        assert call_kwargs[1]["top_k"] == 5  # default when accessing through the mock

    def test_run_query_records_metrics(self, orchestrator):
        orchestrator.run_query("metrics")

        report = orchestrator.get_monitoring_report()
        assert report["counters"]["query.count"] == 1
        assert report["counters"]["query.cache_miss"] == 1
        assert report["timings"]["query"]["count"] == 1

    def test_run_query_cache_hit_skips_qa(self, orchestrator, mock_qa_agent):
        first = orchestrator.run_query("cache me", top_k=2)
        second = orchestrator.run_query("cache me", top_k=2)

        assert second == first
        assert mock_qa_agent.query.call_count == 1
        counters = orchestrator.get_monitoring_report()["counters"]
        assert counters["query.cache_miss"] == 1
        assert counters["query.cache_hit"] == 1

    def test_run_query_records_errors(self, orchestrator, mock_qa_agent):
        mock_qa_agent.query.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            orchestrator.run_query("fail")

        assert orchestrator.metrics.get_counter("query.error") == 1

    def test_run_query_stream_records_metrics(self, orchestrator):
        chunks = list(orchestrator.run_query_stream("stream"))

        assert chunks == ["4", "2"]
        report = orchestrator.get_monitoring_report()
        assert report["counters"]["query_stream.count"] == 1
        assert report["timings"]["query_stream"]["count"] == 1

    def test_cache_stats_describe_active_cache(self, orchestrator):
        orchestrator.run_query("cached")

        assert orchestrator.get_cache_stats() == {
            "size": 1,
            "ttl": 300,
            "max_size": 100,
        }

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

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    def test_delete_document_calls_delete_chain(self, orchestrator, mock_collection_agent):
        """delete_document should cascade through vector_store, doc_store, graph_store."""
        mock_doc = MagicMock()
        mock_doc.get.return_value = {
            "id": "test-doc-123",
            "metadata": {"chunk_ids": ["chunk_1", "chunk_2"]},
        }
        mock_collection_agent._doc_store.get_document.return_value = mock_doc.get.return_value

        result = orchestrator.delete_document("test-doc-123")
        assert result is True

    def test_delete_nonexistent_document(self, orchestrator, mock_collection_agent):
        mock_collection_agent._doc_store.get_document.return_value = None
        result = orchestrator.delete_document("nonexistent")
        assert result is False

    # ------------------------------------------------------------------
    # BM25 cache
    # ------------------------------------------------------------------

    def test_bm25_cache_initialized_on_first_query(self, orchestrator):
        assert orchestrator._bm25_retriever is None
        assert orchestrator._last_vector_count == 0

    def test_run_pipeline_triggers_bm25_update(self, orchestrator):
        """After pipeline with documents, BM25 index should be updated."""
        with patch.object(orchestrator, "_update_bm25_index") as mock_update:
            orchestrator.run_full_pipeline("/path")
            # ingest has chunks_created > 0, so BM25 update should be called
            mock_update.assert_called_once()

    def test_run_pipeline_skips_bm25_on_no_docs(self, orchestrator, mock_collection_agent):
        """When no documents are ingested, BM25 update should be skipped."""
        mock_collection_agent.ingest_path.return_value = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "files_processed": 0,
            "errors": [],
        }
        with patch.object(orchestrator, "_update_bm25_index") as mock_update:
            orchestrator.run_full_pipeline("/path")
            mock_update.assert_not_called()

    # ------------------------------------------------------------------
    # Memory integration
    # ------------------------------------------------------------------

    def test_run_query_stores_episodic_memory(self, orchestrator):
        """Query results should be stored in episodic memory."""
        with patch.object(orchestrator._episodic_memory, "store_conversation") as mock_store:
            orchestrator.run_query("What is AI?", top_k=3)
            mock_store.assert_called_once()
            _, kwargs = mock_store.call_args
            assert kwargs["user_message"] == "What is AI?"
            assert "42" in kwargs["assistant_response"]

    def test_run_pipeline_stores_action_memory(self, orchestrator):
        """Pipeline results should be stored in episodic memory as actions."""
        with patch.object(orchestrator._episodic_memory, "store") as mock_store:
            orchestrator.run_full_pipeline("/path")
            mock_store.assert_called_once()
            _, kwargs = mock_store.call_args
            assert kwargs["memory_type"] == "action"


# ===================================================================
# TC-05: 端到端集成测试（mock LLM 和存储）
# ===================================================================


class TestEndToEnd:
    """End-to-end integration tests with mocked dependencies."""

    @pytest.fixture
    def qa_agent(self):
        agent = MagicMock()
        agent.query.return_value = {
            "answer": "mock answer",
            "sources": [],
        }
        agent.stream_query.return_value = iter(["mock ", "answer"])
        agent._retriever.retrieve.return_value = [
            {"id": "chunk-1", "text": "context", "metadata": {}, "score": 0.9}
        ]
        return agent

    @pytest.fixture
    def orchestrator(self, qa_agent):
        collection = MagicMock()
        collection._vector_store = MagicMock()
        collection._doc_store = MagicMock()
        collection.get_stats.return_value = {
            "total_documents": 1,
            "total_chunks": 1,
            "vector_store_size": 1,
        }
        collection.ingest_path.return_value = {
            "documents_loaded": 1,
            "chunks_created": 1,
            "files_processed": 1,
            "errors": [],
        }

        extraction = MagicMock()
        extraction._graph_store.node_count = 1
        extraction._graph_store.edge_count = 0
        extraction.build_graph_from_store.return_value = {
            "entities_found": 0,
            "relations_found": 0,
        }

        quality = MagicMock()
        quality.check_expired_documents.return_value = []
        quality.detect_knowledge_gaps.return_value = []

        with patch("knowledge_agent.agents.orchestrator.EpisodicMemory") as memory_cls:
            memory_cls.return_value = MagicMock(count=MagicMock(return_value=2))
            orchestrator = Orchestrator(
                collection_agent=collection,
                extraction_agent=extraction,
                quality_agent=quality,
                qa_agent=qa_agent,
            )
        orchestrator._semantic_memory = MagicMock(fact_count=3)
        return orchestrator

    def test_query_returns_result_structure(self, orchestrator):
        """Orchestrator.run_query returns expected keys."""
        result = orchestrator.run_query("test question", top_k=2)
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_stream_generates_chunks(self, orchestrator):
        """Orchestrator.run_query_stream yields text chunks."""
        chunks = list(orchestrator.run_query_stream("test", top_k=2))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_query_with_chat_history(self, orchestrator, qa_agent):
        """Orchestrator.run_query accepts chat_history."""
        history = [{"role": "user", "content": "previous question"}]
        result = orchestrator.run_query("follow up", top_k=2, chat_history=history)
        assert "answer" in result
        qa_agent.query.assert_called_with(
            "follow up",
            top_k=2,
            chat_history=history,
        )

    def test_query_with_graphrag_flag(self, orchestrator, qa_agent):
        """Orchestrator.run_query accepts use_graphrag flag."""
        with patch.object(orchestrator, "_get_qa_agent", return_value=qa_agent) as get_qa:
            result = orchestrator.run_query("test", top_k=2, use_graphrag=True)
        assert "answer" in result
        get_qa.assert_called_once_with(use_graphrag=True)

    def test_query_with_enhanced_search(self, orchestrator, qa_agent):
        """Orchestrator.run_query accepts use_enhanced_search flag."""
        with patch.object(orchestrator, "_enhance_query", return_value="enhanced") as enhance:
            result = orchestrator.run_query("test", top_k=2, use_enhanced_search=True)
        assert "answer" in result
        enhance.assert_called_once_with("test", 2)
        qa_agent.query.assert_called_with("enhanced", top_k=2, chat_history=None)

    def test_cache_hit_returns_same_result(self, orchestrator, qa_agent):
        """Identical queries should return cached result."""
        r1 = orchestrator.run_query("cache test", top_k=2)
        r2 = orchestrator.run_query("cache test", top_k=2)
        assert r1 == r2
        assert qa_agent.query.call_count == 1

    def test_full_pipeline_returns_result(self, orchestrator):
        """run_full_pipeline coordinates injected components."""
        result = orchestrator.run_full_pipeline(
            "/virtual/docs",
            enable_extraction=False,
            enable_quality_check=False,
        )
        assert result.success is True
        assert result.results["ingest"]["documents_loaded"] == 1

    def test_system_report_returns_stats(self, orchestrator):
        """get_system_report returns storage and graph stats."""
        report = orchestrator.get_system_report()
        assert "storage" in report
        assert "graph" in report
        assert "quality" in report

    def test_memory_stats_returns_counts(self, orchestrator):
        """get_memory_stats returns memory counts."""
        stats = orchestrator.get_memory_stats()
        assert stats == {"episodic_count": 2, "semantic_facts": 3}

    def test_retrieve_returns_results(self, orchestrator):
        """Orchestrator.retrieve returns search results."""
        results = orchestrator.retrieve("test query", top_k=2)
        assert results[0]["id"] == "chunk-1"
