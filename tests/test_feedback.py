"""Tests for feedback and evolution modules: FeedbackCollector, FreshnessManager, KnowledgeScorer."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.feedback.collector import FeedbackCollector
from knowledge_agent.feedback.freshness import FreshnessManager
from knowledge_agent.feedback.scorer import KnowledgeScorer


# ===================================================================
# FeedbackCollector
# ===================================================================

class TestFeedbackCollector:
    """Tests for SQLite-backed FeedbackCollector."""

    @pytest.fixture
    def collector(self, tmp_path):
        db_path = str(tmp_path / "feedback.db")
        return FeedbackCollector(db_path=db_path)

    def test_record_and_get_stats(self, collector):
        collector.record("What is AI?", rating="useful")
        collector.record("Tell me a joke", rating="useless")
        collector.record("How are you?", rating="partial")

        stats = collector.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["useful"] == 1
        assert stats["useless"] == 1
        assert stats["partial"] == 1
        assert stats["usefulness_rate"] == pytest.approx(1 / 3, abs=1e-3)

    def test_get_stats_empty(self, collector):
        stats = collector.get_stats()
        assert stats["total_feedback"] == 0
        assert stats["usefulness_rate"] == 0.0

    def test_record_only_useful(self, collector):
        collector.record("Test query", rating="useful")
        stats = collector.get_stats()
        assert stats["total_feedback"] == 1
        assert stats["usefulness_rate"] == 1.0

    def test_get_recent_ordered(self, collector):
        collector.record("First query", rating="useful")
        collector.record("Second query", rating="useless")
        collector.record("Third query", rating="partial")

        recent = collector.get_recent(limit=10)
        assert len(recent) == 3
        # Most recent should be first
        assert recent[0]["query_text"] == "Third query"
        assert recent[-1]["query_text"] == "First query"

    def test_get_recent_limit(self, collector):
        for i in range(5):
            collector.record(f"Query {i}", rating="useful")

        recent = collector.get_recent(limit=2)
        assert len(recent) == 2

    def test_record_returns_id(self, collector):
        fid = collector.record("test query", rating="useful")
        assert isinstance(fid, str)
        assert len(fid) > 0

    def test_record_with_all_fields(self, collector):
        collector.record(
            query_text="test query",
            answer_text="test answer",
            rating="useful",
            comment="Great answer!",
            source_doc_ids=["doc1", "doc2"],
        )
        recent = collector.get_recent(limit=1)
        assert recent[0]["answer_text"] == "test answer"
        assert recent[0]["comment"] == "Great answer!"
        assert recent[0]["source_doc_ids"] == ["doc1", "doc2"]

    def test_get_unhelpful_queries(self, collector):
        collector.record("Good one", rating="useful")
        collector.record("Bad one", rating="useless")
        collector.record("Partial one", rating="partial")

        unhelpful = collector.get_unhelpful_queries(limit=10)
        assert len(unhelpful) == 2
        ratings = {r["rating"] for r in unhelpful}
        assert ratings == {"useless", "partial"}


# ===================================================================
# FreshnessManager
# ===================================================================

class TestFreshnessManager:
    """Tests for FreshnessManager (time-decay scoring)."""

    @pytest.fixture
    def mock_doc_store(self):
        return MagicMock()

    @pytest.fixture
    def freshness(self, mock_doc_store):
        return FreshnessManager(doc_store=mock_doc_store)

    def test_calculate_returns_value_in_range(self, freshness):
        now = datetime.now(timezone.utc).isoformat()
        score = freshness.calculate(now, reference_count=0)
        assert 0.0 <= score <= 1.0

    def test_calculate_older_is_lower(self, freshness):
        recent = datetime.now(timezone.utc).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
        score_recent = freshness.calculate(recent, reference_count=0)
        score_old = freshness.calculate(old, reference_count=0)
        assert score_recent > score_old

    def test_calculate_higher_refs_higher_score(self, freshness):
        now = datetime.now(timezone.utc).isoformat()
        low_ref = freshness.calculate(now, reference_count=0)
        high_ref = freshness.calculate(now, reference_count=100)
        assert high_ref > low_ref

    def test_calculate_parses_age_correctly(self, freshness):
        one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        score = freshness.calculate(one_year_ago, reference_count=0)
        # After one year (half life 180 days), score should be < 0.5
        assert score < 0.5

    def test_calculate_with_invalid_date(self, freshness):
        score = freshness.calculate("invalid-date")
        assert 0.0 <= score <= 1.0

    def test_calculate_batch_adds_freshness_score(self, freshness):
        items = [
            {"id": "1", "ingested_at": datetime.now(timezone.utc).isoformat(), "reference_count": 0},
            {"id": "2", "ingested_at": (datetime.now(timezone.utc) - timedelta(days=500)).isoformat(), "reference_count": 0},
        ]
        result = freshness.calculate_batch(items)
        assert len(result) == 2
        for item in result:
            assert "freshness_score" in item
        # Newer item should have higher score -> appears first
        assert result[0]["id"] == "1"

    def test_get_stale_documents(self, freshness, mock_doc_store):
        now = datetime.now(timezone.utc)
        mock_doc_store.list_documents.return_value = [
            {"id": "old_unused", "ingested_at": (now - timedelta(days=400)).isoformat(), "chunk_count": 0},
            {"id": "new_used", "ingested_at": now.isoformat(), "chunk_count": 10},
        ]
        stale = freshness.get_stale_documents(min_age_days=365, max_references=2)
        assert len(stale) == 1
        assert stale[0]["id"] == "old_unused"

    def test_get_stale_documents_empty_when_none_stale(self, freshness, mock_doc_store):
        now = datetime.now(timezone.utc).isoformat()
        mock_doc_store.list_documents.return_value = [
            {"id": "recent", "ingested_at": now, "chunk_count": 5},
        ]
        stale = freshness.get_stale_documents(min_age_days=365, max_references=2)
        assert stale == []

    def test_get_decay_schedule(self, freshness):
        items = [
            {"id": "1", "ingested_at": datetime.now(timezone.utc).isoformat(), "reference_count": 5},
        ]
        schedule = freshness.get_decay_schedule(items, days_list=[30, 90])
        assert 30 in schedule
        assert 90 in schedule
        assert len(schedule[30]) == 1
        assert "predicted_score" in schedule[30][0]
        # Score at 30 days should be higher than at 90 days
        assert schedule[30][0]["predicted_score"] > schedule[90][0]["predicted_score"]

    def test_get_all_with_freshness(self, freshness, mock_doc_store):
        now = datetime.now(timezone.utc).isoformat()
        mock_doc_store.list_documents.return_value = [
            {"id": "doc1", "ingested_at": now, "chunk_count": 3},
        ]
        docs = freshness.get_all_with_freshness()
        assert len(docs) == 1
        assert "freshness_score" in docs[0]


# ===================================================================
# KnowledgeScorer
# ===================================================================

class TestKnowledgeScorer:
    """Tests for KnowledgeScorer (quality scoring)."""

    @pytest.fixture
    def mock_feedback(self):
        fb = MagicMock()
        fb.get_stats.return_value = {"usefulness_rate": 0.8}
        return fb

    @pytest.fixture
    def scorer(self, mock_feedback):
        return KnowledgeScorer(feedback_collector=mock_feedback)

    def test_score_document_returns_float(self, scorer):
        score = scorer.score_document("doc1", usefulness_rate=1.0, citation_count=5, age_days=0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_document_high_quality(self, scorer):
        score = scorer.score_document("doc1", usefulness_rate=1.0, citation_count=100, age_days=0)
        assert score > 0.5

    def test_score_document_low_quality(self, scorer):
        score = scorer.score_document("doc1", usefulness_rate=0.0, citation_count=0, age_days=365)
        assert score == 0.0

    def test_score_clamped_to_1(self, scorer):
        score = scorer.score_document("doc1", usefulness_rate=1.0, citation_count=1000, age_days=0)
        assert score <= 1.0

    def test_score_clamped_to_0(self, scorer):
        score = scorer.score_document("doc1", usefulness_rate=-1.0, citation_count=0, age_days=0)
        assert score >= 0.0

    def test_score_uses_feedback_when_unavailable(self, scorer, mock_feedback):
        score = scorer.score_document("doc1", citation_count=5, age_days=30)
        # Should use mock feedback rate of 0.8
        assert 0.0 <= score <= 1.0

    def test_score_batch_sorts_by_score(self, scorer):
        docs = [
            {"id": "good", "doc_id": "good", "citation_count": 100, "age_days": 0},
            {"id": "bad", "doc_id": "bad", "citation_count": 0, "age_days": 1000},
        ]
        scored = scorer.score_batch(docs)
        assert len(scored) == 2
        assert scored[0]["id"] == "good"  # higher quality
        assert scored[1]["id"] == "bad"
        for doc in scored:
            assert "quality_score" in doc

    def test_score_batch_empty(self, scorer):
        assert scorer.score_batch([]) == []

    def test_get_top_documents(self, scorer):
        docs = [
            {"id": "a", "doc_id": "a", "citation_count": 2, "age_days": 0},
            {"id": "b", "doc_id": "b", "citation_count": 1, "age_days": 0},
            {"id": "c", "doc_id": "c", "citation_count": 5, "age_days": 0},
        ]
        top = scorer.get_top_documents(docs, top_k=2)
        assert len(top) == 2
        assert top[0]["id"] == "c"

    def test_get_low_quality_documents(self, scorer):
        docs = [
            {"doc_id": "good", "citation_count": 100, "age_days": 0},
            {"doc_id": "poor", "citation_count": 0, "age_days": 1000},
        ]
        low = scorer.get_low_quality_documents(docs, threshold=0.3)
        assert len(low) == 1
        assert low[0]["doc_id"] == "poor"

    def test_uses_doc_id_fallback(self, scorer):
        """When 'doc_id' key is absent, falls back to 'id'."""
        doc = {"id": "fallback", "citation_count": 0, "age_days": 0}
        scored = scorer.score_batch([doc])
        assert len(scored) == 1
