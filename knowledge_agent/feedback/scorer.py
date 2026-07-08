"""知识质量评分模块 — 基于反馈和引用频率的知识质量评估."""

from __future__ import annotations

import math
from typing import Any


class KnowledgeScorer:
    """知识质量评分器.

    综合用户反馈、引用频率和时间因素计算每条知识的质量分数。
    """

    def __init__(self, feedback_collector=None) -> None:
        from knowledge_agent.feedback.collector import FeedbackCollector

        self._feedback = feedback_collector or FeedbackCollector()

    # ------------------------------------------------------------------
    # 评分
    # ------------------------------------------------------------------

    def score_document(
        self,
        doc_id: str,
        usefulness_rate: float | None = None,
        citation_count: int = 0,
        age_days: int = 0,
    ) -> float:
        """计算单个文档的综合质量分数.

        评分公式: Q = U * log(1 + citations) * e^(-age/365)

        其中 U 基于用户反馈评分的加权平均 (useful=1.0, partial=0.5, useless=0.0)。

        Args:
            doc_id: 文档 ID.
            usefulness_rate: 预计算的有用率，None 时从反馈统计获取.
            citation_count: 被引用次数.
            age_days: 文档存在天数.

        Returns:
            质量分数 (0.0 ~ 1.0).
        """
        if usefulness_rate is None:
            usefulness_rate = self._get_usefulness_for_doc(doc_id)

        citation_bonus = math.log(1 + max(citation_count, 0))
        age_decay = math.exp(-max(age_days, 0) / 365.0)

        score = usefulness_rate * citation_bonus * age_decay
        return round(min(max(score, 0.0), 1.0), 4)

    def score_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量评分.

        Args:
            documents: 文档信息列表，每项含 doc_id、citation_count、age_days 等.

        Returns:
            排序后的评分列表（从高到低），每项含 score 字段.
        """
        scored: list[dict[str, Any]] = []
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id", ""))
            score = self.score_document(
                doc_id=doc_id,
                citation_count=doc.get("citation_count", 0),
                age_days=doc.get("age_days", 0),
            )
            scored.append({**doc, "quality_score": score})

        scored.sort(key=lambda d: d["quality_score"], reverse=True)
        return scored

    def get_top_documents(
        self,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """获取质量最高的 top_k 文档.

        Args:
            documents: 文档列表.
            top_k: 返回数量.

        Returns:
            top_k 高质量文档列表.
        """
        scored = self.score_batch(documents)
        return scored[:top_k]

    def get_low_quality_documents(
        self,
        documents: list[dict[str, Any]],
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """获取低于质量阈值的文档（用于人工审核或淘汰）.

        Args:
            documents: 文档列表.
            threshold: 质量阈值.

        Returns:
            低质量文档列表.
        """
        scored = self.score_batch(documents)
        return [d for d in scored if d["quality_score"] < threshold]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_usefulness_for_doc(self, doc_id: str) -> float:
        """从反馈记录中获取指定文档的有用率.

        优先使用按文档粒度的反馈统计；若文档尚无反馈记录，则回退到全局统计。
        """
        try:
            doc_stats = self._feedback.get_stats_for_doc(doc_id)
            if doc_stats["total_feedback"] > 0:
                return doc_stats["usefulness_rate"]
            # 文档无反馈记录时回退到全局有用率
            global_stats = self._feedback.get_stats()
            return global_stats.get("usefulness_rate", 0.5)
        except Exception:
            return 0.5
