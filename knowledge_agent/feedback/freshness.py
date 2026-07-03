"""知识新鲜度管理 — 时间衰减与动态权重调整."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from knowledge_agent.storage.doc_store import DocStore


class FreshnessManager:
    """知识新鲜度管理器.

    基于时间和引用频率动态调整知识条目的权重。
    旧知识和低频引用的知识权重逐渐降低。
    """

    DEFAULT_HALF_LIFE_DAYS = 180  # 默认半衰期：180 天

    def __init__(self, doc_store: DocStore | None = None) -> None:
        self._doc_store = doc_store or DocStore()

    # ------------------------------------------------------------------
    # 新鲜度计算
    # ------------------------------------------------------------------

    def calculate(self, ingested_at: str, reference_count: int = 0) -> float:
        """计算单条知识的新鲜度.

        公式: F = (1 + log(1+refs)) * e^(-t * ln(2) / half_life)

        Args:
            ingested_at: ISO 格式的摄入时间.
            reference_count: 引用次数.

        Returns:
            新鲜度评分 (0.0 ~ 1.0+).
        """
        age_days = self._parse_age_days(ingested_at)
        ref_bonus = 1.0 + math.log(1.0 + max(reference_count, 0))
        decay = math.exp(-age_days * math.log(2) / self.DEFAULT_HALF_LIFE_DAYS)
        return round(ref_bonus * decay, 4)

    def calculate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量计算新鲜度.

        Args:
            items: 条目列表，每项需含 ingested_at 和可选的 reference_count.

        Returns:
            带 freshness_score 字段的条目列表（按新鲜度降序）.
        """
        for item in items:
            item["freshness_score"] = self.calculate(
                ingested_at=item.get("ingested_at", ""),
                reference_count=item.get("reference_count", item.get("chunk_count", 0)),
            )

        items.sort(key=lambda i: i.get("freshness_score", 0), reverse=True)
        return items

    # ------------------------------------------------------------------
    # 文档级管理
    # ------------------------------------------------------------------

    def get_all_with_freshness(self) -> list[dict[str, Any]]:
        """获取所有文档及其新鲜度评分.

        Returns:
            包含 freshness_score 的文档列表.
        """
        docs = self._doc_store.list_documents()
        return self.calculate_batch(docs)

    def get_stale_documents(
        self,
        min_age_days: int = 365,
        max_references: int = 3,
    ) -> list[dict[str, Any]]:
        """获取陈旧文档 — 老且引用少的条目.

        Args:
            min_age_days: 最小存在天数.
            max_references: 最大引用数阈值.

        Returns:
            陈旧文档列表.
        """
        docs = self.get_all_with_freshness()

        stale: list[dict[str, Any]] = []
        for doc in docs:
            age = self._parse_age_days(doc.get("ingested_at", ""))
            refs = doc.get("reference_count", doc.get("chunk_count", 0))
            if age >= min_age_days and refs <= max_references:
                stale.append(doc)

        return stale

    def get_decay_schedule(
        self,
        items: list[dict[str, Any]],
        days_list: list[int] | None = None,
    ) -> dict[int, list[dict[str, Any]]]:
        """预测知识在未来各时间点的新鲜度衰减趋势.

        Args:
            items: 条目列表.
            days_list: 预测的时间点列表，默认 [30, 90, 180, 365].

        Returns:
            {天数: 带 predicted_score 的条目列表}.
        """
        if days_list is None:
            days_list = [30, 90, 180, 365]

        schedule: dict[int, list[dict[str, Any]]] = {}

        for days in days_list:
            decay = math.exp(-days * math.log(2) / self.DEFAULT_HALF_LIFE_DAYS)
            predicted = []
            for item in items:
                refs = item.get("reference_count", item.get("chunk_count", 1))
                ref_bonus = 1.0 + math.log(1.0 + max(refs, 0))
                predicted.append({**item, "predicted_score": round(ref_bonus * decay, 4)})
            schedule[days] = predicted

        return schedule

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_age_days(ingested_at: str) -> int:
        """解析摄入时间为天龄.

        Args:
            ingested_at: ISO 格式时间字符串.

        Returns:
            距今的天数.
        """
        try:
            ingested = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max((now - ingested).days, 0)
        except (ValueError, TypeError):
            return 365
