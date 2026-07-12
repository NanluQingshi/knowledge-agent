"""检索质量指标 — MRR、Recall@k、Precision@k、NDCG@k."""

from __future__ import annotations

import math
from typing import Any


class RetrievalMetrics:
    """检索质量指标计算.

    所有指标均在给定查询的检索结果列表上计算：
    - ``MRR``: Mean Reciprocal Rank — 第一个相关结果在排序中的倒数位置
    - ``Recall@k``: 前 k 个结果中召回的相关文档比例
    - ``Precision@k``: 前 k 个结果中相关文档的比例
    - ``NDCG@k``: 归一化折损累计增益 — 考虑排序位置的相关性

    用法示例：:

        metrics = RetrievalMetrics()
        result = metrics.evaluate(
            retrieved_ids=["doc_a", "doc_b", "doc_c"],
            relevant_ids=["doc_b", "doc_d"],
            k=3,
        )
        # result = {"mrr": 0.5, "recall@3": 0.5, "precision@3": 0.333, "ndcg@3": 0.5}
    """

    @staticmethod
    def evaluate(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int | None = None,
    ) -> dict[str, float]:
        """在单次检索上计算所有质量指标.

        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按排序顺序）.
            relevant_ids: 与查询相关的（期望的）文档 ID 列表.
            k: 截断深度；默认使用检索结果的总数.

        Returns:
            包含 mrr、recall@k、precision@k、ndcg@k 的字典.
        """
        if not retrieved_ids or not relevant_ids:
            return {"mrr": 0.0, f"recall@{k or len(retrieved_ids)}": 0.0,
                    f"precision@{k or len(retrieved_ids)}": 0.0,
                    f"ndcg@{k or len(retrieved_ids)}": 0.0}

        if k is None:
            k = len(retrieved_ids)

        k = min(k, len(retrieved_ids))
        top_k_ids = retrieved_ids[:k]
        relevant_set = set(relevant_ids)

        mrr = RetrievalMetrics._mrr(retrieved_ids, relevant_set)
        recall = RetrievalMetrics._recall(top_k_ids, relevant_set, relevant_ids)
        precision = RetrievalMetrics._precision(top_k_ids, relevant_set)
        ndcg = RetrievalMetrics._ndcg(top_k_ids, relevant_set)

        return {
            "mrr": round(mrr, 4),
            f"recall@{k}": round(recall, 4),
            f"precision@{k}": round(precision, 4),
            f"ndcg@{k}": round(ndcg, 4),
        }

    @staticmethod
    def evaluate_batch(
        results: list[dict[str, Any]],
        k: int | None = None,
    ) -> dict[str, float]:
        """在批量检索结果上计算聚合指标.

        Args:
            results: 检索结果列表，每项含 retrieved_ids 和 relevant_ids.
            k: 截断深度.

        Returns:
            各指标在所有查询上的平均值.
        """
        if not results:
            return {"mrr": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0}

        total_mrr = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_ndcg = 0.0
        n = 0

        for r in results:
            retrieved = r.get("retrieved_ids", [])
            relevant = r.get("relevant_ids", [])
            if not retrieved or not relevant:
                continue

            metrics = RetrievalMetrics.evaluate(retrieved, relevant, k=k)
            k_used = k or len(retrieved)
            total_mrr += metrics["mrr"]
            total_recall += metrics.get(f"recall@{k_used}", 0.0)
            total_precision += metrics.get(f"precision@{k_used}", 0.0)
            total_ndcg += metrics.get(f"ndcg@{k_used}", 0.0)
            n += 1

        if n == 0:
            return {"mrr": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0}

        return {
            "mrr": round(total_mrr / n, 4),
            "recall": round(total_recall / n, 4),
            "precision": round(total_precision / n, 4),
            "ndcg": round(total_ndcg / n, 4),
            "num_queries": n,
        }

    # ------------------------------------------------------------------
    # Internal 指标计算
    # ------------------------------------------------------------------

    @staticmethod
    def _mrr(retrieved: list[str], relevant_set: set[str]) -> float:
        """Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _recall(top_k: list[str], relevant_set: set[str], all_relevant: list[str]) -> float:
        """Recall@k."""
        if not all_relevant:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return hits / len(all_relevant)

    @staticmethod
    def _precision(top_k: list[str], relevant_set: set[str]) -> float:
        """Precision@k."""
        if not top_k:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return hits / len(top_k)

    @staticmethod
    def _ndcg(top_k: list[str], relevant_set: set[str]) -> float:
        """NDCG@k.

        使用二元相关性（相关=1，不相关=0），DCG 以 log2(rank+1) 折损。
        """
        dcg = 0.0
        idcg = 0.0

        for i, doc_id in enumerate(top_k, start=1):
            gain = 1.0 if doc_id in relevant_set else 0.0
            dcg += gain / math.log2(i + 1)

        # 理想情况：所有相关文档都在最前面
        num_relevant = min(len(relevant_set), len(top_k))
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0
