"""混合检索器 — 向量检索与 BM25 的融合检索."""

from __future__ import annotations

from typing import Any

from knowledge_agent.config import settings
from knowledge_agent.retrieval.bm25_retriever import BM25Retriever
from knowledge_agent.retrieval.vector_retriever import VectorRetriever

# Reciprocal Rank Fusion 中使用的固定常量 k
_RRF_K = 60.0


class HybridRetriever:
    """混合检索器，融合向量检索与 BM25 稀疏检索的结果.

    使用 Reciprocal Rank Fusion (RRF) 算法合并两路检索结果，
    权重通过 settings.hybrid_weight_vector 和 settings.hybrid_weight_bm25 配置。
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        """初始化 HybridRetriever.

        Args:
            vector_retriever: 向量检索器实例.
            bm25_retriever: BM25 检索器实例（需已调用 index()）.
        """
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """执行混合检索，返回融合排序后的结果.

        Args:
            query: 用户查询字符串.
            top_k: 返回结果数量上限；默认使用 settings.retrieval_top_k.

        Returns:
            融合检索结果列表，每个元素包含：
                - id: 文档唯一标识
                - text: 文档文本
                - metadata: 文档元数据
                - score: RRF 融合得分
                - sources: 来源标记，取值 "vector"、"bm25" 或 "both"
        """
        k = top_k if top_k is not None else settings.retrieval_top_k

        # 分别执行两路检索
        vector_results = self._vector_retriever.retrieve(query, top_k=k)
        bm25_results = self._bm25_retriever.retrieve(query, top_k=k)

        # 构建字典映射：id -> {结果信息, 向量排名, BM25 排名}
        merged: dict[str, Any] = {}

        for rank_v, result in enumerate(vector_results, start=1):
            doc_id = result["id"]
            merged[doc_id] = {
                "id": doc_id,
                "text": result["text"],
                "metadata": result.get("metadata", {}),
                "rank_v": rank_v,
                "rank_b": None,
                "in_vector": True,
                "in_bm25": False,
            }

        for rank_b, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id in merged:
                merged[doc_id]["rank_b"] = rank_b
                merged[doc_id]["in_bm25"] = True
            else:
                merged[doc_id] = {
                    "id": doc_id,
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "rank_v": None,
                    "rank_b": rank_b,
                    "in_vector": False,
                    "in_bm25": True,
                }

        # 计算 RRF 融合得分并确定来源
        weight_v = settings.hybrid_weight_vector
        weight_b = settings.hybrid_weight_bm25

        scored: list[dict[str, Any]] = []
        for entry in merged.values():
            score_v = weight_v / (_RRF_K + entry["rank_v"]) if entry["rank_v"] is not None else 0.0
            score_b = weight_b / (_RRF_K + entry["rank_b"]) if entry["rank_b"] is not None else 0.0

            sources: list[str] = []
            if entry["in_vector"]:
                sources.append("vector")
            if entry["in_bm25"]:
                sources.append("bm25")

            scored.append(
                {
                    "id": entry["id"],
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": score_v + score_b,
                    "sources": sources if len(sources) < 2 else ["vector", "bm25"],
                }
            )

        # 按 RRF 得分降序排序，取 top_k
        scored.sort(key=lambda item: item["score"], reverse=True)

        return scored[:k]
