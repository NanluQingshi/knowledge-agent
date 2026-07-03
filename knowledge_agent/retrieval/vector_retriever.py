"""向量检索器 — 基于 Embedding 的语义检索."""

from __future__ import annotations

from typing import Any

from knowledge_agent.config import settings
from knowledge_agent.embeddings.embedder import Embedder
from knowledge_agent.storage.vector_store import VectorStore


class VectorRetriever:
    """基于向量相似度的检索器.

    将查询文本编码为向量后在 VectorStore 中执行 ANN 搜索。
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        """初始化 VectorRetriever.

        Args:
            vector_store: 已初始化的向量存储实例.
            embedder: 文本向量化器.
        """
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对查询文本执行向量检索.

        Args:
            query: 用户查询字符串.
            top_k: 返回结果数量上限；默认使用 settings.retrieval_top_k.

        Returns:
            VectorStore.search 返回的结果列表，每个元素包含
            id、text、metadata、distance 四个键.
        """
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else settings.retrieval_top_k

        query_embedding = self._embedder.embed_single(query)

        return self._vector_store.search(query_embedding=query_embedding, top_k=k)
