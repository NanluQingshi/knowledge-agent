"""BM25 检索器 — 基于词袋模型的稀疏检索."""

from __future__ import annotations

import re
from typing import Any

from knowledge_agent.config import settings

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False
    BM25Okapi = None  # type: ignore[assignment]


def _tokenize(text: str) -> list[str]:
    """将文本切分为 token 列表.

    使用简单规则分词：按非字母数字字符分割，转为小写，过滤空串。
    可根据需求替换为 jieba 等更强大的分词器。

    Args:
        text: 原始文本.

    Returns:
        Token 列表.
    """
    return [t for t in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower()) if t]


class BM25Retriever:
    """基于 BM25 算法的稀疏检索器.

    使用 rank_bm25 库中的 BM25Okapi 实现。
    需要先调用 index() 构建索引后才能进行检索。
    """

    def __init__(self) -> None:
        """初始化 BM25Retriever，索引为空."""
        self._index: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """基于文档语料构建 BM25 索引.

        Args:
            corpus: 文档字典列表，每个字典须包含 "text" 键，可选 "metadata" 键.

        Raises:
            RuntimeError: rank_bm25 库未安装时抛出.
            ValueError: corpus 为空时抛出.
        """
        if not _BM25_AVAILABLE:
            raise RuntimeError(
                "rank_bm25 is not installed. "
                "Install it with: pip install rank_bm25"
            )

        if not corpus:
            raise ValueError("Corpus must not be empty")

        self._corpus = corpus
        self._tokenized_corpus = [_tokenize(doc.get("text", "")) for doc in corpus]
        self._index = BM25Okapi(self._tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对查询文本执行 BM25 检索.

        Args:
            query: 用户查询字符串.
            top_k: 返回结果数量上限；默认使用 settings.retrieval_top_k.

        Returns:
            检索结果列表，每个元素包含 id、text、metadata、score 四个键.

        Raises:
            RuntimeError: 尚未调用 index() 构建索引时抛出.
        """
        if self._index is None:
            raise RuntimeError(
                "BM25 index has not been built yet. Call index(corpus) first."
            )

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else settings.retrieval_top_k
        query_tokens = _tokenize(query)

        scores = self._index.get_scores(query_tokens)

        # 将 (index, score) 配对并按得分降序排序
        scored_indices = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for idx, score in scored_indices[:k]:
            doc = self._corpus[idx]
            results.append(
                {
                    "id": doc.get("id", str(idx)),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(score),
                }
            )

        return results

    @property
    def is_indexed(self) -> bool:
        """索引是否已构建."""
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        """索引中的文档总数."""
        return len(self._corpus)
