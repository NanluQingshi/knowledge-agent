"""Cross-Encoder 重排序器 — 对检索结果进行精排."""

from __future__ import annotations

from typing import Any

from knowledge_agent.config import settings


class CrossEncoderReranker:
    """Cross-Encoder 重排序器.

    对混合检索（RRF 融合）后的结果使用 Cross-Encoder 模型
    进行二次精排，提升排序质量。

    支持两种模式：
    - ``local``: 使用 sentence-transformers 本地 cross-encoder 模型
      （默认: ``cross-encoder/ms-marco-MiniLM-L-6-v2``）
    - ``api``: 使用 LLM API 逐对打分（更准确但更慢、更贵）

    当模型不可用时自动降级为无操作（返回原排序）。
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        mode: str = "local",
    ) -> None:
        """初始化 CrossEncoderReranker.

        Args:
            model_name: Cross-Encoder 模型名称.
            mode: "local"（本地模型）或 "api"（LLM API 打分）.
        """
        self._model_name = model_name
        self._mode = mode
        self._model = None  # 延迟加载

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对检索结果进行重排序.

        Args:
            query: 原始查询.
            results: 检索结果列表（需包含 text 字段）.
            top_k: 最终返回数量；默认取输入数量.

        Returns:
            重排序后的结果列表，每项增加 ``rerank_score`` 字段.
        """
        if not results:
            return []

        k = top_k if top_k is not None else len(results)
        texts = [r.get("text", "") for r in results]

        if not texts or not any(texts):
            # 无有效文本，原样返回
            for r in results[:k]:
                r["rerank_score"] = r.get("score", 0.0)
            return results[:k]

        if self._mode == "api":
            scores = self._score_via_api(query, texts, k)
        else:
            scores = self._score_via_local(query, texts, k)

        # 合并分数
        reranked = []
        for r, score in zip(results, scores):
            r["rerank_score"] = round(float(score), 6)
            reranked.append(r)

        # 按重排序分数降序排列
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:k]

    def _score_via_local(
        self,
        query: str,
        texts: list[str],
        top_k: int,
    ) -> list[float]:
        """使用本地 Cross-Encoder 模型打分."""
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)

            pairs = [[query, text] for text in texts]
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]
        except Exception:
            # 模型加载或推理失败，降级为等权返回
            return [1.0 / (i + 1) for i in range(len(texts))]

    def _score_via_api(
        self,
        query: str,
        texts: list[str],
        top_k: int,
    ) -> list[float]:
        """使用 LLM API 逐对打分（当前为简单相关性评分）."""
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        except Exception:
            return [1.0 / (i + 1) for i in range(len(texts))]

        if not settings.openai_api_key:
            return [1.0 / (i + 1) for i in range(len(texts))]

        scores: list[float] = []
        for text in texts:
            try:
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance judge. Rate how relevant the following "
                                "document is to the given query on a scale of 0 to 10. "
                                "Return only a number."
                            ),
                        },
                        {"role": "user", "content": f"Query: {query}\n\nDocument: {text[:2000]}"},
                    ],
                    temperature=0.0,
                    max_tokens=4,
                )
                raw = response.choices[0].message.content.strip()
                score = float(raw) / 10.0
                scores.append(min(max(score, 0.0), 1.0))
            except Exception:
                scores.append(0.5)

        return scores
