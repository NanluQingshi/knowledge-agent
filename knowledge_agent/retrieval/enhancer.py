"""搜索增强模块 — Query Rewriting、HyDE 检索、多查询融合."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from knowledge_agent.config import settings

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a search query optimization expert. Given the original question, 
generate {num_variations} different versions that could help retrieve better results.

Rules:
1. Keep the core meaning of the question
2. Use different wording and phrasing
3. Add relevant keywords that might appear in relevant documents
4. Make some versions more specific, others more general
5. Return one query per line, no numbering, no extra text

Original question: {question}"""

_HYDE_PROMPT = """You are a domain expert. Given a question, write a short paragraph that 
answers it. This paragraph will be used as a search query to find similar documents, 
so write it in a factual, informative style as if it were a real document.

Question: {question}

Write a paragraph that answers this question:"""


class QueryRewriter:
    """查询改写器 — 使用 LLM 生成多个查询变体提升检索召回.

    将用户问题改写为多个不同角度的版本，分别检索后融合结果。
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.llm_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def rewrite(self, question: str, num_variations: int = 3) -> list[str]:
        """生成多个查询变体.

        Args:
            question: 用户原始问题.
            num_variations: 生成的变体数量.

        Returns:
            查询变体列表（含原始问题）.
        """
        if not question or not question.strip():
            return []

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _REWRITE_PROMPT.format(
                        question=question, num_variations=num_variations,
                    )},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            variations = [q.strip() for q in content.split("\n") if q.strip()]
        except Exception as exc:
            logger.warning("Query rewriting failed: %s", exc)
            variations = []

        # 去重 + 去空 + 合并原始问题
        seen: set[str] = set()
        result = [question]
        seen.add(question.lower().strip())

        for v in variations:
            key = v.lower().strip()
            if key and key not in seen and len(v) > 5:
                seen.add(key)
                result.append(v)

        return result[: num_variations + 1]


class HyDEGenerator:
    """HyDE (Hypothetical Document Embeddings) 生成器.

    先让 LLM 生成一个假设的"理想文档"（回答问题的段落），
    然后用这个文档的向量进行检索，提升语义匹配质量。
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.llm_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def generate(self, question: str) -> str:
        """生成假设文档.

        Args:
            question: 用户问题.

        Returns:
            假设文档文本；API 失败时返回原始问题.
        """
        if not question or not question.strip():
            return ""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYDE_PROMPT.format(question=question)},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            hypo_doc = response.choices[0].message.content or ""
            return hypo_doc.strip() if hypo_doc.strip() else question
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return question


class MultiQueryFusion:
    """多查询检索融合器.

    将查询改写 + HyDE 生成的多个检索结果使用 RRF 算法融合。
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        hyde: HyDEGenerator | None = None,
    ) -> None:
        self._rewriter = rewriter or QueryRewriter()
        self._hyde = hyde or HyDEGenerator()

    def expand_queries(self, question: str) -> list[str]:
        """扩展查询：生成改写 + HyDE 文档.

        Args:
            question: 用户原始问题.

        Returns:
            扩展后的查询列表.
        """
        queries = [question]

        # 查询改写
        try:
            variations = self._rewriter.rewrite(question, num_variations=3)
            queries.extend(v for v in variations if v != question)
        except Exception:
            pass

        # HyDE 生成
        try:
            hypo_doc = self._hyde.generate(question)
            if hypo_doc and hypo_doc != question:
                queries.append(hypo_doc)
        except Exception:
            pass

        return queries

    @staticmethod
    def fuse_results(
        all_results: list[list[dict[str, Any]]],
        top_k: int = 5,
        rrf_k: float = 60.0,
    ) -> list[dict[str, Any]]:
        """RRF 融合多路检索结果.

        Args:
            all_results: 多路检索结果列表，每路为 [{"id": ..., "text": ..., ...}, ...].
            top_k: 最终返回结果数量.
            rrf_k: RRF 常数，默认 60.

        Returns:
            融合后的结果列表，按 RRF 得分降序.
        """
        merged: dict[str, dict[str, Any]] = {}

        for results in all_results:
            for rank, result in enumerate(results, start=1):
                doc_id = result["id"]
                if doc_id in merged:
                    merged[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
                    merged[doc_id]["sources"].append(result.get("source", "fusion"))
                else:
                    merged[doc_id] = {
                        "id": doc_id,
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                        "rrf_score": 1.0 / (rrf_k + rank),
                        "sources": [result.get("source", "fusion")],
                    }

        scored = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        return scored[:top_k]