"""评估运行器 — 编排检索质量评估与 RAGAS 答案质量评估."""

from __future__ import annotations

from typing import Any

from knowledge_agent.agents.orchestrator import Orchestrator
from knowledge_agent.evaluation.dataset import EvaluationDataset
from knowledge_agent.evaluation.metrics import RetrievalMetrics


class EvaluationRunner:
    """评估运行器.

    支持两种评估模式：
    1. **检索质量评估**: 基于期望文档 ID 计算 MRR/Recall/Precision/NDCG
    2. **答案质量评估**: 基于 RAGAS 计算 Faithfulness/Answer Relevancy 等
       （需安装 ragas 包，否则会以纯文本形式输出供人工评估）
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        dataset: EvaluationDataset | None = None,
    ) -> None:
        """初始化 EvaluationRunner.

        Args:
            orchestrator: 要评估的 Orchestrator 实例；默认新建.
            dataset: 评估数据集；默认新建.
        """
        self._orchestrator = orchestrator or Orchestrator()
        self._dataset = dataset or EvaluationDataset()

    # ------------------------------------------------------------------
    # 检索质量评估
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        top_k: int = 5,
        category: str | None = None,
    ) -> dict[str, Any]:
        """评估检索质量.

        对数据集中的每个 query 执行混合检索，将返回的文档 ID
        与期望文档 ID 对比，计算 MRR/Recall/Precision/NDCG。

        Args:
            top_k: 检索深度.
            category: 可选类别过滤.

        Returns:
            聚合评估报告，包含各项指标和逐条详情.
        """
        items = self._dataset.list_items(category=category)
        if not items:
            return {
                "status": "no_data",
                "message": "评估数据集为空，请先添加评估条目。",
                "metrics": {},
                "details": [],
            }

        detail_results: list[dict[str, Any]] = []

        for item in items:
            query = item["query"]
            expected_ids = item.get("expected_doc_ids", [])
            if not expected_ids:
                continue

            # 执行检索
            retrieved = self._orchestrator._get_qa_agent()._retriever.retrieve(query, top_k=top_k)
            retrieved_ids = [r.get("id", "") for r in retrieved]

            # 计算指标
            metrics = RetrievalMetrics.evaluate(retrieved_ids, expected_ids, k=top_k)

            detail_results.append({
                "query": query,
                "query_id": item["id"],
                "category": item.get("category", "general"),
                "difficulty": item.get("difficulty", "medium"),
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected_ids,
                "metrics": metrics,
            })

        # 聚合
        agg = RetrievalMetrics.evaluate_batch(
            [{"retrieved_ids": d["retrieved_ids"], "relevant_ids": d["expected_ids"]}
             for d in detail_results],
            k=top_k,
        )

        summary_parts = [
            f"Retrieval evaluation on {len(detail_results)} queries (top_k={top_k})",
            f"  MRR         = {agg.get('mrr', 0):.4f}",
            f"  Recall@{top_k}   = {agg.get('recall', 0):.4f}",
            f"  Precision@{top_k} = {agg.get('precision', 0):.4f}",
            f"  NDCG@{top_k}     = {agg.get('ndcg', 0):.4f}",
        ]

        return {
            "status": "ok",
            "num_queries": len(detail_results),
            "top_k": top_k,
            "metrics": agg,
            "summary": "\n".join(summary_parts),
            "details": detail_results,
        }

    # ------------------------------------------------------------------
    # 答案质量评估（RAGAS）
    # ------------------------------------------------------------------

    def evaluate_answer_quality(
        self,
        top_k: int = 5,
        category: str | None = None,
    ) -> dict[str, Any]:
        """评估答案质量.

        对数据集中的每个 query 执行 RAG 问答，尝试使用 RAGAS
        计算 Faithfulness、Answer Relevancy 等指标。
        若 ragas 未安装，则以纯文本输出供人工评估。

        Args:
            top_k: 检索深度.
            category: 可选类别过滤.

        Returns:
            评估报告.
        """
        items = self._dataset.list_items(category=category)
        if not items:
            return {
                "status": "no_data",
                "message": "评估数据集为空，请先添加评估条目。",
                "results": [],
            }

        # 执行问答
        qa_results: list[dict[str, Any]] = []
        for item in items:
            query = item["query"]
            try:
                result = self._orchestrator.run_query(query, top_k=top_k)
                qa_results.append({
                    "query": query,
                    "query_id": item["id"],
                    "answer": result["answer"],
                    "contexts": [s.get("text", "") for s in result.get("sources", [])],
                    "expected_answer": item.get("expected_answer", ""),
                    "category": item.get("category", "general"),
                })
            except Exception as exc:
                qa_results.append({
                    "query": query,
                    "query_id": item["id"],
                    "error": str(exc),
                })

        # 尝试使用 RAGAS
        try:
            return self._evaluate_with_ragas(qa_results)
        except ImportError:
            return self._evaluate_without_ragas(qa_results)

    # ------------------------------------------------------------------
    # Internal: RAGAS 集成
    # ------------------------------------------------------------------

    def _evaluate_with_ragas(
        self, qa_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """使用 RAGAS 评估答案质量."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            # 回退到纯文本输出
            return self._evaluate_without_ragas(qa_results)

        # 过滤掉没有答案或出错的条目
        valid = [r for r in qa_results if "answer" in r and r.get("answer")]
        if not valid:
            return {
                "status": "no_valid_data",
                "message": "没有有效的问答结果可供评估。",
                "results": qa_results,
            }

        try:
            data = {
                "question": [r["query"] for r in valid],
                "answer": [r["answer"] for r in valid],
                "contexts": [r.get("contexts", []) for r in valid],
            }
            # 如果提供了期望答案，添加 ground_truth
            if any(r.get("expected_answer") for r in valid):
                data["ground_truth"] = [
                    r.get("expected_answer", "") for r in valid
                ]

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            scores = evaluate(dataset, metrics=metrics)

            result_dict = {m: round(float(scores[m]), 4) for m in scores}

            summary_parts = [
                f"Answer quality evaluation on {len(valid)} queries (via RAGAS)",
            ]
            for metric, score in result_dict.items():
                summary_parts.append(f"  {metric:25s} = {score:.4f}")

            return {
                "status": "ok",
                "evaluator": "ragas",
                "num_queries": len(valid),
                "scores": result_dict,
                "summary": "\n".join(summary_parts),
                "details": valid,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"RAGAS evaluation failed: {exc}",
                "results": qa_results,
            }

    @staticmethod
    def _evaluate_without_ragas(
        qa_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """无 RAGAS 时的回退输出，供人工评估."""
        lines = [
            "Answer quality evaluation (manual — install `ragas` for auto metrics)",
            f"Total queries: {len(qa_results)}",
            "",
        ]
        for r in qa_results:
            lines.append(f"--- Query: {r['query'][:80]} ---")
            if "error" in r:
                lines.append(f"  ERROR: {r['error']}")
            else:
                lines.append(f"  Answer: {r.get('answer', '')[:200]}")
                if r.get("expected_answer"):
                    lines.append(f"  Expected: {r['expected_answer'][:200]}")

        return {
            "status": "ok",
            "evaluator": "manual",
            "num_queries": len(qa_results),
            "summary": "\n".join(lines),
            "details": qa_results,
        }
