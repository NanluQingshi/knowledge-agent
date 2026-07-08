"""编排 Agent — 多 Agent 协作工作流协调器.

支持三种工作流模式：
- ingest: 采集 → 抽取（可选） → 质检（可选）
- query: 检索增强问答
- full_pipeline: 采集 → 抽取 → 质检 → 返回完整报告
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from knowledge_agent.agents.collection_agent import CollectionAgent
from knowledge_agent.agents.extraction_agent import ExtractionAgent
from knowledge_agent.agents.qa_agent import QAAgent
from knowledge_agent.agents.quality_agent import QualityAgent
from knowledge_agent.retrieval.bm25_retriever import BM25Retriever
from knowledge_agent.retrieval.graphrag_retriever import GraphRAGRetriever
from knowledge_agent.retrieval.hybrid_retriever import HybridRetriever
from knowledge_agent.retrieval.vector_retriever import VectorRetriever


class WorkflowStep(str, Enum):
    """工作流步骤."""

    COLLECT = "collect"
    EXTRACT = "extract"
    QUALITY_CHECK = "quality_check"
    QUERY = "query"


@dataclass
class WorkflowResult:
    """工作流执行结果."""

    success: bool = True
    steps_completed: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""


class Orchestrator:
    """多 Agent 编排器.

    协调 CollectionAgent、ExtractionAgent、QAAgent、QualityAgent
    完成完整的知识沉淀工作流。

    用法::

        orchestrator = Orchestrator()
        result = orchestrator.run_full_pipeline("./docs/")
        result = orchestrator.run_query("什么是 GraphRAG？")
    """

    def __init__(
        self,
        collection_agent: CollectionAgent | None = None,
        extraction_agent: ExtractionAgent | None = None,
        qa_agent: QAAgent | None = None,
        quality_agent: QualityAgent | None = None,
    ) -> None:
        self._collection = collection_agent or CollectionAgent()
        self._extraction = extraction_agent or ExtractionAgent()
        self._quality = quality_agent or QualityAgent()

        # QAAgent 需要检索器 — 默认按需构建
        self._qa_agent = qa_agent  # None 时延迟构建

    # ------------------------------------------------------------------
    # 全流程管道
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        path: str | Path,
        *,
        enable_extraction: bool = True,
        enable_quality_check: bool = True,
        quality_max_age_days: int = 90,
    ) -> WorkflowResult:
        """执行完整知识沉淀管道：采集 → 抽取 → 质检.

        Args:
            path: 文档路径（文件或目录）.
            enable_extraction: 是否执行知识抽取步骤.
            enable_quality_check: 是否执行质检步骤.
            quality_max_age_days: 过期检测的最大天数.

        Returns:
            WorkflowResult 包含各步骤的结果摘要.
        """
        result = WorkflowResult()

        # Step 1: 采集
        try:
            ingest_result = self._collection.ingest_path(path)
            result.results["ingest"] = ingest_result
            result.steps_completed.append(WorkflowStep.COLLECT.value)

            if ingest_result.get("errors"):
                result.errors.extend(ingest_result["errors"])
        except Exception as exc:
            result.errors.append({"step": "collect", "error": str(exc)})
            result.success = False
            result.summary = f"Ingest failed: {exc}"
            return result

        # Step 2: 抽取
        if enable_extraction:
            try:
                extract_result = self._extraction.build_graph_from_store(
                    self._collection._vector_store,
                )
                result.results["extraction"] = extract_result
                result.steps_completed.append(WorkflowStep.EXTRACT.value)
            except Exception as exc:
                result.errors.append({"step": "extract", "error": str(exc)})

        # Step 3: 质检
        if enable_quality_check:
            try:
                expired = self._quality.check_expired_documents(max_age_days=quality_max_age_days)
                gaps = self._quality.detect_knowledge_gaps()
                result.results["quality"] = {
                    "expired_documents": len(expired),
                    "expired_list": expired,
                    "knowledge_gaps": len(gaps),
                    "gap_entities": [g["entity_name"] for g in gaps],
                }
                result.steps_completed.append(WorkflowStep.QUALITY_CHECK.value)
            except Exception as exc:
                result.errors.append({"step": "quality_check", "error": str(exc)})

        # 生成摘要
        parts = [f"Pipeline completed with {len(result.steps_completed)} steps"]
        ingest = result.results.get("ingest", {})
        parts.append(
            f"  - Ingested {ingest.get('documents_loaded', 0)} documents, "
            f"{ingest.get('chunks_created', 0)} chunks"
        )
        if "extraction" in result.results:
            ext = result.results["extraction"]
            parts.append(
                f"  - Extracted {ext.get('entities_found', 0)} entities, "
                f"{ext.get('relations_found', 0)} relations"
            )
        if "quality" in result.results:
            q = result.results["quality"]
            parts.append(
                f"  - Quality: {q.get('expired_documents', 0)} expired, "
                f"{q.get('knowledge_gaps', 0)} knowledge gaps"
            )
        if result.errors:
            parts.append(f"  - {len(result.errors)} error(s) encountered")

        result.summary = "\n".join(parts)
        return result

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    def run_query(
        self,
        question: str,
        top_k: int = 5,
        use_graphrag: bool = False,
    ) -> dict[str, Any]:
        """执行 RAG 问答.

        Args:
            question: 用户问题.
            top_k: 检索结果数量.
            use_graphrag: 是否同时使用 GraphRAG 检索增强.

        Returns:
            QAAgent.query() 返回的结果字典.
        """
        qa = self._get_qa_agent(use_graphrag=use_graphrag)
        return qa.query(question, top_k=top_k)

    def run_query_stream(
        self,
        question: str,
        top_k: int = 5,
        use_graphrag: bool = False,
    ):
        """执行流式 RAG 问答.

        Args:
            question: 用户问题.
            top_k: 检索结果数量.
            use_graphrag: 是否同时使用 GraphRAG 检索增强.

        Yields:
            LLM 文本片段.
        """
        qa = self._get_qa_agent(use_graphrag=use_graphrag)
        yield from qa.stream_query(question, top_k=top_k)

    # ------------------------------------------------------------------
    # 统计分析
    # ------------------------------------------------------------------

    def get_system_report(self) -> dict[str, Any]:
        """生成系统整体状态报告.

        Returns:
            包含向量库、图谱、文档库统计的字典.
        """
        stats = self._collection.get_stats()

        graph_stats = {
            "nodes": self._extraction._graph_store.node_count,
            "edges": self._extraction._graph_store.edge_count,
        }

        quality_report = {
            "expired_documents": len(self._quality.check_expired_documents()),
            "knowledge_gaps": len(self._quality.detect_knowledge_gaps()),
        }

        return {
            "storage": stats,
            "graph": graph_stats,
            "quality": quality_report,
        }

    # ------------------------------------------------------------------
    # 反馈与进化
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        query_text: str,
        answer_text: str = "",
        rating: str = "useful",
        comment: str = "",
        source_doc_ids: list[str] | None = None,
    ) -> str:
        """记录用户对回答的反馈.

        Args:
            query_text: 用户问题.
            answer_text: 模型回答.
            rating: useful / useless / partial.
            comment: 用户评注.
            source_doc_ids: 引用的来源文档.

        Returns:
            反馈 ID.
        """
        from knowledge_agent.feedback.collector import FeedbackCollector

        collector = FeedbackCollector()
        return collector.record(
            query_text=query_text,
            answer_text=answer_text,
            rating=rating,
            comment=comment,
            source_doc_ids=source_doc_ids,
        )

    def get_feedback_stats(self) -> dict:
        """获取用户反馈统计."""
        from knowledge_agent.feedback.collector import FeedbackCollector

        collector = FeedbackCollector()
        return collector.get_stats()

    def get_knowledge_health(self) -> dict:
        """全面评估知识库健康状况.

        包含：新鲜度评分、质量评分、缺口数量、过期文档数。
        """
        from knowledge_agent.feedback.freshness import FreshnessManager

        freshness = FreshnessManager()
        docs_with_freshness = freshness.get_all_with_freshness()
        stale = freshness.get_stale_documents()

        return {
            "total_documents": len(docs_with_freshness),
            "stale_documents": len(stale),
            "expired_documents": len(self._quality.check_expired_documents()),
            "knowledge_gaps": len(self._quality.detect_knowledge_gaps()),
            "freshness_distribution": {
                "high (>0.7)": sum(1 for d in docs_with_freshness if d.get("freshness_score", 0) > 0.7),
                "medium (0.3-0.7)": sum(
                    1 for d in docs_with_freshness if 0.3 <= d.get("freshness_score", 0) <= 0.7
                ),
                "low (<0.3)": sum(1 for d in docs_with_freshness if d.get("freshness_score", 0) < 0.3),
            },
        }

    def run_maintenance(self) -> dict:
        """执行自动维护：识别并报告需要关注的知识条目.

        Returns:
            维护报告，包含陈旧文档、低质量条目、知识缺口的列表.
        """
        from knowledge_agent.feedback.freshness import FreshnessManager

        freshness = FreshnessManager()
        stale = freshness.get_stale_documents(min_age_days=180, max_references=2)

        return {
            "stale_documents": [
                {"filename": d.get("filename", ""), "age_days": d.get("age_days", 0)}
                for d in stale
            ],
            "knowledge_gaps": [
                {"entity": g.get("entity_name", ""), "connections": g.get("current_connections", 0)}
                for g in self._quality.detect_knowledge_gaps()
            ],
            "expired": [
                {"filename": d.get("filename", ""), "ingested_at": d.get("ingested_at", "")}
                for d in self._quality.check_expired_documents()
            ],
            "recommendation": (
                f"Found {len(stale)} stale documents, "
                f"{len(self._quality.detect_knowledge_gaps())} knowledge gaps, "
                f"{len(self._quality.check_expired_documents())} expired documents."
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_qa_agent(self, use_graphrag: bool = False) -> QAAgent:
        """获取或构建 QAAgent 实例."""
        if self._qa_agent is not None and not use_graphrag:
            return self._qa_agent

        from knowledge_agent.embeddings.embedder import Embedder
        from knowledge_agent.graph.community_detector import CommunityDetector
        from knowledge_agent.graph.graph_retriever import GraphRetriever
        from knowledge_agent.storage.vector_store import VectorStore

        vector_store = VectorStore()
        embedder = Embedder()
        vector_retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        bm25_retriever = BM25Retriever()

        # 尝试为 BM25 构建索引
        if vector_store.count() > 0:
            try:
                all_results = vector_store.get_all_documents()
                if all_results:
                    bm25_retriever.index(all_results)
            except Exception:
                pass

        hybrid = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
        qa = QAAgent(hybrid_retriever=hybrid)

        # 如果需要 GraphRAG 增强，可以在此集成
        # (暂不改变默认检索器，后续 Phase 中深化集成)

        self._qa_agent = qa
        return qa
