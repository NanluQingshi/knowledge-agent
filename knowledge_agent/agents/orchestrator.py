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

from knowledge_agent.memory.episodic_memory import EpisodicMemory
from knowledge_agent.memory.semantic_memory import SemanticMemory


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

        # 记忆系统
        self._episodic_memory = EpisodicMemory(
            vector_store=self._collection._vector_store,
        )
        self._semantic_memory = SemanticMemory(
            graph_store=self._extraction._graph_store,
        )

        # QAAgent 需要检索器 — 默认按需构建
        self._qa_agent = qa_agent  # None 时延迟构建

        # BM25 检索器缓存 — 增量维护避免每次查询重建
        self._bm25_retriever: BM25Retriever | None = None
        self._last_vector_count: int = 0

        # 监控
        from knowledge_agent.monitoring.metrics import MetricsCollector
        from knowledge_agent.monitoring.tracer import Tracer
        self._metrics = MetricsCollector()
        self._tracer = Tracer()

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

        # Step 3: 记忆 — 将抽取结果存入语义记忆
        if enable_extraction and "extraction" in result.results:
            try:
                ext = result.results["extraction"]
                entities = ext.get("entities_found", 0)
                relations = ext.get("relations_found", 0)
                if entities > 0 or relations > 0:
                    self._episodic_memory.store(
                        content=f"Ingested documents from {path}, extracted {entities} entities and {relations} relations",
                        memory_type="action",
                        metadata={"pipeline_run": True, "path": str(path)},
                    )
                result.steps_completed.append("memory_record")
            except Exception as exc:
                result.errors.append({"step": "memory", "error": str(exc)})

        # Step 4: 质检
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

        # 采集完成后刷新 BM25 索引缓存
        if ingest_result.get("chunks_created", 0) > 0:
            try:
                self._update_bm25_index()
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    def run_query(
        self,
        question: str,
        top_k: int = 5,
        use_graphrag: bool = False,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """执行 RAG 问答.

        Args:
            question: 用户问题.
            top_k: 检索结果数量.
            use_graphrag: 是否同时使用 GraphRAG 检索增强.
            chat_history: 可选的历史对话记录，每项为 {"role": ..., "content": ...}.

        Returns:
            QAAgent.query() 返回的结果字典.
        """
        qa = self._get_qa_agent(use_graphrag=use_graphrag)

        self._tracer.start()
        self._metrics.increment("query.count")
        with self._metrics.timeit("query"):
            result = qa.query(question, top_k=top_k, chat_history=chat_history)

        # 将问答记录存入情景记忆
        try:
            self._episodic_memory.store_conversation(
                user_message=question,
                assistant_response=result.get("answer", ""),
                metadata={
                    "top_k": top_k,
                    "use_graphrag": use_graphrag,
                },
            )
        except Exception:
            pass

        return result

    def run_query_stream(
        self,
        question: str,
        top_k: int = 5,
        use_graphrag: bool = False,
        chat_history: list[dict[str, str]] | None = None,
    ):
        """执行流式 RAG 问答.

        Args:
            question: 用户问题.
            top_k: 检索结果数量.
            use_graphrag: 是否同时使用 GraphRAG 检索增强.
            chat_history: 可选的历史对话记录，每项为 {"role": ..., "content": ...}.

        Yields:
            LLM 文本片段.
        """
        qa = self._get_qa_agent(use_graphrag=use_graphrag)

        self._tracer.start()
        self._metrics.increment("query.count")

        # 收集流式输出并存入记忆
        full_answer = ""
        with self._metrics.timeit("query_stream"):
            for chunk in qa.stream_query(question, top_k=top_k, chat_history=chat_history):
                full_answer += chunk
                yield chunk
        for chunk in qa.stream_query(question, top_k=top_k, chat_history=chat_history):
            full_answer += chunk
            yield chunk

        # 流结束后存入情景记忆
        if full_answer:
            try:
                self._episodic_memory.store_conversation(
                    user_message=question,
                    assistant_response=full_answer,
                    metadata={
                        "top_k": top_k,
                        "use_graphrag": use_graphrag,
                        "streamed": True,
                    },
                )
            except Exception:
                pass

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
    # 记忆系统
    # ------------------------------------------------------------------

    def recall_memories(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = "conversation",
    ) -> list[dict[str, Any]]:
        """检索相关情景记忆.

        Args:
            query: 查询文本.
            top_k: 返回数量.
            memory_type: 记忆类型过滤 (conversation / action / observation).

        Returns:
            情景记忆列表.
        """
        try:
            return self._episodic_memory.recall(
                query=query,
                top_k=top_k,
                memory_type=memory_type,
            )
        except Exception:
            return []

    def get_memory_stats(self) -> dict[str, int]:
        """获取记忆系统统计.

        Returns:
            包含 episodic_count、semantic_facts 的字典.
        """
        try:
            episodic_count = self._episodic_memory.count()
        except Exception:
            episodic_count = 0
        try:
            semantic_facts = self._semantic_memory.fact_count
        except Exception:
            semantic_facts = 0

        return {
            "episodic_count": episodic_count,
            "semantic_facts": semantic_facts,
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
    # 检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """执行混合检索（向量 + BM25），返回检索结果.

        公开的检索接口，供 EvaluationRunner 等外部模块使用，
        无需访问内部私有属性。

        Args:
            query: 查询文本.
            top_k: 返回结果数量；默认使用 settings.retrieval_top_k.

        Returns:
            检索结果列表，每项含 id、text、metadata、score.
        """
        qa = self._get_qa_agent()
        return qa._retriever.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """删除指定文档及其关联数据.

        从 VectorStore 中删除对应 chunk，从 DocStore 中删除元数据记录，
        并从 GraphStore 中移除与该文档关联的图谱数据。

        Args:
            doc_id: 文档 ID.

        Returns:
            是否成功删除（文档不存在时返回 False）.
        """
        # 1. 从 DocStore 获取文档信息
        doc = self._collection._doc_store.get_document(doc_id)
        if doc is None:
            return False

        chunk_ids_meta = doc.get("metadata", {}).get("chunk_ids", [])
        if chunk_ids_meta:
            try:
                self._collection._vector_store.delete(chunk_ids_meta)
            except Exception:
                pass
        else:
            # 没有记录 chunk_ids 时，通过 metadata 中的 doc_id 前缀删除
            try:
                collection = self._collection._vector_store.collection
                all_data = collection.get(
                    where={"doc_id": doc_id},
                    include=["ids"],
                )
                ids_to_delete = all_data.get("ids", [])
                if ids_to_delete:
                    self._collection._vector_store.delete(ids_to_delete)
            except Exception:
                pass

        # 2. 从 DocStore 删除元数据
        try:
            self._collection._doc_store.delete_document(doc_id)
        except Exception:
            pass

        # 3. 从 GraphStore 移除关联实体
        try:
            entity_id = doc_id.lower().replace("-", "_")
            self._extraction._graph_store.delete_entity(entity_id)
        except Exception:
            pass

        return True

    # ------------------------------------------------------------------
    # 文档版本管理
    # ------------------------------------------------------------------

    def get_document_versions(self, doc_id: str) -> list[dict[str, Any]]:
        """获取文档的版本历史.

        Args:
            doc_id: 文档 ID.

        Returns:
            版本历史列表，从最新到最旧.
        """
        try:
            return self._collection._doc_store.get_document_versions(doc_id)
        except Exception:
            return []

    def rollback_document(self, doc_id: str) -> dict[str, Any] | None:
        """回滚文档到指定版本.

        Args:
            doc_id: 要回滚到的文档版本 ID.

        Returns:
            回滚后的文档元数据.
        """
        return self._collection._doc_store.rollback_document(doc_id)

    # ------------------------------------------------------------------
    # 监控
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        """监控指标收集器."""
        return self._metrics

    def get_monitoring_report(self) -> dict[str, Any]:
        """获取监控报告.

        Returns:
            包含 timings、counters 和汇总信息的字典.
        """
        return self._metrics.get_report()

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

        # 使用缓存的 BM25 检索器，必要时刷新
        bm25 = self._bm25_retriever
        if bm25 is None:
            bm25 = BM25Retriever()
            self._update_bm25_index()
            self._bm25_retriever = bm25
        else:
            # 检查向量库是否有新增数据
            current_count = vector_store.count()
            if current_count > self._last_vector_count:
                self._update_bm25_index()

        # 可选：加载 Cross-Encoder 重排序器
        reranker = None
        try:
            from knowledge_agent.retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        except Exception:
            pass

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25,
            reranker=reranker,
        )
        qa = QAAgent(hybrid_retriever=hybrid)

        self._qa_agent = qa
        return qa

    def _update_bm25_index(self) -> None:
        """刷新 BM25 索引（增量更新）."""
        from knowledge_agent.storage.vector_store import VectorStore

        vector_store = VectorStore()
        current_count = vector_store.count()
        if current_count == 0:
            self._last_vector_count = 0
            return

        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever()

        corpus = vector_store.get_all_documents()
        if corpus:
            self._bm25_retriever.index(corpus)
            self._last_vector_count = current_count
