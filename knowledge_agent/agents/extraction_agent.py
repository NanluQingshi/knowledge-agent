"""抽取 Agent — 负责 NER、关系抽取、知识图谱构建与文本总结."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from knowledge_agent.chunkers.base import Chunk
from knowledge_agent.config import settings
from knowledge_agent.extraction.triple_extractor import TripleExtractor
from knowledge_agent.graph.graph_store import GraphStore
from knowledge_agent.storage.vector_store import VectorStore

_SUMMARIZE_SYSTEM_PROMPT = """你是一个专业的文本总结助手。

要求：
1. 提炼原文的核心要点，保留关键信息（人物、事件、数据、结论等）。
2. 保持客观，不添加原文没有的信息。
3. 语言简洁清晰，逻辑连贯。"""


class ExtractionAgent:
    """知识抽取 Agent.

    负责：
    - 从文本或文档块中提取实体、关系和 SPO 三元组
    - 将抽取的知识写入 GraphStore
    - 对文本进行 LLM 摘要生成
    - 从 VectorStore 中批量构建知识图谱
    """

    def __init__(
        self,
        triple_extractor: TripleExtractor | None = None,
        graph_store: GraphStore | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        """初始化 ExtractionAgent.

        Args:
            triple_extractor: SPO 三元组提取器，默认新建 TripleExtractor().
            graph_store: 知识图谱存储，默认 GraphStore().
            openai_client: OpenAI 客户端实例；若未提供，则从 settings 自动创建.
        """
        self._triple_extractor = triple_extractor or TripleExtractor()
        self._graph_store = graph_store or GraphStore()
        self._client = openai_client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    # ------------------------------------------------------------------
    # 单文本处理
    # ------------------------------------------------------------------

    def process_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理单段文本，提取知识并写入图数据库.

        Args:
            text: 输入文本.
            metadata: 可选元数据（当前保留，暂未使用）.

        Returns:
            统计字典：``entities_found``、``relations_found``、``triples_found``.
        """
        if not text or not text.strip():
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        try:
            result = self._triple_extractor.extract(text)
        except RuntimeError:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        entities = result.get("entities", [])
        relations = result.get("relations", [])
        triples = result.get("triples", [])

        self._populate_graph(entities, relations)

        return {
            "entities_found": len(entities),
            "relations_found": len(relations),
            "triples_found": len(triples),
        }

    # ------------------------------------------------------------------
    # 多块聚合处理
    # ------------------------------------------------------------------

    def process_chunks(self, chunks: list[Chunk]) -> dict[str, Any]:
        """批量处理文档块，聚合知识后写入图数据库.

        Args:
            chunks: Chunk 对象列表.

        Returns:
            聚合后的统计字典.
        """
        if not chunks:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        try:
            result = self._triple_extractor.extract_from_chunks(chunks)
        except RuntimeError:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        self._populate_graph(entities, relations)

        return {
            "entities_found": len(entities),
            "relations_found": len(relations),
            "triples_found": len(relations),  # triples count == relations count after dedup
        }

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------

    def summarize(self, text: str, max_length: int = 200) -> str:
        """对文本进行 LLM 摘要生成.

        Args:
            text: 输入文本.
            max_length: 摘要最大字数，默认 200.

        Returns:
            生成的摘要字符串；输入为空或 API 失败时返回空字符串.
        """
        if not text or not text.strip():
            return ""

        try:
            response = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{_SUMMARIZE_SYSTEM_PROMPT}\n"
                            f"请将以下文本总结为不超过 {max_length} 字的中文摘要。"
                        ),
                    },
                    {"role": "user", "content": text[:8000]},  # 防止超长文本超出 token 限制
                ],
                temperature=0.3,
                max_tokens=max(256, max_length * 2),
            )
        except Exception:
            return ""

        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # 从向量库构建图谱
    # ------------------------------------------------------------------

    def build_graph_from_store(
        self,
        vector_store: VectorStore,
        top_k: int = 100,
    ) -> dict[str, Any]:
        """从向量存储中拉取文档块，批量提取知识并填充图数据库.

        使用 ChromaDB collection 的 get() 接口直接获取文档块，
        提取实体/关系后写入 GraphStore 并持久化.

        Args:
            vector_store: 已初始化的 VectorStore 实例.
            top_k: 最多处理的文档块数量.

        Returns:
            统计字典.
        """
        try:
            collection = vector_store.collection
            all_data = collection.get(
                include=["documents", "metadatas"],
                limit=top_k,
            )
        except Exception:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        doc_texts: list[str] = all_data.get("documents", []) or []
        doc_metadatas: list[dict[str, Any]] = all_data.get("metadatas", []) or []

        if not doc_texts:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        # 填充缺失的 metadata
        while len(doc_metadatas) < len(doc_texts):
            doc_metadatas.append({})

        chunks = [
            Chunk(text=text, metadata=meta or {}, chunk_index=i)
            for i, (text, meta) in enumerate(zip(doc_texts, doc_metadatas))
        ]

        try:
            result = self._triple_extractor.extract_from_chunks(chunks)
        except RuntimeError:
            return {"entities_found": 0, "relations_found": 0, "triples_found": 0}

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        self._populate_graph(entities, relations)

        return {
            "entities_found": len(entities),
            "relations_found": len(relations),
            "triples_found": len(relations),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_graph(
        self,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> None:
        """将实体和关系写入 GraphStore.

        Args:
            entities: 实体列表，每项至少包含 ``name`` 字段.
            relations: 关系列表，每项至少包含 ``subject``、``predicate``、``object`` 字段.
        """
        # 写入实体
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            entity_id = name.lower().replace(" ", "_")
            try:
                self._graph_store.add_entity(
                    entity_id=entity_id,
                    name=name,
                    entity_type=entity.get("type", "unknown"),
                    properties={
                        "description": entity.get("description", ""),
                        "mentions": entity.get("mentions", []),
                    },
                )
            except Exception:
                continue

        # 写入关系
        for rel in relations:
            subj = rel.get("subject", "").strip()
            pred = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            if not subj or not pred or not obj:
                continue

            subj_id = subj.lower().replace(" ", "_")
            obj_id = obj.lower().replace(" ", "_")
            try:
                self._graph_store.add_relation(
                    subject_id=subj_id,
                    predicate=pred,
                    object_id=obj_id,
                    weight=float(rel.get("confidence", 1.0)),
                    evidence=rel.get("evidence", ""),
                )
            except Exception:
                continue

        # 持久化
        try:
            self._graph_store.save()
        except Exception:
            pass
