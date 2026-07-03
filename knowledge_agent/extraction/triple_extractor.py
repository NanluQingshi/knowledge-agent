"""SPO 三元组提取模块 — 整合实体识别与关系抽取."""

from __future__ import annotations

from typing import Any

from knowledge_agent.chunkers.base import Chunk
from knowledge_agent.extraction.entity_extractor import EntityExtractor
from knowledge_agent.extraction.relation_extractor import RelationExtractor


class TripleExtractor:
    """SPO 三元组提取器.

    整合 :class:`EntityExtractor` 与 :class:`RelationExtractor`，
    输出标准化的 Subject-Predicate-Object 三元组，同时保留完整的
    实体与关系元数据用于后续知识图谱构建。

    Attributes:
        entity_extractor: 实体识别器实例.
        relation_extractor: 关系抽取器实例.
    """

    def __init__(self) -> None:
        """初始化 TripleExtractor，创建实体与关系提取器实例。"""
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

    # ------------------------------------------------------------------
    # 单文本提取
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, Any]:
        """从文本中提取实体、关系和 SPO 三元组。

        Args:
            text: 输入文本。

        Returns:
            包含以下键的字典：

            - **entities** — 实体列表，每项含 ``name``, ``type``, ``description``, ``mentions``
            - **relations** — 关系列表，每项含 ``subject``, ``predicate``, ``object``,
              ``confidence``, ``evidence``
            - **triples** — SPO 三元组列表，每项为 ``{"subject": ..., "predicate": ..., "object": ...}``
        """
        entities = self.entity_extractor.extract(text)
        relations = self.relation_extractor.extract(text, entities)

        triples = [
            {
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],
            }
            for rel in relations
            if rel.get("subject") and rel.get("predicate") and rel.get("object")
        ]

        return {
            "entities": entities,
            "relations": relations,
            "triples": triples,
        }

    # ------------------------------------------------------------------
    # 多块聚合
    # ------------------------------------------------------------------

    def extract_from_chunks(self, chunks: list) -> dict[str, Any]:
        """从多个文档块中提取并聚合知识。

        遍历 ``chunks`` 中的每个 :class:`~knowledge_agent.chunkers.base.Chunk`，
        提取三元组后按实体名称和关系签名合并去重。

        Args:
            chunks: :class:`~knowledge_agent.chunkers.base.Chunk` 对象列表。

        Returns:
            聚合后的字典，包含 ``entities``, ``relations``, ``triples`` 三个键，
            语义与 :meth:`extract` 一致。
        """
        seen_entities: dict[str, dict[str, Any]] = {}
        seen_relation_sigs: set[tuple[str, str, str]] = set()
        all_triples: list[dict[str, str]] = []

        for chunk in chunks:
            if not isinstance(chunk, Chunk) or not chunk.text.strip():
                continue

            try:
                result = self.extract(chunk.text)
            except RuntimeError:
                # 单块失败不中断整体流程，跳过即可
                continue

            # --- 合并实体（按名称小写去重，保留首次出现的实体） ---
            for entity in result.get("entities", []):
                key = entity.get("name", "").strip().lower()
                if key and key not in seen_entities:
                    seen_entities[key] = entity

            # --- 合并关系（按 subject-predicate-object 签名去重） ---
            for rel in result.get("relations", []):
                sig = (
                    rel.get("subject", "").strip().lower(),
                    rel.get("predicate", "").strip().lower(),
                    rel.get("object", "").strip().lower(),
                )
                if sig not in seen_relation_sigs:
                    seen_relation_sigs.add(sig)
                    all_triples.append({
                        "subject": rel["subject"],
                        "predicate": rel["predicate"],
                        "object": rel["object"],
                    })

        return {
            "entities": list(seen_entities.values()),
            "relations": [
                {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]}
                for t in all_triples
            ],
            "triples": all_triples,
        }
