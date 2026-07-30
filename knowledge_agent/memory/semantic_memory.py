"""语义记忆模块 — 类比概念知识，存储提炼后的事实和规则.

包装知识图谱 GraphStore，提供事实查询、规则推理接口。
"""

from __future__ import annotations

from typing import Any

from knowledge_agent.graph.graph_store import GraphStore


class SemanticMemory:
    """语义记忆 — 提炼后的事实、规则与概念关系.

    基于知识图谱实现：
    - 事实存储与查询
    - 规则推断（简单图遍历推理）
    - 概念关联探索
    """

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        self._graph = graph_store or GraphStore()

    # ------------------------------------------------------------------
    # 事实存储
    # ------------------------------------------------------------------

    def remember_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """存储一个事实三元组.

        Args:
            subject: 主体.
            predicate: 谓词.
            obj: 客体.
            confidence: 置信度 (0.0 ~ 1.0).
            source: 证据来源.
        """
        subj_id = self._to_id(subject)
        obj_id = self._to_id(obj)

        self._graph.add_entity(subj_id, subject, "concept")
        self._graph.add_entity(obj_id, obj, "concept")
        self._graph.add_relation(subj_id, predicate, obj_id, weight=confidence, evidence=source)
        self._graph.save()

    def recall_facts(self, entity_name: str, depth: int = 1) -> list[dict[str, Any]]:
        """查询与指定实体相关的事实.

        Args:
            entity_name: 实体名称.
            depth: 邻域搜索深度.

        Returns:
            相关事实列表.
        """
        eid = self._to_id(entity_name)
        entity = self._graph.get_entity(eid)
        if entity is None:
            results = self._graph.search_entities(entity_name)
            if not results:
                return []
            eid = results[0].get("id", eid)

        neighbors = self._graph.get_neighbors(eid, depth=depth)
        facts: list[dict[str, Any]] = []
        for n in neighbors:
            facts.append(
                {
                    "subject": entity_name,
                    "relation": n.get("relation", "related_to"),
                    "object": n.get("name", str(n.get("id", ""))),
                    "detail": n,
                }
            )
        return facts

    # ------------------------------------------------------------------
    # 概念探索
    # ------------------------------------------------------------------

    def find_connections(
        self,
        entity_a: str,
        entity_b: str,
    ) -> list[dict[str, Any]]:
        """查找两个实体之间的直接关系.

        Args:
            entity_a: 实体 A 名称.
            entity_b: 实体 B 名称.

        Returns:
            关系列表.
        """
        id_a = self._to_id(entity_a)
        id_b = self._to_id(entity_b)
        return self._graph.get_relations_between(id_a, id_b)

    def search_concepts(self, query: str) -> list[dict[str, Any]]:
        """按名称搜索概念实体.

        Args:
            query: 搜索关键词.

        Returns:
            匹配的实体列表.
        """
        return self._graph.search_entities(query)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def fact_count(self) -> int:
        """事实总数（边数）."""
        return self._graph.edge_count

    @property
    def concept_count(self) -> int:
        """概念总数（节点数）."""
        return self._graph.node_count

    def get_all_facts(self) -> list[dict[str, Any]]:
        """获取所有事实关系.

        Returns:
            关系列表.
        """
        return self._graph.get_all_relations()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_id(name: str) -> str:
        """实体名 → 内部 ID."""
        return name.lower().replace(" ", "_")
