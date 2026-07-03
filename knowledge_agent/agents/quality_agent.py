"""质检 / 更新 Agent — 知识过期检测、冲突发现与版本管理."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from knowledge_agent.config import settings
from knowledge_agent.graph.graph_store import GraphStore
from knowledge_agent.storage.doc_store import DocStore

_CONFLICT_PAIRS = {
    ("supports", "opposes"),
    ("creates", "destroys"),
    ("increases", "decreases"),
    ("approves", "rejects"),
    ("depends_on", "conflicts_with"),
    ("leads", "reports_to"),
    ("causes", "prevents"),
}


class QualityAgent:
    """知识质检与维护 Agent.

    负责过期检测、知识冲突发现、知识缺口识别和新鲜度评分。
    """

    def __init__(
        self,
        doc_store: DocStore | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._doc_store = doc_store or DocStore()
        self._graph_store = graph_store or GraphStore()

    # ------------------------------------------------------------------
    # 过期检测
    # ------------------------------------------------------------------

    def check_expired_documents(self, max_age_days: int = 90) -> list[dict[str, Any]]:
        """检测超过指定天数的过期文档.

        Args:
            max_age_days: 文档保留最大天数，默认 90 天.

        Returns:
            过期文档元数据列表.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        all_docs = self._doc_store.list_documents()

        expired: list[dict[str, Any]] = []
        for doc in all_docs:
            ingested_at = doc.get("ingested_at", "")
            if ingested_at and ingested_at < cutoff:
                expired.append(doc)

        return expired

    # ------------------------------------------------------------------
    # 冲突检测
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        new_knowledge: dict[str, Any],
        existing_knowledge: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """检测新旧知识之间的冲突.

        比较新提取的事实与已有知识，发现矛盾关系。

        Args:
            new_knowledge: 新知识，含 triples 列表的字典.
            existing_knowledge: 已有知识列表；若为 None 则从 GraphStore 查询.

        Returns:
            冲突列表，每项含 {entity_1, entity_2, new_claim, existing_claim, severity}.
        """
        new_triples = new_knowledge.get("triples", [])
        if not new_triples:
            return []

        conflicts: list[dict[str, Any]] = []

        for triple in new_triples:
            subj = triple.get("subject", "")
            pred = triple.get("predicate", "")
            obj = triple.get("object", "")

            existing = self._find_existing_relations(subj, obj, existing_knowledge)

            for exist_rel in existing:
                exist_pred = exist_rel.get("predicate", exist_rel.get("relation", ""))
                pair_key = (pred.lower(), exist_pred.lower())
                reverse_key = (exist_pred.lower(), pred.lower())

                if pair_key in _CONFLICT_PAIRS or reverse_key in _CONFLICT_PAIRS:
                    severity = "high"
                elif pred.lower() != exist_pred.lower():
                    severity = "medium"
                else:
                    continue

                conflicts.append(
                    {
                        "entity_1": subj,
                        "entity_2": obj,
                        "new_claim": f"{subj} {pred} {obj}",
                        "existing_claim": f"{subj} {exist_pred} {obj}",
                        "severity": severity,
                    }
                )

        return conflicts

    def _find_existing_relations(
        self,
        subject: str,
        obj: str,
        existing: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """查找两个实体间的已有关系."""
        if existing is not None:
            return [
                r
                for r in existing
                if (r.get("subject", "") == subject and r.get("object", "") == obj)
                or (r.get("subject", "") == obj and r.get("object", "") == subject)
            ]

        relations = self._graph_store.get_all_relations()
        results: list[dict[str, Any]] = []
        for r in relations:
            s = r.get("subject", r.get("source", ""))
            t = r.get("object", r.get("target", ""))
            if (s == subject and t == obj) or (s == obj and t == subject):
                results.append(r)

        return results

    # ------------------------------------------------------------------
    # 知识缺口检测
    # ------------------------------------------------------------------

    def detect_knowledge_gaps(self, graph_store: GraphStore | None = None) -> list[dict[str, Any]]:
        """检测知识图谱中的缺口 — 孤立节点或连接稀少的实体.

        Args:
            graph_store: 要分析的图谱存储；默认使用实例自身持有的实例.

        Returns:
            缺口列表，每项含 entity 信息和建议问题.
        """
        gs = graph_store or self._graph_store
        if gs.node_count == 0:
            return []

        entities = gs.get_all_entities()
        gaps: list[dict[str, Any]] = []

        for entity in entities:
            eid = entity.get("id", "")
            neighbors = gs.get_neighbors(eid, depth=1)
            degree = len(neighbors)

            if degree <= 1:
                name = entity.get("name", eid)
                etype = entity.get("type", "unknown")
                suggestions = [
                    f"What is {name}?",
                    f"How does {name} relate to other concepts?",
                    f"What are the key properties of {name}?",
                ]

                gaps.append(
                    {
                        "entity_id": eid,
                        "entity_name": name,
                        "entity_type": etype,
                        "current_connections": degree,
                        "suggested_questions": suggestions,
                    }
                )

        return gaps

    # ------------------------------------------------------------------
    # 新鲜度评分
    # ------------------------------------------------------------------

    def calculate_freshness_score(self, doc_metadata: dict[str, Any]) -> float:
        """计算文档的新鲜度评分（0.0 ~ 1.0).

        基于指数时间衰减和引用次数加权。

        Args:
            doc_metadata: 包含 ingested_at 和可选 reference_count 的文档元数据.

        Returns:
            新鲜度分数，越新 / 越常引用的文档分数越高.
        """
        ref_count = doc_metadata.get("reference_count", doc_metadata.get("chunk_count", 1))
        ingested_str = doc_metadata.get("ingested_at", "")

        try:
            ingested_dt = datetime.fromisoformat(ingested_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = (now - ingested_dt).days
        except (ValueError, TypeError):
            age_days = 365

        # 指数衰减: ref_count * e^(-age / 180)
        decay = math.exp(-max(age_days, 0) / 180.0)
        score = float(ref_count) * decay

        return round(min(score, 1.0), 4)
