"""GraphRAG 检索器 — 知识图谱增强的全局/局部/混合检索."""

from __future__ import annotations

import re
from typing import Any

from knowledge_agent.graph.community_detector import CommunityDetector
from knowledge_agent.graph.graph_retriever import GraphRetriever


# 匹配连续大写开头的单词短语（用于实体提及识别）
_CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+")


class GraphRAGRetriever:
    """GraphRAG 检索器.

    结合知识图谱的局部邻域检索与全局社区检索，支持三种模式：
        - local：从查询中提取实体提及，对每个实体执行 Local Search
        - global：基于社区摘要执行 Global Search
        - hybrid：融合两种模式的结果
    """

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        community_detector: CommunityDetector,
    ) -> None:
        """初始化 GraphRAGRetriever.

        Args:
            graph_retriever: 已初始化的 GraphRetriever 实例.
            community_detector: 已初始化的 CommunityDetector 实例.
        """
        self._graph_retriever = graph_retriever
        self._community_detector = community_detector

    def retrieve(
        self,
        query: str,
        mode: str = "local",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """执行 GraphRAG 检索.

        Args:
            query: 用户查询字符串.
            mode: 检索模式，"local" / "global" / "hybrid"；默认 "local".
            top_k: 每种模式返回的最大结果数，默认 5.

        Returns:
            检索结果列表，每个元素包含：
                - text: 格式化后的上下文文本
                - metadata: 实体或社区的元数据
                - type: 结果类型，"entity" 或 "community"

        Raises:
            ValueError: mode 参数非法时抛出.
        """
        if not query or not query.strip():
            return []

        mode = mode.strip().lower()
        if mode not in ("local", "global", "hybrid"):
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of: 'local', 'global', 'hybrid'"
            )

        if mode == "local":
            return self._retrieve_local(query, top_k)
        elif mode == "global":
            return self._retrieve_global(query, top_k)
        else:  # hybrid
            return self._retrieve_hybrid(query, top_k)

    def entity_mentions_in_query(self, query: str) -> list[str]:
        """从查询中提取潜在实体提及.

        使用启发式规则：匹配连续大写开头的多词短语。
        例如 "Apple Inc."、"New York" 都会被识别。

        Args:
            query: 用户查询字符串.

        Returns:
            提取到的实体提及短语列表.
        """
        if not query or not query.strip():
            return []

        matches = _CAPITALIZED_PHRASE_RE.findall(query.strip())
        # 去重并保持原始顺序
        seen: set[str] = set()
        unique: list[str] = []
        for m in matches:
            m_stripped = m.strip()
            if m_stripped and m_stripped not in seen:
                seen.add(m_stripped)
                unique.append(m_stripped)
        return unique

    # ------------------------------------------------------------------
    # 内部检索方法
    # ------------------------------------------------------------------

    def _retrieve_local(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """局部检索：提取实体提及后对每个实体执行 local_search.

        Args:
            query: 用户查询.
            top_k: 最大结果数.

        Returns:
            实体级检索结果列表.
        """
        mentions = self.entity_mentions_in_query(query)

        # 如果没有识别到实体提及，将整个查询作为实体名称尝试一次检索
        if not mentions:
            result = self._graph_retriever.local_search(query.strip(), depth=2)
            if result["entity"] is not None:
                return [self._format_entity_result(result)]
            return []

        results: list[dict[str, Any]] = []
        seen_entity_ids: set[str] = set()

        for mention in mentions:
            result = self._graph_retriever.local_search(mention, depth=2)
            entity = result.get("entity")
            if entity is None:
                continue

            entity_id = entity.get("id")
            if entity_id and entity_id not in seen_entity_ids:
                seen_entity_ids.add(entity_id)
                results.append(self._format_entity_result(result))

            if len(results) >= top_k:
                break

        return results[:top_k]

    def _retrieve_global(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """全局检索：基于社区摘要的检索.

        Args:
            query: 用户查询.
            top_k: 最大结果数.

        Returns:
            社区级检索结果列表.
        """
        communities = self._graph_retriever.global_search(
            community_detector=self._community_detector,
            query=query,
            top_k=top_k,
        )

        results: list[dict[str, Any]] = []
        for community in communities:
            entity_names = [e.get("name", "") for e in community.get("entities", [])]
            entity_list_str = ", ".join(entity_names[:10])

            summary = community.get("summary", "")
            text_parts = [f"Community #{community['community_id']}"]
            if summary:
                text_parts.append(f"Summary: {summary}")
            text_parts.append(f"Entities ({community['size']}): {entity_list_str}")
            if len(entity_names) > 10:
                text_parts[-1] += f" ... and {len(entity_names) - 10} more"

            results.append({
                "text": "\n".join(text_parts),
                "metadata": {
                    "community_id": community["community_id"],
                    "size": community["size"],
                    "summary": summary,
                    "relevance_score": community.get("relevance_score", 0.0),
                },
                "type": "community",
            })

        return results

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """混合检索：融合局部与全局检索结果.

        Args:
            query: 用户查询.
            top_k: 最大结果数.

        Returns:
            按类型交替排列的混合结果列表.
        """
        local_results = self._retrieve_local(query, top_k)
        global_results = self._retrieve_global(query, top_k)

        # 去重合并：对 community 类型按 community_id 去重，对 entity 按 entity id 去重
        seen_entities: set[str] = set()
        seen_communities: set[int] = set()

        merged: list[dict[str, Any]] = []
        # 交替插入，最大化信息多样性
        max_len = max(len(local_results), len(global_results))
        for i in range(max_len):
            if i < len(local_results):
                item = local_results[i]
                eid = item.get("metadata", {}).get("id")
                if eid not in seen_entities:
                    seen_entities.add(eid)
                    merged.append(item)
            if i < len(global_results):
                item = global_results[i]
                cid = item.get("metadata", {}).get("community_id")
                if cid not in seen_communities:
                    seen_communities.add(cid)
                    merged.append(item)

        return merged[:top_k]

    @staticmethod
    def _format_entity_result(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """将 local_search 的结果格式化为标准输出格式.

        Args:
            result: local_search 返回的原始结果.

        Returns:
            包含 text、metadata、type 的标准结果字典.
        """
        entity = result.get("entity", {})
        neighbors = result.get("neighbors", [])

        entity_name = entity.get("name", "") if entity else ""
        entity_type = entity.get("type", "") if entity else ""

        # 构建可读文本
        text_parts = [f"Entity: {entity_name} (type: {entity_type})"]
        if neighbors:
            neighbor_lines = []
            for nb in neighbors[:10]:
                nb_name = nb.get("name", "")
                nb_type = nb.get("type", "")
                nb_rel = nb.get("relation", "")
                neighbor_lines.append(f"  - {nb_name} ({nb_type}) -- [{nb_rel}]")
            text_parts.append("Neighbors:")
            text_parts.extend(neighbor_lines)
            if len(neighbors) > 10:
                text_parts.append(f"  ... and {len(neighbors) - 10} more neighbors")

        return {
            "text": "\n".join(text_parts),
            "metadata": {
                "id": entity.get("id", "") if entity else "",
                "name": entity_name,
                "type": entity_type,
                "subgraph_size": result.get("subgraph_size", 0),
            },
            "type": "entity",
        }
