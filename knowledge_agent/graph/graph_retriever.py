"""图检索模块 — 基于知识图谱的 Local / Global GraphRAG 检索."""

from __future__ import annotations

from collections import deque
from typing import Any

from knowledge_agent.graph.community_detector import CommunityDetector
from knowledge_agent.graph.graph_store import GraphStore


class GraphRetriever:
    """基于知识图谱的检索器.

    提供三种检索模式：
        - local_search：围绕单个实体的邻域检索（Local GraphRAG）
        - global_search：基于社区摘要的全局检索（Global GraphRAG）
        - traverse：按关系类型的有向路径遍历
    """

    def __init__(self, graph_store: GraphStore) -> None:
        """初始化 GraphRetriever.

        Args:
            graph_store: 已填充数据的 GraphStore 实例.
        """
        self._graph_store = graph_store

    def local_search(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """执行局部检索：围绕指定实体的邻域检索.

        实现 GraphRAG 的 "Local Search" 模式 — 通过实体名称找到对应节点，
        并获取其邻域子图信息。

        Args:
            entity_name: 实体名称（支持不区分大小写的子串匹配）.
            depth: BFS 邻域深度，默认 2.

        Returns:
            检索结果字典，包含：
                - entity: 匹配到的实体信息
                - neighbors: 邻居实体列表，每项含 id、name、type、relation
                - subgraph_size: 子图大小（1 个中心实体 + 邻居数量）
        """
        # 尝试精确 / 子串匹配实体名称
        matched = self._graph_store.search_entities(entity_name)
        if not matched:
            return {
                "entity": None,
                "neighbors": [],
                "subgraph_size": 0,
            }

        # 取第一个匹配的实体
        entity = matched[0]
        entity_id = entity["id"]

        neighbors = self._graph_store.get_neighbors(entity_id, depth=depth)
        subgraph_size = 1 + len(neighbors)

        return {
            "entity": entity,
            "neighbors": neighbors,
            "subgraph_size": subgraph_size,
        }

    def global_search(
        self,
        community_detector: CommunityDetector,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """执行全局检索：基于社区摘要的检索.

        实现 GraphRAG 的 "Global Search" 模式 — 先对整个图执行社区检测，
        然后通过查询关键词与社区摘要的匹配度排序，返回最相关的若干个社区。

        Args:
            community_detector: CommunityDetector 实例.
            query: 用户查询字符串.
            top_k: 返回的最相关社区数量，默认 3.

        Returns:
            按相关性排序的社区列表，每个元素包含：
                - community_id: 社区 ID
                - entities: 社区内实体列表
                - summary: 社区摘要
                - size: 社区大小
                - relevance_score: 与查询的相关性得分
        """
        if not query or not query.strip():
            return []

        try:
            communities = community_detector.detect(self._graph_store)
        except ValueError:
            return []

        # 确保社区摘要已生成
        communities = community_detector.generate_summaries(self._graph_store, communities)

        # 将查询分词用于关键词匹配
        query_lower = query.strip().lower()
        query_tokens = set(query_lower.split())

        # 计算每个社区与查询的相关性得分
        scored_communities: list[tuple[float, dict[str, Any]]] = []

        graph = self._graph_store.graph
        for cid, info in communities.items():
            score = _compute_relevance(
                query_tokens=query_tokens,
                query_lower=query_lower,
                summary=info.get("summary", ""),
                entity_ids=info["entity_ids"],
                graph=graph,
            )
            scored_communities.append((score, {
                "community_id": cid,
                "entities": [
                    {
                        "id": eid,
                        "name": graph.nodes[eid].get("name", eid),
                        "type": graph.nodes[eid].get("type", "unknown"),
                    }
                    for eid in info["entity_ids"]
                ],
                "summary": info.get("summary", ""),
                "size": info["size"],
                "relevance_score": round(score, 4),
            }))

        # 按得分降序排序，取 top_k
        scored_communities.sort(key=lambda x: x[0], reverse=True)
        return [info for _, info in scored_communities[:top_k]]

    def traverse(
        self,
        entity_id: str,
        relation: str | None = None,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """从指定实体出发，按关系类型进行有向路径遍历（BFS）.

        Args:
            entity_id: 起始实体 ID.
            relation: 可选的关系谓词过滤；为 None 时遍历所有关系.
            max_depth: 最大遍历深度，默认 3.

        Returns:
            路径步骤列表，每个元素包含：
                - entity_id: 当前实体 ID
                - entity_name: 当前实体名称
                - entity_type: 当前实体类型
                - relation: 到达当前实体的关系（起点无此字段）
                - depth: 当前深度
        """
        if entity_id not in self._graph_store.graph:
            return []

        graph = self._graph_store.graph
        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque()
        path: list[dict[str, Any]] = []

        # 起点
        start_data = graph.nodes[entity_id]
        path.append({
            "entity_id": entity_id,
            "entity_name": start_data.get("name", entity_id),
            "entity_type": start_data.get("type", "unknown"),
            "relation": None,
            "depth": 0,
        })

        for successor in graph.successors(entity_id):
            if successor not in visited:
                edge_data = graph.get_edge_data(entity_id, successor)
                pred = edge_data.get("predicate", "") if edge_data else ""
                if relation is None or pred == relation:
                    visited.add(successor)
                    node_data = graph.nodes[successor]
                    path.append({
                        "entity_id": successor,
                        "entity_name": node_data.get("name", successor),
                        "entity_type": node_data.get("type", "unknown"),
                        "relation": pred,
                        "depth": 1,
                    })
                    queue.append((successor, 1))

        # BFS 扩展
        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth >= max_depth:
                continue

            for successor in graph.successors(current_node):
                if successor not in visited:
                    edge_data = graph.get_edge_data(current_node, successor)
                    pred = edge_data.get("predicate", "") if edge_data else ""
                    if relation is None or pred == relation:
                        visited.add(successor)
                        node_data = graph.nodes[successor]
                        path.append({
                            "entity_id": successor,
                            "entity_name": node_data.get("name", successor),
                            "entity_type": node_data.get("type", "unknown"),
                            "relation": pred,
                            "depth": current_depth + 1,
                        })
                        queue.append((successor, current_depth + 1))

        return path


def _compute_relevance(
    query_tokens: set[str],
    query_lower: str,
    summary: str,
    entity_ids: list[str],
    graph: Any,
) -> float:
    """计算一个社区与查询的相关性得分.

    得分由两部分组成：
        1. 摘要文本的关键词匹配占比
        2. 实体名称的关键词匹配占比

    Args:
        query_tokens: 查询的分词集合.
        query_lower: 查询原文小写.
        summary: 社区摘要文本.
        entity_ids: 社区包含的实体 ID 列表.
        graph: NetworkX 图实例.

    Returns:
        综合相关性得分（0 ~ 2.0）.
    """
    score = 0.0

    # 摘要匹配
    if summary:
        summary_lower = summary.lower()
        summary_matches = sum(1 for token in query_tokens if token in summary_lower)
        if query_tokens and summary_matches > 0:
            score += summary_matches / len(query_tokens)

    # 实体名称匹配
    entity_name_matches = 0
    for eid in entity_ids:
        name = str(graph.nodes[eid].get("name", ""))
        if query_lower in name.lower():
            entity_name_matches += 1
    if entity_ids:
        score += entity_name_matches / len(entity_ids)

    return score
