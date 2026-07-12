"""图存储模块 — 基于 NetworkX 的知识图谱存储."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from knowledge_agent.config import settings


class GraphStore:
    """基于 NetworkX DiGraph 的知识图谱存储.

    提供实体/关系的增删查、邻域搜索、子串匹配以及 JSON 持久化能力。
    """

    def __init__(self, path: str | None = None) -> None:
        """初始化 GraphStore.

        如果指定路径（或 settings.graph_db_path）下存在 JSON 文件，则从该文件恢复图；
        否则创建空的有向图。

        Args:
            path: 图数据文件的路径；默认为 None，此时使用 settings.graph_db_path.
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._path = Path(path or settings.graph_db_path)

        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # 内部序列化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """从 JSON 文件加载图数据."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(data, directed=True, multigraph=False)
        except (json.JSONDecodeError, KeyError, ValueError):
            self._graph = nx.DiGraph()

    def save(self, path: str | None = None) -> None:
        """将图数据持久化到 JSON 文件.

        Args:
            path: 保存路径；默认为初始化时使用的路径.
        """
        save_path = Path(path or self._path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 实体操作
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """添加一个实体节点.

        Args:
            entity_id: 实体唯一标识.
            name: 实体名称.
            entity_type: 实体类型（如 person、organization 等）.
            properties: 附加属性字典，须可 JSON 序列化.
        """
        self._graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            properties=properties or {},
        )

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """根据 ID 获取实体信息.

        Args:
            entity_id: 实体唯一标识.

        Returns:
            包含 id 及所有节点属性的字典；若不存在则返回 None.
        """
        if entity_id not in self._graph:
            return None
        attrs = dict(self._graph.nodes[entity_id])
        attrs["id"] = entity_id
        return attrs

    def get_all_entities(self) -> list[dict[str, Any]]:
        """返回所有实体的列表.

        Returns:
            每个元素为包含 id、name、type、properties 的字典.
        """
        return [
            {
                "id": node,
                "name": data.get("name", ""),
                "type": data.get("type", ""),
                "properties": data.get("properties", {}),
            }
            for node, data in self._graph.nodes(data=True)
        ]

    def search_entities(self, query: str) -> list[dict[str, Any]]:
        """对实体名称执行不区分大小写的子串匹配.

        Args:
            query: 搜索关键字.

        Returns:
            匹配的实体列表.
        """
        if not query or not query.strip():
            return []

        q_lower = query.strip().lower()
        results: list[dict[str, Any]] = []
        for node, data in self._graph.nodes(data=True):
            name = (data.get("name", "") or "")
            if q_lower in name.lower():
                results.append({
                    "id": node,
                    "name": name,
                    "type": data.get("type", ""),
                    "properties": data.get("properties", {}),
                })
        return results

    # ------------------------------------------------------------------
    # 关系操作
    # ------------------------------------------------------------------

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        weight: float = 1.0,
        evidence: str = "",
    ) -> None:
        """添加一条有向关系.

        如果 subject 或 object 节点尚不存在，则自动创建对应的实体节点。

        Args:
            subject_id: 主体实体 ID.
            predicate: 关系谓词（如 "works_at", "located_in"）.
            object_id: 客体实体 ID.
            weight: 关系权重，默认 1.0.
            evidence: 关系证据文本.
        """
        if subject_id not in self._graph:
            self._graph.add_node(subject_id, name=subject_id, type="unknown", properties={})
        if object_id not in self._graph:
            self._graph.add_node(object_id, name=object_id, type="unknown", properties={})

        self._graph.add_edge(
            subject_id,
            object_id,
            predicate=predicate,
            weight=weight,
            evidence=evidence,
        )

    def get_relations_between(
        self,
        entity_id_1: str,
        entity_id_2: str,
    ) -> list[dict[str, Any]]:
        """返回两个实体之间的所有关系（双向）.

        Args:
            entity_id_1: 第一个实体 ID.
            entity_id_2: 第二个实体 ID.

        Returns:
            关系字典列表，每个元素包含 subject_id、predicate、object_id、
            weight、evidence、direction 字段.
        """
        results: list[dict[str, Any]] = []

        # entity_id_1 -> entity_id_2
        if self._graph.has_edge(entity_id_1, entity_id_2):
            data = self._graph.get_edge_data(entity_id_1, entity_id_2)
            results.append({
                "subject_id": entity_id_1,
                "predicate": data.get("predicate", ""),
                "object_id": entity_id_2,
                "weight": data.get("weight", 1.0),
                "evidence": data.get("evidence", ""),
                "direction": "outgoing",
            })

        # entity_id_2 -> entity_id_1
        if self._graph.has_edge(entity_id_2, entity_id_1):
            data = self._graph.get_edge_data(entity_id_2, entity_id_1)
            results.append({
                "subject_id": entity_id_2,
                "predicate": data.get("predicate", ""),
                "object_id": entity_id_1,
                "weight": data.get("weight", 1.0),
                "evidence": data.get("evidence", ""),
                "direction": "incoming",
            })

        return results

    def get_all_relations(self) -> list[dict[str, Any]]:
        """返回所有关系的列表.

        Returns:
            每个元素包含 subject_id、predicate、object_id、weight、evidence.
        """
        return [
            {
                "subject_id": u,
                "predicate": data.get("predicate", ""),
                "object_id": v,
                "weight": data.get("weight", 1.0),
                "evidence": data.get("evidence", ""),
            }
            for u, v, data in self._graph.edges(data=True)
        ]

    # ------------------------------------------------------------------
    # 邻域 / 遍历
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """获取指定实体在给定深度内的邻居（BFS）.

        BFS 遍历从 entity_id 出发，在 depth 步范围内（包含进/出两个方向）收集邻居实体，
        并记录它们与原点的关系路径。

        Args:
            entity_id: 起始实体 ID.
            depth: BFS 深度，默认 1（仅直接邻居）.

        Returns:
            邻居字典列表，每个元素包含 id、name、type、relation（谓词 + 方向）.
        """
        if entity_id not in self._graph:
            return []

        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque()
        neighbors: list[dict[str, Any]] = []

        # 先加入所有直接邻居到队列
        for neighbor in self._graph.successors(entity_id):
            if neighbor not in visited:
                visited.add(neighbor)
                edge_data = self._graph.get_edge_data(entity_id, neighbor)
                node_data = self._graph.nodes[neighbor]
                neighbors.append({
                    "id": neighbor,
                    "name": node_data.get("name", neighbor),
                    "type": node_data.get("type", ""),
                    "relation": f"{edge_data.get('predicate', '')} (outgoing)"
                    if edge_data
                    else "(outgoing)",
                })
                queue.append((neighbor, 1))

        for predecessor in self._graph.predecessors(entity_id):
            if predecessor not in visited:
                visited.add(predecessor)
                edge_data = self._graph.get_edge_data(predecessor, entity_id)
                node_data = self._graph.nodes[predecessor]
                neighbors.append({
                    "id": predecessor,
                    "name": node_data.get("name", predecessor),
                    "type": node_data.get("type", ""),
                    "relation": f"{edge_data.get('predicate', '')} (incoming)"
                    if edge_data
                    else "(incoming)",
                })
                queue.append((predecessor, 1))

        # BFS 扩展
        if depth > 1:
            while queue:
                current_node, current_depth = queue.popleft()
                if current_depth >= depth:
                    continue

                # 后继
                for successor in self._graph.successors(current_node):
                    if successor not in visited:
                        visited.add(successor)
                        edge_data = self._graph.get_edge_data(current_node, successor)
                        node_data = self._graph.nodes[successor]
                        neighbors.append({
                            "id": successor,
                            "name": node_data.get("name", successor),
                            "type": node_data.get("type", ""),
                            "relation": f"{edge_data.get('predicate', '')} (outgoing)"
                            if edge_data
                            else "(outgoing)",
                        })
                        queue.append((successor, current_depth + 1))

                # 前驱
                for predecessor in self._graph.predecessors(current_node):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        edge_data = self._graph.get_edge_data(predecessor, current_node)
                        node_data = self._graph.nodes[predecessor]
                        neighbors.append({
                            "id": predecessor,
                            "name": node_data.get("name", predecessor),
                            "type": node_data.get("type", ""),
                            "relation": f"{edge_data.get('predicate', '')} (incoming)"
                            if edge_data
                            else "(incoming)",
                        })
                        queue.append((predecessor, current_depth + 1))

        return neighbors

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_entity(self, entity_id: str) -> bool:
        """删除一个实体及其所有关联关系.

        Args:
            entity_id: 实体 ID.

        Returns:
            是否成功删除.
        """
        if entity_id not in self._graph:
            return False
        # 删除以该实体为 subject 或 object 的所有边
        edges_to_remove = [(u, v) for u, v in self._graph.edges() if u == entity_id or v == entity_id]
        self._graph.remove_edges_from(edges_to_remove)
        self._graph.remove_node(entity_id)
        self.save()
        return True

    def delete_relation(self, subject_id: str, predicate: str, object_id: str) -> bool:
        """删除指定的关系边.

        Args:
            subject_id: 主体 ID.
            predicate: 谓词.
            object_id: 客体 ID.

        Returns:
            是否成功删除.
        """
        edges_to_remove = [
            (u, v, k)
            for u, v, k, d in self._graph.edges(data=True, keys=True)
            if u == subject_id and v == object_id and d.get("predicate") == predicate
        ]
        if not edges_to_remove:
            return False
        self._graph.remove_edges_from(edges_to_remove)
        self.save()
        return True

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """图中实体节点数量."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """图中关系边的数量."""
        return self._graph.number_of_edges()

    @property
    def graph(self) -> nx.DiGraph:
        """暴露底层 NetworkX 有向图，供社区检测等模块使用."""
        return self._graph
