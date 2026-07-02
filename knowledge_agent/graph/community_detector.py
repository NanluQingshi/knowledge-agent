"""社区检测模块 — 基于 Louvain 算法的知识图谱社区发现."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from knowledge_agent.graph.graph_store import GraphStore

try:
    import community as community_louvain  # type: ignore[import-untyped]

    _LOUVAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LOUVAIN_AVAILABLE = False
    community_louvain = None  # type: ignore[assignment]


def _default_entity_types(graph: nx.DiGraph) -> dict[str, str]:
    """快速获取图中所有节点的类型映射."""
    return {
        node: data.get("type", "unknown")
        for node, data in graph.nodes(data=True)
    }


class CommunityDetector:
    """基于 Louvain 算法的社区检测器.

    对知识图谱执行社区发现，支持社区摘要生成与层级信息提取。
    """

    def detect(self, graph_store: GraphStore) -> dict[int, dict[str, Any]]:
        """执行社区检测.

        将图转换为无向图后运行 Louvain 算法（best_partition），
        并按社区 ID 聚合实体。

        Args:
            graph_store: 已填充数据的 GraphStore 实例.

        Returns:
            字典，键为社区 ID (int)，值为包含以下字段的字典：
                - entity_ids: 该社区包含的实体 ID 列表
                - summary: 摘要文本（初始为空字符串）
                - size: 社区包含的实体数量

        Raises:
            RuntimeError: python-louvain 库未安装时抛出.
            ValueError: 图为空时抛出.
        """
        if not _LOUVAIN_AVAILABLE:
            raise RuntimeError(
                "python-louvain is not installed. "
                "Install it with: pip install python-louvain"
            )

        graph = graph_store.graph
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot detect communities in an empty graph")

        # 转换为无向图（Louvain 要求）
        undirected = graph.to_undirected()

        # 运行 Louvain 社区发现
        partition: dict[str, int] = community_louvain.best_partition(undirected)

        # 按社区 ID 聚合实体
        communities: dict[int, dict[str, Any]] = {}
        community_entities: dict[int, list[str]] = defaultdict(list)

        for entity_id, community_id in partition.items():
            community_entities[community_id].append(entity_id)

        for community_id, entity_ids in community_entities.items():
            communities[community_id] = {
                "entity_ids": entity_ids,
                "summary": "",
                "size": len(entity_ids),
            }

        return communities

    def generate_summaries(
        self,
        graph_store: GraphStore,
        communities: dict[int, dict[str, Any]],
        llm_client: Any = None,
    ) -> dict[int, dict[str, Any]]:
        """为每个社区生成文本摘要.

        如果提供了 llm_client，则使用 LLM 生成 1-2 句自然语言总结；
        否则根据实体类型统计生成简单的结构化摘要。

        Args:
            graph_store: 数据源 GraphStore.
            communities: detect() 方法返回的社区字典，会被原地更新.
            llm_client: 可选的支持 chat.completions.create 的 LLM 客户端.

        Returns:
            更新后的 communities 字典，其中的 summary 字段已被填充.
        """
        graph = graph_store.graph

        for community_id, info in communities.items():
            entity_ids = info["entity_ids"]
            entity_names: list[str] = []
            type_counts: dict[str, int] = defaultdict(int)

            for eid in entity_ids:
                if eid in graph:
                    node_data = graph.nodes[eid]
                    name = node_data.get("name", eid)
                    entity_names.append(str(name))
                    etype = node_data.get("type", "unknown")
                    type_counts[etype] += 1

            if llm_client is not None:
                summary = self._llm_summary(
                    llm_client=llm_client,
                    entity_names=entity_names,
                    type_counts=dict(type_counts),
                )
            else:
                # 根据实体类型统计生成简单摘要
                sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                top_types_str = ", ".join(f"{t[0]} ({t[1]})" for t in sorted_types[:3])
                summary = f"Community focused on {top_types_str}" if top_types_str else ""

            info["summary"] = summary

        return communities

    def _llm_summary(
        self,
        llm_client: Any,
        entity_names: list[str],
        type_counts: dict[str, int],
    ) -> str:
        """使用 LLM 生成社区摘要.

        Args:
            llm_client: 实现了 chat.completions.create 的客户端.
            entity_names: 社区中的实体名称列表.
            type_counts: 实体类型 -> 数量 的统计.

        Returns:
            LLM 返回的摘要文本；若调用失败则回退到结构化摘要.
        """
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        top_types_str = ", ".join(f"{t} ({c})" for t, c in sorted_types[:3])

        prompt = (
            "You are a knowledge graph analyst. Summarize the following community "
            "of entities in 1-2 sentences, describing the common theme or domain "
            "they belong to.\n\n"
            f"Entity types: {top_types_str}\n"
            f"Entities ({len(entity_names)}): {', '.join(entity_names[:20])}\n\n"
            "Summary:"
        )

        try:
            response = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )
            summary = response.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception:
            pass

        # 回退到结构化摘要
        return f"Community focused on {top_types_str}" if top_types_str else ""

    def get_hierarchy(self, graph_store: GraphStore) -> list[dict[str, Any]]:
        """获取社区层级信息.

        对每个社区，列出其包含的实体以及社区间的边（跨社区关系）。

        Args:
            graph_store: 数据源 GraphStore.

        Returns:
            社区层级信息列表，每个元素包含：
                - community_id: 社区 ID
                - entities: 该社区中的实体列表
                - inter_community_edges: 连接该社区实体到其他社区实体的边列表
        """
        graph = graph_store.graph

        # 先检测社区
        communities = self.detect(graph_store)

        # 构建实体 -> 社区 ID 的映射
        entity_to_community: dict[str, int] = {}
        for cid, info in communities.items():
            for eid in info["entity_ids"]:
                entity_to_community[eid] = cid

        result: list[dict[str, Any]] = []
        for cid, info in communities.items():
            entity_list = []
            for eid in info["entity_ids"]:
                node_data = graph.nodes.get(eid, {})
                entity_list.append({
                    "id": eid,
                    "name": node_data.get("name", eid),
                    "type": node_data.get("type", "unknown"),
                })

            # 收集跨社区边
            inter_edges: list[dict[str, Any]] = []
            for eid in info["entity_ids"]:
                for _, target, edge_data in graph.out_edges(eid, data=True):
                    target_cid = entity_to_community.get(target)
                    if target_cid is not None and target_cid != cid:
                        inter_edges.append({
                            "source_id": eid,
                            "target_id": target,
                            "target_community": target_cid,
                            "predicate": edge_data.get("predicate", ""),
                            "weight": edge_data.get("weight", 1.0),
                        })

            result.append({
                "community_id": cid,
                "entities": entity_list,
                "inter_community_edges": inter_edges,
                "size": info["size"],
                "summary": info["summary"],
            })

        return result
