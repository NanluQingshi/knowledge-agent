"""情景记忆模块 — 类比具体事件记忆，存储历史对话与操作日志.

使用 ChromaDB（语义检索）+ 时间戳实现时序检索。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from knowledge_agent.config import settings
from knowledge_agent.embeddings.embedder import Embedder
from knowledge_agent.storage.vector_store import VectorStore


class EpisodicMemory:
    """情景记忆 — 存储和检索历史对话、操作日志.

    每条记忆以 Chunk 形式存入独立 ChromaDB collection，
    附带时间戳用于时序过滤。
    """

    _COLLECTION_NAME = "episodic_memory"

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._vector_store = vector_store or VectorStore()
        # 使用独立的 collection 存储情景记忆
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        persist_dir = settings.chroma_persist_dir
        client = chromadb.PersistentClient(path=persist_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self._collection = client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            embedding_function=ef,
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        memory_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """存储一条情景记忆.

        Args:
            content: 记忆内容文本.
            memory_type: 类型标签 (conversation / action / observation).
            metadata: 附加元数据.

        Returns:
            记忆 ID.
        """
        memory_id = str(uuid.uuid4())
        meta = {
            "memory_type": memory_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        embedding = self._embedder.embed_single(content)
        self._collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[memory_id],
        )

        return memory_id

    def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """存储一轮对话.

        Args:
            user_message: 用户消息.
            assistant_response: 助手回复.
            metadata: 附加元数据.

        Returns:
            [user_memory_id, assistant_memory_id].
        """
        ts = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []

        for role, content in [("user", user_message), ("assistant", assistant_response)]:
            mid = str(uuid.uuid4())
            meta = {
                "memory_type": "conversation",
                "role": role,
                "timestamp": ts,
                **(metadata or {}),
            }
            embedding = self._embedder.embed_single(content)
            self._collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[meta],
                ids=[mid],
            )
            ids.append(mid)

        return ids

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索情景记忆.

        Args:
            query: 查询文本.
            top_k: 返回数量.
            memory_type: 可选类型过滤.

        Returns:
            记忆列表，每项含 id、text、metadata、distance.
        """
        query_emb = self._embedder.embed_single(query)
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type}

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        memories: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            memories.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
            )
        return memories

    def recall_recent(
        self,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """获取最近存储的记忆（按时间戳倒序）.

        Args:
            limit: 返回数量.
            memory_type: 可选类型过滤.

        Returns:
            记忆列表.
        """
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type}

        all_data = self._collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        if not all_data["ids"]:
            return []

        memories = []
        for i in range(len(all_data["ids"])):
            meta = (all_data["metadatas"] or [{}])[i]
            memories.append(
                {
                    "id": all_data["ids"][i],
                    "text": (all_data["documents"] or [""])[i],
                    "metadata": meta,
                }
            )

        memories.sort(key=lambda m: m["metadata"].get("timestamp", ""), reverse=True)
        return memories[:limit]

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> None:
        """删除指定记忆.

        Args:
            memory_id: 记忆 ID.
        """
        self._collection.delete(ids=[memory_id])

    def count(self) -> int:
        """返回记忆总数."""
        return self._collection.count()

    def clear(self) -> None:
        """清空所有情景记忆."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
