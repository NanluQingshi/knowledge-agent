"""ChromaDB 向量存储封装."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from knowledge_agent.chunkers.base import Chunk
from knowledge_agent.config import settings


class VectorStore:
    """ChromaDB 持久化向量存储封装.

    默认从全局 settings 读取 chroma_persist_dir。
    使用 sentence-transformers 的 all-MiniLM-L6-v2 作为 ChromaDB 内置嵌入函数，
    这样 add() 时传入的 embeddings 参数会覆盖内置函数的行为。
    """

    _LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
    _COLLECTION_NAME = "knowledge_base"

    def __init__(self, persist_dir: str | None = None) -> None:
        """初始化 VectorStore.

        Args:
            persist_dir: ChromaDB 持久化目录，默认 settings.chroma_persist_dir.
        """
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        persist_path = Path(self._persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self._LOCAL_EMBED_MODEL,
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            embedding_function=self._embedding_function,
        )

    @property
    def collection(self) -> chromadb.Collection:
        """当前 ChromaDB Collection 实例."""
        return self._collection

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """将分块数据及其向量添加至集合.

        Args:
            chunks: Chunk 对象列表，其 .text 作为文档内容.
            embeddings: 与 chunks 对应的向量列表.
            metadatas: 与 chunks 对应的元数据字典列表.
            ids: 唯一标识符列表.

        Raises:
            ValueError: 各参数长度不一致时抛出.
        """
        n = len(chunks)
        if not (len(embeddings) == len(metadatas) == len(ids) == n):
            raise ValueError(
                f"All arguments must have the same length: "
                f"chunks={n}, embeddings={len(embeddings)}, "
                f"metadatas={len(metadatas)}, ids={len(ids)}"
            )

        documents = [chunk.text for chunk in chunks]
        metadatas_merged: list[dict[str, Any]] = []
        for chunk, meta in zip(chunks, metadatas):
            merged = {**meta, **chunk.metadata}
            metadatas_merged.append(merged)

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas_merged,
            ids=ids,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """按向量相似度搜索.

        Args:
            query_embedding: 查询向量.
            top_k: 返回的最相似结果数量，默认 5.

        Returns:
            包含 id、text、metadata、distance 的字典列表.
        """
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not raw["ids"] or not raw["ids"][0]:
            return []

        results: list[dict[str, Any]] = []
        for i in range(len(raw["ids"][0])):
            results.append(
                {
                    "id": raw["ids"][0][i],
                    "text": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                    "distance": raw["distances"][0][i] if raw["distances"] else 0.0,
                }
            )
        return results

    def delete(self, ids: list[str]) -> None:
        """按 ID 列表删除文档.

        Args:
            ids: 要删除的文档 ID 列表.
        """
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        """返回集合中的文档总数.

        Returns:
            文档数量.
        """
        return self._collection.count()
