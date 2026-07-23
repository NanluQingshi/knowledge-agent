"""SQLite 文档元数据存储封装."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from knowledge_agent.config import settings


class DocStore:
    """SQLite 持久化的文档元数据存储.

    默认从全局 settings 读取 doc_db_path。
    提供文档元数据的增删改查以及总量统计能力。
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_type TEXT,
        chunk_count INTEGER DEFAULT 0,
        content_hash TEXT DEFAULT '',
        version INTEGER DEFAULT 1,
        previous_version_id TEXT DEFAULT '',
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT
    )
    """

    _COLUMNS = ["id", "source", "filename", "file_type", "chunk_count", "content_hash",
                 "version", "previous_version_id", "ingested_at", "metadata_json"]

    def __init__(self, db_path: str | None = None) -> None:
        """初始化 DocStore.

        Args:
            db_path: SQLite 数据库文件路径，默认 settings.doc_db_path.
        """
        self._db_path = db_path or settings.doc_db_path
        db_file = Path(self._db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        """创建并返回一个数据库连接 (上下文管理器用法请使用 _with_cursor)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """创建 documents 表（如尚不存在）."""
        with self._connection() as conn:
            conn.execute(self._CREATE_TABLE_SQL)
            conn.commit()

    def add_document(
        self,
        doc_id: str,
        source: str,
        filename: str,
        file_type: str,
        chunk_count: int,
        content_hash: str = "",
        metadata: dict[str, Any] | None = None,
        previous_version_id: str = "",
    ) -> None:
        """添加一条文档元数据记录.

        Args:
            doc_id: 文档唯一标识.
            source: 文档来源（如文件路径、URL）.
            filename: 文档文件名.
            file_type: 文档类型（如 pdf、md、txt）.
            chunk_count: 文档被切分出的 Chunk 数量.
            content_hash: 文档内容的 SHA256 哈希（用于去重和版本追踪）.
            metadata: 附加元数据，会被序列化为 JSON 字符串.
            previous_version_id: 前一版本的文档 ID，用于版本链追踪.

        Raises:
            sqlite3.IntegrityError: 主键冲突时抛出.
        """
        # 计算版本号
        version = 1
        if previous_version_id:
            prev = self.get_document(previous_version_id)
            if prev:
                version = (prev.get("version", 0) or 0) + 1

        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO documents (id, source, filename, file_type, chunk_count,
                   content_hash, version, previous_version_id, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, source, filename, file_type, chunk_count, content_hash,
                 version, previous_version_id, metadata_json),
            )
            conn.commit()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """根据 ID 查询单条文档元数据.

        Args:
            doc_id: 文档唯一标识.

        Returns:
            文档元数据字典，未找到时返回 None.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def list_documents(self) -> list[dict[str, Any]]:
        """列出所有文档元数据，按摄入时间倒序排列.

        Returns:
            文档元数据字典列表.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC",
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete_document(self, doc_id: str) -> None:
        """删除指定 ID 的文档元数据记录.

        Args:
            doc_id: 文档唯一标识.
        """
        with self._connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

    def find_by_hash(self, content_hash: str) -> list[dict[str, Any]]:
        """根据内容哈希查找文档.

        Args:
            content_hash: SHA256 内容哈希.

        Returns:
            匹配的文档列表（可能多条，相同内容来自不同来源）.
        """
        if not content_hash:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY ingested_at DESC",
                (content_hash,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def find_by_source(self, source: str) -> list[dict[str, Any]]:
        """根据来源路径查找文档（按版本倒序）.

        Args:
            source: 文档来源路径.

        Returns:
            匹配的文档版本列表，最新版本在前.
        """
        if not source:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE source = ? ORDER BY version DESC, ingested_at DESC",
                (source,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_document_versions(self, doc_id: str) -> list[dict[str, Any]]:
        """获取指定文档的所有版本历史.

        通过 previous_version_id 链追踪所有版本。

        Args:
            doc_id: 当前文档 ID.

        Returns:
            版本历史列表，从最新到最旧.
        """
        versions = []
        current = self.get_document(doc_id)
        if current is None:
            return []

        # 向前追溯所有版本
        while current is not None:
            versions.append(current)
            prev_id = current.get("previous_version_id", "")
            if prev_id:
                current = self.get_document(prev_id)
            else:
                current = None

        return versions

    def get_latest_version(self, source: str) -> dict[str, Any] | None:
        """获取指定来源的最新版本文档.

        Args:
            source: 文档来源路径.

        Returns:
            最新版本文档，或 None（无记录时）.
        """
        docs = self.find_by_source(source)
        return docs[0] if docs else None

    def search_documents(self, keyword: str) -> list[dict[str, Any]]:
        """按关键词搜索文档（匹配文件名和来源）.

        Args:
            keyword: 搜索关键词.

        Returns:
            匹配的文档列表.
        """
        if not keyword or not keyword.strip():
            return self.list_documents()
        kw = f"%{keyword.strip()}%"
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE filename LIKE ? OR source LIKE ? "
                "ORDER BY ingested_at DESC",
                (kw, kw),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """按标签搜索文档（标签存储在 metadata_json 中）.

        Args:
            tag: 标签名称.

        Returns:
            匹配的文档列表.
        """
        if not tag:
            return []
        all_docs = self.list_documents()
        tag_lower = tag.strip().lower()
        result = []
        for doc in all_docs:
            meta = doc.get("metadata", {})
            tags = meta.get("tags", [])
            if any(t.lower() == tag_lower for t in tags):
                result.append(doc)
        return result

    def add_tag(self, doc_id: str, tag: str) -> bool:
        """为文档添加标签.

        Args:
            doc_id: 文档 ID.
            tag: 标签名称.

        Returns:
            是否成功.
        """
        doc = self.get_document(doc_id)
        if doc is None:
            return False
        meta = doc.get("metadata", {})
        tags = meta.get("tags", [])
        if tag not in tags:
            tags.append(tag)
        meta["tags"] = tags
        metadata_json = json.dumps(meta, ensure_ascii=False)
        with self._connection() as conn:
            conn.execute(
                "UPDATE documents SET metadata_json = ? WHERE id = ?",
                (metadata_json, doc_id),
            )
            conn.commit()
        return True

    def get_all_tags(self) -> list[str]:
        """获取所有文档中使用的标签列表.

        Returns:
            去重后的标签列表.
        """
        all_docs = self.list_documents()
        tag_set: set[str] = set()
        for doc in all_docs:
            meta = doc.get("metadata", {})
            tags = meta.get("tags", [])
            tag_set.update(tags)
        return sorted(tag_set)

    def rollback_document(self, doc_id: str) -> dict[str, Any] | None:
        """回滚到指定版本（标记为当前激活版本，保留版本链）.

        Args:
            doc_id: 要回滚到的文档版本 ID.

        Returns:
            回滚后的文档元数据，或 None（未找到时）.
        """
        target = self.get_document(doc_id)
        if target is None:
            return None
        # 更新元数据标记为回滚状态
        meta = target.get("metadata", {})
        meta["rolled_back"] = True
        meta["rolled_back_at"] = __import__("datetime").datetime.now().isoformat()
        metadata_json = json.dumps(meta, ensure_ascii=False)
        with self._connection() as conn:
            conn.execute(
                "UPDATE documents SET metadata_json = ? WHERE id = ?",
                (metadata_json, doc_id),
            )
            conn.commit()
        return self.get_document(doc_id)

    def get_total_chunks(self) -> int:
        """返回所有文档的 Chunk 总数.

        Returns:
            chunk_count 列的 SUM，若无记录则返回 0.
        """
        with self._connection() as conn:
            row = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) AS total FROM documents").fetchone()
        return int(row["total"]) if row else 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """将 sqlite3.Row 转为字典，并反序列化 metadata_json 字段."""
        d = dict(row)
        # 将 metadata_json 字符串解析回字典
        raw = d.pop("metadata_json", None)
        d["metadata"] = json.loads(raw) if raw else {}
        return d
