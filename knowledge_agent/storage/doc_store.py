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
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT
    )
    """

    _COLUMNS = ["id", "source", "filename", "file_type", "chunk_count", "content_hash", "ingested_at", "metadata_json"]

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
    ) -> None:
        """添加一条文档元数据记录.

        Args:
            doc_id: 文档唯一标识.
            source: 文档来源（如文件路径、URL）.
            filename: 文档文件名.
            file_type: 文档类型（如 pdf、md、txt）.
            chunk_count: 文档被切分出的 Chunk 数量.
            content_hash: 文档内容的 SHA256 哈希（用于去重）.
            metadata: 附加元数据，会被序列化为 JSON 字符串.

        Raises:
            sqlite3.IntegrityError: 主键冲突时抛出.
        """
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO documents (id, source, filename, file_type, chunk_count, content_hash, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, source, filename, file_type, chunk_count, content_hash, metadata_json),
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
