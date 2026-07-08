"""反馈收集器 — 用户反馈的采集、存储与聚合."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_agent.config import settings


class FeedbackCollector:
    """用户反馈收集与聚合.

    将用户标记的"有用/无用"评分持久化到 SQLite，
    支持按文档/查询维度的聚合统计。
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        query_text TEXT NOT NULL,
        answer_text TEXT,
        rating TEXT NOT NULL CHECK(rating IN ('useful', 'useless', 'partial')),
        comment TEXT,
        source_doc_ids TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(Path(settings.data_dir) / "feedback.db")
        self._db_path = path
        db_file = Path(path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(self._CREATE_TABLE_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # 写入反馈
    # ------------------------------------------------------------------

    def record(
        self,
        query_text: str,
        answer_text: str = "",
        rating: str = "useful",
        comment: str = "",
        source_doc_ids: list[str] | None = None,
    ) -> str:
        """记录一条用户反馈.

        Args:
            query_text: 触发查询的问题.
            answer_text: 模型给出的回答.
            rating: 评分 (useful / useless / partial).
            comment: 用户备注.
            source_doc_ids: 引用的来源文档 ID 列表.

        Returns:
            反馈 ID.
        """
        import uuid

        fid = str(uuid.uuid4())
        source_json = json.dumps(source_doc_ids or [], ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            conn.execute(
                """INSERT INTO feedback (id, query_text, answer_text, rating, comment, source_doc_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (fid, query_text, answer_text, rating, comment, source_json, now),
            )
            conn.commit()

        return fid

    # ------------------------------------------------------------------
    # 聚合查询
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取整体反馈统计.

        Returns:
            包含总数、各评级数量、有用率的字典.
        """
        with self._connection() as conn:
            total_row = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()
            total = total_row["c"] if total_row else 0

            rating_counts: dict[str, int] = {}
            for row in conn.execute("SELECT rating, COUNT(*) as c FROM feedback GROUP BY rating").fetchall():
                rating_counts[row["rating"]] = row["c"]

        useful = rating_counts.get("useful", 0)
        useless = rating_counts.get("useless", 0)
        partial = rating_counts.get("partial", 0)

        return {
            "total_feedback": total,
            "useful": useful,
            "useless": useless,
            "partial": partial,
            "usefulness_rate": round(useful / total, 4) if total > 0 else 0.0,
        }

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近的反馈记录.

        Args:
            limit: 返回条数.

        Returns:
            反馈列表，按时间倒序.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            raw = d.pop("source_doc_ids", "[]")
            try:
                d["source_doc_ids"] = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                d["source_doc_ids"] = []
            results.append(d)

        return results

    def get_unhelpful_queries(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取被标记为 useless 或 partial 的查询，用于分析改进点.

        Args:
            limit: 返回条数.

        Returns:
            不理想反馈列表.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE rating IN ('useless', 'partial') ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 按文档维度的反馈查询
    # ------------------------------------------------------------------

    def get_stats_for_doc(self, doc_id: str) -> dict[str, Any]:
        """获取指定文档的反馈统计.

        在 source_doc_ids JSON 字段中搜索包含给定 doc_id 的记录，
        汇总这些记录中各评级数量。

        Args:
            doc_id: 文档 ID.

        Returns:
            包含 total、useful、useless、partial、usefulness_rate 的字典.
        """
        with self._connection() as conn:
            all_rows = conn.execute(
                "SELECT rating, source_doc_ids FROM feedback",
            ).fetchall()

        useful = 0
        useless = 0
        partial = 0

        for row in all_rows:
            raw = row["source_doc_ids"]
            try:
                ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except (json.JSONDecodeError, TypeError):
                ids = []
            if doc_id in ids:
                rating = row["rating"]
                if rating == "useful":
                    useful += 1
                elif rating == "useless":
                    useless += 1
                elif rating == "partial":
                    partial += 1

        total = useful + useless + partial
        return {
            "total_feedback": total,
            "useful": useful,
            "useless": useless,
            "partial": partial,
            "usefulness_rate": round(useful / total, 4) if total > 0 else 0.5,
        }
