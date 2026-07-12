"""评估数据集 — 加载/保存/管理评估用 QA 数据集."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_agent.config import settings


class EvaluationDataset:
    """评估数据集管理.

    每个条目包含：
    - query: 用户问题
    - expected_doc_ids: 期望检索到的文档 ID 列表（可选，用于检索质量评估）
    - expected_answer: 期望回答（可选，用于答案质量评估）
    - metadata: 附加元数据（类别、难度等）
    """

    def __init__(self, path: str | None = None) -> None:
        """初始化 EvaluationDataset.

        Args:
            path: 数据集 JSON 文件路径；默认 settings.data_dir/eval_dataset.json.
        """
        self._path = Path(path or str(Path(settings.data_dir) / "eval_dataset.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # 数据集管理
    # ------------------------------------------------------------------

    def add_item(
        self,
        query: str,
        expected_doc_ids: list[str] | None = None,
        expected_answer: str = "",
        category: str = "general",
        difficulty: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加一个评估条目.

        Args:
            query: 问题.
            expected_doc_ids: 期望检索到的文档 ID 列表.
            expected_answer: 期望的回答文本.
            category: 类别标签.
            difficulty: 难度 (easy / medium / hard).
            metadata: 附加元数据.

        Returns:
            条目 ID.
        """
        item_id = str(uuid.uuid4())
        self._items.append({
            "id": item_id,
            "query": query,
            "expected_doc_ids": expected_doc_ids or [],
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return item_id

    def remove_item(self, item_id: str) -> bool:
        """删除指定评估条目.

        Args:
            item_id: 条目 ID.

        Returns:
            是否成功删除.
        """
        for i, item in enumerate(self._items):
            if item["id"] == item_id:
                self._items.pop(i)
                self._save()
                return True
        return False

    def list_items(
        self,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出评估条目，支持按类别和难度过滤.

        Args:
            category: 可选类别过滤.
            difficulty: 可选难度过滤.

        Returns:
            条目列表.
        """
        items = list(self._items)
        if category:
            items = [i for i in items if i.get("category") == category]
        if difficulty:
            items = [i for i in items if i.get("difficulty") == difficulty]
        return items

    @property
    def size(self) -> int:
        """条目总数."""
        return len(self._items)

    def clear(self) -> None:
        """清空所有条目."""
        self._items.clear()
        self._save()

    # ------------------------------------------------------------------
    # 导入/导出
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> EvaluationDataset:
        """从外部 JSON 文件导入数据集（不覆盖默认路径）.

        Args:
            path: 外部 JSON 文件路径.

        Returns:
            加载了数据的 EvaluationDataset 实例（临时路径）.
        """
        ds = cls(path=str(path))
        return ds

    def export_to(self, path: str | Path) -> None:
        """导出数据集到指定路径.

        Args:
            path: 输出路径.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)
