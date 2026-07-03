"""程序记忆模块 — 类比技能习惯，存储工作流模板与最佳实践.

以 JSON/YAML 配置形式持久化到本地文件。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_agent.config import settings


class ProceduralMemory:
    """程序记忆 — 工作流模板、操作步骤与最佳实践.

    存储为 JSON 文件，支持模板的增删改查和执行记录的追踪。
    """

    def __init__(self, storage_path: str | None = None) -> None:
        path = storage_path or str(Path(settings.data_dir) / "procedural_memory.json")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # 模板管理
    # ------------------------------------------------------------------

    def add_template(
        self,
        name: str,
        steps: list[dict[str, Any]],
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """注册一个工作流模板.

        Args:
            name: 模板名称.
            steps: 步骤列表，每项含 action、params 等字段.
            description: 模板描述.
            tags: 标签列表.

        Returns:
            模板 ID.
        """
        tid = name.lower().replace(" ", "_")
        self._templates[tid] = {
            "id": tid,
            "name": name,
            "description": description,
            "steps": steps,
            "tags": tags or [],
            "usage_count": 0,
            "last_used": None,
            "success_rate": 1.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return tid

    def get_template(self, template_id: str) -> dict[str, Any] | None:
        """获取模板详情.

        Args:
            template_id: 模板 ID.

        Returns:
            模板字典，未找到时返回 None.
        """
        return self._templates.get(template_id)

    def list_templates(self, tag: str | None = None) -> list[dict[str, Any]]:
        """列出所有模板，可按标签过滤.

        Args:
            tag: 可选标签过滤.

        Returns:
            模板列表，按使用次数降序排列.
        """
        templates = list(self._templates.values())
        if tag:
            templates = [t for t in templates if tag in t.get("tags", [])]
        return sorted(templates, key=lambda t: t.get("usage_count", 0), reverse=True)

    def delete_template(self, template_id: str) -> bool:
        """删除模板.

        Args:
            template_id: 模板 ID.

        Returns:
            是否成功删除.
        """
        if template_id in self._templates:
            del self._templates[template_id]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # 执行追踪
    # ------------------------------------------------------------------

    def record_execution(
        self,
        template_id: str,
        success: bool,
        duration_seconds: float = 0.0,
    ) -> None:
        """记录一次模板执行.

        Args:
            template_id: 模板 ID.
            success: 是否执行成功.
            duration_seconds: 执行耗时.
        """
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            return

        total = tmpl.get("usage_count", 0)
        prev_rate = tmpl.get("success_rate", 1.0)
        tmpl["usage_count"] = total + 1
        tmpl["success_rate"] = round((prev_rate * total + (1.0 if success else 0.0)) / (total + 1), 4)
        tmpl["last_used"] = datetime.now(timezone.utc).isoformat()
        tmpl["last_duration"] = duration_seconds
        self._save()

    def get_best_practices(self, min_success_rate: float = 0.8) -> list[dict[str, Any]]:
        """获取成功率高于阈值的最佳实践模板.

        Args:
            min_success_rate: 最小成功率阈值.

        Returns:
            符合条件的模板列表.
        """
        return [
            t
            for t in self._templates.values()
            if t.get("success_rate", 0.0) >= min_success_rate
        ]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        """从 JSON 文件加载模板."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        """保存模板到 JSON 文件."""
        self._path.write_text(json.dumps(self._templates, ensure_ascii=False, indent=2), encoding="utf-8")
