"""JSON 加载器 — 支持 .json 文件."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document


class JSONLoader(BaseLoader):
    """JSON 文件加载器，将结构化的 JSON 数据格式化为文本.

    支持两种模式：
    - 对象/数组：格式化为易读的 key: value 文本
    - 纯文本字段：直接提取 "content" 或 "text" 字段
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".json"

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(file_path.read_text(encoding="latin-1"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file {file_path}: {exc}")

        content = self._format_json(data)
        if not content.strip():
            return []

        stat = file_path.stat()
        return [
            Document(
                content=content,
                metadata={
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "type": "json",
                    "json_type": type(data).__name__,
                },
                source=str(file_path.resolve()),
            )
        ]

    @staticmethod
    def _format_json(data: object, indent: int = 0) -> str:
        """将 JSON 数据格式化为可读文本."""
        prefix = "  " * indent
        lines: list[str] = []

        if isinstance(data, dict):
            # 尝试提取 content/text 字段作为直接内容
            if indent == 0:
                for key in ("content", "text", "body", "description"):
                    if key in data and isinstance(data[key], str):
                        return data[key].strip()

            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}- {key}:")
                    lines.append(JSONLoader._format_json(value, indent + 1))
                else:
                    lines.append(f"{prefix}- {key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(JSONLoader._format_json(item, indent + 1))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)