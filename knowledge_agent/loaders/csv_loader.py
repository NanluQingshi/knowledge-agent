"""CSV 加载器 — 支持 .csv 文件."""

from __future__ import annotations

from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document


class CSVLoader(BaseLoader):
    """CSV 文件加载器，将每行视为一条记录，整体作为一个文档.

    将 CSV 内容格式化为表格文本（含表头），便于后续分块和检索。
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._format_csv(file_path)
        if not content.strip():
            return []

        stat = file_path.stat()
        return [
            Document(
                content=content,
                metadata={
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "type": "csv",
                },
                source=str(file_path.resolve()),
            )
        ]

    @staticmethod
    def _format_csv(path: Path) -> str:
        """将 CSV 格式化为易读的表格文本."""
        try:
            import csv
            import io

            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = path.read_text(encoding="latin-1")

            reader = csv.reader(io.StringIO(raw))
            rows = list(reader)
            if not rows:
                return ""

            lines: list[str] = []
            # 表头
            header = rows[0]
            lines.append(" | ".join(header))
            lines.append("-" * len(lines[0]))
            # 数据行
            for row in rows[1:]:
                # 补齐长度
                while len(row) < len(header):
                    row.append("")
                lines.append(" | ".join(row[: len(header)]))

            return "\n".join(lines)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")