"""HTML 加载器 — 支持 .html / .htm 文件."""

from __future__ import annotations

from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document


class HTMLLoader(BaseLoader):
    """HTML 文档加载器，提取页面文本内容.

    依赖 beautifulsoup4 包，未安装时回退到简单的 HTML 标签剥离。
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".html", ".htm"}

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_html = self._read_file(file_path)
        content = self._extract_text(raw_html)

        if not content.strip():
            return []

        stat = file_path.stat()
        return [
            Document(
                content=content,
                metadata={
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "type": "html",
                },
                source=str(file_path.resolve()),
            )
        ]

    @staticmethod
    def _read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    @staticmethod
    def _extract_text(html: str) -> str:
        """从 HTML 中提取纯文本."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            # 移除 script 和 style 标签
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            # 清理多余空白行
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except ImportError:
            # 回退：简单的 HTML 标签剥离
            import re

            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text