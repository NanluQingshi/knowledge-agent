"""纯文本加载器 — 支持 .txt/.log/.csv/.json 文件."""
from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document

SUPPORTED_EXTENSIONS = {".txt", ".log", ".csv", ".json"}


class TextLoader(BaseLoader):
    """纯文本文件加载器，逐文件读取为一个文档."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file(file_path)
        stat = file_path.stat()

        doc = Document(
            content=content,
            metadata={
                "filename": file_path.name,
                "size": stat.st_size,
                "type": file_path.suffix.lower(),
            },
            source=str(file_path.resolve()),
        )
        return [doc]

    @staticmethod
    def _read_file(path: Path) -> str:
        """读取文本文件，自动处理编码.

        优先使用 UTF-8 解码；若失败则回退到 Latin-1（不会抛出解码异常）。
        """
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
