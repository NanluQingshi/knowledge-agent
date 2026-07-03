"""Markdown 加载器 — 支持按 ## 二级标题分割文档."""
from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    """Markdown 文件加载器，按 ``##`` 二级标题分割为多个文档.

    每个 ``##`` 标题及其后续内容构成一个独立文档，元数据中包含标题名称。
    标题之前的内容（前言）作为第一个文档（无 heading 元数据）。
    若文件中无 ``##`` 标题，则整体作为一个文档返回。
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".md"

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file(file_path)
        stat = file_path.stat()
        base_metadata: dict = {
            "filename": file_path.name,
            "size": stat.st_size,
            "type": ".md",
        }
        source = str(file_path.resolve())

        sections = self._split_into_sections(content)

        documents: list[Document] = []
        for heading, section_content in sections:
            if not section_content:
                continue
            metadata = dict(base_metadata)
            if heading:
                metadata["heading"] = heading
            doc = Document(
                content=section_content,
                metadata=metadata,
                source=source,
            )
            documents.append(doc)

        return documents if documents else [
            Document(
                content="",
                metadata=base_metadata,
                source=source,
            )
        ]

    @staticmethod
    def _split_into_sections(content: str) -> list[tuple[str, str]]:
        """将 Markdown 内容按 ``## `` 二级标题分割。

        Returns:
            列表，每个元素为 (标题, 正文) 元组。标题可能为空字符串（表示前言部分）。
        """
        sections: list[tuple[str, str]] = []
        lines = content.split("\n")
        current_heading: str = ""
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_lines:
                    sections.append(
                        (current_heading, "\n".join(current_lines).strip())
                    )
                current_heading = line.removeprefix("## ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 处理最后一个标题之后的内容
        remaining = "\n".join(current_lines).strip()
        sections.append((current_heading, remaining))

        return sections

    @staticmethod
    def _read_file(path: Path) -> str:
        """读取 Markdown 文件，自动处理编码."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
