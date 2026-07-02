"""PDF 加载器 — 使用 pypdf 按页提取文本."""
from pathlib import Path

from pypdf import PdfReader

from knowledge_agent.loaders.base import BaseLoader, Document


class PDFLoader(BaseLoader):
    """PDF 文件加载器，每页作为一个独立文档.

    使用 pypdf 库逐页提取文本，元数据中包含页码和总页数。
    跳过无文本内容的空白页。
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(file_path))
        stat = file_path.stat()
        base_metadata: dict = {
            "filename": file_path.name,
            "size": stat.st_size,
            "type": ".pdf",
            "total_pages": len(reader.pages),
        }
        source = str(file_path.resolve())

        documents: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            metadata = dict(base_metadata)
            metadata["page"] = i + 1  # 1-based 页码
            doc = Document(
                content=text.strip(),
                metadata=metadata,
                source=source,
            )
            documents.append(doc)

        return documents
