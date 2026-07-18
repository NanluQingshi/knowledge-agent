"""文档加载器 — 支持多格式文档的加载和解析."""

from knowledge_agent.loaders.base import BaseLoader
from knowledge_agent.loaders.text_loader import TextLoader
from knowledge_agent.loaders.markdown_loader import MarkdownLoader
from knowledge_agent.loaders.pdf_loader import PDFLoader
from knowledge_agent.loaders.docx_loader import DocxLoader
from knowledge_agent.loaders.html_loader import HTMLLoader
from knowledge_agent.loaders.csv_loader import CSVLoader
from knowledge_agent.loaders.json_loader import JSONLoader
from knowledge_agent.loaders.url_loader import UrlLoader
from knowledge_agent.loaders.image_loader import ImageLoader


def all_loaders() -> list[BaseLoader]:
    """返回默认的文档加载器列表.

    Returns:
        包含所有内置加载器实例的列表，按通用性降序排列.
    """
    return [
        TextLoader(),
        MarkdownLoader(),
        PDFLoader(),
        DocxLoader(),
        HTMLLoader(),
        CSVLoader(),
        JSONLoader(),
        ImageLoader(),
    ]


__all__ = [
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "DocxLoader",
    "HTMLLoader",
    "CSVLoader",
    "JSONLoader",
    "UrlLoader",
    "ImageLoader",
    "all_loaders",
]
