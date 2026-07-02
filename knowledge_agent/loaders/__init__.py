"""文档加载器 — 支持多格式文档的加载和解析."""

from knowledge_agent.loaders.base import BaseLoader
from knowledge_agent.loaders.text_loader import TextLoader
from knowledge_agent.loaders.markdown_loader import MarkdownLoader
from knowledge_agent.loaders.pdf_loader import PDFLoader

__all__ = ["BaseLoader", "TextLoader", "MarkdownLoader", "PDFLoader"]
