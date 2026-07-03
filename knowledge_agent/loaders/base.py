"""文档加载器抽象基类与数据模型."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """加载后的文档."""

    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""


class BaseLoader(ABC):
    """文档加载器基类 — 所有加载器需实现 can_handle 和 load."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """判断此加载器是否可处理指定文件."""
        ...

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """加载文件并返回文档列表."""
        ...
