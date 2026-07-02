"""分块器基类与 Chunk 数据模型."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """文档分块后的数据单元."""

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class BaseChunker(ABC):
    """分块器抽象基类 — 所有分块策略需实现 chunk 方法."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """将输入文本切分为 Chunk 列表.

        Args:
            text: 原始文档文本.
            metadata: 文档级别的元数据，会合并到每个 Chunk 中.

        Returns:
            Chunk 对象列表.
        """
        ...
