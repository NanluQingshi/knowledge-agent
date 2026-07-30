"""分块器基类与 Chunk 数据模型."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class _CharacterEncoding:
    """tiktoken 不可用时的离线字符级降级编码器."""

    @staticmethod
    def encode(text: str) -> list[str]:
        return list(text)

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return "".join(tokens)


@lru_cache(maxsize=1)
def get_token_encoding() -> Any:
    """延迟加载 tokenizer，失败时使用不访问网络的字符级实现."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning("Falling back to character tokenizer: %s", exc)
        return _CharacterEncoding()


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

    @staticmethod
    def count_tokens(text: str) -> int:
        """计算文本的 token 数量.

        Args:
            text: 输入文本.

        Returns:
            token 数量.
        """
        return len(get_token_encoding().encode(text))
