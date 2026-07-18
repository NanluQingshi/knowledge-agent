"""查询结果缓存 — 相同提问直接返回缓存，避免重复计算."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

_DEFAULT_TTL = 300  # 默认缓存 5 分钟
_MAX_SIZE = 100  # 最大缓存条目数


class QueryCache:
    """LRU 查询结果缓存.

    相同提问在 TTL 内直接返回缓存结果，避免重复检索和 LLM 调用。
    使用 OrderedDict 实现 LRU 淘汰策略。
    """

    def __init__(self, ttl: int = _DEFAULT_TTL, max_size: int = _MAX_SIZE) -> None:
        """初始化 QueryCache.

        Args:
            ttl: 缓存有效期（秒），默认 300 秒（5 分钟）.
            max_size: 最大缓存条目数，默认 100.
        """
        self._ttl = ttl
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        """获取缓存结果.

        Args:
            key: 缓存键（通常为查询文本）.

        Returns:
            缓存的结果字典，未命中或已过期时返回 None.
        """
        if key not in self._cache:
            return None

        timestamp, result = self._cache[key]
        if time.monotonic() - timestamp > self._ttl:
            # 过期，删除
            del self._cache[key]
            return None

        # 移动到末尾（最近使用）
        self._cache.move_to_end(key)
        return result

    def set(self, key: str, result: dict[str, Any]) -> None:
        """设置缓存结果.

        Args:
            key: 缓存键.
            result: 结果字典.
        """
        # 达到上限时淘汰最久未使用的条目
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (time.monotonic(), result)
        self._cache.move_to_end(key)

    def invalidate(self, key: str) -> None:
        """使指定键的缓存失效.

        Args:
            key: 缓存键.
        """
        self._cache.pop(key, None)

    def clear(self) -> None:
        """清空所有缓存."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """当前缓存条目数."""
        return len(self._cache)

    @property
    def keys(self) -> list[str]:
        """当前缓存的所有键."""
        return list(self._cache.keys())