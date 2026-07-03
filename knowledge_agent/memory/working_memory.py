"""工作记忆模块 — 类比短期注意力，管理当前对话上下文."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_TOKENS = 8000
APPROX_CHARS_PER_TOKEN = 4  # 粗略估算，用于字符数到 token 数的转换


class WorkingMemory:
    """当前对话上下文管理器.

    维护一个有限容量的消息窗口，自动驱逐最早的消息以保持在 token 预算内。
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._max_tokens = max_tokens
        self._messages: list[dict[str, Any]] = []
        self._context_vars: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到工作记忆.

        Args:
            role: 消息角色 (user / assistant / system).
            content: 消息内容.
        """
        self._messages.append({"role": role, "content": content})
        self._trim()

    def add_messages(self, messages: list[dict[str, str]]) -> None:
        """批量添加消息.

        Args:
            messages: 消息列表，每项含 role 和 content.
        """
        for msg in messages:
            self._messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        self._trim()

    def get_messages(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """获取当前消息列表.

        Args:
            last_n: 只返回最近 n 条，None 表示全部.

        Returns:
            消息列表.
        """
        if last_n is None:
            return list(self._messages)
        return list(self._messages[-last_n:])

    def clear(self) -> None:
        """清空所有消息."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # 上下文变量
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """设置上下文变量.

        Args:
            key: 变量名.
            value: 对应值.
        """
        self._context_vars[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文变量.

        Args:
            key: 变量名.
            default: 默认值.

        Returns:
            变量值.
        """
        return self._context_vars.get(key, default)

    def get_all_context(self) -> dict[str, Any]:
        """获取所有上下文变量的副本.

        Returns:
            上下文变量字典.
        """
        return dict(self._context_vars)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        """当前消息数量."""
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        """估算当前上下文的 token 数."""
        total_chars = sum(len(m.get("content", "")) for m in self._messages)
        return total_chars // APPROX_CHARS_PER_TOKEN

    @property
    def max_tokens(self) -> int:
        """当前 token 预算上限."""
        return self._max_tokens

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """驱逐最早的消息直到 token 数在预算内."""
        while self.estimated_tokens > self._max_tokens and len(self._messages) > 1:
            self._messages.pop(0)
