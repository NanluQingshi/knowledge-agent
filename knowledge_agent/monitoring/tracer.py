"""请求追踪 — 上下文 Trace ID 管理."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def generate_trace_id() -> str:
    """生成新的 Trace ID."""
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文的 Trace ID.

    Args:
        trace_id: 指定的 Trace ID；不传时自动生成.

    Returns:
        设置的 Trace ID.
    """
    tid = trace_id or generate_trace_id()
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前上下文的 Trace ID.

    Returns:
        当前 Trace ID，未设置时返回空字符串.
    """
    return _trace_id.get()


class Tracer:
    """请求追踪器，为每个请求分配唯一 Trace ID 并可在上下文中传递."""

    @staticmethod
    def start(trace_id: str | None = None) -> str:
        """开始一个新的追踪.

        Args:
            trace_id: 可选的指定 Trace ID.

        Returns:
            Trace ID.
        """
        return set_trace_id(trace_id)

    @staticmethod
    def current() -> str:
        """获取当前 Trace ID.

        Returns:
            当前 Trace ID，未设置时返回空字符串.
        """
        return get_trace_id()

    @staticmethod
    def reset() -> None:
        """重置当前 Trace ID."""
        _trace_id.set("")