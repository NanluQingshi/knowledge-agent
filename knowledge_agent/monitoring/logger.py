"""结构化日志配置 — 统一的 JSON 格式日志输出."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from knowledge_agent.monitoring.tracer import get_trace_id


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器，输出 JSON 格式日志."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id() or "",
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra") and record.extra:
            log_entry["extra"] = record.extra

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """配置全局结构化日志.

    Args:
        level: 日志级别，默认 INFO.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 移除已有 handler，避免重复
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)

    # 第三方库的日志级别设为 WARNING，减少噪音
    for lib in ("openai", "httpx", "chromadb", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器.

    Args:
        name: 日志器名称，通常使用 __name__.

    Returns:
        配置好的 Logger 实例.
    """
    return logging.getLogger(name)