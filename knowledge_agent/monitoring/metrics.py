"""性能指标收集器 — 追踪操作耗时、计数、P50/P95 延迟."""

from __future__ import annotations

import time
import statistics
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """性能指标收集器.

    支持两种指标类型：
    - **timing**: 操作耗时（ms），支持 P50/P95/P99 统计
    - **counter**: 计数器，如请求次数、错误次数

    所有指标存储在内存中，可通过 Web UI 监控面板查看。
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def record_timing(self, operation: str, duration_ms: float) -> None:
        """记录一次操作耗时.

        Args:
            operation: 操作名称，如 "query", "ingest", "retrieve", "embed".
            duration_ms: 耗时（毫秒）.
        """
        self._timings[operation].append(duration_ms)

    def timeit(self, operation: str):
        """上下文管理器，自动记录耗时.

        用法::

            with metrics.timeit("query"):
                result = orchestrator.run_query(...)
        """
        return _TimeitContext(self, operation)

    def get_timing_stats(self, operation: str | None = None) -> dict[str, Any]:
        """获取耗时统计.

        Args:
            operation: 可选操作名称过滤；None 返回所有操作.

        Returns:
            包含 count、mean、p50、p95、p99、min、max 的字典.
        """
        if operation:
            timings = self._timings.get(operation, [])
            return self._compute_stats(operation, timings)

        result = {}
        for op, timings in self._timings.items():
            result[op] = self._compute_stats(op, timings)
        return result

    def get_all_timing_stats(self) -> dict[str, Any]:
        """获取所有操作的耗时统计汇总."""
        return self.get_timing_stats()

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------

    def increment(self, counter: str, value: int = 1) -> None:
        """增加计数器.

        Args:
            counter: 计数器名称，如 "query.count", "error.count".
            value: 增加值，默认 1.
        """
        self._counters[counter] += value

    def get_counter(self, counter: str) -> int:
        """获取计数器当前值.

        Args:
            counter: 计数器名称.

        Returns:
            当前计数值.
        """
        return self._counters.get(counter, 0)

    def get_all_counters(self) -> dict[str, int]:
        """获取所有计数器."""
        return dict(self._counters)

    # ------------------------------------------------------------------
    # 汇总报告
    # ------------------------------------------------------------------

    def get_report(self) -> dict[str, Any]:
        """生成完整的监控报告."""
        return {
            "timings": self.get_all_timing_stats(),
            "counters": self.get_all_counters(),
        }

    def reset(self) -> None:
        """重置所有指标."""
        self._timings.clear()
        self._counters.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(name: str, timings: list[float]) -> dict[str, Any]:
        """计算一组时序数据的统计量."""
        if not timings:
            return {
                "operation": name,
                "count": 0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_t = sorted(timings)
        n = len(sorted_t)

        return {
            "operation": name,
            "count": n,
            "mean": round(statistics.mean(sorted_t), 2),
            "p50": round(sorted_t[int(n * 0.50)], 2),
            "p95": round(sorted_t[int(n * 0.95)], 2),
            "p99": round(sorted_t[int(n * 0.99)], 2),
            "min": round(sorted_t[0], 2),
            "max": round(sorted_t[-1], 2),
        }


class _TimeitContext:
    """计时上下文管理器."""

    def __init__(self, collector: MetricsCollector, operation: str) -> None:
        self._collector = collector
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "_TimeitContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        duration = (time.perf_counter() - self._start) * 1000  # ms
        self._collector.record_timing(self._operation, duration)