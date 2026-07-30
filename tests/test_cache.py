"""Tests for QueryCache — LRU eviction, TTL expiration."""

from __future__ import annotations

import time


from knowledge_agent.cache import QueryCache


class TestQueryCache:
    """Tests for the LRU query result cache."""

    def test_set_and_get(self):
        cache = QueryCache(ttl=300, max_size=10)
        cache.set("key1", {"answer": "test"})
        result = cache.get("key1")
        assert result is not None
        assert result["answer"] == "test"

    def test_missing_key_returns_none(self):
        cache = QueryCache()
        assert cache.get("nonexistent") is None

    def test_expired_ttl(self):
        cache = QueryCache(ttl=0.1, max_size=10)
        cache.set("key1", {"answer": "test"})
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        cache = QueryCache(ttl=300, max_size=3)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.set("c", {"v": 3})
        cache.set("d", {"v": 4})  # 超过上限，a 应被淘汰
        assert cache.get("a") is None
        assert cache.get("d") is not None

    def test_lru_reorders_on_access(self):
        cache = QueryCache(ttl=300, max_size=3)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.set("c", {"v": 3})
        cache.get("a")  # 访问 a，a 移到末尾
        cache.set("d", {"v": 4})  # 淘汰 b（最久未使用）
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_invalidate(self):
        cache = QueryCache(ttl=300, max_size=10)
        cache.set("key1", {"answer": "test"})
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = QueryCache(ttl=300, max_size=10)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_size_property(self):
        cache = QueryCache(ttl=300, max_size=10)
        assert cache.size == 0
        cache.set("a", {"v": 1})
        assert cache.size == 1

    def test_keys_property(self):
        cache = QueryCache(ttl=300, max_size=10)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        assert set(cache.keys) == {"a", "b"}

    def test_zero_ttl_expires_immediately(self):
        cache = QueryCache(ttl=0, max_size=10)
        cache.set("key1", {"answer": "test"})
        time.sleep(0.01)
        assert cache.get("key1") is None
