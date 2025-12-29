"""
Shared cache primitives for VoiceForge.

Keeps caching logic consistent across model caches.
"""

import time
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class TTLCache(Generic[K, V]):
    """A small, thread-safe TTL cache."""

    def __init__(self, ttl_seconds: float = 10.0):
        self._ttl = float(ttl_seconds)
        self._data: Dict[K, Tuple[float, V]] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: K) -> Optional[V]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return None
            ts, value = item
            if now - ts > self._ttl:
                self._data.pop(key, None)
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)

    def get_or_set(self, key: K, factory: Callable[[], V]) -> V:
        existing = self.get(key)
        if existing is not None:
            return existing
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def set_ttl(self, ttl_seconds: float) -> None:
        with self._lock:
            self._ttl = float(ttl_seconds)

