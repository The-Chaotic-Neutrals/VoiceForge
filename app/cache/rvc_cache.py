# cache/rvc_cache.py
"""RVC model caching for efficient reuse (local RVC only)."""

import os
import time
import threading
import gc
import logging
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if local RVC is available
try:
    from rvc_loader import get_loader as get_rvc_loader, is_available as is_rvc_local_available
    RVC_LOCAL_AVAILABLE = is_rvc_local_available()
except ImportError:
    RVC_LOCAL_AVAILABLE = False

    def get_rvc_loader(only_cpu=False):
        raise ImportError("Local RVC dependencies not installed")


class RVCModelCache:
    """Thread-safe cache for RVC model loaders with LRU eviction."""

    def __init__(self, max_size: int = 3):
        self._cache: Dict[str, Any] = {}  # BaseLoader instances
        self._last_used: Dict[str, float] = {}  # LRU tracking
        self._lock = threading.Lock()
        self._max_size = max_size

    def get_loader(self, model_name: str, only_cpu: bool = False) -> Any:
        """
        Get or create a loader for the specified model.

        NOTE: This requires local RVC dependencies. If you use the RVC microservice,
        you do NOT need this cache.
        """
        if not RVC_LOCAL_AVAILABLE:
            raise ImportError(
                "Local RVC dependencies not installed. "
                "RVC cache requires local dependencies. "
                "Consider using the RVC server instead."
            )

        with self._lock:
            if model_name in self._cache:
                self._last_used[model_name] = time.time()
                return self._cache[model_name]

            if len(self._cache) >= self._max_size:
                self._evict_lru()

            loader = get_rvc_loader(only_cpu=only_cpu)
            self._cache[model_name] = loader
            self._last_used[model_name] = time.time()
            return loader

    def _evict_lru(self):
        if not self._cache:
            return
        lru_model = min(self._last_used.items(), key=lambda x: x[1])[0]
        self._unload_model(lru_model)

    def _unload_model(self, model_name: str):
        if model_name not in self._cache:
            return

        loader = self._cache.pop(model_name)
        self._last_used.pop(model_name, None)

        try:
            loader.unload_models()
        except Exception as e:
            logger.warning(f"Error unloading {model_name}: {e}")

        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, model_name: str) -> bool:
        with self._lock:
            if model_name in self._cache:
                self._unload_model(model_name)
                return True
            return False

    def clear(self):
        with self._lock:
            for model_name in list(self._cache.keys()):
                self._unload_model(model_name)

    def remove(self, model_name: str):
        self.unload(model_name)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def max_size(self) -> int:
        return self._max_size

    def get_loaded_models(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())

    def get_lru_order(self) -> List[Tuple[str, float]]:
        with self._lock:
            return sorted(self._last_used.items(), key=lambda x: x[1])

    def record_use(self, model_name: str):
        with self._lock:
            if model_name in self._cache:
                self._last_used[model_name] = time.time()

    @staticmethod
    def is_available() -> bool:
        return RVC_LOCAL_AVAILABLE


_global_cache: Optional[RVCModelCache] = None
_cache_lock = threading.Lock()


def get_rvc_model_cache(max_size: Optional[int] = None) -> RVCModelCache:
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                size = max_size or int(os.getenv("RVC_CACHE_MAX_SIZE", "3"))
                _global_cache = RVCModelCache(max_size=size)
    return _global_cache

