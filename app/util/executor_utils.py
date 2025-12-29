"""
Shared ThreadPoolExecutor management.

Eliminates duplicate executor creation patterns across routers.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


_executor: Optional[ThreadPoolExecutor] = None


def get_shared_executor() -> ThreadPoolExecutor:
    """
    Get or create a shared ThreadPoolExecutor for blocking operations.
    
    Uses MAX_WORKERS environment variable or CPU count.
    """
    global _executor
    if _executor is None:
        max_workers = int(os.getenv("MAX_WORKERS", os.cpu_count() or 4))
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


def shutdown_executor():
    """Shutdown the shared executor (for cleanup)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None

