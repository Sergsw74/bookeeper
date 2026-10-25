"""
Memory management and consumption safety guard.
Monitors process resident set size (RSS) and enforces a maximum ceiling (default: 10 GB).
Provides memory threshold checks, proactive garbage collection, and dynamic chunk/batch sizing.
"""

import gc
import logging
import os
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# Default ceiling: 10 GB (configurable via BOOKEEPER_MEMORY_LIMIT_GB)
DEFAULT_MAX_MEMORY_BYTES = int(
    float(os.getenv("BOOKEEPER_MEMORY_LIMIT_GB", "10.0")) * 1024 * 1024 * 1024
)


def get_process_memory_bytes() -> int:
    """Return current process resident set size (RSS) in bytes."""
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            pass
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except Exception:
        return 0


def get_process_memory_gb() -> float:
    """Return current process resident set size (RSS) in gigabytes."""
    return get_process_memory_bytes() / (1024.0**3)


def check_memory_limit(
    limit_bytes: Optional[int] = None,
    context: str = "",
    raise_on_breach: bool = True,
) -> float:
    """
    Check current process memory consumption against limit (default 10 GB).
    If memory usage reaches 85% of limit, triggers proactive garbage collection.
    If memory usage exceeds the limit, raises MemoryError (or logs error if raise_on_breach=False).

    Returns:
        Current memory usage in GB.
    """
    ceiling = limit_bytes if limit_bytes is not None else DEFAULT_MAX_MEMORY_BYTES
    curr_bytes = get_process_memory_bytes()
    curr_gb = curr_bytes / (1024.0**3)
    limit_gb = ceiling / (1024.0**3)

    # 1. Proactive cleanup threshold (85% of limit, e.g. 8.5 GB)
    if curr_bytes >= int(0.85 * ceiling):
        gc.collect()
        curr_bytes = get_process_memory_bytes()
        curr_gb = curr_bytes / (1024.0**3)
        logger.warning(
            f"[MemoryGuard] High memory usage detected: {curr_gb:.2f} GB / {limit_gb:.1f} GB ({context}). "
            f"Proactive garbage collection triggered."
        )

    # 2. Hard limit breach (10 GB)
    if curr_bytes > ceiling:
        msg = (
            f"[MemoryGuard] Process memory consumption ({curr_gb:.2f} GB) exceeded the maximum "
            f"safety ceiling of {limit_gb:.1f} GB during {context}."
        )
        logger.error(msg)
        if raise_on_breach:
            raise MemoryError(msg)

    return curr_gb


def cleanup_memory() -> None:
    """Force an immediate garbage collection to release unreferenced data blocks."""
    gc.collect()
