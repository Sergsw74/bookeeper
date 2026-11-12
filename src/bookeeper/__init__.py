"""
bookeeper: Calibre library ingester, LLM metadata normalizer, and Concept Knowledge Graph extractor.
"""

import os
from bookeeper.config import Settings, get_settings

__version__ = os.getenv("BOOKEEPER_VERSION", "0.1.2")
__release_date__ = os.getenv("BOOKEEPER_RELEASE_DATE") or os.getenv("BOOKEEPER_BUILD_DATE", "2026-11-09")
__all__ = ["Settings", "get_settings", "__version__", "__release_date__"]
