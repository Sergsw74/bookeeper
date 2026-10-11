#!/usr/bin/env python3
"""
Convenience launcher for Archive Downloader.
Usage:
    python run.py download --limit 100
    python run.py status
    python run.py inspect
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from archive_downloader.cli import app

if __name__ == "__main__":
    app()
