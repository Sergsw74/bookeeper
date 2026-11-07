#!/usr/bin/env python3
"""
scripts/stamp_version.py
Helper utility to get, bump, or stamp version and release date across Bookeeper workspace files.
"""

import argparse
import datetime
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_current_version() -> str:
    init_file = REPO_ROOT / "src" / "bookeeper" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*(?:os\.getenv\([^,]+,\s*)?["\']([^"\']+)["\']', content)
        if m:
            return m.group(1)
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if m:
            return m.group(1)
    return "0.1.1"


def bump_patch(version: str) -> str:
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except Exception:
        return f"{version}.1"


def stamp_files(version: str, release_date: str) -> None:
    # 1. src/bookeeper/__init__.py
    init_file = REPO_ROOT / "src" / "bookeeper" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        content = re.sub(
            r'__version__\s*=\s*os\.getenv\([\'"]BOOKEEPER_VERSION[\'"],\s*[\'"][^\'"]+[\'"]\)',
            f'__version__ = os.getenv("BOOKEEPER_VERSION", "{version}")',
            content,
        )
        content = re.sub(
            r'__version__\s*=\s*[\'"][0-9.]+[\'"]',
            f'__version__ = os.getenv("BOOKEEPER_VERSION", "{version}")',
            content,
        )
        content = re.sub(
            r'__release_date__\s*=\s*os\.getenv\([\'"]BOOKEEPER_RELEASE_DATE[\'"]\)\s*or\s*os\.getenv\([\'"]BOOKEEPER_BUILD_DATE[\'"],\s*[\'"][^\'"]+[\'"]\)',
            f'__release_date__ = os.getenv("BOOKEEPER_RELEASE_DATE") or os.getenv("BOOKEEPER_BUILD_DATE", "{release_date}")',
            content,
        )
        content = re.sub(
            r'__release_date__\s*=\s*[\'"][0-9-]+[\'"]',
            f'__release_date__ = os.getenv("BOOKEEPER_RELEASE_DATE") or os.getenv("BOOKEEPER_BUILD_DATE", "{release_date}")',
            content,
        )
        init_file.write_text(content, encoding="utf-8")

    # 2. pyproject.toml
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        content = re.sub(r'version\s*=\s*["\'][^"\']+["\']', f'version = "{version}"', content, count=1)
        pyproject.write_text(content, encoding="utf-8")

    # 3. archive_downloader/pyproject.toml
    ad_pyproject = REPO_ROOT / "archive_downloader" / "pyproject.toml"
    if ad_pyproject.exists():
        content = ad_pyproject.read_text(encoding="utf-8")
        content = re.sub(r'version\s*=\s*["\'][^"\']+["\']', f'version = "{version}"', content, count=1)
        ad_pyproject.write_text(content, encoding="utf-8")

    # 4. archive_downloader/src/archive_downloader/__init__.py
    ad_init = REPO_ROOT / "archive_downloader" / "src" / "archive_downloader" / "__init__.py"
    if ad_init.exists():
        content = ad_init.read_text(encoding="utf-8")
        content = re.sub(r'__version__\s*=\s*["\'][^"\']+["\']', f'__version__ = "{version}"', content)
        ad_init.write_text(content, encoding="utf-8")

    # 5. src/bookeeper/ui/static/index.html
    html_file = REPO_ROOT / "src" / "bookeeper" / "ui" / "static" / "index.html"
    if html_file.exists():
        html = html_file.read_text(encoding="utf-8")
        html = re.sub(r'id="debug-sys-date"[^>]*>[^<]+<', f'id="debug-sys-date" style="color:#e2e8f0; font-family:monospace;">{release_date}<', html)
        html = re.sub(r'id="debug-sys-version"[^>]*>[^<]+<', f'id="debug-sys-version" style="color:#38bdf8; font-weight:600; font-family:monospace;">{version}<', html)
        html = re.sub(r'Release v[0-9.]+', f'Release v{version}', html)
        html = re.sub(r'<span>20[0-9]{2}-[0-9]{2}-[0-9]{2}</span>', f'<span>{release_date}</span>', html)
        html = re.sub(r'v[0-9.]+(?=</span>\s*<span style="color:#64748b;">•)', f'v{version}', html)
        html_file.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stamp or inspect Bookeeper workspace version & release date.")
    parser.add_argument("--get", action="store_true", help="Print current version")
    parser.add_argument("--bump", action="store_true", help="Bump patch version and stamp files")
    parser.add_argument("--version", type=str, default="", help="Set specific version")
    parser.add_argument("--date", type=str, default="", help="Set release date (default: today)")
    args = parser.parse_args()

    current_ver = get_current_version()

    if args.get:
        print(current_ver)
        return

    rel_date = args.date.strip() or datetime.date.today().isoformat()
    target_ver = current_ver

    if args.bump:
        target_ver = bump_patch(current_ver)
    elif args.version:
        target_ver = args.version.strip()

    stamp_files(target_ver, rel_date)
    print(f"Stamped Bookeeper: version={target_ver}, release_date={rel_date}")


if __name__ == "__main__":
    main()
