"""
jepa_downloader.py
==================
Utilities for resolving, restoring from backups, and pulling JEPA model weights
(e.g., jepa_head_weights.pt, jepa_768_weights.npz) from local backup archives
or Docker/OCI image registries (such as 192.168.50.20:5000/bk-jepa-ai:latest).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Standard filenames for JEPA weight artifacts
JEPA_PT_WEIGHTS = "jepa_head_weights.pt"
JEPA_NPZ_WEIGHTS = "jepa_768_weights.npz"


def parse_image_ref(image_ref: str, default_registry: str = "192.168.50.20:5000") -> Tuple[str, str, str, str]:
    """
    Parse an OCI/Docker image reference into (scheme, registry, repository, tag).
    Supports:
      - 192.168.50.20:5000/bk-jepa-ai:latest
      - http://192.168.50.20:5000/bk-jepa-ai:latest
      - bk-jepa-ai:latest (defaults to default_registry)
      - registry.example.com/org/repo:tag
    """
    scheme = ""
    ref = image_ref.strip()

    if ref.startswith("http://"):
        scheme = "http"
        ref = ref[7:]
    elif ref.startswith("https://"):
        scheme = "https"
        ref = ref[8:]

    parts = ref.split("/", 1)
    if len(parts) == 1:
        # No registry prefix, use default_registry
        registry = default_registry
        repo_and_tag = parts[0]
    else:
        # First part might be a registry if it contains '.' or ':' or 'localhost'
        first, rest = parts[0], parts[1]
        if "." in first or ":" in first or first == "localhost":
            registry = first
            repo_and_tag = rest
        else:
            registry = default_registry
            repo_and_tag = f"{first}/{rest}"

    if ":" in repo_and_tag:
        repository, tag = repo_and_tag.rsplit(":", 1)
    else:
        repository, tag = repo_and_tag, "latest"

    if not scheme:
        # Private LAN IPs, port 5000, or localhost default to plain HTTP
        if re.match(r"^(localhost|127\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$", registry):
            scheme = "http"
        elif ":" in registry and not registry.endswith(":443"):
            scheme = "http"
        else:
            scheme = "https"

    return scheme, registry, repository, tag


def download_weights_from_oci_registry(
    image_ref: str,
    output_dir: Union[str, Path] = "output",
    timeout: int = 60,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download JEPA weight files directly from an OCI/Docker Registry V2 endpoint.
    Streams layers without requiring Docker daemon or Docker CLI.
    """
    scheme, registry, repo, tag = parse_image_ref(image_ref)
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    extracted: Dict[str, Path] = {}
    target_pt = out_path / JEPA_PT_WEIGHTS
    target_npz = out_path / JEPA_NPZ_WEIGHTS

    if not force and target_pt.is_file() and target_npz.is_file():
        logger.info(f"JEPA weight files already exist in {out_path}; skipping download.")
        return {JEPA_PT_WEIGHTS: target_pt, JEPA_NPZ_WEIGHTS: target_npz}

    accept_headers = (
        "application/vnd.oci.image.index.v1+json, "
        "application/vnd.docker.distribution.manifest.list.v2+json, "
        "application/vnd.oci.image.manifest.v1+json, "
        "application/vnd.docker.distribution.manifest.v2+json"
    )

    manifest_url = f"{scheme}://{registry}/v2/{repo}/manifests/{tag}"
    logger.info(f"Fetching OCI manifest from {manifest_url}...")

    req = urllib.request.Request(manifest_url, headers={"Accept": accept_headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # If HTTPS failed and no explicit scheme was set, attempt HTTP fallback
        if scheme == "https":
            logger.debug(f"HTTPS connection failed ({e}), attempting fallback to HTTP...")
            scheme = "http"
            manifest_url = f"http://{registry}/v2/{repo}/manifests/{tag}"
            req = urllib.request.Request(manifest_url, headers={"Accept": accept_headers})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        else:
            raise RuntimeError(f"Failed to fetch manifest from {manifest_url}: {e}") from e

    # Resolve manifest if manifest list / index was returned
    if "manifests" in data:
        # Select first available manifest (architecture agnostic data weights)
        child_digest = data["manifests"][0]["digest"]
        logger.info(f"Resolved OCI index, picking child manifest {child_digest}...")
        c_url = f"{scheme}://{registry}/v2/{repo}/manifests/{child_digest}"
        c_req = urllib.request.Request(c_url, headers={"Accept": accept_headers})
        with urllib.request.urlopen(c_req, timeout=timeout) as c_resp:
            data = json.loads(c_resp.read().decode("utf-8"))

    layers = data.get("layers", [])
    if not layers:
        raise RuntimeError(f"No layers found in image manifest for {image_ref}")

    # Inspect layers in reverse (weight layer was created last)
    for layer in reversed(layers):
        digest = layer.get("digest")
        if not digest:
            continue

        blob_url = f"{scheme}://{registry}/v2/{repo}/blobs/{digest}"
        logger.info(f"Downloading image layer blob {digest[:19]}...")
        b_req = urllib.request.Request(blob_url)
        with urllib.request.urlopen(b_req, timeout=timeout) as b_resp:
            blob = b_resp.read()

        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
                for member in tar.getmembers():
                    base_name = Path(member.name).name
                    if base_name in (JEPA_PT_WEIGHTS, JEPA_NPZ_WEIGHTS):
                        dest_file = out_path / base_name
                        if not force and dest_file.is_file():
                            extracted[base_name] = dest_file
                            continue

                        extracted_file = tar.extractfile(member)
                        if extracted_file:
                            with open(dest_file, "wb") as f_out:
                                shutil.copyfileobj(extracted_file, f_out)
                            extracted[base_name] = dest_file
                            logger.info(f"Extracted {base_name} ({member.size:,} bytes) -> {dest_file}")
        except Exception as err:
            logger.debug(f"Layer {digest[:12]} was not a readable tarball or failed: {err}")

        if JEPA_PT_WEIGHTS in extracted and JEPA_NPZ_WEIGHTS in extracted:
            break

    return extracted


def restore_weights_from_backups(
    search_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path] = "output",
    force: bool = False,
) -> Dict[str, Path]:
    """
    Search local backup archives (.zip or .7z) for JEPA weight files and restore them to output_dir.
    """
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    extracted: Dict[str, Path] = {}
    target_pt = out_path / JEPA_PT_WEIGHTS
    target_npz = out_path / JEPA_NPZ_WEIGHTS

    if not force and target_pt.is_file() and target_npz.is_file():
        return {JEPA_PT_WEIGHTS: target_pt, JEPA_NPZ_WEIGHTS: target_npz}

    # Collect backup archives from provided search directories
    candidate_archives: List[Path] = []
    for s_dir in search_dirs:
        p = Path(s_dir).expanduser().resolve()
        if not p.is_dir():
            if p.is_file() and p.suffix.lower() in (".zip", ".7z"):
                candidate_archives.append(p)
            continue

        for ext in ("*.zip", "*.7z"):
            candidate_archives.extend(p.glob(ext))
            candidate_archives.extend((p / "backups").glob(ext))

    # Sort archives newest first
    candidate_archives = sorted(set(candidate_archives), key=lambda x: x.stat().st_mtime, reverse=True)

    for arch in candidate_archives:
        ext = arch.suffix.lower()
        if ext == ".zip":
            try:
                with zipfile.ZipFile(arch, "r") as zf:
                    for name in zf.namelist():
                        base = Path(name).name
                        if base in (JEPA_PT_WEIGHTS, JEPA_NPZ_WEIGHTS):
                            dest_file = out_path / base
                            if not force and dest_file.is_file():
                                extracted[base] = dest_file
                                continue

                            with zf.open(name) as src, open(dest_file, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            extracted[base] = dest_file
                            logger.info(f"Restored {base} from zip backup {arch.name} -> {dest_file}")
            except Exception as e:
                logger.debug(f"Could not inspect zip backup {arch}: {e}")

        elif ext == ".7z":
            try:
                import py7zr

                with py7zr.SevenZipFile(arch, mode="r") as sz:
                    all_names = sz.getnames()
                    targets_in_7z = [n for n in all_names if Path(n).name in (JEPA_PT_WEIGHTS, JEPA_NPZ_WEIGHTS)]
                    if targets_in_7z:
                        sz.extract(path=str(out_path), targets=targets_in_7z)
                        for n in targets_in_7z:
                            base = Path(n).name
                            extracted_path = out_path / n
                            dest_file = out_path / base
                            if extracted_path != dest_file and extracted_path.is_file():
                                shutil.move(str(extracted_path), str(dest_file))
                            extracted[base] = dest_file
                            logger.info(f"Restored {base} from 7z backup {arch.name} -> {dest_file}")
            except Exception as e:
                logger.debug(f"Could not inspect 7z backup {arch}: {e}")

        if JEPA_PT_WEIGHTS in extracted and JEPA_NPZ_WEIGHTS in extracted:
            break

    return extracted


def ensure_jepa_weights(
    output_dir: Union[str, Path] = "output",
    pt_weights_path: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    weights_image: Optional[str] = None,
    backup_search_dirs: Optional[List[Union[str, Path]]] = None,
    auto_download: bool = True,
    force: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Ensure JEPA weights exist using resolution cascade:
      1. Existing files in output_dir (or specified pt_weights_path / weights_path)
      2. Restore from existing backup archives (*.zip, *.7z) in output/backups or workspace
      3. Existing local fallback folders (models/, /app/models/)
      4. Auto-download from OCI/Docker image (e.g., 192.168.50.20:5000/bk-jepa-ai:latest)
    """
    out_dir = Path(output_dir).expanduser().resolve()
    target_pt = Path(pt_weights_path) if pt_weights_path else out_dir / JEPA_PT_WEIGHTS
    target_npz = Path(weights_path) if weights_path else out_dir / JEPA_NPZ_WEIGHTS

    # Step 1: Check existing files
    if not force and target_pt.is_file():
        logger.debug(f"Found existing JEPA PyTorch checkpoint at {target_pt}")
        return target_pt, (target_npz if target_npz.is_file() else None)

    # Step 2: Check backup archives
    search_dirs = list(backup_search_dirs or [])
    search_dirs.extend([out_dir, out_dir / "backups", Path(".")])
    restored = restore_weights_from_backups(search_dirs, output_dir=out_dir, force=force)
    if target_pt.is_file():
        logger.info(f"Successfully restored JEPA weights from backup into {out_dir}")
        return target_pt, (target_npz if target_npz.is_file() else None)

    # Step 3: Check local model fallbacks
    for fallback_dir in (Path("models"), Path("/app/models")):
        fb_pt = fallback_dir / JEPA_PT_WEIGHTS
        fb_npz = fallback_dir / JEPA_NPZ_WEIGHTS
        if fb_pt.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            if not target_pt.is_file():
                shutil.copy2(fb_pt, target_pt)
            if fb_npz.is_file() and not target_npz.is_file():
                shutil.copy2(fb_npz, target_npz)
            logger.info(f"Copied JEPA weights from fallback {fallback_dir} -> {out_dir}")
            return target_pt, (target_npz if target_npz.is_file() else None)

    # Step 4: Download from OCI weights image
    image = weights_image or os.getenv("BOOKEEPER_JEPA_WEIGHTS_IMAGE") or "192.168.50.20:5000/bk-jepa-ai:latest"
    if auto_download and image:
        try:
            logger.info(f"Weights missing. Auto-downloading JEPA weights from OCI image {image}...")
            downloaded = download_weights_from_oci_registry(image, output_dir=out_dir, force=force)
            if target_pt.is_file():
                logger.info(f"Successfully downloaded JEPA weights from {image} into {out_dir}")
                return target_pt, (target_npz if target_npz.is_file() else None)
        except Exception as e:
            logger.warning(f"Could not auto-download weights from OCI image {image}: {e}")

    return (target_pt if target_pt.is_file() else None), (target_npz if target_npz.is_file() else None)
