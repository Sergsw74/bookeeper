"""
Tests for JEPA weight downloader, backup restore, and OCI image reference parsing.
"""

from pathlib import Path
import pytest
from bookeeper.evaluation.jepa_downloader import parse_image_ref, ensure_jepa_weights


def test_parse_image_ref():
    scheme, reg, repo, tag = parse_image_ref("192.168.50.20:5000/bk-jepa-ai:latest")
    assert scheme == "http"
    assert reg == "192.168.50.20:5000"
    assert repo == "bk-jepa-ai"
    assert tag == "latest"

    scheme, reg, repo, tag = parse_image_ref("http://192.168.50.20:5000/bk-jepa-ai:0.1.2")
    assert scheme == "http"
    assert reg == "192.168.50.20:5000"
    assert repo == "bk-jepa-ai"
    assert tag == "0.1.2"

    scheme, reg, repo, tag = parse_image_ref("bk-jepa-ai:v1", default_registry="192.168.50.20:5000")
    assert scheme == "http"
    assert reg == "192.168.50.20:5000"
    assert repo == "bk-jepa-ai"
    assert tag == "v1"


def test_ensure_jepa_weights_from_existing(tmp_path):
    pt_file = tmp_path / "jepa_head_weights.pt"
    npz_file = tmp_path / "jepa_768_weights.npz"

    pt_file.write_bytes(b"mock_pt_weights")
    npz_file.write_bytes(b"mock_npz_weights")

    res_pt, res_npz = ensure_jepa_weights(
        output_dir=tmp_path,
        auto_download=False,
    )
    assert res_pt == pt_file
    assert res_npz == npz_file
