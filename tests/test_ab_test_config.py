import pytest
import yaml
from pathlib import Path
import sys

# Ensure scripts dir is on sys.path to import ABTestRunner
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ab_test import ABTestRunner


def test_ab_test_runner_deep_merge_dicts():
    base = {
        "llm_model": "qwen2.5:3b",
        "failover_cooldown_seconds": 600,
        "verification": {
            "enabled": True,
            "model": "llama3.1:8b",
            "percent": 1.0,
            "max_examples": 20,
        },
        "servers": ["srv1", "srv2"],
    }
    overlay = {
        "llm_model": "llama3.1:8b",
        "verification_cooldown_seconds": 45,
        "verification": {
            "percent": 5.0,
            "cooldown_seconds": 30,
        },
    }

    merged = ABTestRunner.deep_merge_dicts(base, overlay)

    # Overwritten top-level
    assert merged["llm_model"] == "llama3.1:8b"
    # Preserved top-level
    assert merged["failover_cooldown_seconds"] == 600
    assert merged["servers"] == ["srv1", "srv2"]
    # Added top-level from overlay
    assert merged["verification_cooldown_seconds"] == 45
    # Nested dict merged
    assert merged["verification"]["enabled"] is True
    assert merged["verification"]["model"] == "llama3.1:8b"
    assert merged["verification"]["max_examples"] == 20
    assert merged["verification"]["percent"] == 5.0
    assert merged["verification"]["cooldown_seconds"] == 30


def test_prepare_branch_config_auto_detect_and_overlay(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    base_config = {
        "calibre_library_path": "/path/to/calibre",
        "failover_cooldown_seconds": 600,
        "verification_cooldown_seconds": 60,
        "verification": {
            "enabled": True,
            "model": "llama3.1:8b",
            "percent": 1.0,
        },
    }
    with open(repo_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_config, f)

    # config-a overlay
    config_a = {
        "llm_model": "model-a",
        "verification": {
            "percent": 2.5,
        },
    }
    with open(repo_dir / "config-a.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_a, f)

    # config-b overlay
    config_b = {
        "llm_model": "model-b",
        "verification_cooldown_seconds": 15,
    }
    with open(repo_dir / "config-b.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_b, f)

    runner = ABTestRunner(
        branch1="main",
        branch2="feature",
        repo_dir=str(repo_dir),
        dry_run=True,
    )

    # Branch A
    branch_a_dir = tmp_path / "run" / "branch_A"
    branch_a_dir.mkdir(parents=True)
    res_a = runner.prepare_branch_config("A", branch_a_dir)
    assert res_a is not None
    assert res_a.is_file()

    with open(res_a, "r", encoding="utf-8") as f:
        merged_a = yaml.safe_load(f)

    assert merged_a["llm_model"] == "model-a"
    assert merged_a["calibre_library_path"] == "/path/to/calibre"
    assert merged_a["verification"]["enabled"] is True
    assert merged_a["verification"]["percent"] == 2.5

    # Branch B
    branch_b_dir = tmp_path / "run" / "branch_B"
    branch_b_dir.mkdir(parents=True)
    res_b = runner.prepare_branch_config("B", branch_b_dir)
    assert res_b is not None
    assert res_b.is_file()

    with open(res_b, "r", encoding="utf-8") as f:
        merged_b = yaml.safe_load(f)

    assert merged_b["llm_model"] == "model-b"
    assert merged_b["calibre_library_path"] == "/path/to/calibre"
    assert merged_b["verification_cooldown_seconds"] == 15
    assert merged_b["verification"]["percent"] == 1.0


def test_prepare_branch_config_explicit_flags(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    base_cfg_path = tmp_path / "custom_base.yaml"
    with open(base_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"base_key": "base_val", "shared": 1}, f)

    overlay_a_path = tmp_path / "custom_overlay_a.yaml"
    with open(overlay_a_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"shared": 10, "extra_a": True}, f)

    runner = ABTestRunner(
        branch1="main",
        branch2="feature",
        repo_dir=str(repo_dir),
        base_config=str(base_cfg_path),
        config_a=str(overlay_a_path),
        dry_run=True,
    )

    branch_a_dir = tmp_path / "run_branch_A"
    branch_a_dir.mkdir()
    res_a = runner.prepare_branch_config("A", branch_a_dir)
    assert res_a is not None

    with open(res_a, "r", encoding="utf-8") as f:
        merged = yaml.safe_load(f)

    assert merged["base_key"] == "base_val"
    assert merged["shared"] == 10
    assert merged["extra_a"] is True
