"""
Unit tests for multi-verification support in config.yaml and CLI commands.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import Settings, SingleVerifierConfig


def test_single_verifier_config_properties():
    """Verify SingleVerifierConfig default values and resolved properties."""
    v = SingleVerifierConfig(mode="ideas")
    assert v.mode == "ideas"
    assert v.enabled is True
    assert v.percent is None
    assert v.resolved_distance_threshold is None

    v2 = SingleVerifierConfig(mode="chunking", threshold=0.15)
    assert v2.resolved_distance_threshold == 0.15

    v3 = SingleVerifierConfig(mode="chunking", distance_threshold=0.22)
    assert v3.resolved_distance_threshold == 0.22

    # Threshold overrides distance_threshold if both specified
    v4 = SingleVerifierConfig(mode="chunking", distance_threshold=0.22, threshold=0.18, custom_param="custom_val")
    assert v4.resolved_distance_threshold == 0.18
    assert v4.custom_param == "custom_val"


def test_settings_verifiers_sync_from_yaml(tmp_path: Path):
    """Verify that verifiers array in YAML initializes Settings.verifiers and provides default Settings.verification."""
    cfg_data = {
        "verifiers": [
            {
                "mode": "ideas",
                "model": "gemma4:12b",
                "percent": 2.6,
                "temperature": 0.0,
                "max_examples": 15,
                "cooldown_seconds": 12,
            },
            {
                "mode": "chunking",
                "distance_threshold": 0.18,
                "max_examples": 10,
            },
        ]
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    settings = Settings.from_yaml(cfg_file)

    # Verifiers array populated
    assert len(settings.verifiers) == 2
    assert settings.verifiers[0].mode == "ideas"
    assert settings.verifiers[0].percent == 2.6
    assert settings.verifiers[0].model == "gemma4:12b"
    assert settings.verifiers[1].mode == "chunking"
    assert settings.verifiers[1].resolved_distance_threshold == 0.18

    # Default verification settings inherited from first verifier
    assert settings.verification.mode == "ideas"
    assert settings.verification.model == "gemma4:12b"
    assert settings.verification.percent == 2.6
    assert settings.verification.temperature == 0.0
    assert settings.verification.max_examples == 15
    assert settings.verification.cooldown_seconds == 12
    assert settings.resolved_verifier_model == "gemma4:12b"
    assert settings.resolved_verification_cooldown_seconds == 12


def test_settings_backward_compat_legacy_verification(tmp_path: Path):
    """Verify backward compatibility: if only legacy verification block is provided, verifiers is auto-populated."""
    cfg_data = {
        "verification": {
            "enabled": True,
            "model": "llama3.1:8b",
            "percent": 3.5,
            "mode": "chunk",
            "temperature": 0.1,
            "max_examples": 25,
            "distance_threshold": 0.25,
            "cooldown_seconds": 45,
        }
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    settings = Settings.from_yaml(cfg_file)

    assert len(settings.verifiers) == 1
    v0 = settings.verifiers[0]
    assert v0.mode == "chunk"
    assert v0.model == "llama3.1:8b"
    assert v0.percent == 3.5
    assert v0.temperature == 0.1
    assert v0.max_examples == 25
    assert v0.distance_threshold == 0.25
    assert v0.cooldown_seconds == 45


def test_settings_nested_verification_verifiers(tmp_path: Path):
    """Verify nested verification.verifiers in YAML is supported and synced."""
    cfg_data = {
        "verification": {
            "verifiers": [
                {"mode": "ideas", "percent": 1.5},
                {"mode": "chunking", "threshold": 0.22},
            ]
        }
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    settings = Settings.from_yaml(cfg_file)
    assert len(settings.verifiers) == 2
    assert settings.verifiers[0].mode == "ideas"
    assert settings.verifiers[0].percent == 1.5
    assert settings.verifiers[1].mode == "chunking"
    assert settings.verifiers[1].resolved_distance_threshold == 0.22
    assert settings.verification.mode == "ideas"
    assert settings.verification.percent == 1.5


def test_cli_verify_without_parameters_runs_multi_verification(tmp_path: Path):
    """Test that `bookeeper verify` without mode runs all configured verifiers and saves combined report."""
    cfg_data = {
        "output_dir": str(tmp_path / "output"),
        "verifiers": [
            {"mode": "ideas", "percent": 2.0},
            {"mode": "chunking", "distance_threshold": 0.15},
        ],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    runner = CliRunner()

    with patch("bookeeper.cli._execute_single_verification") as mock_exec:
        mock_exec.side_effect = [
            {"mode": "ideas", "stats": {"verified_ideas": 10}},
            {"mode": "chunking", "stats": {"coherent_expansions": 20}},
        ]

        result = runner.invoke(app, ["verify", "--config", str(cfg_file)])
        assert result.exit_code == 0
        assert mock_exec.call_count == 2

        # Check call arguments for each verifier
        call_0 = mock_exec.call_args_list[0].kwargs
        assert call_0["mode"] == "ideas"
        assert call_0["percent"] == 2.0

        call_1 = mock_exec.call_args_list[1].kwargs
        assert call_1["mode"] == "chunking"
        assert call_1["threshold"] == 0.15

        # Check that combined report was written
        combined_report_file = tmp_path / "output" / "verification_report.json"
        assert combined_report_file.is_file()
        combined_data = json.loads(combined_report_file.read_text(encoding="utf-8"))
        assert "multi_verification" in combined_data
        assert len(combined_data["multi_verification"]) == 2
        assert combined_data["multi_verification"][0]["mode"] == "ideas"
        assert combined_data["multi_verification"][1]["mode"] == "chunking"


def test_cli_verifiers_alias_command(tmp_path: Path):
    """Test that `bookeeper verifiers` is recognized and executes multi-verification."""
    cfg_data = {
        "output_dir": str(tmp_path / "output"),
        "verifiers": [
            {"mode": "ideas", "percent": 1.0},
            {"mode": "chunking", "distance_threshold": 0.20},
        ],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    runner = CliRunner()

    with patch("bookeeper.cli._execute_single_verification") as mock_exec:
        mock_exec.side_effect = [
            {"mode": "ideas", "stats": {}},
            {"mode": "chunking", "stats": {}},
        ]

        result = runner.invoke(app, ["verifiers", "--config", str(cfg_file)])
        assert result.exit_code == 0
        assert mock_exec.call_count == 2


def test_cli_verify_with_explicit_mode_runs_only_single(tmp_path: Path):
    """Test that when an explicit --mode is passed, only that single verifier is executed."""
    cfg_data = {
        "output_dir": str(tmp_path / "output"),
        "verifiers": [
            {"mode": "ideas", "percent": 2.0},
            {"mode": "chunking", "distance_threshold": 0.15},
        ],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    runner = CliRunner()

    with patch("bookeeper.cli._execute_single_verification") as mock_exec:
        mock_exec.return_value = {"mode": "chunking", "stats": {}}

        result = runner.invoke(app, ["verify", "--config", str(cfg_file), "--mode", "chunking"])
        assert result.exit_code == 0
        assert mock_exec.call_count == 1
        assert mock_exec.call_args.kwargs["mode"] == "chunking"


def test_cli_test_run_auto_triggers_multi_verification(tmp_path: Path):
    """Test that `bookeeper test-run` automatically triggers multi-verification in stage 2 when mode is not specified."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_data = {
        "output_dir": str(out_dir),
        "verifiers": [
            {"mode": "ideas", "percent": 2.0},
            {"mode": "chunking", "distance_threshold": 0.15},
        ],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    # Create dummy knowledge_graph.json with 1 Book node so validation passes
    kg_file = out_dir / "knowledge_graph.json"
    dummy_kg = {
        "nodes": [{"id": "book:1", "type": "Book", "title": "Test Book", "author": "Author"}],
        "edges": [],
    }
    kg_file.write_text(json.dumps(dummy_kg), encoding="utf-8")

    runner = CliRunner()

    with patch("bookeeper.cli.build_graph") as mock_build, patch(
        "bookeeper.cli._execute_single_verification"
    ) as mock_exec:
        mock_exec.side_effect = [
            {"mode": "ideas", "stats": {}},
            {"mode": "chunking", "stats": {}},
        ]

        result = runner.invoke(app, ["test-run", "1", "--config", str(cfg_file)])
        assert result.exit_code == 0
        assert mock_build.call_count == 1
        # Multi-verification was automatically triggered: 2 verifiers executed
        assert mock_exec.call_count == 2
        assert mock_exec.call_args_list[0].kwargs["mode"] == "ideas"
        assert mock_exec.call_args_list[1].kwargs["mode"] == "chunking"
