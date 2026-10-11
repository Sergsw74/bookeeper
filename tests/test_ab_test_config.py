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


def test_runner_resolves_config_percent_and_mode(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cfg = {
        "request_timeout": 45,
        "verification": {
            "percent": 0.3,
            "mode": "chunk",
        },
    }
    with open(repo_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 1. When omitted, inherits 0.3%, 'chunk', 45s from config.yaml
    runner_default = ABTestRunner(
        branch1="main",
        branch2="feat",
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_default.percent == 0.3
    assert runner_default.mode == "chunk"
    assert runner_default.timeout == 45

    # 2. When overridden with explicit parameters, honors explicit parameters
    runner_explicit = ABTestRunner(
        branch1="main",
        branch2="feat",
        percent=2.5,
        mode="ideas",
        timeout=90,
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_explicit.percent == 2.5
    assert runner_explicit.mode == "ideas"
    assert runner_explicit.timeout == 90


def test_runner_same_branch_resolution(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # 1. No branches provided: defaults to active branch (e.g. master) for both
    runner_no_branch = ABTestRunner(
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_no_branch.is_same_branch is True
    assert runner_no_branch.branch1 == runner_no_branch.branch2
    assert "cfgA_vs_cfgB" in runner_no_branch.session_dir.name

    # 2. Single branch provided: sets branch2 equal to branch1
    runner_single_branch = ABTestRunner(
        branch1="feature-eval",
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_single_branch.is_same_branch is True
    assert runner_single_branch.branch1 == "feature-eval"
    assert runner_single_branch.branch2 == "feature-eval"
    assert "feature-eval_cfgA_vs_cfgB" in runner_single_branch.session_dir.name

    # 3. Explicit same_branch=True with different branch2 argument: forces same branch
    runner_forced_same = ABTestRunner(
        branch1="main",
        branch2="other",
        same_branch=True,
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_forced_same.is_same_branch is True
    assert runner_forced_same.branch1 == "main"
    assert runner_forced_same.branch2 == "main"

    # 4. Two different branches: standard multi-branch mode
    runner_two_branches = ABTestRunner(
        branch1="main",
        branch2="feature-x",
        repo_dir=str(repo_dir),
        dry_run=True,
    )
    assert runner_two_branches.is_same_branch is False
    assert runner_two_branches.branch1 == "main"
    assert runner_two_branches.branch2 == "feature-x"
    assert "main_vs_feature-x" in runner_two_branches.session_dir.name


def test_runner_same_branch_execute_flow_dry_run(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    base_cfg = {"llm_model": "base-model", "verification": {"percent": 1.0}}
    with open(repo_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f)

    overlay_b = {"llm_model": "model-b-custom"}
    with open(repo_dir / "config-b.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(overlay_b, f)

    out_dir = tmp_path / "ab_test_runs"
    runner = ABTestRunner(
        branch1="main",
        repo_dir=str(repo_dir),
        output_dir=str(out_dir),
        dry_run=True,
    )
    session_dir = runner.execute_flow()
    assert session_dir.is_dir()

    # Check generated files
    sum_json = session_dir / "comparison_summary.json"
    sum_md = session_dir / "comparison_summary.md"
    assert sum_json.is_file()
    assert sum_md.is_file()

    import json
    with open(sum_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["is_same_branch"] is True
    assert data["baseline"] == "main"
    assert data["candidate"] == "main"

    with open(sum_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "Same-Branch Config Comparison" in md_text
    assert "Case A (Config A)" in md_text
    assert "Case B (Config B)" in md_text


def test_ab_test_cli_same_branch_invocations(tmp_path: Path):
    import subprocess
    script_path = Path(__file__).parent.parent / "scripts" / "ab_test.py"
    repo_dir = Path(__file__).parent.parent

    # Test 1: No branch argument
    res = subprocess.run(
        [sys.executable, str(script_path), "--dry-run", "--output-dir", str(tmp_path / "out1")],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Same-Branch Configuration Comparison" in res.stdout
    assert "CASE A" in res.stdout
    assert "CASE B" in res.stdout

    # Test 2: Single branch argument
    res = subprocess.run(
        [sys.executable, str(script_path), "master", "--dry-run", "--output-dir", str(tmp_path / "out2")],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Same-Branch Configuration Comparison" in res.stdout
    assert "Target Branch:" in res.stdout
    assert "master" in res.stdout

    # Test 3: Number of books without branch (e.g. '3 1.0')
    res = subprocess.run(
        [sys.executable, str(script_path), "3", "1.0", "--dry-run", "--output-dir", str(tmp_path / "out3")],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Same-Branch Configuration Comparison" in res.stdout
    assert "Target Books:" in res.stdout
    assert "3" in res.stdout

    # Test 4: --same-branch flag
    res = subprocess.run(
        [sys.executable, str(script_path), "--same-branch", "--dry-run", "--output-dir", str(tmp_path / "out4")],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Same-Branch Configuration Comparison" in res.stdout


def test_ab_test_runs_verifiers_from_config_yaml_and_cross_check(tmp_path: Path):
    """Test that A/B test runs all verifiers from config.yaml, auto-executes cross-check, and generates ab_test_summary.md with timing stats."""
    import json
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cfg = {
        "llm_model": "test-model",
        "verifiers": [
            {"mode": "ideas", "percent": 1.0},
            {"mode": "chunk", "percent": 10.0},
        ],
    }
    with open(repo_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    out_dir = tmp_path / "ab_test_runs"
    runner = ABTestRunner(
        branch1="main",
        repo_dir=str(repo_dir),
        output_dir=str(out_dir),
        dry_run=True,
        blocks=5,
    )
    session_dir = runner.execute_flow()
    assert session_dir.is_dir()

    # 1. Cross-check report generated at end of verification
    cc_json = session_dir / "cross_check_report.json"
    cc_md = session_dir / "cross_check_report.md"
    assert cc_json.is_file()
    assert cc_md.is_file()

    # 2. Both summary markdown files generated
    sum_json = session_dir / "comparison_summary.json"
    sum_md = session_dir / "comparison_summary.md"
    ab_md = session_dir / "ab_test_summary.md"
    assert sum_json.is_file()
    assert sum_md.is_file()
    assert ab_md.is_file()

    # 3. Verify comparison_summary.json structure
    with open(sum_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Timing metrics
    assert "timing" in data
    timing = data["timing"]
    assert "baseline" in timing and "candidate" in timing
    assert timing["baseline"]["build_duration_seconds"] > 0
    assert timing["baseline"]["verification_duration_seconds"] > 0
    assert timing["baseline"]["pipeline_duration_seconds"] > 0
    assert timing["cross_check_duration_seconds"] >= 0
    assert timing["total_overall_duration_seconds"] > 0

    # Cross-check summary included
    assert "cross_check" in data
    assert data["cross_check"]["total_samples"] == 5
    assert "mean_delta_recall" in data["cross_check"]
    assert "total_old_supported" in data["cross_check"]

    # Verifiers breakdown included for multi-verifier config
    assert "verifiers_breakdown" in data
    assert "ideas" in data["verifiers_breakdown"]
    assert "chunk" in data["verifiers_breakdown"]

    # 4. Verify markdown summary contents
    md_content = ab_md.read_text(encoding="utf-8")
    assert "Overall Verdict" in md_content
    assert "Execution Timing & Performance" in md_content
    assert "Knowledge Graph Build Duration" in md_content
    assert "Verification Phase Duration" in md_content
    assert "End-to-End Pipeline Duration" in md_content
    assert "Cross-Check Comparative Audit" in md_content
    assert "Supported Assertions Count (+1.0)" in md_content
    assert "Not Mentioned / Omission Count (-0.25)" in md_content
    assert "Contradicted / Hallucination Count (-1.0)" in md_content
    assert "Mean Net Semantic Integrity" in md_content
    assert "Mean Calibrated Fidelity" in md_content


def test_ab_test_skip_cross_check_flag(tmp_path: Path):
    """Test that --skip-cross-check skips cross check audit generation."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cfg = {"llm_model": "test-model", "verification": {"percent": 1.0}}
    with open(repo_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    out_dir = tmp_path / "ab_test_runs"
    runner = ABTestRunner(
        branch1="main",
        repo_dir=str(repo_dir),
        output_dir=str(out_dir),
        dry_run=True,
        skip_cross_check=True,
    )
    session_dir = runner.execute_flow()
    assert session_dir.is_dir()

    # Cross-check should NOT be generated
    assert not (session_dir / "cross_check_report.json").is_file()
    # Summary should still be generated
    assert (session_dir / "ab_test_summary.md").is_file()



