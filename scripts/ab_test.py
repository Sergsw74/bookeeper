#!/usr/bin/env python3
"""
A/B Testing Automation Runner for Bookeeper.

Performs automated end-to-end A/B testing:
1. Standard Mode: Checks out Branch A, runs `bookeeper test-run`, then checks out
   Branch B, runs `bookeeper test-run`, and compares results.
2. VS Mode: Ingests an existing baseline `verification_report.json` path as Branch A,
   checks out only Branch B to run `bookeeper test-run`, and compares results.

Usage:
    # 1. Standard Branch vs Branch Mode
    ./ab_test.py <branch1> <branch2> [--books 5] [--percent 1.0]

    # 2. VS Mode (Existing Baseline Report vs Candidate Branch)
    ./ab_test.py vs <baseline_report_path> <branch2> [--books 5] [--percent 1.0]
    ./ab_test.py --vs <baseline_report_path> <branch2> [--books 5] [--percent 1.0]
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Terminal ANSI styling helpers
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def log_info(msg: str) -> None:
    print(f"{Colors.CYAN}ℹ{Colors.RESET} {msg}")


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def log_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def log_header(msg: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== {msg} ==={Colors.RESET}\n")


class ABTestRunner:
    """Manages git checkouts, test execution, archiving, and statistical comparison."""

    def __init__(
        self,
        branch1: str,
        branch2: str,
        baseline_report: Optional[str] = None,
        num_books: int = 5,
        percent: float = 1.0,
        mode: str = "ideas",
        timeout: int = 60,
        repo_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        extra_args: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.branch2 = branch2
        self.num_books = num_books
        self.percent = percent
        self.mode = mode
        self.timeout = timeout
        self.extra_args = extra_args or ""
        self.dry_run = dry_run

        # Handle VS mode with pre-existing baseline report
        if baseline_report:
            self.baseline_report_path: Optional[Path] = Path(baseline_report).expanduser().resolve()
            if not self.dry_run and not self.baseline_report_path.is_file():
                raise FileNotFoundError(f"Baseline report not found at {self.baseline_report_path}")
            self.is_vs_mode = True
            # Clean descriptive title for baseline
            stem = self.baseline_report_path.stem
            parent_name = self.baseline_report_path.parent.name
            desc_tag = parent_name if parent_name not in ("output", ".") else stem
            self.branch1 = branch1 or f"Report ({desc_tag})"
        else:
            self.baseline_report_path = None
            self.is_vs_mode = False
            self.branch1 = branch1

        # Resolve repository directory
        if repo_dir:
            self.repo_dir = Path(repo_dir).expanduser().resolve()
        else:
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    self.repo_dir = Path(res.stdout.strip()).resolve()
                else:
                    self.repo_dir = Path("/home/ubuntu/repos/bookeeper").resolve()
            except Exception:
                self.repo_dir = Path("/home/ubuntu/repos/bookeeper").resolve()

        if not (self.repo_dir / ".git").is_dir():
            raise FileNotFoundError(f"Directory {self.repo_dir} is not a valid git repository.")

        # Resolve bookeeper executable
        venv_bin = self.repo_dir / ".venv" / "bin" / "bookeeper"
        if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
            self.bookeeper_cmd = str(venv_bin)
        else:
            self.bookeeper_cmd = "bookeeper"

        # Resolve archive directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_archive = Path(output_dir).expanduser().resolve() if output_dir else self.repo_dir / "ab_test_runs"
        mode_prefix = "vs" if self.is_vs_mode else "run"
        self.session_dir = base_archive / f"{mode_prefix}_{timestamp}_{self._sanitize(self.branch1)}_vs_{self._sanitize(self.branch2)}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.initial_branch: Optional[str] = None

    @staticmethod
    def _sanitize(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        cwd = cwd or self.repo_dir
        return subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=True, check=check)

    def check_repo_clean(self) -> bool:
        """Verify working directory has no uncommitted changes."""
        res = self._run_cmd(["git", "status", "--porcelain"])
        return len(res.stdout.strip()) == 0

    def get_current_branch(self) -> str:
        """Retrieve current active git branch name."""
        res = self._run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return res.stdout.strip()

    def checkout_branch(self, branch: str) -> None:
        """Check out requested git branch."""
        log_info(f"Checking out branch: {Colors.BOLD}{branch}{Colors.RESET}...")
        self._run_cmd(["git", "checkout", branch])
        cur = self.get_current_branch()
        if cur != branch:
            raise RuntimeError(f"Failed to switch branch. Expected '{branch}', currently on '{cur}'.")
        log_success(f"Successfully checked out {Colors.BOLD}{branch}{Colors.RESET}")

    def import_baseline_report(self, dest_dir: Path) -> Dict[str, Any]:
        """Import pre-existing baseline verification report (VS mode)."""
        log_header(f"Importing Baseline Verification Report: {self.branch1}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_report = dest_dir / "verification_report.json"
        copied_artifacts = ["verification_report.json"]

        if self.dry_run and (not self.baseline_report_path or not self.baseline_report_path.is_file()):
            # Synthetic report for dry run
            mock_data = {
                "stats": {
                    "total_ideas_in_graph": 120,
                    "candidate_ideas_with_chunks": 115,
                    "sampled_ideas": 15,
                    "sample_percentage": self.percent,
                    "mode": self.mode,
                    "model_name": "baseline-model",
                    "total_evaluations": 15,
                    "verified_count": 12,
                    "discrepancy_count": 3,
                    "verified_percentage": 80.0,
                    "discrepancy_percentage": 20.0,
                    "total_duration_seconds": 15.0,
                },
                "discrepancies": [
                    {
                        "idea_name": "Baseline Discrepancy",
                        "idea_weight": 5,
                        "book_title": "Baseline Book",
                        "section_title": "Chapter 1",
                        "chunk_id": "b1_c1_p1",
                        "explanation": "Baseline discrepancy sample explanation.",
                        "is_supported": False,
                        "confidence": 0.9,
                    }
                ],
                "verified_samples": [],
            }
            with open(dest_report, "w", encoding="utf-8") as f:
                json.dump(mock_data, f, indent=2)
            src_str = "synthetic_baseline_report.json"
            dur = 15.0
            tot_ideas = 120
            v_rate = 80.0
        else:
            assert self.baseline_report_path is not None
            shutil.copy2(self.baseline_report_path, dest_report)
            src_str = str(self.baseline_report_path)

            # Check for adjacent knowledge_graph.json
            nearby_kg = self.baseline_report_path.parent / "knowledge_graph.json"
            if nearby_kg.is_file():
                shutil.copy2(nearby_kg, dest_dir / "knowledge_graph.json")
                copied_artifacts.append("knowledge_graph.json")

            rep_data = self._load_json(dest_report) or {}
            st = rep_data.get("stats", {})
            dur = float(st.get("total_duration_seconds", 0.0))
            tot_ideas = st.get("total_ideas_in_graph", 0)
            v_rate = st.get("verified_percentage", 0.0)

        meta = {
            "branch": self.branch1,
            "source_file": src_str,
            "exit_code": 0,
            "duration_seconds": dur,
            "timestamp": datetime.datetime.now().isoformat(),
            "is_imported_baseline": True,
            "copied_artifacts": copied_artifacts,
        }
        with open(dest_dir / "run_meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        log_success(
            f"Baseline report loaded ({tot_ideas} ideas in graph, {v_rate:.2f}% pass rate)."
        )
        log_info(f"Copied artifacts to: {dest_dir}")
        return meta

    def run_test_run(self, branch: str, branch_dir: Path) -> Dict[str, Any]:
        """Execute test-run for the currently checked-out branch and archive results."""
        log_header(f"Running Test-Run on Branch: {branch}")
        branch_dir.mkdir(parents=True, exist_ok=True)

        log_file = branch_dir / "test_run.log"
        meta_file = branch_dir / "run_meta.json"

        # Build CLI command
        cmd = [
            self.bookeeper_cmd,
            "test-run",
            str(self.num_books),
            "--percent",
            str(self.percent),
            "--mode",
            str(self.mode),
            "--timeout",
            str(self.timeout),
        ]
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        log_info(f"Executing command: {Colors.DIM}{' '.join(cmd)}{Colors.RESET}")
        t0 = time.time()
        exit_code = 0
        error_msg = None

        if self.dry_run:
            log_warning("Dry-run mode enabled: generating mock verification report.")
            time.sleep(0.5)
            mock_report = {
                "stats": {
                    "total_ideas_in_graph": 155,
                    "candidate_ideas_with_chunks": 150,
                    "sampled_ideas": 15,
                    "sample_percentage": self.percent,
                    "mode": self.mode,
                    "model_name": "candidate-model",
                    "total_evaluations": 15,
                    "verified_count": 14,
                    "discrepancy_count": 1,
                    "verified_percentage": 93.33,
                    "discrepancy_percentage": 6.67,
                    "total_duration_seconds": 12.0,
                },
                "discrepancies": [
                    {
                        "idea_name": "Candidate Discrepancy Sample",
                        "idea_weight": 7,
                        "book_title": "Candidate Book",
                        "section_title": "Chapter 2",
                        "chunk_id": "b1_c2_p1",
                        "explanation": "Slight nuance mismatch with chunk text.",
                        "is_supported": False,
                        "confidence": 0.85,
                    }
                ],
                "verified_samples": [],
            }
            with open(branch_dir / "verification_report.json", "w", encoding="utf-8") as f:
                json.dump(mock_report, f, indent=2)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Dry-run executed successfully for {branch}\n")
        else:
            with open(log_file, "w", encoding="utf-8") as lf:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=self.repo_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in proc.stdout:  # type: ignore
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        lf.write(line)
                    proc.wait()
                    exit_code = proc.returncode
                except Exception as exc:
                    exit_code = -1
                    error_msg = str(exc)
                    lf.write(f"\nExecution failed with error: {exc}\n")

        duration = time.time() - t0

        # Archive artifacts from repos/bookeeper/output
        repo_output_dir = self.repo_dir / "output"
        copied_artifacts = []

        for fname in ["verification_report.json", "knowledge_graph.json", ".bookeeper_state.json"]:
            src = repo_output_dir / fname
            if src.is_file():
                dest = branch_dir / fname
                shutil.copy2(src, dest)
                copied_artifacts.append(fname)

        run_meta = {
            "branch": branch,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.datetime.now().isoformat(),
            "copied_artifacts": copied_artifacts,
            "error": error_msg,
        }
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(run_meta, mf, indent=2)

        if exit_code != 0:
            log_error(f"test-run exited with code {exit_code} on branch '{branch}'.")
        else:
            log_success(f"test-run completed on branch '{branch}' in {duration:.1f}s.")
            log_info(f"Archived artifacts: {', '.join(copied_artifacts) if copied_artifacts else 'none'}")

        return run_meta

    def execute_flow(self) -> Path:
        """Run complete A/B workflow across both baseline and candidate."""
        log_header("A/B Testing Pipeline Initialized")
        if self.is_vs_mode:
            print(f"• Mode:             {Colors.BOLD}{Colors.YELLOW}VS MODE (Report vs Branch){Colors.RESET}")
            print(f"• Baseline Report:  {Colors.BOLD}{self.baseline_report_path}{Colors.RESET}")
        else:
            print(f"• Mode:             {Colors.BOLD}Standard Branch vs Branch{Colors.RESET}")
            print(f"• Branch A:         {Colors.BOLD}{self.branch1}{Colors.RESET}")
        print(f"• Branch B:         {Colors.BOLD}{self.branch2}{Colors.RESET}")
        print(f"• Target Books:     {Colors.BOLD}{self.num_books}{Colors.RESET}")
        print(f"• Verification %:   {Colors.BOLD}{self.percent}%{Colors.RESET} (mode: {self.mode})")
        print(f"• Request Timeout:  {self.timeout}s")
        print(f"• Repository Path:  {self.repo_dir}")
        print(f"• Output Archive:   {self.session_dir}\n")

        # 1. Guard git clean state
        if not self.dry_run and not self.check_repo_clean():
            raise RuntimeError(
                f"Repository at {self.repo_dir} has unstaged or uncommitted changes. "
                "Please commit or stash changes before running A/B testing."
            )

        self.initial_branch = self.get_current_branch()
        log_info(f"Original active branch: {Colors.BOLD}{self.initial_branch}{Colors.RESET}")

        branch1_dir = self.session_dir / f"branch_A_{self._sanitize(self.branch1)}"
        branch2_dir = self.session_dir / f"branch_B_{self._sanitize(self.branch2)}"

        try:
            # 2. Process Baseline (Branch A or imported report)
            if self.is_vs_mode:
                self.import_baseline_report(branch1_dir)
            else:
                if not self.dry_run:
                    self.checkout_branch(self.branch1)
                self.run_test_run(self.branch1, branch1_dir)

            # 3. Process Candidate (Branch B)
            if not self.dry_run:
                self.checkout_branch(self.branch2)
            self.run_test_run(self.branch2, branch2_dir)

        finally:
            # 4. Always restore initial branch
            if not self.dry_run and self.initial_branch:
                log_info(f"Restoring initial branch: {Colors.BOLD}{self.initial_branch}{Colors.RESET}...")
                try:
                    self.checkout_branch(self.initial_branch)
                except Exception as e:
                    log_warning(f"Could not restore initial branch: {e}")

        # 5. Compare verification results
        self.compare_results(branch1_dir, branch2_dir)
        return self.session_dir

    def compare_results(self, dir_a: Path, dir_b: Path) -> Dict[str, Any]:
        """Compute side-by-side comparison between the two verification reports."""
        log_header("Verification A/B Comparison Analysis")

        report_a_file = dir_a / "verification_report.json"
        report_b_file = dir_b / "verification_report.json"
        meta_a_file = dir_a / "run_meta.json"
        meta_b_file = dir_b / "run_meta.json"

        data_a = self._load_json(report_a_file)
        data_b = self._load_json(report_b_file)
        meta_a = self._load_json(meta_a_file) or {}
        meta_b = self._load_json(meta_b_file) or {}

        if not data_a:
            log_error(f"Missing verification report for Baseline ({self.branch1}) at {report_a_file}")
        if not data_b:
            log_error(f"Missing verification report for Branch B ({self.branch2}) at {report_b_file}")

        stats_a = data_a.get("stats", {}) if data_a else {}
        stats_b = data_b.get("stats", {}) if data_b else {}
        disc_a = data_a.get("discrepancies", []) if data_a else []
        disc_b = data_b.get("discrepancies", []) if data_b else []

        # Graph files comparison (if present)
        graph_a = self._load_json(dir_a / "knowledge_graph.json")
        graph_b = self._load_json(dir_b / "knowledge_graph.json")

        def _count_graph(g: Optional[Dict[str, Any]]) -> Tuple[int, int]:
            if not g:
                return 0, 0
            nodes = len(g.get("nodes", []))
            edges = len(g.get("edges", []))
            return nodes, edges

        nodes_a, edges_a = _count_graph(graph_a)
        nodes_b, edges_b = _count_graph(graph_b)

        # Build comparison metrics table
        metrics = [
            (
                "Total Ideas in Graph",
                stats_a.get("total_ideas_in_graph", 0),
                stats_b.get("total_ideas_in_graph", 0),
                "count",
            ),
            (
                "Candidate Ideas (with Chunks)",
                stats_a.get("candidate_ideas_with_chunks", 0),
                stats_b.get("candidate_ideas_with_chunks", 0),
                "count",
            ),
            (
                "Sampled Ideas Audited",
                stats_a.get("sampled_ideas", 0),
                stats_b.get("sampled_ideas", 0),
                "count",
            ),
            (
                "Verified Ideas (Factual Pass)",
                stats_a.get("verified_count", 0),
                stats_b.get("verified_count", 0),
                "count",
            ),
            (
                "Verification Pass Rate (%)",
                stats_a.get("verified_percentage", 0.0),
                stats_b.get("verified_percentage", 0.0),
                "percent",
            ),
            (
                "Discrepancies (Hallucination Fail)",
                stats_a.get("discrepancy_count", 0),
                stats_b.get("discrepancy_count", 0),
                "count_lower_is_better",
            ),
            (
                "Discrepancy Rate (%)",
                stats_a.get("discrepancy_percentage", 0.0),
                stats_b.get("discrepancy_percentage", 0.0),
                "percent_lower_is_better",
            ),
            (
                "Graph Nodes / Edges",
                f"{nodes_a} / {edges_a}",
                f"{nodes_b} / {edges_b}",
                "raw",
            ),
            (
                "End-to-End Pipeline Duration (s)",
                meta_a.get("duration_seconds", 0.0),
                meta_b.get("duration_seconds", 0.0),
                "duration",
            ),
            (
                "Verification Phase Duration (s)",
                stats_a.get("total_duration_seconds", 0.0),
                stats_b.get("total_duration_seconds", 0.0),
                "duration",
            ),
        ]

        # Terminal output formatting
        col_w_metric = 36
        col_w_val = 24
        col_w_delta = 22

        lbl_a = f"BASELINE ({self.branch1})"
        lbl_b = f"CANDIDATE ({self.branch2})"
        header_line = (
            f"{'METRIC':<{col_w_metric}} "
            f"{lbl_a:<{col_w_val}} "
            f"{lbl_b:<{col_w_val}} "
            f"{'DELTA (B - A)':<{col_w_delta}}"
        )
        sep_line = "=" * (col_w_metric + col_w_val + col_w_val + col_w_delta + 3)
        sub_sep = "-" * len(sep_line)

        print(sep_line)
        print(f"{Colors.BOLD}{header_line}{Colors.RESET}")
        print(sub_sep)

        summary_data: List[Dict[str, Any]] = []

        for name, val_a, val_b, m_type in metrics:
            delta_str = ""
            color = Colors.RESET

            if m_type in ("count", "duration", "count_lower_is_better"):
                try:
                    f_a = float(val_a)
                    f_b = float(val_b)
                    diff = f_b - f_a
                    pct_diff = (diff / f_a * 100.0) if f_a > 0 else 0.0
                    sign = "+" if diff > 0 else ""

                    if m_type == "count_lower_is_better" or m_type == "duration":
                        color = Colors.GREEN if diff < 0 else (Colors.RED if diff > 0 else Colors.RESET)
                    else:
                        color = Colors.GREEN if diff > 0 else (Colors.RED if diff < 0 else Colors.RESET)

                    delta_str = f"{sign}{diff:.2f} ({sign}{pct_diff:.1f}%)" if "." in f"{diff}" else f"{sign}{int(diff)}"
                except (ValueError, TypeError):
                    delta_str = "-"

            elif m_type in ("percent", "percent_lower_is_better"):
                try:
                    f_a = float(val_a)
                    f_b = float(val_b)
                    diff = f_b - f_a
                    sign = "+" if diff > 0 else ""
                    if m_type == "percent_lower_is_better":
                        color = Colors.GREEN if diff < 0 else (Colors.RED if diff > 0 else Colors.RESET)
                    else:
                        color = Colors.GREEN if diff > 0 else (Colors.RED if diff < 0 else Colors.RESET)
                    delta_str = f"{sign}{diff:.2f}% pts"
                except (ValueError, TypeError):
                    delta_str = "-"
            else:
                delta_str = "-"

            str_a = f"{val_a:.2f}" if isinstance(val_a, float) else str(val_a)
            str_b = f"{val_b:.2f}" if isinstance(val_b, float) else str(val_b)

            print(
                f"{name:<{col_w_metric}} "
                f"{str_a:<{col_w_val}} "
                f"{str_b:<{col_w_val}} "
                f"{color}{delta_str:<{col_w_delta}}{Colors.RESET}"
            )

            summary_data.append(
                {
                    "metric": name,
                    "branch_a": val_a,
                    "branch_b": val_b,
                    "delta": delta_str,
                }
            )

        print(sep_line)

        # Print Discrepancy Sample Comparison
        if disc_a or disc_b:
            print(f"\n{Colors.BOLD}Discrepancy Details Comparison:{Colors.RESET}")
            print(f"• Baseline ({self.branch1}) Discrepancies: {len(disc_a)}")
            for idx, d in enumerate(disc_a[:3], 1):
                name = d.get("idea_name") or d.get("chunk_id", "Unknown")
                exp = d.get("explanation", "No explanation")
                print(f"   [{idx}] {Colors.YELLOW}{name}{Colors.RESET}: {exp[:100]}")
            if len(disc_a) > 3:
                print(f"   ... and {len(disc_a) - 3} more.")

            print(f"• Candidate ({self.branch2}) Discrepancies: {len(disc_b)}")
            for idx, d in enumerate(disc_b[:3], 1):
                name = d.get("idea_name") or d.get("chunk_id", "Unknown")
                exp = d.get("explanation", "No explanation")
                print(f"   [{idx}] {Colors.YELLOW}{name}{Colors.RESET}: {exp[:100]}")
            if len(disc_b) > 3:
                print(f"   ... and {len(disc_b) - 3} more.")
            print()

        # Generate summary verdict
        rate_a = float(stats_a.get("verified_percentage", 0.0))
        rate_b = float(stats_b.get("verified_percentage", 0.0))
        rate_diff = rate_b - rate_a

        if rate_diff > 0.01:
            verdict = (
                f"Candidate branch '{self.branch2}' outperformed baseline '{self.branch1}' with a "
                f"+{rate_diff:.2f}% higher factual verification rate."
            )
            v_color = Colors.GREEN
        elif rate_diff < -0.01:
            verdict = (
                f"Baseline '{self.branch1}' had a higher factual verification rate "
                f"(+{abs(rate_diff):.2f}% over candidate '{self.branch2}')."
            )
            v_color = Colors.YELLOW
        else:
            verdict = f"Both configurations demonstrated identical factual verification rates ({rate_a:.2f}%)."
            v_color = Colors.CYAN

        print(f"{v_color}{Colors.BOLD}VERDICT: {verdict}{Colors.RESET}\n")

        # Save structured comparison JSON and Markdown report
        comparison_record = {
            "is_vs_mode": self.is_vs_mode,
            "baseline": self.branch1,
            "candidate": self.branch2,
            "baseline_report_path": str(self.baseline_report_path) if self.baseline_report_path else None,
            "num_books": self.num_books,
            "sample_percent": self.percent,
            "mode": self.mode,
            "verdict": verdict,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": summary_data,
            "discrepancies_count": {
                "baseline": len(disc_a),
                "candidate": len(disc_b),
            },
        }

        json_out = self.session_dir / "comparison_summary.json"
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(comparison_record, jf, indent=2, ensure_ascii=False)

        md_out = self.session_dir / "comparison_summary.md"
        self._write_markdown_report(md_out, comparison_record)

        log_success(f"Full comparison JSON saved to: {Colors.BOLD}{json_out}{Colors.RESET}")
        log_success(f"Full comparison Markdown report saved to: {Colors.BOLD}{md_out}{Colors.RESET}")

        return comparison_record

    def _write_markdown_report(self, dest: Path, data: Dict[str, Any]) -> None:
        lines = [
            "# A/B Testing Verification Comparison Report",
            "",
            f"- **Mode:** `{'VS Mode (Report vs Branch)' if data['is_vs_mode'] else 'Standard Branch vs Branch'}`",
            f"- **Baseline:** `{data['baseline']}`" + (f" (`{data['baseline_report_path']}`)" if data.get('baseline_report_path') else ""),
            f"- **Candidate:** `{data['candidate']}`",
            f"- **Target Books:** `{data['num_books']}`",
            f"- **Sample Rate:** `{data['sample_percent']}%` (Mode: `{data['mode']}`)",
            f"- **Generated:** `{data['timestamp']}`",
            "",
            f"### 🎯 Verdict",
            f"> **{data['verdict']}**",
            "",
            "### 📊 Comparative Metrics",
            "",
            "| Metric | Baseline | Candidate | Delta (Candidate - Baseline) |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for m in data.get("metrics", []):
            lines.append(f"| {m['metric']} | {m['branch_a']} | {m['branch_b']} | {m['delta']} |")

        lines.extend([
            "",
            "### 🔍 Discrepancies Summary",
            f"- **Baseline (`{data['baseline']}`):** {data['discrepancies_count']['baseline']} discrepancies",
            f"- **Candidate (`{data['candidate']}`):** {data['discrepancies_count']['candidate']} discrepancies",
            "",
        ])

        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _load_json(path: Path) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent A/B testing runner for Bookeeper branches or Report vs Branch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Standard Branch vs Branch Mode:
  ./ab_test.py master feature-model-adgustments
  ./ab_test.py master feature-model-adgustments --books 10 --percent 2.0

  # 2. VS Mode (Existing Baseline Report vs Candidate Branch):
  ./ab_test.py vs /path/to/verification_report.json feature-model-adgustments
  ./ab_test.py --vs /path/to/verification_report.json feature-model-adgustments
  ./ab_test.py /path/to/verification_report.json feature-model-adgustments
        """,
    )
    parser.add_argument(
        "arg1",
        help="Name of Branch A (baseline), OR mode keyword 'vs', OR path to baseline verification report.",
    )
    parser.add_argument(
        "arg2",
        nargs="?",
        default=None,
        help="Name of Branch B (candidate), OR path to baseline report if arg1 is 'vs'.",
    )
    parser.add_argument(
        "arg3",
        nargs="?",
        default=None,
        help="Name of Branch B if 'vs <report> <branch2>' format is used.",
    )
    parser.add_argument(
        "--vs",
        "--baseline-report",
        dest="vs_report",
        type=str,
        default=None,
        help="Path to pre-existing baseline verification_report.json (skips Branch A test-run, only tests Branch B).",
    )
    parser.add_argument(
        "--books",
        "-b",
        type=int,
        default=5,
        help="Number of books to process from scratch (default: 5).",
    )
    parser.add_argument(
        "--percent",
        "-p",
        type=float,
        default=1.0,
        help="Percentage of ideas/chunks to verify (default: 1.0 for 1%%).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="ideas",
        choices=["ideas", "chunking"],
        help="Verification mode: 'ideas' or 'chunking' (default: ideas).",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60,
        help="Ollama request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to Bookeeper git repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save archived results and comparison reports.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Optional additional arguments to pass to 'bookeeper test-run' (e.g. '--skip-warmup').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the workflow and produce comparison reports without running Ollama LLM queries.",
    )

    args = parser.parse_args()

    # Determine execution mode and target branch/report
    baseline_report: Optional[str] = None
    branch1: str = ""
    branch2: str = ""

    if args.vs_report:
        # Invoked with --vs <report> <branch2>
        baseline_report = args.vs_report
        branch2 = args.arg1
        branch1 = f"Report ({Path(baseline_report).name})"
    elif args.arg1.lower() == "vs":
        # Invoked with: vs <report> <branch2>
        if not args.arg2 or not args.arg3:
            parser.error("In 'vs' mode, specify baseline report path and candidate branch: 'vs <report_path> <branch2>'")
        baseline_report = args.arg2
        branch2 = args.arg3
        branch1 = f"Report ({Path(baseline_report).name})"
    elif os.path.isfile(os.path.expanduser(args.arg1)) or args.arg1.endswith(".json"):
        # Invoked with: <report.json> <branch2>
        if not args.arg2:
            parser.error("Specify candidate branch when providing baseline report: '<report_path> <branch2>'")
        baseline_report = args.arg1
        branch2 = args.arg2
        branch1 = f"Report ({Path(baseline_report).name})"
    else:
        # Standard: <branch1> <branch2>
        if not args.arg2:
            parser.error("Specify both branches: '<branch1> <branch2>' (or use 'vs <report_path> <branch2>')")
        branch1 = args.arg1
        branch2 = args.arg2

    try:
        runner = ABTestRunner(
            branch1=branch1,
            branch2=branch2,
            baseline_report=baseline_report,
            num_books=args.books,
            percent=args.percent,
            mode=args.mode,
            timeout=args.timeout,
            repo_dir=args.repo_dir,
            output_dir=args.output_dir,
            extra_args=args.extra_args,
            dry_run=args.dry_run,
        )
        runner.execute_flow()
    except KeyboardInterrupt:
        log_warning("\nExecution aborted by user.")
        sys.exit(130)
    except Exception as exc:
        log_error(f"A/B test execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
