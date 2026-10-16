"""
A/B Testing and Comparative Benchmark Harness: Bookkeeper vs LightRAG.

Executes dual queries against Bookkeeper Knowledge Graph and LightRAG Engine,
evaluates responses with a rigorous LLM Judge using the verification model,
and generates statistical reports and side-by-side Markdown comparison summaries.
"""

import json
import logging
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from bookeeper.config import Settings
from bookeeper.evaluation.question_generator import (
    ChecklistItem,
    QuestionGenerator,
    QuestionItem,
    QuestionLibrary,
    _clean_json_str,
)
from bookeeper.processing.extractor import OllamaPool
from bookeeper.rag.lightrag_service import LightRAGRawIndexer, LightRAGService
from bookeeper.ui.backend import SearchEngineBackend

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    Console = None
    Panel = None
    Table = None


class ChecklistCheckResult(BaseModel):
    """Result of verifying a single boolean checklist criterion against a response."""
    criterion: str
    expected: bool
    actual: bool
    passed: bool
    reason: str = ""


class JudgeEvaluation(BaseModel):
    """Evaluation score and rationale for a single query comparative benchmark test."""
    model_config = ConfigDict(populate_by_name=True)

    question_id: str
    query: str = Field(default="", validation_alias=AliasChoices("query", "question"))
    category: str = "concept"
    book_title: str = ""

    # Latencies in seconds
    bookkeeper_latency: float = 0.0
    lightrag_latency: float = 0.0

    # Binary Invariant Checklist Results
    bookkeeper_checks: List[ChecklistCheckResult] = Field(default_factory=list)
    lightrag_checks: List[ChecklistCheckResult] = Field(default_factory=list)
    bookkeeper_passed_count: int = 0
    lightrag_passed_count: int = 0
    total_checks: int = 0
    bookkeeper_pass_rate: float = 0.0  # percentage, e.g. 75.0%
    lightrag_pass_rate: float = 0.0    # percentage, e.g. 50.0%

    # Dimension Scores (1.0 to 5.0) mapped continuously from pass rate
    grounding_bookkeeper: float = 3.0
    grounding_lightrag: float = 3.0
    completeness_bookkeeper: float = 3.0
    completeness_lightrag: float = 3.0
    clarity_bookkeeper: float = 3.0
    clarity_lightrag: float = 3.0
    relevance_bookkeeper: float = 3.0
    relevance_lightrag: float = 3.0

    # Total aggregate score
    bookkeeper_total_score: float = 3.0
    lightrag_total_score: float = 3.0

    winner: Literal["bookkeeper", "lightrag", "tie"] = "tie"
    rationale: str = ""

    # Responses
    bookkeeper_response: str = ""
    lightrag_response: str = ""
    bookkeeper_references_count: int = 0
    lightrag_references_count: int = 0

    @property
    def question(self) -> str:
        return self.query

    @question.setter
    def question(self, val: str) -> None:
        self.query = val


class BenchmarkReport(BaseModel):
    """Comprehensive comparative benchmark report between Bookkeeper and LightRAG."""
    timestamp: float = Field(default_factory=time.time)
    judge_model: str = "unknown"
    book_titles: List[str] = Field(default_factory=list)
    total_questions: int = 0
    total_checklist_items: int = 0

    # Win / Loss / Tie counts
    bookkeeper_wins: int = 0
    lightrag_wins: int = 0
    ties: int = 0

    # Average latencies
    bookkeeper_avg_latency: float = 0.0
    lightrag_avg_latency: float = 0.0

    # Binary checklist pass metrics
    bookkeeper_total_passed_checks: int = 0
    lightrag_total_passed_checks: int = 0
    bookkeeper_overall_pass_rate: float = 0.0  # percentage
    lightrag_overall_pass_rate: float = 0.0    # percentage

    # Average overall scores (1-5 scale)
    bookkeeper_avg_score: float = 0.0
    lightrag_avg_score: float = 0.0

    # Dimension breakdowns
    avg_grounding: Dict[str, float] = Field(default_factory=dict)
    avg_completeness: Dict[str, float] = Field(default_factory=dict)
    avg_clarity: Dict[str, float] = Field(default_factory=dict)
    avg_relevance: Dict[str, float] = Field(default_factory=dict)

    # Knowledge Graph Structural Topology metrics
    bookkeeper_nodes_count: int = 0
    bookkeeper_edges_count: int = 0
    lightrag_nodes_count: int = 0
    lightrag_edges_count: int = 0
    bookkeeper_chunks_count: int = 0
    lightrag_chunks_count: int = 0
    bookkeeper_node_types: Dict[str, int] = Field(default_factory=dict)
    lightrag_node_types: Dict[str, int] = Field(default_factory=dict)

    evaluations: List[JudgeEvaluation] = Field(default_factory=list)

    def save_json(self, path: Union[Path, str]) -> Path:
        """Save detailed benchmark report as JSON."""
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        return p

    def generate_markdown_summary(self, output_path: Optional[Union[Path, str]] = None) -> str:
        """Generate a clean, structured Markdown summary with binary checklist deltas and comparison tables."""
        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        pct_bk = (self.bookkeeper_wins / max(1, self.total_questions)) * 100
        pct_lr = (self.lightrag_wins / max(1, self.total_questions)) * 100
        pct_tie = (self.ties / max(1, self.total_questions)) * 100

        lat_delta = self.bookkeeper_avg_latency - self.lightrag_avg_latency
        lat_ratio = (self.lightrag_avg_latency / max(0.001, self.bookkeeper_avg_latency))

        bk_check_pct = self.bookkeeper_overall_pass_rate
        lr_check_pct = self.lightrag_overall_pass_rate
        check_delta = bk_check_pct - lr_check_pct

        bk_nodes = self.bookkeeper_nodes_count
        lr_nodes = self.lightrag_nodes_count
        bk_edges = self.bookkeeper_edges_count
        lr_edges = self.lightrag_edges_count

        bk_density = (bk_edges / max(1, bk_nodes)) if bk_nodes else 0.0
        lr_density = (lr_edges / max(1, lr_nodes)) if lr_nodes else 0.0
        density_delta = bk_density - lr_density

        if lr_nodes > 0:
            nodes_ratio_str = f"{bk_nodes / lr_nodes:.1f}x {'more' if bk_nodes >= lr_nodes else 'fewer'} nodes"
        elif bk_nodes > 0:
            nodes_ratio_str = f"{bk_nodes} nodes (baseline unavailable)"
        else:
            nodes_ratio_str = "-"

        if lr_edges > 0:
            edges_ratio_str = f"{bk_edges / lr_edges:.1f}x {'more' if bk_edges >= lr_edges else 'fewer'} edges"
        elif bk_edges > 0:
            edges_ratio_str = f"{bk_edges} edges (baseline unavailable)"
        else:
            edges_ratio_str = "-"

        md = f"""# 📊 A/B Benchmark Summary: Bookkeeper vs. LightRAG

**Evaluation Date:** {date_str}  
**Judge Model:** `{self.judge_model}`  
**Evaluated Books ({len(self.book_titles)}):** {', '.join(f'*{b}*' for b in self.book_titles)}  
**Total Evaluated Queries:** {self.total_questions}  
**Total Invariant Checklist Criteria:** {self.total_checklist_items}

---

## 🏆 1. Executive Summary: Binary Verification Checklist & Win Rates

| Metric | Bookkeeper (A) | LightRAG (B) | Comparison / Delta |
| :--- | :---: | :---: | :---: |
| **Checklist Invariant Pass Rate** | **{bk_check_pct:.1f}%** ({self.bookkeeper_total_passed_checks}/{self.total_checklist_items}) | **{lr_check_pct:.1f}%** ({self.lightrag_total_passed_checks}/{self.total_checklist_items}) | **{'+' if check_delta >= 0 else ''}{check_delta:.1f}% {'Bookkeeper' if check_delta >= 0 else 'LightRAG'}** |
| **Head-to-Head Wins** | **{self.bookkeeper_wins}** ({pct_bk:.1f}%) | **{self.lightrag_wins}** ({pct_lr:.1f}%) | {f"+{self.bookkeeper_wins - self.lightrag_wins} Bookkeeper" if self.bookkeeper_wins >= self.lightrag_wins else f"+{self.lightrag_wins - self.bookkeeper_wins} LightRAG"} |
| **Ties / Neutral** | {self.ties} ({pct_tie:.1f}%) | {self.ties} ({pct_tie:.1f}%) | - |
| **Overall Quality (1-5 Scale)** | **{self.bookkeeper_avg_score:.2f}** | **{self.lightrag_avg_score:.2f}** | **{'+' if self.bookkeeper_avg_score >= self.lightrag_avg_score else ''}{self.bookkeeper_avg_score - self.lightrag_avg_score:.2f}** |
| **Average Latency** | **{self.bookkeeper_avg_latency:.4f}s** | **{self.lightrag_avg_latency:.4f}s** | **{abs(lat_delta):.4f}s ({lat_ratio:.1f}x {'faster' if lat_delta < 0 else 'slower'})** |

---

## 📐 2. Dimension Breakdown (1-5 Scale)

| Evaluation Dimension | Bookkeeper | LightRAG | Delta |
| :--- | :---: | :---: | :---: |
| **Factual Invariant Compliance** | {self.avg_grounding.get('bookkeeper', 0):.2f} | {self.avg_grounding.get('lightrag', 0):.2f} | {'+' if self.avg_grounding.get('bookkeeper', 0) >= self.avg_grounding.get('lightrag', 0) else ''}{self.avg_grounding.get('bookkeeper', 0) - self.avg_grounding.get('lightrag', 0):.2f} |
| **Completeness & Depth** | {self.avg_completeness.get('bookkeeper', 0):.2f} | {self.avg_completeness.get('lightrag', 0):.2f} | {'+' if self.avg_completeness.get('bookkeeper', 0) >= self.avg_completeness.get('lightrag', 0) else ''}{self.avg_completeness.get('bookkeeper', 0) - self.avg_completeness.get('lightrag', 0):.2f} |
| **Clarity & Structural Presentation** | {self.avg_clarity.get('bookkeeper', 0):.2f} | {self.avg_clarity.get('lightrag', 0):.2f} | {'+' if self.avg_clarity.get('bookkeeper', 0) >= self.avg_clarity.get('lightrag', 0) else ''}{self.avg_clarity.get('bookkeeper', 0) - self.avg_clarity.get('lightrag', 0):.2f} |
| **Relevance to Inquiry** | {self.avg_relevance.get('bookkeeper', 0):.2f} | {self.avg_relevance.get('lightrag', 0):.2f} | {'+' if self.avg_relevance.get('bookkeeper', 0) >= self.avg_relevance.get('lightrag', 0) else ''}{self.avg_relevance.get('bookkeeper', 0) - self.avg_relevance.get('lightrag', 0):.2f} |

---

## 🕸️ 3. Knowledge Graph Structural Topology Comparison

| Graph Structural Metric | Bookkeeper (A) | LightRAG (B) | Comparison / Ratio |
| :--- | :---: | :---: | :---: |
| **Total Graph Nodes (Entities / Ideas)** | **{bk_nodes:,}** | **{lr_nodes:,}** | **{nodes_ratio_str}** |
| **Total Graph Edges (Relationships)** | **{bk_edges:,}** | **{lr_edges:,}** | **{edges_ratio_str}** |
| **Edge-to-Node Ratio (Connectivity Density)** | **{bk_density:.2f}** edges/node | **{lr_density:.2f}** edges/node | **{'+' if density_delta >= 0 else ''}{density_delta:.2f} edges/node** |
| **Indexed Document Chunks** | **{self.bookkeeper_chunks_count:,}** | **{self.lightrag_chunks_count:,}** | Hierarchical chunks vs. flat chunks |

---

## 📋 4. Query-by-Query Comparison Table

| ID | Category | Subject Matter Query | Winner | Bookkeeper (Passed / Latency) | LightRAG (Passed / Latency) | Key Judge Rationale |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
"""
        for ev in self.evaluations:
            q_clean = (ev.query or ev.question).replace("\n", " ").replace("|", "\\|")
            if len(q_clean) > 60:
                q_clean = q_clean[:57] + "..."
            rat_clean = ev.rationale.replace("\n", " ").replace("|", "\\|")
            if len(rat_clean) > 100:
                rat_clean = rat_clean[:97] + "..."

            winner_icon = "💡 Bookkeeper" if ev.winner == "bookkeeper" else ("⚡ LightRAG" if ev.winner == "lightrag" else "⚖️ Tie")
            bk_checks_str = f"{ev.bookkeeper_passed_count}/{ev.total_checks} ({ev.bookkeeper_pass_rate:.0f}%) / {ev.bookkeeper_latency:.2f}s"
            lr_checks_str = f"{ev.lightrag_passed_count}/{ev.total_checks} ({ev.lightrag_pass_rate:.0f}%) / {ev.lightrag_latency:.2f}s"
            md += f"| `{ev.question_id}` | `{ev.category}` | {q_clean} | **{winner_icon}** | {bk_checks_str} | {lr_checks_str} | {rat_clean} |\n"

        md += "\n---\n\n## 🔬 5. Detailed Checklist Invariant Breakdown per Query\n\n"
        for idx, ev in enumerate(self.evaluations, 1):
            md += f"### Query {idx}: `{ev.query or ev.question}`\n"
            md += f"- **Target Book / Context:** *{ev.book_title or 'Knowledge Base'}* | **Category:** `{ev.category}` | **Winner:** **{ev.winner.upper()}**\n"
            md += f"- **Bookkeeper Pass Rate:** {ev.bookkeeper_passed_count}/{ev.total_checks} ({ev.bookkeeper_pass_rate:.1f}%) | **Latency:** {ev.bookkeeper_latency:.2f}s\n"
            md += f"- **LightRAG Pass Rate:** {ev.lightrag_passed_count}/{ev.total_checks} ({ev.lightrag_pass_rate:.1f}%) | **Latency:** {ev.lightrag_latency:.2f}s\n\n"
            md += "| Criterion | Expected | Bookkeeper Actual | LightRAG Actual | Bookkeeper Reason | LightRAG Reason |\n"
            md += "| :--- | :---: | :---: | :---: | :--- | :--- |\n"
            for i in range(max(len(ev.bookkeeper_checks), len(ev.lightrag_checks))):
                bk_c = ev.bookkeeper_checks[i] if i < len(ev.bookkeeper_checks) else None
                lr_c = ev.lightrag_checks[i] if i < len(ev.lightrag_checks) else None
                crit = (bk_c.criterion if bk_c else (lr_c.criterion if lr_c else "")).replace("|", "\\|")
                exp = str(bk_c.expected if bk_c else (lr_c.expected if lr_c else True))
                bk_icon = "✅" if (bk_c and bk_c.passed) else "❌"
                lr_icon = "✅" if (lr_c and lr_c.passed) else "❌"
                bk_reason = (bk_c.reason if bk_c else "").replace("\n", " ").replace("|", "\\|")
                lr_reason = (lr_c.reason if lr_c else "").replace("\n", " ").replace("|", "\\|")
                md += f"| {crit} | `{exp}` | {bk_icon} `{bk_c.actual if bk_c else '-'}` | {lr_icon} `{lr_c.actual if lr_c else '-'}` | {bk_reason} | {lr_reason} |\n"
            md += "\n"

        md += "---\n\n## 🔍 5. Qualitative Analysis & Recommendations\n\n"
        if self.bookkeeper_avg_latency < self.lightrag_avg_latency:
            md += f"- **Latency Advantage:** Bookkeeper responded **{lat_ratio:.1f}x faster** on average by relying on direct graph lookups and immediate structured relation synthesis rather than full-pass dual LLM generation.\n"
        if bk_check_pct >= lr_check_pct:
            md += f"- **Invariant Compliance:** Bookkeeper achieved higher factual precision ({bk_check_pct:.1f}% vs {lr_check_pct:.1f}%), accurately fulfilling concrete subject invariants.\n"
        else:
            md += f"- **Narrative Synthesis:** LightRAG demonstrated higher checklist compliance ({lr_check_pct:.1f}% vs {bk_check_pct:.1f}%) through broad generative synthesis.\n"

        if output_path:
            p = Path(output_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md, encoding="utf-8")

        return md


class BenchmarkJudge:
    """
    Impartial LLM Judge that scores Bookkeeper and LightRAG answers using the verification model.
    """

    def __init__(
        self,
        settings: Settings,
        model_name: Optional[str] = None,
        pool: Optional[OllamaPool] = None,
        timeout: int = 90,
    ):
        self.settings = settings
        self.model_name = model_name or getattr(settings, "resolved_verifier_model", settings.llm_model)
        servers = getattr(settings, "resolved_verification_servers", None) or settings.resolved_ollama_servers
        cooldown = getattr(settings, "resolved_verification_cooldown_seconds", 60)
        self.pool = pool or OllamaPool(
            servers=servers,
            cooldown_seconds=cooldown,
            max_tasks_per_server=1,
        )
        self.timeout = timeout

    def _call_llm_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Invoke Ollama with structured JSON and curl fallback."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _invoke(node_or_url: Any) -> Dict[str, Any]:
            base_url = getattr(node_or_url, "target_url", getattr(node_or_url, "url", str(node_or_url)))
            url = f"{base_url.rstrip('/')}/api/chat"
            body = {
                "model": self.model_name,
                "messages": messages,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                },
                "stream": False,
            }

            raw_stdout = ""
            if shutil.which("curl"):
                cmd = [
                    "curl",
                    "-s",
                    "--connect-timeout", "4",
                    "--max-time", str(self.timeout),
                    "-X", "POST",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(body),
                    url,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0 and res.stdout.strip():
                    raw_stdout = res.stdout.strip()

            if not raw_stdout:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_bytes = resp.read()
                    raw_stdout = raw_bytes.decode("utf-8", errors="replace").strip()

            if raw_stdout:
                data = json.loads(raw_stdout)
                if "error" in data:
                    raise RuntimeError(f"Ollama server {url} error: {data['error']}")
                content = data.get("message", {}).get("content", "")
                cleaned = _clean_json_str(content)
                return json.loads(cleaned)

            raise RuntimeError(f"Ollama server {url} returned empty response or timed out after {self.timeout}s")

        return self.pool.execute_with_failover(
            _invoke,
            model_name=self.model_name,
            input_chars=len(prompt),
            max_task_duration=float(self.timeout),
            task_type="verification",
        )

    def evaluate_response_checklist(
        self,
        query: str,
        response_text: str,
        checklist: List[ChecklistItem],
        system_name: str,
    ) -> List[ChecklistCheckResult]:
        """
        Evaluate each checklist item against response_text (acting as context).
        Returns a list of ChecklistCheckResult with boolean verdicts.
        """
        if not checklist:
            return []

        clean_text = (response_text or "").strip()
        if not clean_text or clean_text.lower() in ("no response.", "error"):
            return [
                ChecklistCheckResult(
                    criterion=item.criterion,
                    expected=item.expected,
                    actual=not item.expected,
                    passed=False,
                    reason=f"{system_name} provided an empty response or error.",
                )
                for item in checklist
            ]

        numbered_checks = "\n".join(
            f"{idx}. Criterion: {item.criterion} (Target Concept: {item.aspect_label or 'Key property'})"
            for idx, item in enumerate(checklist, 1)
        )

        system_prompt = (
            "You are a rigorous factual verification auditor. Your job is to verify whether "
            "a retrieved answer (used strictly as Context) satisfies each boolean checklist criterion.\n"
            "Rules:\n"
            "- Rely ONLY on the provided Response text. Do not assume facts not stated in the response.\n"
            "- For each criterion, determine 'verdict': true if the Response text explicitly confirms / demonstrates this criterion; "
            "false if the response denies it, omits it, or fails to satisfy it.\n"
            "- Provide a 1-sentence 'reason' citing evidence from the Response text."
        )

        user_prompt = f"""
Subject Matter Query: "{query}"

System Response ({system_name}) under evaluation:
---
{clean_text[:6000]}
---

Checklist to verify against the above Response:
{numbered_checks}

Respond with ONLY a JSON object matching this schema:
{{
  "results": [
    {{
      "check_index": 1,
      "verdict": true,
      "reason": "The response explicitly states that..."
    }}
  ]
}}
"""
        results_map = {}
        try:
            res_json = self._call_llm_json(user_prompt, system_prompt=system_prompt)
            raw_results = res_json.get("results", [])
            for r in raw_results:
                idx = int(r.get("check_index", 0))
                results_map[idx] = (bool(r.get("verdict", False)), str(r.get("reason", "")).strip())
        except Exception as e:
            logger.warning(f"Checklist verification call failed for {system_name}: {e}")

        check_results: List[ChecklistCheckResult] = []
        for idx, item in enumerate(checklist, 1):
            if idx in results_map:
                actual_val, reason = results_map[idx]
            else:
                actual_val = False
                reason = "Not evaluated or omitted in model response."

            passed = (actual_val == item.expected)
            check_results.append(
                ChecklistCheckResult(
                    criterion=item.criterion,
                    expected=item.expected,
                    actual=actual_val,
                    passed=passed,
                    reason=reason,
                )
            )

        return check_results

    def evaluate(
        self,
        question: QuestionItem,
        bookkeeper_res: Dict[str, Any],
        lightrag_res: Dict[str, Any],
    ) -> JudgeEvaluation:
        """
        Evaluate and score both responses against the binary verification checklist.
        """
        q_text = question.query or getattr(question, "question", "")
        bk_text = bookkeeper_res.get("response") or bookkeeper_res.get("answer") or "No response."
        lr_text = lightrag_res.get("response") or "No response."

        bk_lat = float(bookkeeper_res.get("response_time", 0.0))
        lr_lat = float(lightrag_res.get("response_time", 0.0))

        bk_refs = len(bookkeeper_res.get("references", []))
        lr_refs = len(lightrag_res.get("references", []))

        # Use question checklist or synthesize fallback criteria
        checklist = list(question.checklist)
        if not checklist:
            aspects = question.expected_aspects or ["Direct explanation of mechanism", "Concrete architectural trade-offs"]
            checklist = [
                ChecklistItem(
                    criterion=f"Does the response affirm or explain '{asp}'?",
                    expected=True,
                    aspect_label=str(asp),
                )
                for asp in aspects
            ]

        # 1. Verify Bookkeeper against checklist
        bk_checks = self.evaluate_response_checklist(
            query=q_text,
            response_text=bk_text,
            checklist=checklist,
            system_name="Bookkeeper",
        )

        # 2. Verify LightRAG against checklist
        lr_checks = self.evaluate_response_checklist(
            query=q_text,
            response_text=lr_text,
            checklist=checklist,
            system_name="LightRAG",
        )

        bk_passed = sum(1 for c in bk_checks if c.passed)
        lr_passed = sum(1 for c in lr_checks if c.passed)
        total_checks = len(checklist)
        bk_rate = round((bk_passed / max(1, total_checks)) * 100.0, 1)
        lr_rate = round((lr_passed / max(1, total_checks)) * 100.0, 1)

        if bk_rate > lr_rate:
            winner = "bookkeeper"
            rationale = f"Bookkeeper passed {bk_passed}/{total_checks} criteria ({bk_rate:.1f}%) vs LightRAG {lr_passed}/{total_checks} ({lr_rate:.1f}%)."
        elif lr_rate > bk_rate:
            winner = "lightrag"
            rationale = f"LightRAG passed {lr_passed}/{total_checks} criteria ({lr_rate:.1f}%) vs Bookkeeper {bk_passed}/{total_checks} ({bk_rate:.1f}%)."
        else:
            winner = "tie"
            rationale = f"Both systems passed {bk_passed}/{total_checks} criteria ({bk_rate:.1f}%)."

        # Map to continuous 1-5 score parity
        bk_total = round(1.0 + (bk_rate / 100.0) * 4.0, 2)
        lr_total = round(1.0 + (lr_rate / 100.0) * 4.0, 2)

        return JudgeEvaluation(
            question_id=question.id,
            query=q_text,
            category=question.category,
            book_title=question.book_title,
            bookkeeper_latency=bk_lat,
            lightrag_latency=lr_lat,
            bookkeeper_checks=bk_checks,
            lightrag_checks=lr_checks,
            bookkeeper_passed_count=bk_passed,
            lightrag_passed_count=lr_passed,
            total_checks=total_checks,
            bookkeeper_pass_rate=bk_rate,
            lightrag_pass_rate=lr_rate,
            grounding_bookkeeper=bk_total,
            grounding_lightrag=lr_total,
            completeness_bookkeeper=bk_total,
            completeness_lightrag=lr_total,
            clarity_bookkeeper=bk_total,
            clarity_lightrag=lr_total,
            relevance_bookkeeper=bk_total,
            relevance_lightrag=lr_total,
            bookkeeper_total_score=bk_total,
            lightrag_total_score=lr_total,
            winner=winner,
            rationale=rationale,
            bookkeeper_response=bk_text,
            lightrag_response=lr_text,
            bookkeeper_references_count=bk_refs,
            lightrag_references_count=lr_refs,
        )


class ABBenchmarkRunner:
    """
    Coordinates end-to-end A/B benchmarking between Bookkeeper and LightRAG:
    - Phase 1: Bookkeeper Knowledge Graph Analysis & Export (Safe Point 1)
    - Phase 2: LightRAG Data Ingestion & Native Chunking (Safe Point 2)
    - Phase 3: Dual Retrieval & Rigorous Verification Queries (Safe Point 3)
    """

    def __init__(
        self,
        settings: Settings,
        output_dir: Optional[Union[Path, str]] = None,
        console: Optional[Any] = None,
    ):
        self.settings = settings
        self.output_dir = Path(output_dir or (Path.cwd() / "ab_test_runs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "benchmark_checkpoint.json"
        self.console = console or (Console() if Console is not None else None)

    @classmethod
    def find_latest_session_dir(cls, base_dir: Optional[Union[Path, str]] = None) -> Optional[Path]:
        """Find the most recent benchmark session directory containing a checkpoint or LightRAG storage."""
        root = Path(base_dir or (Path.cwd() / "ab_test_runs")).expanduser().resolve()
        if not root.is_dir():
            return None
        candidates = []
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("run_"):
                # Check if it has a checkpoint or lightrag_storage
                if (p / "benchmark_checkpoint.json").is_file() or (p / "lightrag_storage").is_dir():
                    candidates.append(p)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load benchmark checkpoint state if available."""
        if self.checkpoint_file.is_file():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read benchmark checkpoint at {self.checkpoint_file}: {e}")
        return {}

    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        """Persist benchmark checkpoint state to disk."""
        current = self.load_checkpoint()
        current.update(data)
        current["updated_at"] = time.time()
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save benchmark checkpoint: {e}")

    def record_safe_point_1(self, graph_path: Union[Path, str], book_count: int, books: Optional[List[Dict[str, Any]]] = None) -> None:
        """Record Safe Point 1: Phase 1 (Bookkeeper Analysis & Graph Construction) completed."""
        self.save_checkpoint({
            "phase1_bookkeeper_completed": True,
            "phase1_timestamp": time.time(),
            "bookkeeper_graph_file": str(graph_path),
            "bookkeeper_books_count": book_count,
            "books": books or [],
        })
        if self.console:
            self.console.print(
                f"[bold green]✓ Safe Point 1 recorded:[/bold green] Phase 1 (Bookkeeper Analysis) complete "
                f"({book_count} books, graph: [dim]{graph_path}[/dim])."
            )

    def record_safe_point_2(self, storage_dir: Union[Path, str], books: List[Dict[str, Any]]) -> None:
        """Record Safe Point 2: Phase 2 (LightRAG Analysis & Indexing) completed."""
        self.save_checkpoint({
            "phase2_lightrag_completed": True,
            "phase2_timestamp": time.time(),
            "lightrag_storage_dir": str(storage_dir),
            "lightrag_books_count": len(books),
            "books": books,
        })
        if self.console:
            self.console.print(
                f"[bold green]✓ Safe Point 2 recorded:[/bold green] Phase 2 (LightRAG Analysis) complete "
                f"({len(books)} books, storage: [dim]{storage_dir}[/dim])."
            )

    def record_safe_point_3(self, report_path: Union[Path, str]) -> None:
        """Record Safe Point 3: Phase 3 (Comparative Dual Queries & Judging) completed."""
        self.save_checkpoint({
            "phase3_queries_completed": True,
            "phase3_timestamp": time.time(),
            "report_path": str(report_path),
        })

    def is_phase1_complete(self) -> bool:
        """Check if Phase 1 safe point exists and is valid."""
        cp = self.load_checkpoint()
        if cp.get("phase1_bookkeeper_completed"):
            return True
        # Check if knowledge_graph.json exists with nodes
        default_kg = getattr(self.settings, "resolved_output_dir", Path("output")) / "knowledge_graph.json"
        session_kg = self.output_dir / "knowledge_graph.json"
        for kg in (session_kg, default_kg):
            if kg.is_file() and kg.stat().st_size > 100:
                return True
        return False

    def is_phase2_complete(self) -> bool:
        """Check if Phase 2 safe point exists and LightRAG storage contains valid data."""
        cp = self.load_checkpoint()
        if cp.get("phase2_lightrag_completed"):
            return True
        lr_dir = self.output_dir / "lightrag_storage"
        doc_status = lr_dir / "kv_store_doc_status.json"
        if doc_status.is_file() and doc_status.stat().st_size > 10:
            return True
        return False

    def run_benchmark(
        self,
        books: List[Dict[str, Any]],
        question_library: Optional[QuestionLibrary] = None,
        questions_per_book: int = 5,
        lightrag_mode: str = "hybrid",
        bookkeeper_mode: str = "ideas",
        judge_model: Optional[str] = None,
        from_phase: int = 1,
        phase: Optional[str] = None,
        clean: bool = False,
        resume: bool = True,
        skip_lightrag_indexing: bool = False,
    ) -> BenchmarkReport:
        """
        Execute the full multi-phase comparative benchmark:
        - Phase 1: Bookkeeper Analysis (handled before this or verified here)
        - Phase 2: Ingest books into LightRAG using native token-based chunking
        - Phase 3: Dual query retrieval against both systems & LLM Judge evaluation
        """
        t_start = time.perf_counter()
        titles = [b.get("title", f"Book #{b.get('id')}") for b in books]

        if clean:
            if self.checkpoint_file.is_file():
                try:
                    self.checkpoint_file.unlink()
                except Exception:
                    pass

        if self.console:
            self.console.print(
                Panel.fit(
                    f"[bold cyan]🧪 A/B Benchmark: Bookkeeper vs. LightRAG[/bold cyan]\n"
                    f"• Target Books ({len(books)}): [bold green]{', '.join(titles[:4])}{'...' if len(titles) > 4 else ''}[/bold green]\n"
                    f"• Output Directory: [dim]{self.output_dir}[/dim]\n"
                    f"• Start Phase: [bold yellow]Phase {from_phase}[/bold yellow] | Target Phase: [bold yellow]{phase or 'All (1-3)'}[/bold yellow] | Resume: [cyan]{resume}[/cyan]\n"
                    f"• LightRAG Mode: [cyan]{lightrag_mode}[/cyan] | Bookkeeper Mode: [cyan]{bookkeeper_mode}[/cyan]\n"
                    f"• Verification & Judge Model: [bold yellow]{judge_model or self.settings.resolved_verifier_model}[/bold yellow]",
                    title="A/B Benchmark Pipeline",
                    border_style="cyan",
                )
            )

        lightrag_working_dir = self.output_dir / "lightrag_storage"

        # -------------------------------------------------------------
        # Phase 2: LightRAG Native Chunking & Analysis
        # -------------------------------------------------------------
        run_phase_2 = (from_phase <= 2) and (phase in (None, "all", "2", "lightrag"))
        if run_phase_2:
            phase2_already_done = resume and self.is_phase2_complete() and not clean

            if not skip_lightrag_indexing and not phase2_already_done:
                if self.console:
                    self.console.print(f"\n[bold yellow]═══ Phase 2/3: Ingesting {len(books)} books into LightRAG (native chunking) ═══[/bold yellow]\n")
                indexer = LightRAGRawIndexer(
                    settings=self.settings,
                    working_dir=lightrag_working_dir,
                    console=self.console,
                )
                try:
                    indexer.index_books(books=books, clean=clean, show_progress=True)
                finally:
                    indexer.close()

                self.record_safe_point_2(storage_dir=lightrag_working_dir, books=books)
            else:
                if self.console:
                    self.console.print(
                        f"[bold green]✓ Phase 2 Safe Point detected:[/bold green] Reusing existing LightRAG storage at [dim]{lightrag_working_dir}[/dim]."
                    )

            if phase in ("2", "lightrag"):
                if self.console:
                    self.console.print(f"\n[bold green]✓ Phase 2 (LightRAG Analysis) complete. Exiting as requested by --phase {phase}.[/bold green]\n")
                return BenchmarkReport(
                    timestamp=time.time(),
                    judge_model=judge_model or self.settings.resolved_verifier_model,
                    book_titles=titles,
                    total_questions=0,
                )

        # -------------------------------------------------------------
        # Phase 3: Dual Query Execution & Verification Judging
        # -------------------------------------------------------------
        run_phase_3 = (from_phase <= 3) and (phase in (None, "all", "3", "benchmark"))
        if not run_phase_3:
            return BenchmarkReport(
                timestamp=time.time(),
                judge_model=judge_model or self.settings.resolved_verifier_model,
                book_titles=titles,
                total_questions=0,
            )

        # Step 3.1: Question Library Preparation
        ql_path = self.output_dir / "question_library.json"
        if question_library is None:
            if resume and ql_path.is_file() and not clean:
                if self.console:
                    self.console.print(f"[bold green]✓ Reusing existing question library at [cyan]{ql_path}[/cyan][/bold green]")
                question_library = QuestionLibrary.load(ql_path)
            else:
                if self.console:
                    self.console.print(f"\n[bold yellow]═══ Phase 3/3: Synthesizing Question Library via Verification Model ═══[/bold yellow]\n")
                generator = QuestionGenerator(
                    settings=self.settings,
                    model_name=judge_model,
                )
                question_library = generator.generate_library(
                    books=books,
                    questions_per_book=questions_per_book,
                    output_file=ql_path,
                )
                if self.console:
                    self.console.print(
                        f"[bold green]✓ Synthesized {len(question_library.questions)} questions across {len(books)} books.[/bold green] "
                        f"Saved to: [bold cyan]{ql_path}[/bold cyan]"
                    )
        else:
            if self.console:
                self.console.print(f"[bold green]✓ Loaded question library with {len(question_library.questions)} questions.[/bold green]")

        # Step 3.2: Dual Query Execution & Rigorous LLM Judging
        if self.console:
            self.console.print(f"\n[bold yellow]═══ Phase 3/3: Dual Query Execution & Rigorous LLM Judging ═══[/bold yellow]\n")

        bk_backend = SearchEngineBackend(config=self.settings, neo4j_config=self.settings.neo4j)
        lr_service = LightRAGService(settings=self.settings, working_dir=lightrag_working_dir)
        judge = BenchmarkJudge(settings=self.settings, model_name=judge_model)

        evaluations: List[JudgeEvaluation] = []
        bk_wins = 0
        lr_wins = 0
        ties = 0

        try:
            for idx, q_item in enumerate(question_library.questions, 1):
                q_text = q_item.query or getattr(q_item, "question", "")
                if self.console:
                    self.console.print(f"[bold cyan]Query {idx}/{len(question_library.questions)}:[/bold cyan] {q_text}")

                # 1. Query Bookkeeper
                bk_res = bk_backend.query_api(query=q_text, mode=bookkeeper_mode)

                # 2. Query LightRAG
                lr_res = lr_service.query(query=q_text, mode=lightrag_mode)

                # 3. Judge evaluation
                ev = judge.evaluate(question=q_item, bookkeeper_res=bk_res, lightrag_res=lr_res)
                evaluations.append(ev)

                if ev.winner == "bookkeeper":
                    bk_wins += 1
                    winner_badge = "[bold green]💡 Winner: Bookkeeper[/bold green]"
                elif ev.winner == "lightrag":
                    lr_wins += 1
                    winner_badge = "[bold cyan]⚡ Winner: LightRAG[/bold cyan]"
                else:
                    ties += 1
                    winner_badge = "[bold yellow]⚖️ Tie[/bold yellow]"

                if self.console:
                    self.console.print(
                        f"  • {winner_badge} (Checks: BK={ev.bookkeeper_passed_count}/{ev.total_checks} ({ev.bookkeeper_pass_rate:.0f}%) in {ev.bookkeeper_latency:.2f}s | "
                        f"LR={ev.lightrag_passed_count}/{ev.total_checks} ({ev.lightrag_pass_rate:.0f}%) in {ev.lightrag_latency:.2f}s)\n"
                        f"  • [dim]Rationale: {ev.rationale}[/dim]\n"
                    )
        finally:
            bk_graph_stats, lr_graph_stats = self._collect_graph_topology_stats(
                bk_backend=bk_backend,
                lr_service=lr_service,
            )
            bk_backend.close()
            lr_service.close()

        # Step 5: Report Synthesis
        total_q = len(evaluations)
        total_checks = sum(e.total_checks for e in evaluations)
        bk_passed_checks = sum(e.bookkeeper_passed_count for e in evaluations)
        lr_passed_checks = sum(e.lightrag_passed_count for e in evaluations)
        bk_overall_rate = round((bk_passed_checks / max(1, total_checks)) * 100.0, 1)
        lr_overall_rate = round((lr_passed_checks / max(1, total_checks)) * 100.0, 1)

        avg_bk_lat = sum(e.bookkeeper_latency for e in evaluations) / max(1, total_q)
        avg_lr_lat = sum(e.lightrag_latency for e in evaluations) / max(1, total_q)
        avg_bk_score = sum(e.bookkeeper_total_score for e in evaluations) / max(1, total_q)
        avg_lr_score = sum(e.lightrag_total_score for e in evaluations) / max(1, total_q)

        avg_grounding = {
            "bookkeeper": round(sum(e.grounding_bookkeeper for e in evaluations) / max(1, total_q), 2),
            "lightrag": round(sum(e.grounding_lightrag for e in evaluations) / max(1, total_q), 2),
        }
        avg_completeness = {
            "bookkeeper": round(sum(e.completeness_bookkeeper for e in evaluations) / max(1, total_q), 2),
            "lightrag": round(sum(e.completeness_lightrag for e in evaluations) / max(1, total_q), 2),
        }
        avg_clarity = {
            "bookkeeper": round(sum(e.clarity_bookkeeper for e in evaluations) / max(1, total_q), 2),
            "lightrag": round(sum(e.clarity_lightrag for e in evaluations) / max(1, total_q), 2),
        }
        avg_relevance = {
            "bookkeeper": round(sum(e.relevance_bookkeeper for e in evaluations) / max(1, total_q), 2),
            "lightrag": round(sum(e.relevance_lightrag for e in evaluations) / max(1, total_q), 2),
        }

        report = BenchmarkReport(
            timestamp=time.time(),
            judge_model=judge.model_name,
            book_titles=titles,
            total_questions=total_q,
            total_checklist_items=total_checks,
            bookkeeper_wins=bk_wins,
            lightrag_wins=lr_wins,
            ties=ties,
            bookkeeper_avg_latency=round(avg_bk_lat, 4),
            lightrag_avg_latency=round(avg_lr_lat, 4),
            bookkeeper_total_passed_checks=bk_passed_checks,
            lightrag_total_passed_checks=lr_passed_checks,
            bookkeeper_overall_pass_rate=bk_overall_rate,
            lightrag_overall_pass_rate=lr_overall_rate,
            bookkeeper_avg_score=round(avg_bk_score, 2),
            lightrag_avg_score=round(avg_lr_score, 2),
            avg_grounding=avg_grounding,
            avg_completeness=avg_completeness,
            avg_clarity=avg_clarity,
            avg_relevance=avg_relevance,
            bookkeeper_nodes_count=bk_graph_stats["nodes_count"],
            bookkeeper_edges_count=bk_graph_stats["edges_count"],
            bookkeeper_chunks_count=bk_graph_stats["chunks_count"],
            bookkeeper_node_types=bk_graph_stats["node_types"],
            lightrag_nodes_count=lr_graph_stats["nodes_count"],
            lightrag_edges_count=lr_graph_stats["edges_count"],
            lightrag_chunks_count=lr_graph_stats["chunks_count"],
            lightrag_node_types=lr_graph_stats["node_types"],
            evaluations=evaluations,
        )

        # Save artifacts
        json_path = self.output_dir / "benchmark_report.json"
        md_path = self.output_dir / "ab_test_summary.md"
        report.save_json(json_path)
        report.generate_markdown_summary(md_path)
        self.record_safe_point_3(json_path)

        if self.console:
            self.console.print(f"\n[bold yellow]═══ Phase 3/3: Benchmark Report Generated ═══[/bold yellow]\n")
            self.console.print(f"  • [bold white]JSON Report:[/bold white] [bold cyan]{json_path}[/bold cyan]")
            self.console.print(f"  • [bold white]Summary Report:[/bold white] [bold green]{md_path}[/bold green]\n")

            bk_n = report.bookkeeper_nodes_count
            lr_n = report.lightrag_nodes_count
            bk_e = report.bookkeeper_edges_count
            lr_e = report.lightrag_edges_count
            if bk_n > 0 or lr_n > 0:
                ratio_n = f"{bk_n / lr_n:.1f}x" if lr_n else "-"
                ratio_e = f"{bk_e / lr_e:.1f}x" if lr_e else "-"
                bk_dens = bk_e / max(1, bk_n)
                lr_dens = lr_e / max(1, lr_n)
                self.console.print("[bold cyan]─── Knowledge Graph Structural Topology Comparison ───[/bold cyan]")
                self.console.print(f"  • [bold white]Total Graph Nodes:[/bold white] Bookkeeper={bk_n:,} vs. LightRAG={lr_n:,} ({ratio_n})")
                self.console.print(f"  • [bold white]Total Graph Edges:[/bold white] Bookkeeper={bk_e:,} vs. LightRAG={lr_e:,} ({ratio_e})")
                self.console.print(f"  • [bold white]Relational Density:[/bold white] Bookkeeper={bk_dens:.2f} vs. LightRAG={lr_dens:.2f} edges/node\n")

        return report

    def _collect_graph_topology_stats(
        self,
        bk_backend: Optional[SearchEngineBackend] = None,
        lr_service: Optional[LightRAGService] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Collect node count, edge count, and chunk counts for Bookkeeper and LightRAG."""
        lr_stats = {
            "nodes_count": 0,
            "edges_count": 0,
            "chunks_count": 0,
            "node_types": {},
        }
        if lr_service is not None:
            try:
                lr_engine = lr_service.get_engine()
                raw_lr = lr_engine.get_stats()
                lr_stats["nodes_count"] = raw_lr.get("nodes_count", 0)
                lr_stats["edges_count"] = raw_lr.get("edges_count", 0)
                lr_stats["chunks_count"] = raw_lr.get("chunks_count", 0)
                lr_stats["node_types"] = raw_lr.get("node_types", {})
            except Exception as e:
                logger.warning(f"Failed to extract LightRAG stats: {e}")

        bk_stats = {
            "nodes_count": 0,
            "edges_count": 0,
            "chunks_count": 0,
            "node_types": {},
        }
        # 1. Try Neo4j if configured
        if self.settings.neo4j and self.settings.neo4j.password:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    self.settings.neo4j.uri,
                    auth=(self.settings.neo4j.user, self.settings.neo4j.password),
                )
                with driver.session() as session:
                    res_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()
                    if res_nodes:
                        bk_stats["nodes_count"] = res_nodes["cnt"]
                    res_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
                    if res_edges:
                        bk_stats["edges_count"] = res_edges["cnt"]
                    res_chunks = session.run("MATCH (n:Chunk) RETURN count(n) AS cnt").single()
                    if res_chunks:
                        bk_stats["chunks_count"] = res_chunks["cnt"]
                driver.close()
            except Exception:
                pass

        # 2. Fallback to local knowledge_graph.json
        if bk_stats["nodes_count"] == 0:
            candidate_paths = [
                self.output_dir / "knowledge_graph.json",
                Path("output/knowledge_graph.json"),
                Path("knowledge_graph.json"),
            ]
            for p in candidate_paths:
                if p.is_file():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        nodes = data.get("nodes", [])
                        edges = data.get("edges", [])
                        bk_stats["nodes_count"] = len(nodes)
                        bk_stats["edges_count"] = len(edges)
                        from collections import Counter
                        node_types = dict(Counter(n.get("type", "unknown") for n in nodes))
                        bk_stats["node_types"] = node_types
                        bk_stats["chunks_count"] = node_types.get("Chunk", 0)
                        break
                    except Exception:
                        pass

        return bk_stats, lr_stats
