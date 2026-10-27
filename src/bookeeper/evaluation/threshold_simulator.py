"""
Knowledge Base Threshold Sweep & Simulation Engine.

Simulates varying (low_threshold, high_confidence_threshold) boundary pairs for chunk
routing over cached book state and chunk files without invoking LLMs. Evaluates each
variation using graph embedding space metrics (mean Chamfer proximity mu, regularity sigma,
worst-decile p10, coverage at tau, compression leverage factor L_comp, and Vendi spectral entropy),
and produces comparative Rich tables and Markdown visual reports with Mermaid xychart-beta plots.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.extractor import Concept
from bookeeper.processing.idea_postprocessor import IdeaPostProcessor
from bookeeper.processing.graph_evaluator import (
    GraphSpaceMetrics,
    PostProcessingDriftMetrics,
    calculate_postprocessing_drift,
    evaluate_graph_space,
)
from bookeeper.processing.memory_guard import cleanup_memory
from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex, normalize_sort_date

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Threshold Pair Models & Input Parsing
# ==============================================================================


class ThresholdPair(BaseModel):
    """
    Pair of semantic similarity thresholds for chunk routing:
    - low: boundary below which chunks are classified as UNASSIGNED (processed for new ideas).
    - high: boundary at or above which chunks are classified as HIGH_CONFIDENCE (direct assignment).
    Chunks between [low, high) fall into LIMBO.
    """

    low: float = Field(..., ge=0.0, le=1.0, description="Unassigned lower boundary threshold.")
    high: float = Field(..., ge=0.0, le=1.0, description="High-confidence upper boundary threshold.")
    label: str = Field("", description="Human-readable label for this variation.")

    def model_post_init(self, __context: Any) -> None:
        if self.low > self.high:
            raise ValueError(f"Lower boundary low={self.low} cannot exceed upper boundary high={self.high}")
        if not self.label:
            self.label = f"{self.low:.2f}:{self.high:.2f}"

    def __str__(self) -> str:
        return self.label or f"{self.low:.2f}:{self.high:.2f}"


def parse_threshold_pairs(
    pairs_input: Union[str, Sequence[Tuple[float, float]], Sequence[Dict[str, Any]], Sequence[ThresholdPair]],
) -> List[ThresholdPair]:
    """
    Parse various threshold pair representations into a clean list of ThresholdPair objects.

    Supported formats:
    - CLI string: "0.60:0.90, 0.65:0.90, 0.70:0.90, 0.65:0.95"
    - CLI string with dashes: "0.60-0.90,0.65-0.90"
    - CLI string with tuples: "(0.60, 0.90), (0.65, 0.90)"
    - List of tuples: [(0.60, 0.90), (0.65, 0.90)]
    - List of dicts: [{"low": 0.60, "high": 0.90}]
    - List of ThresholdPair objects.
    """
    if not pairs_input:
        return [
            ThresholdPair(low=0.60, high=0.90, label="0.60:0.90"),
            ThresholdPair(low=0.65, high=0.90, label="0.65:0.90 (Default)"),
            ThresholdPair(low=0.70, high=0.90, label="0.70:0.90"),
            ThresholdPair(low=0.75, high=0.90, label="0.75:0.90"),
            ThresholdPair(low=0.65, high=0.95, label="0.65:0.95"),
        ]

    results: List[ThresholdPair] = []

    if isinstance(pairs_input, str):
        # Clean enclosing brackets/parentheses
        raw_str = pairs_input.strip()
        # Find all number pairs: e.g. "0.60:0.90", "0.60, 0.90", "(0.60, 0.90)", "0.60-0.90"
        # We can split by top-level delimiters (commas/semicolons) or regex pattern
        # Pattern matching (float separator float)
        pair_pattern = re.compile(r"\(?([0-9]*\.?[0-9]+)\s*[:,\-\/]\s*([0-9]*\.?[0-9]+)\)?")
        matches = pair_pattern.findall(raw_str)
        if matches:
            for m_low, m_high in matches:
                try:
                    f_low = float(m_low)
                    f_high = float(m_high)
                    results.append(ThresholdPair(low=f_low, high=f_high))
                except (ValueError, Exception) as e:
                    logger.warning(f"Skipping invalid threshold pair string '{m_low}:{m_high}': {e}")
        else:
            raise ValueError(f"Could not parse any valid threshold pairs from input: '{pairs_input}'")

    elif isinstance(pairs_input, (list, tuple)):
        for item in pairs_input:
            if isinstance(item, ThresholdPair):
                results.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                results.append(ThresholdPair(low=float(item[0]), high=float(item[1])))
            elif isinstance(item, dict):
                low_val = float(item.get("low") or item.get("unassigned_threshold") or item.get("min") or 0.0)
                high_val = float(item.get("high") or item.get("high_confidence_threshold") or item.get("max") or 1.0)
                lbl = str(item.get("label") or "")
                results.append(ThresholdPair(low=low_val, high=high_val, label=lbl))
            else:
                logger.warning(f"Unrecognized threshold pair item: {item}")

    if not results:
        raise ValueError("No valid threshold pairs could be parsed.")

    return results


# ==============================================================================
# 2. Simulation Results & Markdown Report Generation
# ==============================================================================


class SimulationVariationResult(BaseModel):
    """
    Detailed metrics for a single threshold pair variation run.
    """

    pair: ThresholdPair
    label: str
    tau: float = 0.75
    total_chunks: int = 0
    unassigned_count: int = 0
    high_confidence_count: int = 0
    limbo_count: int = 0
    nodes_count: int = 0

    # Graph Space Distributional Metrics
    chamfer_mean: float = Field(..., description="Size-weighted mean Chamfer proximity (mu).")
    chamfer_std: float = Field(..., description="Size-weighted regularity standard deviation (sigma).")
    median_p50: float = Field(..., description="Median proximity (50th percentile).")
    worst_decile_p10: float = Field(..., description="Worst-decile proximity (10th percentile).")
    coverage_at_tau_pct: float = Field(..., description="Threshold coverage (Cov_tau >= tau) %.")
    worst_case_gap: float = Field(..., description="Directed Hausdorff gap (1.0 - min_sim).")
    vendi_diversity_score: float = Field(..., description="Vendi score (orthogonal concept modes).")

    # Skipping Leverage Metrics
    parent_chunks_count: int = 0
    skipped_chunks_count: int = 0
    parent_chunk_ratio_pct: float = Field(0.0, description="Extraction ratio rho_p = W_p / W_total %.")
    skipped_retention_pct: float = Field(0.0, description="Skipped text retention R_skipped %.")
    generalization_surplus_pct: float = Field(0.0, description="Generalization surplus Delta_cov %.")
    compression_leverage_factor: float = Field(0.0, description="Compression leverage L_comp.")

    # Post-Processing Idea Drift Metrics (RAW vs POST Ideas)
    postprocessed_nodes_count: Optional[int] = Field(None, description="Number of surviving concepts after post-processing deduplication.")
    forward_retention: Optional[float] = Field(None, description="Forward retention S_{R -> P}: average cosine similarity of nearest surviving post idea.")
    backward_fidelity: Optional[float] = Field(None, description="Backward fidelity S_{P -> R}: average cosine similarity of nearest original raw idea.")
    symmetric_idea_drift_score: Optional[float] = Field(None, description="Harmonic mean of retention and fidelity in [0, 1].")
    weighted_retention: Optional[float] = Field(None, description="Evidence-weighted forward retention.")
    centroid_cosine_alignment: Optional[float] = Field(None, description="Macro centroid cosine alignment between RAW and POST ideas.")
    p10_retention: Optional[float] = Field(None, description="Bottom 10% worst-preserved raw ideas similarity.")
    min_retention: Optional[float] = Field(None, description="Minimum cosine similarity observed for any raw idea.")
    dropped_raw_count: Optional[int] = Field(None, description="Count of raw ideas whose best post match falls below tau_loss.")
    dropped_raw_ratio_pct: Optional[float] = Field(None, description="Percentage of raw ideas dropped below tau_loss.")

    duration_seconds: float = 0.0


class SimulationSuiteReport(BaseModel):
    """
    Aggregated simulation suite report containing all variations and Markdown visual charts.
    """

    tau: float = 0.75
    total_corpus_chunks: int = 0
    total_corpus_chars: int = 0
    total_books: int = 0
    total_raw_concepts: int = 0
    skip_limbo: bool = True
    variations: List[SimulationVariationResult] = Field(default_factory=list)

    def to_rich_table(self) -> Table:
        """Render a comparative Rich terminal table displaying one row per threshold pair."""
        caption_items = [
            f"Total Universe Chunks: {self.total_corpus_chunks:,}",
            f"Total Text Volume: {self.total_corpus_chars:,} chars",
            f"Books: {self.total_books}",
        ]
        if self.total_raw_concepts > 0:
            caption_items.append(f"Corpus Raw Concepts: {self.total_raw_concepts:,}")

        table = Table(
            title=f"Knowledge Base Threshold Simulation Benchmark (τ = {self.tau:.2f}, Skip Limbo = {self.skip_limbo})",
            caption=" | ".join(caption_items),
            header_style="bold cyan",
            border_style="dim",
        )

        has_drift = any(v.forward_retention is not None for v in self.variations)

        table.add_column("Variation (Low : High)", justify="left", style="bold")
        table.add_column("Nodes (M)", justify="right")
        if has_drift:
            table.add_column("Post (Mp)", justify="right")
        table.add_column("Parent (Np)", justify="right")
        table.add_column("Skipped (Ns)", justify="right")
        table.add_column("Compute (ρp)", justify="right", style="cyan")
        table.add_column("Cov (Cov_τ)", justify="right", style="bold green")
        table.add_column("Mean (μ)", justify="right")
        table.add_column("Std (σ)", justify="right")
        table.add_column("Decile (p10)", justify="right", style="yellow")
        if has_drift:
            table.add_column("Ret (S_R→P)", justify="right", style="green")
            table.add_column("Fid (S_P→R)", justify="right", style="green")
            table.add_column("Centroid", justify="right", style="cyan")
            table.add_column("Drop (%)", justify="right", style="red")
        table.add_column("Leverage (Lcomp)", justify="right", style="magenta")
        table.add_column("Vendi", justify="right")
        table.add_column("Time", justify="right", style="dim")

        for v in self.variations:
            row_items = [
                v.label,
                f"{v.nodes_count:,}",
            ]
            if has_drift:
                row_items.append(f"{v.postprocessed_nodes_count:,}" if v.postprocessed_nodes_count is not None else "-")
            row_items.extend([
                f"{v.parent_chunks_count:,}",
                f"{v.skipped_chunks_count:,}",
                f"{v.parent_chunk_ratio_pct:.2f}%",
                f"{v.coverage_at_tau_pct:.2f}%",
                f"{v.chamfer_mean:.4f}",
                f"{v.chamfer_std:.4f}",
                f"{v.worst_decile_p10:.4f}",
            ])
            if has_drift:
                row_items.extend([
                    f"{v.forward_retention:.4f}" if v.forward_retention is not None else "-",
                    f"{v.backward_fidelity:.4f}" if v.backward_fidelity is not None else "-",
                    f"{v.centroid_cosine_alignment:.4f}" if v.centroid_cosine_alignment is not None else "-",
                    f"{v.dropped_raw_ratio_pct:.1f}%" if v.dropped_raw_ratio_pct is not None else "-",
                ])
            row_items.extend([
                f"{v.compression_leverage_factor:.2f}x",
                f"{v.vendi_diversity_score:.1f}",
                f"{v.duration_seconds:.2f}s",
            ])
            table.add_row(*row_items)

        return table

    def to_markdown(self) -> str:
        """
        Generate comprehensive GitHub-flavored Markdown report with:
        1. Summary metadata table.
        2. Comparative table (one pair per row, sigma, tau, mu, p10, coverage, leverage).
        3. Visual plotting graphs using Mermaid xychart-beta diagrams.
        4. Pareto-optimal analysis and recommendations.
        """
        lines: List[str] = []
        lines.append("# Knowledge Base Threshold Simulation & Optimization Report")
        lines.append("")
        lines.append(f"> **Evaluation Radius (τ)**: `{self.tau:.2f}`  ")
        lines.append(f"> **Corpus Chunks Universe (N)**: `{self.total_corpus_chunks:,}`  ")
        lines.append(f"> **Corpus Text Weight**: `{self.total_corpus_chars:,}` characters  ")
        lines.append(f"> **Books Simulated**: `{self.total_books}`  ")
        if self.total_raw_concepts > 0:
            lines.append(f"> **Corpus Raw Concepts**: `{self.total_raw_concepts:,}`  ")
        lines.append(f"> **Limbo Policy**: `{'Bypass / Skip Limbo' if self.skip_limbo else 'LLM Extraction'}`  ")
        lines.append(f"> **Clean Start**: Each variation is constructed from scratch with an empty graph store and isolated vector index (zero state leakage).  ")
        lines.append("")

        # 1. Comparative Table
        lines.append("## 1. Threshold Variation Comparative Summary")
        lines.append("")
        has_drift = any(v.forward_retention is not None for v in self.variations)

        if has_drift:
            lines.append("| Threshold Variation | Nodes ($M$) | Post ($M_p$) | Parent ($N_p$) | Skipped ($N_s$) | Compute ($\\rho_p$) | Coverage ($\\text{Cov}_\\tau$) | Mean ($\\mu$) | $p_{10}$ | Ret ($S_{R\\to P}$) | Fid ($S_{P\\to R}$) | Centroid | Drop (%) | Leverage ($L_{\\text{comp}}$) | Vendi | Time |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for v in self.variations:
                mp_str = f"{v.postprocessed_nodes_count:,}" if v.postprocessed_nodes_count is not None else "-"
                ret_str = f"{v.forward_retention:.4f}" if v.forward_retention is not None else "-"
                fid_str = f"{v.backward_fidelity:.4f}" if v.backward_fidelity is not None else "-"
                cen_str = f"{v.centroid_cosine_alignment:.4f}" if v.centroid_cosine_alignment is not None else "-"
                drop_str = f"{v.dropped_raw_ratio_pct:.1f}%" if v.dropped_raw_ratio_pct is not None else "-"
                lines.append(
                    f"| **{v.label}** | {v.nodes_count:,} | {mp_str} | {v.parent_chunks_count:,} | {v.skipped_chunks_count:,} | "
                    f"{v.parent_chunk_ratio_pct:.2f}% | **{v.coverage_at_tau_pct:.2f}%** | {v.chamfer_mean:.4f} | "
                    f"{v.worst_decile_p10:.4f} | {ret_str} | {fid_str} | {cen_str} | {drop_str} | **{v.compression_leverage_factor:.2f}x** | "
                    f"{v.vendi_diversity_score:.1f} | {v.duration_seconds:.2f}s |"
                )
        else:
            lines.append("| Threshold Variation | Nodes ($M$) | Parent Chunks ($N_p$) | Skipped ($N_s$) | Compute Ratio ($\\rho_p$) | Coverage ($\\text{Cov}_\\tau$) | Mean ($\\mu$) | Regularity ($\\sigma$) | Worst-Decile ($p_{10}$) | Leverage ($L_{\\text{comp}}$) | Vendi Score | Sim Time |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

            for v in self.variations:
                lines.append(
                    f"| **{v.label}** | {v.nodes_count:,} | {v.parent_chunks_count:,} | {v.skipped_chunks_count:,} | "
                    f"{v.parent_chunk_ratio_pct:.2f}% | **{v.coverage_at_tau_pct:.2f}%** | {v.chamfer_mean:.4f} | "
                    f"{v.chamfer_std:.4f} | {v.worst_decile_p10:.4f} | **{v.compression_leverage_factor:.2f}x** | "
                    f"{v.vendi_diversity_score:.1f} | {v.duration_seconds:.2f}s |"
                )

        lines.append("")

        # 2. Mermaid Visual Plots (xychart-beta)
        lines.append("## 2. Visual Metric Plots Across Threshold Variations")
        lines.append("")

        if self.variations:
            labels_json = json.dumps([f"{v.pair.low:.2f}:{v.pair.high:.2f}" for v in self.variations])
            cov_vals = [round(v.coverage_at_tau_pct, 2) for v in self.variations]
            comp_vals = [round(v.parent_chunk_ratio_pct, 2) for v in self.variations]
            mu_vals = [round(v.chamfer_mean, 4) for v in self.variations]
            sigma_vals = [round(v.chamfer_std, 4) for v in self.variations]
            p10_vals = [round(v.worst_decile_p10, 4) for v in self.variations]
            lev_vals = [round(v.compression_leverage_factor, 2) for v in self.variations]

            max_pct = max(100.0, max(cov_vals + comp_vals, default=100.0))
            max_lev = max(3.0, max(lev_vals, default=2.0) * 1.2)
            min_mu = max(0.0, min(mu_vals + p10_vals, default=0.5) - 0.05)
            max_mu = min(1.0, max(mu_vals, default=0.9) + 0.05)

            # Chart 1: Coverage vs Compute Ratio
            lines.append("### Threshold Coverage ($\\text{Cov}_\\tau$) vs Extraction Compute Ratio ($\\rho_p$)")
            lines.append("```mermaid")
            lines.append("xychart-beta")
            lines.append('    title "Coverage vs Compute Ratio Across Variations"')
            lines.append(f"    x-axis {labels_json}")
            lines.append(f'    y-axis "Percentage (%)" 0 --> {int(np.ceil(max_pct))}')
            lines.append(f"    bar {cov_vals}")
            lines.append(f"    line {comp_vals}")
            lines.append("```")
            lines.append("> **Bar**: Coverage $\\text{Cov}_\\tau$ (% literature text volume within $\\tau=0.75$)  ")
            lines.append("> **Line**: Compute Extraction Ratio $\\rho_p$ (% text volume sent for LLM extraction)  ")
            lines.append("")

            # Chart 2: Compression Leverage Multiplier
            lines.append("### Compression Leverage Factor ($L_{\\text{comp}} = \\text{Cov}_\\tau / \\rho_p$)")
            lines.append("```mermaid")
            lines.append("xychart-beta")
            lines.append('    title "Compression Leverage Multiplier Across Variations"')
            lines.append(f"    x-axis {labels_json}")
            lines.append(f'    y-axis "Leverage Factor (x)" 0 --> {round(max_lev, 1)}')
            lines.append(f"    bar {lev_vals}")
            lines.append("```")
            lines.append("> **Leverage Multiplier**: Measures coverage retained per unit of extraction compute. Higher is more computationally efficient.")
            lines.append("")

            # Chart 3: Proximity Distribution (Mean mu vs Worst-Decile p10)
            lines.append("### Cosine Proximity Spectrum: Mean ($\\mu$) vs Worst-Decile ($p_{10}$)")
            lines.append("```mermaid")
            lines.append("xychart-beta")
            lines.append('    title "Mean Proximity (mu) vs Worst-Decile (p10)"')
            lines.append(f"    x-axis {labels_json}")
            lines.append(f'    y-axis "Cosine Similarity" {round(min_mu, 2)} --> {round(max_mu, 2)}')
            lines.append(f"    line {mu_vals}")
            lines.append(f"    line {p10_vals}")
            lines.append("```")
            lines.append("> **Line 1 (Top)**: Mean Chamfer Proximity $\\mu$  ")
            lines.append("> **Line 2 (Bottom)**: Worst-Decile Proximity $p_{10}$ (10th percentile lowest similarity)  ")
            lines.append("")

            # Chart 4: Dispersion Regularity (Std Dev sigma)
            lines.append("### Dispersion Regularity: Standard Deviation ($\\sigma$)")
            lines.append("```mermaid")
            lines.append("xychart-beta")
            lines.append('    title "Embedding Space Dispersion Regularity (sigma)"')
            lines.append(f"    x-axis {labels_json}")
            lines.append(f'    y-axis "Standard Deviation" 0 --> {round(max(sigma_vals, default=0.2) * 1.3, 3)}')
            lines.append(f"    line {sigma_vals}")
            lines.append("```")
            lines.append("> **Std Dev ($\\sigma$)**: Lower standard deviation indicates tighter, more uniform semantic coverage without clustering holes.")
            lines.append("")

            # Charts 5 & 6: Post-Processing Drift & Concept Degradation
            if has_drift:
                ret_vals = [round(v.forward_retention, 4) if v.forward_retention is not None else 0.0 for v in self.variations]
                fid_vals = [round(v.backward_fidelity, 4) if v.backward_fidelity is not None else 0.0 for v in self.variations]
                min_drift = max(0.0, min(ret_vals + fid_vals, default=0.5) - 0.05)
                max_drift = min(1.0, max(ret_vals + fid_vals, default=1.0) + 0.02)

                # Chart 5: Retention vs Fidelity
                lines.append("### Post-Processing Semantic Retention ($S_{R\\to P}$) vs Backward Fidelity ($S_{P\\to R}$)")
                lines.append("```mermaid")
                lines.append("xychart-beta")
                lines.append('    title "Semantic Retention vs Backward Fidelity Across Variations"')
                lines.append(f"    x-axis {labels_json}")
                lines.append(f'    y-axis "Cosine Similarity" {round(min_drift, 2)} --> {round(max_drift, 2)}')
                lines.append(f"    line {ret_vals}")
                lines.append(f"    line {fid_vals}")
                lines.append("```")
                lines.append("> **Line 1 (Top/Blue)**: Forward Retention $S_{R\\to P}$ (how faithfully original raw ideas are preserved in deduplicated KG).  ")
                lines.append("> **Line 2**: Backward Fidelity $S_{P\\to R}$ (how strictly KG ideas remain grounded in original raw literature extractions).  ")
                lines.append("")

                # Chart 6: Concept Nodes Count: Raw (M) vs Surviving Post-Processed (Mp)
                m_vals = [v.nodes_count for v in self.variations]
                mp_vals = [v.postprocessed_nodes_count if v.postprocessed_nodes_count is not None else v.nodes_count for v in self.variations]
                max_m = max(m_vals + mp_vals, default=10)

                lines.append("### Concept Compression: Raw Extracted ($M$) vs Surviving Post ($M_p$) Nodes")
                lines.append("```mermaid")
                lines.append("xychart-beta")
                lines.append('    title "Raw Concepts (M) vs Post-Processed Concepts (Mp)"')
                lines.append(f"    x-axis {labels_json}")
                lines.append(f'    y-axis "Concept Count" 0 --> {int(np.ceil(max_m * 1.15))}')
                lines.append(f"    bar {m_vals}")
                lines.append(f"    bar {mp_vals}")
                lines.append("```")
                lines.append("> **Bar 1**: Raw Ideas Count $M$ extracted from parent chunks before deduplication.  ")
                lines.append("> **Bar 2**: Surviving Knowledge Graph Ideas $M_p$ after post-processing deduplication.  ")
                lines.append("")

                if any((v.dropped_raw_count or 0) > 0 for v in self.variations):
                    drop_ratios = [round(v.dropped_raw_ratio_pct or 0.0, 2) for v in self.variations]
                    max_dr = max(5.0, max(drop_ratios, default=5.0) * 1.2)
                    lines.append("### Dropped Raw Ideas Ratio Below $\\tau_{\\text{loss}}$ (%)")
                    lines.append("```mermaid")
                    lines.append("xychart-beta")
                    lines.append('    title "Dropped Raw Ideas Ratio Below tau_loss (%)"')
                    lines.append(f"    x-axis {labels_json}")
                    lines.append(f'    y-axis "Dropped (%)" 0 --> {round(max_dr, 1)}')
                    lines.append(f"    bar {drop_ratios}")
                    lines.append("```")
                    lines.append("> **Dropped Ratio**: Percentage of raw extracted ideas whose nearest surviving concept falls below the semantic cutoff $\\tau_{\\text{loss}}$.")
                    lines.append("")

        # 3. Pareto Analysis & Insights
        lines.append("## 3. Comparative Observations & Pareto Analysis")
        lines.append("")
        if self.variations:
            best_cov = max(self.variations, key=lambda x: x.coverage_at_tau_pct)
            best_lev = max(self.variations, key=lambda x: x.compression_leverage_factor)
            best_p10 = max(self.variations, key=lambda x: x.worst_decile_p10)

            lines.append("### Graph Space & Compression Observations")
            lines.append(f"- **Maximum Semantic Coverage**: Variation `{best_cov.label}` achieved the highest coverage at **{best_cov.coverage_at_tau_pct:.2f}%** (with compute ratio $\\rho_p = {best_cov.parent_chunk_ratio_pct:.2f}\\%$).")
            lines.append(f"- **Maximum Compression Leverage**: Variation `{best_lev.label}` delivered the highest leverage at **{best_lev.compression_leverage_factor:.2f}x** (covering {best_lev.coverage_at_tau_pct:.2f}% while requiring only {best_lev.parent_chunk_ratio_pct:.2f}% extraction compute).")
            lines.append(f"- **Best Tail Robustness**: Variation `{best_p10.label}` provided the strongest worst-decile floor with $p_{{10}} = {best_p10.worst_decile_p10:.4f}$.")
            lines.append("")

            if has_drift:
                best_ret = max(self.variations, key=lambda x: x.forward_retention or 0.0)
                best_fid = max(self.variations, key=lambda x: x.backward_fidelity or 0.0)
                best_f1 = max(self.variations, key=lambda x: x.symmetric_idea_drift_score or 0.0)

                lines.append("### Post-Processing Idea Drift & Semantic Degradation Observations")
                lines.append(f"- **Highest Forward Retention**: Variation `{best_ret.label}` achieved $S_{{R\\to P}} = {best_ret.forward_retention:.4f}$ (indicating lossless semantic preservation of raw ideas).")
                lines.append(f"- **Highest Backward Fidelity**: Variation `{best_fid.label}` achieved $S_{{P\\to R}} = {best_fid.backward_fidelity:.4f}$ (strictly anchored to ground-truth raw concepts without hallucination drift).")
                lines.append(f"- **Best Symmetric Alignment ($F_1$)**: Variation `{best_f1.label}` scored $F_1 = {best_f1.symmetric_idea_drift_score:.4f}$ harmonic drift score.")

                if any((v.dropped_raw_count or 0) > 0 for v in self.variations):
                    worst_drop = max(self.variations, key=lambda x: x.dropped_raw_ratio_pct or 0.0)
                    lines.append(f"- **Concept Attrition Alert**: Variation `{worst_drop.label}` dropped {worst_drop.dropped_raw_count} raw ideas ({worst_drop.dropped_raw_ratio_pct:.1f}% below $\\tau_{{\\text{{loss}}}}$).")
                else:
                    lines.append("- **Zero Concept Loss**: Across all simulated variations, 0% of raw ideas were dropped below the $\\tau_{\\text{loss}}$ semantic threshold.")
                lines.append("")

        return "\n".join(lines)


# ==============================================================================
# 3. Preloaded Corpus Cache & Fast Offline Replay
# ==============================================================================


class PreloadedBookData(BaseModel):
    """
    Cached data for a single book loaded into memory for instant repeated simulation.
    """

    book_id: str
    book_title: str
    publication_date: str = "9999-12-31"
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    extractions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


def preload_simulation_data(
    state_cache_dir: Path | str,
    chunks_dir: Optional[Path | str] = None,
    graph_path: Optional[Path | str] = None,
    max_books: Optional[int] = None,
    embeddings_client: Optional[Any] = None,
) -> Tuple[List[PreloadedBookData], Dict[str, List[float]]]:
    """
    Scan and load book state files, companion chunks, and precomputed concept embeddings
    into memory once. This enables running dozens of threshold variations completely
    offline at native C++ speed (<1 second per variation).

    Parameters:
        state_cache_dir: Directory containing book_*_state.json files.
        chunks_dir: Optional directory containing book_*_chunks.json files.
        graph_path: Optional path to output/knowledge_graph.json to extract existing concept embeddings.
        max_books: Optional cap on number of books to preload for quick testing.
        embeddings_client: Optional embeddings client to pre-embed any extracted concepts missing from graph.

    Returns:
        Tuple of (preloaded_books, concept_embedding_map).
    """
    s_dir = Path(state_cache_dir).expanduser().resolve()
    candidate_state_dirs = [s_dir]
    sub_states = s_dir / ".book_states"
    if sub_states.is_dir():
        candidate_state_dirs.append(sub_states)

    state_files: Dict[str, Path] = {}
    for cdir in candidate_state_dirs:
        for f in cdir.glob("book_*_state.json"):
            stem = f.stem
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "book" and parts[-1] == "state":
                bid = "_".join(parts[1:-1])
                if bid not in state_files:
                    state_files[bid] = f

    # Chunks candidate directories
    chunk_dirs: List[Path] = []
    if chunks_dir:
        chunk_dirs.append(Path(chunks_dir).expanduser().resolve())
    chunk_dirs.extend([
        s_dir / "chunks",
        s_dir.parent / "chunks",
        s_dir,
    ])

    loaded_books: List[PreloadedBookData] = []
    concept_map: Dict[str, List[float]] = {}

    # 1. Preload concept embeddings from existing knowledge_graph.json if available
    g_path = Path(graph_path).expanduser().resolve() if graph_path else (s_dir.parent / "knowledge_graph.json")
    if not g_path.is_file() and (s_dir / "knowledge_graph.json").is_file():
        g_path = s_dir / "knowledge_graph.json"

    if g_path.is_file():
        try:
            with open(g_path, "r", encoding="utf-8") as gf:
                g_data = json.load(gf)
            nodes = g_data.get("nodes", [])
            for n in nodes:
                name = str(n.get("name") or "").strip().lower()
                emb = n.get("embedding")
                if name and isinstance(emb, list) and len(emb) > 1:
                    concept_map[name] = [float(x) for x in emb]
            logger.info(f"Loaded {len(concept_map):,} precomputed concept embeddings from {g_path}")
        except Exception as e:
            logger.warning(f"Could not load precomputed concept embeddings from {g_path}: {e}")

    # 2. Iterate and preload state files
    for bid, sf_path in state_files.items():
        if max_books and len(loaded_books) >= max_books:
            break
        try:
            with open(sf_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
        except Exception as e:
            logger.warning(f"Failed to read state file {sf_path}: {e}")
            continue

        btitle = str(s_data.get("book_title") or f"Book {bid}")
        bpub = normalize_sort_date(s_data.get("publication_date"))

        # Extract cached extractions
        extractions: Dict[str, List[Dict[str, Any]]] = {}
        for cid, prec in s_data.get("processed_chunks", {}).items():
            ext = prec.get("extraction")
            if ext and isinstance(ext, dict):
                c_list = ext.get("concepts") or []
                if isinstance(c_list, list) and c_list:
                    extractions[str(cid)] = c_list

        # Load chunks (from state embedded chunks or companion chunk JSON)
        raw_chunks: List[Dict[str, Any]] = []
        if s_data.get("chunks"):
            raw_chunks = s_data["chunks"]

        if not raw_chunks:
            # Look in chunk dirs
            for cdir in chunk_dirs:
                cp = cdir / f"book_{bid}_chunks.json"
                if cp.is_file():
                    try:
                        with open(cp, "r", encoding="utf-8") as cf:
                            c_data = json.load(cf)
                        raw_chunks = c_data.get("chunks", [])
                        if raw_chunks:
                            break
                    except Exception:
                        pass

        # Filter chunks that have valid embeddings and character counts
        valid_chunks: List[Dict[str, Any]] = []
        for c in raw_chunks:
            emb = c.get("embedding")
            if isinstance(emb, (list, np.ndarray)) and len(emb) > 1:
                # Normalize embedding to unit length
                arr = np.asarray(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                unit_emb = (arr / norm).tolist() if norm > 0 else arr.tolist()

                c_dict = {
                    "chunk_id": str(c.get("chunk_id") or ""),
                    "chunk_idx": int(c.get("chunk_idx") or 0),
                    "section_title": str(c.get("section_title") or ""),
                    "char_count": int(c.get("char_count") or len(str(c.get("text") or ""))),
                    "embedding": unit_emb,
                }
                valid_chunks.append(c_dict)

        if valid_chunks:
            # Sort chunks by chunk_idx
            valid_chunks.sort(key=lambda x: x["chunk_idx"])
            loaded_books.append(
                PreloadedBookData(
                    book_id=str(bid),
                    book_title=btitle,
                    publication_date=bpub,
                    chunks=valid_chunks,
                    extractions=extractions,
                )
            )

    # Sort books chronologically by publication date, tie-breaking on book ID
    loaded_books.sort(key=lambda b: (b.publication_date, b.book_id))

    # 3. If embeddings_client is provided, batch-embed any extracted concepts missing from concept_map
    if embeddings_client is not None and loaded_books:
        missing_names = set()
        for b in loaded_books:
            for cid, clist in b.extractions.items():
                for c in clist:
                    cname = str(c.get("name") or "").strip()
                    if cname and cname.lower() not in concept_map:
                        missing_names.add(cname)
        if missing_names:
            logger.info(f"Computing embeddings for {len(missing_names)} extracted concepts missing from knowledge graph...")
            missing_list = list(missing_names)
            try:
                new_vecs = embeddings_client.embed_documents(missing_list, batch_size=32)
                for name, v in zip(missing_list, new_vecs):
                    if v and len(v) > 1:
                        arr = np.asarray(v, dtype=np.float32)
                        norm = float(np.linalg.norm(arr))
                        unit_v = (arr / norm).tolist() if norm > 0 else arr.tolist()
                        concept_map[name.lower()] = unit_v
                logger.info(f"Successfully embedded and cached {len(missing_names)} concepts into simulation index.")
            except Exception as e:
                logger.warning(f"Failed to batch-embed missing concepts: {e}")

    logger.info(f"Preloaded {len(loaded_books)} books with {sum(len(b.chunks) for b in loaded_books):,} chunks for simulation.")
    return (loaded_books, concept_map)


def extract_corpus_raw_vectors_and_weights(
    preloaded_books: List[PreloadedBookData],
    concept_embedding_map: Dict[str, List[float]],
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract all raw concept embeddings and evidence quote weights across all preloaded corpus state files.
    """
    raw_vecs: List[List[float]] = []
    raw_weights: List[int] = []
    for b in preloaded_books:
        for cid, clist in b.extractions.items():
            for c in clist:
                c_name = str(c.get("name") or "").strip().lower()
                c_emb = concept_embedding_map.get(c_name)
                if c_emb:
                    raw_vecs.append(c_emb)
                    q = str(c.get("supporting_quote") or c.get("source_quote") or "")
                    raw_weights.append(len(q) if q else 100)
    if raw_vecs:
        return np.asarray(raw_vecs, dtype=np.float32), raw_weights
    return np.empty((0, 0), dtype=np.float32), []


# ==============================================================================
# 4. Single Variation Simulation Runner
# ==============================================================================


def simulate_variation(
    pair: ThresholdPair,
    preloaded_books: List[PreloadedBookData],
    concept_embedding_map: Dict[str, List[float]],
    tau: float = 0.75,
    skip_limbo: bool = True,
    embeddings_client: Optional[Any] = None,
    pool: Optional[Any] = None,
    postprocess: bool = True,
    postprocess_threshold: float = 0.92,
    tau_loss: float = 0.70,
    corpus_raw_embeddings: Optional[np.ndarray] = None,
    corpus_raw_weights: Optional[List[int]] = None,
) -> SimulationVariationResult:
    """
    Execute an isolated simulation of chunk routing and knowledge graph generation
    for a specific (low, high) threshold pair using in-memory native C++ usearch index.
    Guarantees a 100% clean start from scratch: previous graph nodes, indexes, and
    allocations are forcefully cleaned before and after the run.
    When postprocess is True, applies semantic deduplication to generated concepts
    and evaluates idea drift (forward retention, backward fidelity, symmetric score,
    weighted retention, and macro centroid alignment) comparing RAW vs POST ideas.

    Parameters:
        pair: ThresholdPair (unassigned lower bound, high-confidence upper bound).
        preloaded_books: List of preloaded books sorted chronologically.
        concept_embedding_map: Pre-indexed map of concept_name.lower() -> embedding vector.
        tau: Semantic relevance threshold for graph space evaluation (default: 0.75).
        skip_limbo: If True, limbo chunks are bypassed/skipped; if False, extracted.
        embeddings_client: Optional LangChain embeddings client for any missing concept embeddings.
        postprocess: If True, performs postprocessing deduplication and drift calculation.
        postprocess_threshold: Cosine similarity cutoff for merging redundant concepts (default 0.92).
        tau_loss: Cosine similarity threshold below which a raw idea is considered dropped (default 0.70).

    Returns:
        SimulationVariationResult containing full graph space & leverage metrics.
    """
    # 1. Enforce pristine memory & state clean start before beginning variation
    cleanup_memory()

    t_start = time.perf_counter()

    # Initialize brand-new, completely isolated in-memory graph store and HNSW index from scratch
    store = ConceptGraphStore()
    index = IdeaEmbeddingIndex()

    # Strict invariant verification: must be 100% clean and empty
    assert store.graph.number_of_nodes() == 0, "Graph store must have 0 nodes at variation start"
    assert len(index.idea_names) == 0, "Idea index names must be empty at variation start"
    assert index.index is None or len(index.index) == 0, "Index must be empty at variation start"

    total_chunks = 0
    unassigned_count = 0
    high_confidence_count = 0
    limbo_count = 0

    all_universe_vectors: List[List[float]] = []
    all_universe_weights: List[int] = []
    parent_mask: List[bool] = []

    # Dynamic concept embedding cache local to this simulation run (isolated deep copy)
    local_concept_map: Dict[str, List[float]] = {k: list(v) for k, v in concept_embedding_map.items()}

    # Replay books in chronological sequence
    for book in preloaded_books:
        for chk in book.chunks:
            total_chunks += 1
            chunk_vec = chk["embedding"]
            chunk_weight = max(1, chk["char_count"])
            chunk_id = chk["chunk_id"]

            all_universe_vectors.append(chunk_vec)
            all_universe_weights.append(chunk_weight)

            # Route chunk against current graph state
            sim = 0.0
            best_idea = None
            best_summary = ""

            if index.index is not None and len(index.index) > 0 and len(index.idea_names) > 0:
                matches = index.query_batch([chunk_vec], top_k=1)
                if matches and matches[0]:
                    top_m = matches[0][0]
                    sim = float(top_m.get("score") or 0.0)
                    best_idea = top_m.get("name")
                    best_summary = top_m.get("summary") or ""

            # Routing Decision Matrix
            if sim < pair.low:
                # Priority 1: UNASSIGNED (Below lower threshold)
                unassigned_count += 1
                parent_mask.append(True)

                # Retrieve raw extraction concepts from state cache
                c_list = book.extractions.get(chunk_id, [])
                for c in c_list:
                    c_name = str(c.get("name") or "").strip()
                    if not c_name:
                        continue
                    c_norm = c_name.lower()
                    c_summary = str(c.get("summary") or c.get("brief_description") or c_name)

                    c_emb = local_concept_map.get(c_norm)
                    if c_emb is None and embeddings_client is not None:
                        try:
                            emb_res = embeddings_client.embed_query(c_summary)
                            if emb_res and len(emb_res) > 1:
                                arr_e = np.asarray(emb_res, dtype=np.float32)
                                norm_e = float(np.linalg.norm(arr_e))
                                c_emb = (arr_e / norm_e).tolist() if norm_e > 0 else arr_e.tolist()
                                local_concept_map[c_norm] = c_emb
                        except Exception:
                            pass

                    # NEVER fall back to raw chunk_vec as a concept embedding!
                    # Raw paragraph vectors act as artificial 1.0 self-anchors that distort
                    # cosine proximity and inflate coverage metrics. Skip if un-embedded.
                    if c_emb is None:
                        logger.debug(f"Skipping un-embedded concept '{c_name}' in simulation")
                        continue

                    brief = str(c.get("brief_description") or c_summary)
                    detailed = str(c.get("detailed_explanation") or c_summary)
                    cat = str(c.get("category") or "Idea")
                    weight = int(c.get("weight") or 5)
                    sq = str(c.get("supporting_quote") or c.get("quote") or "")

                    index.add_idea(name=c_name, summary=c_summary, vector=c_emb)
                    store.add_concept(
                        Concept(name=c_name, summary=c_summary, brief_description=brief, detailed_explanation=detailed, category=cat, weight=weight),
                        embedding=c_emb,
                    )
                    if not store.graph.has_node(f"chunk:{chunk_id}"):
                        store.graph.add_node(f"chunk:{chunk_id}", type="Chunk", chunk_id=chunk_id)
                    store.graph.add_edge(f"concept:{c_name}", f"chunk:{chunk_id}", relation="SUPPORTED_BY", quote=sq)
                    store.graph.add_edge(f"chunk:{chunk_id}", f"concept:{c_name}", relation="SUPPORTS_IDEA", quote=sq)

            elif sim >= pair.high:
                # Priority 2: HIGH CONFIDENCE (At or above upper threshold)
                high_confidence_count += 1
                parent_mask.append(False)
                # Direct assignment: no new concepts added to index

            else:
                # Priority 3: LIMBO [low, high)
                limbo_count += 1
                if skip_limbo:
                    # Bypassed/skipped by heuristic
                    parent_mask.append(False)
                else:
                    # Processed / validated via LLM
                    parent_mask.append(True)
                    c_list = book.extractions.get(chunk_id, [])
                    for c in c_list:
                        c_name = str(c.get("name") or "").strip()
                        if not c_name:
                            continue
                        c_norm = c_name.lower()
                        c_summary = str(c.get("summary") or c.get("brief_description") or c_name)
                        c_emb = local_concept_map.get(c_norm)
                        if c_emb is None and embeddings_client is not None:
                            try:
                                emb_res = embeddings_client.embed_query(c_summary)
                                if emb_res and len(emb_res) > 1:
                                    arr_e = np.asarray(emb_res, dtype=np.float32)
                                    norm_e = float(np.linalg.norm(arr_e))
                                    c_emb = (arr_e / norm_e).tolist() if norm_e > 0 else arr_e.tolist()
                                    local_concept_map[c_norm] = c_emb
                            except Exception:
                                pass

                        if c_emb is None:
                            continue

                        brief = str(c.get("brief_description") or c_summary)
                        detailed = str(c.get("detailed_explanation") or c_summary)
                        cat = str(c.get("category") or "Idea")
                        weight = int(c.get("weight") or 5)
                        sq = str(c.get("supporting_quote") or c.get("quote") or "")

                        index.add_idea(name=c_name, summary=c_summary, vector=c_emb)
                        store.add_concept(
                            Concept(name=c_name, summary=c_summary, brief_description=brief, detailed_explanation=detailed, category=cat, weight=weight),
                            embedding=c_emb,
                        )
                        if not store.graph.has_node(f"chunk:{chunk_id}"):
                            store.graph.add_node(f"chunk:{chunk_id}", type="Chunk", chunk_id=chunk_id)
                        store.graph.add_edge(f"concept:{c_name}", f"chunk:{chunk_id}", relation="SUPPORTED_BY", quote=sq)
                        store.graph.add_edge(f"chunk:{chunk_id}", f"concept:{c_name}", relation="SUPPORTS_IDEA", quote=sq)

    # 4. Evaluate Embedding Space against Universe
    U_arr = np.array(all_universe_vectors, dtype=np.float32) if all_universe_vectors else np.empty((0, 0), dtype=np.float32)
    G_arr = index.vectors if (index.vectors is not None and len(index.vectors) > 0) else np.empty((0, 0), dtype=np.float32)
    W_arr = np.array(all_universe_weights, dtype=np.float64) if all_universe_weights else np.empty(0, dtype=np.float64)
    P_mask = np.array(parent_mask, dtype=bool) if parent_mask else np.empty(0, dtype=bool)

    raw_metrics = evaluate_graph_space(
        universe_embeddings=U_arr,
        graph_embeddings=G_arr,
        tau=tau,
        parent_chunk_mask=P_mask,
        chunk_weights=W_arr,
    )

    # 4. Optional Post-Processing Step & Drift Metrics Calculation
    post_nodes_cnt: Optional[int] = None
    drift_dict: Optional[Dict[str, Any]] = None

    if G_arr.shape[0] > 0:
        if postprocess:
            # Execute IdeaPostProcessor on the simulated graph store
            postprocessor = IdeaPostProcessor(
                similarity_threshold=postprocess_threshold,
                embeddings=embeddings_client,
                pool=pool,
                resume=False,
            )
            postprocessor.process_graph(store, console=None)

            # Collect surviving post-processed concept vectors directly from the graph store
            post_vec_list: List[np.ndarray] = []
            for nid, attrs in store.graph.nodes(data=True):
                if (attrs.get("type") == "Concept" or attrs.get("is_idea")) and attrs.get("embedding"):
                    v = np.asarray(attrs["embedding"], dtype=np.float32)
                    norm = float(np.linalg.norm(v))
                    post_vec_list.append((v / norm) if norm > 0 else v)

            if post_vec_list:
                post_vectors = np.array(post_vec_list, dtype=np.float32)
                post_nodes_cnt = len(post_vectors)
            else:
                post_vectors = G_arr
                post_nodes_cnt = len(G_arr)
        else:
            post_vectors = G_arr
            post_nodes_cnt = len(G_arr)

        # Compute post-processing idea drift comparing corpus RAW ideas vs POST KG concepts
        raw_target = corpus_raw_embeddings if (corpus_raw_embeddings is not None and len(corpus_raw_embeddings) > 0) else G_arr
        raw_w_target = corpus_raw_weights if (corpus_raw_embeddings is not None and len(corpus_raw_embeddings) > 0) else None

        drift_dict = calculate_postprocessing_drift(
            raw_embeddings=raw_target,
            post_embeddings=post_vectors,
            raw_weights=raw_w_target,
            tau_loss=tau_loss,
        )

    duration = time.perf_counter() - t_start

    result = SimulationVariationResult(
        pair=pair,
        label=pair.label,
        tau=tau,
        total_chunks=total_chunks,
        unassigned_count=unassigned_count,
        high_confidence_count=high_confidence_count,
        limbo_count=limbo_count,
        nodes_count=int(raw_metrics.get("nodes_count") or 0),
        chamfer_mean=float(raw_metrics.get("chamfer_mean") or 0.0),
        chamfer_std=float(raw_metrics.get("chamfer_std") or 0.0),
        median_p50=float(raw_metrics.get("median_p50") or 0.0),
        worst_decile_p10=float(raw_metrics.get("worst_decile_p10") or 0.0),
        coverage_at_tau_pct=float(raw_metrics.get("coverage_at_tau_pct") or 0.0),
        worst_case_gap=float(raw_metrics.get("worst_case_gap") or 1.0),
        vendi_diversity_score=float(raw_metrics.get("vendi_diversity_score") or 0.0),
        parent_chunks_count=int(raw_metrics.get("parent_chunks_count") or 0),
        skipped_chunks_count=int(raw_metrics.get("skipped_chunks_count") or 0),
        parent_chunk_ratio_pct=float(raw_metrics.get("parent_chunk_ratio_pct") or 0.0),
        skipped_retention_pct=float(raw_metrics.get("skipped_retention_pct") or 0.0),
        generalization_surplus_pct=float(raw_metrics.get("generalization_surplus_pct") or 0.0),
        compression_leverage_factor=float(raw_metrics.get("compression_leverage_factor") or 0.0),
        postprocessed_nodes_count=post_nodes_cnt,
        forward_retention=float(drift_dict["forward_retention"]) if drift_dict else None,
        backward_fidelity=float(drift_dict["backward_fidelity"]) if drift_dict else None,
        symmetric_idea_drift_score=float(drift_dict["symmetric_idea_drift_score"]) if drift_dict else None,
        weighted_retention=float(drift_dict["weighted_retention"]) if drift_dict else None,
        centroid_cosine_alignment=float(drift_dict["centroid_cosine_alignment"]) if drift_dict else None,
        p10_retention=float(drift_dict["p10_retention"]) if drift_dict else None,
        min_retention=float(drift_dict["min_retention"]) if drift_dict else None,
        dropped_raw_count=int(drift_dict["dropped_raw_count"]) if drift_dict else None,
        dropped_raw_ratio_pct=float(drift_dict["dropped_raw_ratio_pct"]) if drift_dict else None,
        duration_seconds=round(duration, 3),
    )

    # 5. Clean up graph and index memory to guarantee absolute cleanliness for next variation
    del store
    del index
    del all_universe_vectors
    del all_universe_weights
    del parent_mask
    del U_arr
    del G_arr
    del W_arr
    del P_mask
    cleanup_memory()

    return result


# ==============================================================================
# 5. Full Simulation Suite Orchestrator
# ==============================================================================


def run_threshold_simulation_suite(
    pairs: Sequence[ThresholdPair] | str,
    state_cache_dir: Path | str,
    chunks_dir: Optional[Path | str] = None,
    graph_path: Optional[Path | str] = None,
    tau: float = 0.75,
    skip_limbo: bool = True,
    max_books: Optional[int] = None,
    embeddings_client: Optional[Any] = None,
    pool: Optional[Any] = None,
    console: Optional[Console] = None,
    postprocess: bool = True,
    postprocess_threshold: float = 0.92,
    tau_loss: float = 0.70,
) -> SimulationSuiteReport:
    """
    Orchestrate full multi-variation threshold sweep:
    1. Preloads all book state and chunk cache files once into memory.
    2. Runs isolated simulation for each threshold pair from scratch.
    3. Aggregates results into a unified comparative report with Rich table and Markdown plots.
    """
    con = console or Console()
    parsed_pairs = parse_threshold_pairs(pairs)

    con.print(f"[bold cyan]⚡ Preloading corpus state and chunk cache from '{state_cache_dir}'...[/bold cyan]")
    preloaded_books, concept_map = preload_simulation_data(
        state_cache_dir=state_cache_dir,
        chunks_dir=chunks_dir,
        graph_path=graph_path,
        max_books=max_books,
        embeddings_client=embeddings_client,
    )

    if not preloaded_books:
        raise ValueError(f"No valid books or chunks found in cache directory: {state_cache_dir}")

    total_chunks = sum(len(b.chunks) for b in preloaded_books)
    total_chars = sum(sum(c["char_count"] for c in b.chunks) for b in preloaded_books)

    corpus_raw_vectors, corpus_raw_weights = extract_corpus_raw_vectors_and_weights(preloaded_books, concept_map)
    total_raw_concepts = len(corpus_raw_vectors)

    con.print(
        f"[green]✓ Preloaded [bold]{len(preloaded_books)}[/bold] books ([bold]{total_chunks:,}[/bold] chunks, "
        f"[bold]{total_chars:,}[/bold] chars, [bold]{total_raw_concepts:,}[/bold] raw concepts, "
        f"[bold]{len(concept_map):,}[/bold] concept embeddings).[/green]"
    )
    con.print(f"[dim]Executing {len(parsed_pairs)} threshold variation simulation(s)...[/dim]")

    results: List[SimulationVariationResult] = []
    for idx, p in enumerate(parsed_pairs, start=1):
        con.print(f"  [cyan][{idx}/{len(parsed_pairs)}][/cyan] [bold]Clean start from scratch[/bold] for variation [bold]{p.label}[/bold] (low={p.low:.2f}, high={p.high:.2f})...", end=" ")
        t0 = time.perf_counter()
        res = simulate_variation(
            pair=p,
            preloaded_books=preloaded_books,
            concept_embedding_map=concept_map,
            tau=tau,
            skip_limbo=skip_limbo,
            embeddings_client=embeddings_client,
            pool=pool,
            postprocess=postprocess,
            postprocess_threshold=postprocess_threshold,
            tau_loss=tau_loss,
            corpus_raw_embeddings=corpus_raw_vectors,
            corpus_raw_weights=corpus_raw_weights,
        )
        elapsed = time.perf_counter() - t0
        ret_msg = f", Ret: {res.forward_retention:.3f}, Drop: {res.dropped_raw_ratio_pct:.1f}%" if res.forward_retention is not None else ""
        con.print(f"[green]Done in {elapsed:.2f}s[/green] (Nodes: {res.nodes_count:,}, Cov: {res.coverage_at_tau_pct:.1f}%, Leverage: {res.compression_leverage_factor:.2f}x{ret_msg})")
        results.append(res)

    report = SimulationSuiteReport(
        tau=tau,
        total_corpus_chunks=total_chunks,
        total_corpus_chars=total_chars,
        total_books=len(preloaded_books),
        total_raw_concepts=total_raw_concepts,
        skip_limbo=skip_limbo,
        variations=results,
    )

    return report
