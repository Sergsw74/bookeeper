"""
Token Usage Statistics and Multi-Tier Cost Estimation Subsystem.

Tracks token consumption across four segregated categories:
1. Tokens for embeddings
2. Input tokens (prompt)
3. Output tokens (completion)
4. Cache tokens (prompt cache read / hits)

Statistics are stored strictly in token terms in an accumulating stats file,
while human-readable cost estimates (session cost, total graph cost, and 3-tier
pricing) are dynamically calculated and printed in the terminal UI.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Fallback regex for fast token estimation when tiktoken is not available
_WORD_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count using tiktoken (cl100k_base) if available,
    falling back to word/punctuation regex tokenization.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(_WORD_PUNCT_RE.findall(text))


@dataclass
class TokenCounts:
    """Four-way segregated token counts."""
    embedding_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.embedding_tokens + self.input_tokens + self.output_tokens + self.cache_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "embedding_tokens": self.embedding_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_tokens": self.cache_tokens,
            "total_tokens": self.total_tokens,
        }

    def add(self, other: "TokenCounts") -> "TokenCounts":
        return TokenCounts(
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
        )


class TokenTracker:
    """
    Thread-safe tracker for token usage across the Bookkeeper processing pipeline.
    Maintains session tokens and cumulative whole-graph tokens, persisting only token counts.
    """

    _instance: Optional["TokenTracker"] = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self.session_counts = TokenCounts()
        self.cumulative_counts = TokenCounts()
        self.sessions_count: int = 0
        self.stats_file_path: Optional[Path] = None
        self.session_start_time: float = time.time()

    @classmethod
    def get_instance(cls) -> "TokenTracker":
        """Retrieve the global TokenTracker singleton."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> "TokenTracker":
        """Reset the global TokenTracker singleton (useful for isolated tests)."""
        with cls._singleton_lock:
            cls._instance = cls()
            return cls._instance

    def reset_session(self) -> None:
        """Reset current session counts to zero while preserving cumulative whole-graph counts."""
        with self._lock:
            self.session_counts = TokenCounts()
            self.session_start_time = time.time()

    def record_embedding_tokens(self, tokens: int) -> None:
        """Record tokens consumed by embedding models."""
        cnt = max(0, int(tokens))
        if cnt == 0:
            return
        with self._lock:
            self.session_counts.embedding_tokens += cnt

    def record_llm_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_tokens: int = 0,
    ) -> None:
        """Record input, output, and prompt cache tokens consumed by LLM calls."""
        inp = max(0, int(input_tokens))
        out = max(0, int(output_tokens))
        cache = max(0, int(cache_tokens))

        with self._lock:
            self.session_counts.input_tokens += inp
            self.session_counts.output_tokens += out
            self.session_counts.cache_tokens += cache

    def record_response_data(
        self,
        data: Dict[str, Any],
        prompt_text: str = "",
        completion_text: str = "",
        task_type: str = "llm",
    ) -> None:
        """
        Extract token counts from Ollama or OpenAI-compatible JSON response data,
        falling back to local token estimation if provider metrics are omitted.
        """
        if not isinstance(data, dict):
            return

        tt = (task_type or "llm").strip().lower()

        # Check for Ollama / OpenAI embedding response
        if tt == "embedding" or "embeddings" in data:
            emb_tokens = data.get("prompt_eval_count") or data.get("eval_count") or 0
            if not emb_tokens and prompt_text:
                emb_tokens = estimate_token_count(prompt_text)
            self.record_embedding_tokens(emb_tokens)
            return

        # Check for LLM generation response
        prompt_eval = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")

        if not completion_text and isinstance(data.get("message"), dict):
            completion_text = str(data["message"].get("content") or "")

        total_prompt_est = estimate_token_count(prompt_text) if prompt_text else 0
        total_comp_est = estimate_token_count(completion_text) if completion_text else 0

        # OpenAI-style usage dict
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            if prompt_eval is None and "prompt_tokens" in usage:
                prompt_eval = usage.get("prompt_tokens")
            if eval_count is None and "completion_tokens" in usage:
                eval_count = usage.get("completion_tokens")
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        else:
            cached = data.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        # Cache tokens reported by Ollama or other local engines
        if not cached:
            cached = data.get("cache_tokens") or 0

        # If cache tokens were not explicitly reported by provider, check prompt cache differential:
        # (e.g. In Ollama, prompt_eval_count reflects only non-cached newly evaluated prompt tokens)
        if cached == 0 and prompt_eval is not None and total_prompt_est > 0:
            if total_prompt_est > prompt_eval + 8:
                cached = max(0, total_prompt_est - prompt_eval)
            elif prompt_eval == 0:
                cached = total_prompt_est

        # Fallback estimation if response did not include token telemetry
        if prompt_eval is None or (prompt_eval == 0 and cached == 0 and total_prompt_est > 0):
            prompt_eval = total_prompt_est
        if eval_count is None or (eval_count == 0 and total_comp_est > 0):
            eval_count = total_comp_est

        self.record_llm_tokens(
            input_tokens=prompt_eval or 0,
            output_tokens=eval_count or 0,
            cache_tokens=cached or 0,
        )

    def load_cumulative(self, path: Union[Path, str]) -> None:
        """
        Load cumulative token statistics from the persistent stats file.
        Only tokens are read from storage.
        """
        p = Path(path).expanduser().resolve()
        self.stats_file_path = p
        if not p.is_file():
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            cum = data.get("cumulative", {})
            with self._lock:
                self.cumulative_counts = TokenCounts(
                    embedding_tokens=int(cum.get("embedding_tokens", 0)),
                    input_tokens=int(cum.get("input_tokens", 0)),
                    output_tokens=int(cum.get("output_tokens", 0)),
                    cache_tokens=int(cum.get("cache_tokens", 0)),
                )
                self.sessions_count = int(data.get("sessions_count", 0))
            logger.debug(
                f"Loaded cumulative token statistics: {self.cumulative_counts.total_tokens:,} tokens "
                f"across {self.sessions_count} previous session(s) from {p.name}"
            )
        except Exception as e:
            logger.warning(f"Could not load cumulative token stats from {p}: {e}")

    def save_cumulative(
        self,
        path: Optional[Union[Path, str]] = None,
        books_processed: int = 0,
    ) -> Path:
        """
        Persist token statistics into JSON storage.
        STRICT REQUIREMENT: Only token counts are stored; no prices or currency amounts are saved.
        """
        target_path = Path(path).expanduser().resolve() if path else self.stats_file_path
        if not target_path:
            raise ValueError("No stats file path specified for saving token statistics.")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Advance cumulative counts by current session tokens
            new_cumulative = self.cumulative_counts.add(self.session_counts)
            new_sessions_count = self.sessions_count + 1

            payload = {
                "updated_at": time.time(),
                "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sessions_count": new_sessions_count,
                "cumulative": new_cumulative.to_dict(),
                "last_session": {
                    "duration_seconds": round(time.time() - self.session_start_time, 2),
                    "books_processed": books_processed,
                    "tokens": self.session_counts.to_dict(),
                },
            }

            # Update in-memory state
            self.cumulative_counts = new_cumulative
            self.sessions_count = new_sessions_count

        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_path.replace(target_path)

        logger.info(
            f"Saved updated token statistics to {target_path} (Cumulative: {new_cumulative.total_tokens:,} tokens)"
        )
        return target_path

    def get_summary_snapshot(self) -> Dict[str, Any]:
        """Return a read-only snapshot of current session and cumulative whole-graph tokens."""
        with self._lock:
            combined = self.cumulative_counts.add(self.session_counts)
            return {
                "session": self.session_counts.to_dict(),
                "cumulative_previous": self.cumulative_counts.to_dict(),
                "whole_graph": combined.to_dict(),
                "sessions_count": self.sessions_count,
            }

    @staticmethod
    def calculate_prices_for_section(
        tokens: int,
        unit_price_per_1m: float,
    ) -> float:
        """Calculate approximate monetary cost from token count and unit rate per 1M tokens."""
        return (max(0, tokens) / 1_000_000.0) * max(0.0, unit_price_per_1m)

    def print_summary_table(
        self,
        console: Any,
        pricing_config: Optional[Any] = None,
    ) -> None:
        """
        Print a beautiful, comprehensive Rich table summarizing token metrics across
        the four segregated sections, with three types of prices for each section:
        1. Unit Price Rate ($/1M tokens)
        2. Approximate Session Cost ($)
        3. Approximate Whole Knowledge Graph Cost ($)

        Also renders the multi-tier pricing matrix (Tier 1 Budget, Tier 2 Standard, Tier 3 Flagship)
        for human reliability and financial transparency.
        """
        if console is None:
            return

        try:
            from rich.panel import Panel
            from rich.table import Table
        except ImportError:
            return

        snapshot = self.get_summary_snapshot()
        session_tok = snapshot["session"]
        graph_tok = snapshot["whole_graph"]

        # Resolve unit rates per 1M tokens for the 4 sections
        rates = self._resolve_section_rates(pricing_config)

        # 1. Main Token Breakdown Table (4 Segregated Sections + Total)
        table = Table(
            title="📊 Knowledge Graph Token Usage & Financial Cost Summary",
            header_style="bold cyan",
            border_style="cyan",
            expand=True,
        )

        table.add_column("Token Section", style="bold white", ratio=3)
        table.add_column("Session Tokens", justify="right", style="bold green", ratio=2)
        table.add_column("Whole Graph Tokens", justify="right", style="bold yellow", ratio=2)
        table.add_column("Unit Rate ($/1M)", justify="right", style="dim cyan", ratio=2)
        table.add_column("Session Cost ($)", justify="right", style="bold green", ratio=2)
        table.add_column("Whole Graph Cost ($)", justify="right", style="bold yellow", ratio=2)

        sections = [
            ("1. Tokens for Embeddings", "embedding_tokens", rates["embedding"]),
            ("2. Input Tokens (Prompt)", "input_tokens", rates["input"]),
            ("3. Output Tokens (Completion)", "output_tokens", rates["output"]),
            ("4. Cache Tokens (Prompt Cache Read)", "cache_tokens", rates["cache"]),
        ]

        total_session_cost = 0.0
        total_graph_cost = 0.0

        for label, key, rate in sections:
            s_cnt = session_tok[key]
            g_cnt = graph_tok[key]
            s_cost = self.calculate_prices_for_section(s_cnt, rate)
            g_cost = self.calculate_prices_for_section(g_cnt, rate)
            total_session_cost += s_cost
            total_graph_cost += g_cost

            rate_str = f"${rate:.3f}" if rate > 0 else "$0.000 (free/local)"
            table.add_row(
                label,
                f"{s_cnt:,}",
                f"{g_cnt:,}",
                rate_str,
                f"${s_cost:.4f}",
                f"${g_cost:.4f}",
            )

        table.add_section()
        table.add_row(
            "[bold white]Total Cumulative Tokens[/bold white]",
            f"[bold green]{session_tok['total_tokens']:,}[/bold green]",
            f"[bold yellow]{graph_tok['total_tokens']:,}[/bold yellow]",
            "-",
            f"[bold green]${total_session_cost:.4f}[/bold green]",
            f"[bold yellow]${total_graph_cost:.4f}[/bold yellow]",
        )

        console.print()
        console.print(table)

        # 2. Render 3-Tier Multi-Pricing Reference Matrix
        tier_table = self._build_tier_matrix_table(session_tok, graph_tok, pricing_config)
        if tier_table:
            console.print(tier_table)

        stats_hint = f"output file: [bold cyan]{self.stats_file_path.name if self.stats_file_path else 'token_usage_stats.json'}[/bold cyan]"
        console.print(
            f"  [dim]• Token telemetry persisted to {stats_hint} (pure token metrics; prices are estimated dynamically).[/dim]\n"
        )

    def _resolve_section_rates(self, pricing_config: Any) -> Dict[str, float]:
        """Resolve unit pricing ($/1M) for the 4 sections from config or defaults."""
        rates = {
            "embedding": 0.02,
            "input": 0.15,
            "output": 0.60,
            "cache": 0.075,
        }

        if pricing_config is None:
            return rates

        # If pricing_config is a Pydantic model or dict
        if hasattr(pricing_config, "embedding_price_per_1m") and pricing_config.embedding_price_per_1m is not None:
            rates["embedding"] = float(pricing_config.embedding_price_per_1m)
        elif hasattr(pricing_config, "embedding"):
            rates["embedding"] = float(pricing_config.embedding)

        if hasattr(pricing_config, "input_price_per_1m") and pricing_config.input_price_per_1m is not None:
            rates["input"] = float(pricing_config.input_price_per_1m)
        elif hasattr(pricing_config, "input"):
            rates["input"] = float(pricing_config.input)

        if hasattr(pricing_config, "output_price_per_1m") and pricing_config.output_price_per_1m is not None:
            rates["output"] = float(pricing_config.output_price_per_1m)
        elif hasattr(pricing_config, "output"):
            rates["output"] = float(pricing_config.output)

        if hasattr(pricing_config, "cache_price_per_1m") and pricing_config.cache_price_per_1m is not None:
            rates["cache"] = float(pricing_config.cache_price_per_1m)
        elif hasattr(pricing_config, "cache"):
            rates["cache"] = float(pricing_config.cache)

        if isinstance(pricing_config, dict):
            for k in ("embedding", "input", "output", "cache"):
                if k in pricing_config:
                    rates[k] = float(pricing_config[k])
                elif f"{k}_price_per_1m" in pricing_config:
                    rates[k] = float(pricing_config[f"{k}_price_per_1m"])

        return rates

    def _build_tier_matrix_table(
        self,
        session_tok: Dict[str, int],
        graph_tok: Dict[str, int],
        pricing_config: Any,
    ) -> Optional[Any]:
        """Construct the 3-Tier Multi-Pricing comparative table."""
        try:
            from rich.table import Table
        except ImportError:
            return None

        # 3 Default Industry Standard Pricing Tiers ($ per 1M tokens)
        tiers = [
            (
                "Tier 1: Budget / Local / DeepSeek",
                {"embedding": 0.02, "input": 0.14, "output": 0.28, "cache": 0.014},
                "cyan",
            ),
            (
                "Tier 2: Standard / GPT-4o-mini",
                {"embedding": 0.10, "input": 0.15, "output": 0.60, "cache": 0.075},
                "green",
            ),
            (
                "Tier 3: Flagship / GPT-4o / Claude Sonnet",
                {"embedding": 0.15, "input": 2.50, "output": 10.00, "cache": 1.25},
                "magenta",
            ),
        ]

        # Override tiers if configured in TokenPricingConfig
        if pricing_config is not None:
            for idx, key in enumerate(("tier1", "tier2", "tier3")):
                tier_obj = getattr(pricing_config, key, None)
                tier_name = getattr(pricing_config, f"{key}_name", None)
                if tier_obj is not None:
                    custom_rates = {
                        "embedding": getattr(tier_obj, "embedding", tiers[idx][1]["embedding"]),
                        "input": getattr(tier_obj, "input", tiers[idx][1]["input"]),
                        "output": getattr(tier_obj, "output", tiers[idx][1]["output"]),
                        "cache": getattr(tier_obj, "cache", tiers[idx][1]["cache"]),
                    }
                    c_name = tier_name or tiers[idx][0]
                    tiers[idx] = (c_name, custom_rates, tiers[idx][2])

        tier_table = Table(
            title="🏷️ 3-Tier Pricing Model Estimates (Approximate Equivalent Cloud Costs)",
            header_style="bold magenta",
            border_style="magenta",
            expand=True,
        )
        tier_table.add_column("Pricing Model Tier", style="bold white", ratio=3)
        tier_table.add_column("Rates (Emb / In / Out / Cache per 1M)", style="dim", ratio=3)
        tier_table.add_column("Session Approx Cost", justify="right", style="bold green", ratio=2)
        tier_table.add_column("Whole Graph Approx Cost", justify="right", style="bold yellow", ratio=2)

        for name, r, style in tiers:
            s_cost = (
                self.calculate_prices_for_section(session_tok["embedding_tokens"], r["embedding"])
                + self.calculate_prices_for_section(session_tok["input_tokens"], r["input"])
                + self.calculate_prices_for_section(session_tok["output_tokens"], r["output"])
                + self.calculate_prices_for_section(session_tok["cache_tokens"], r["cache"])
            )
            g_cost = (
                self.calculate_prices_for_section(graph_tok["embedding_tokens"], r["embedding"])
                + self.calculate_prices_for_section(graph_tok["input_tokens"], r["input"])
                + self.calculate_prices_for_section(graph_tok["output_tokens"], r["output"])
                + self.calculate_prices_for_section(graph_tok["cache_tokens"], r["cache"])
            )

            rate_summary = f"${r['embedding']:.2f} / ${r['input']:.2f} / ${r['output']:.2f} / ${r['cache']:.3f}"
            tier_table.add_row(
                f"[{style}]{name}[/{style}]",
                rate_summary,
                f"[{style}]${s_cost:.4f}[/{style}]",
                f"[{style}]${g_cost:.4f}[/{style}]",
            )

        return tier_table
