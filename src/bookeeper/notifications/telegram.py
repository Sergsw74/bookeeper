"""
Telegram notification client for bookeeper job execution and progress tracking.
Provides reliable asynchronous-safe reporting for graph building and verification events.
"""

import html
import logging
from typing import Any, Dict, Optional
import httpx

from bookeeper.config import TelegramConfig

logger = logging.getLogger("bookeeper.notifications.telegram")

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def _format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable string (e.g. 1h 23m 45s)."""
    if seconds < 0:
        return "0s"
    tot_sec = int(round(seconds))
    days = tot_sec // 86400
    hours = (tot_sec % 86400) // 3600
    minutes = (tot_sec % 3600) // 60
    secs = tot_sec % 60
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


class TelegramNotifier:
    """Dispatches notifications to Telegram via the Bot API."""

    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or TelegramConfig()
        self._last_notified_percent: int = 0

    @property
    def is_configured(self) -> bool:
        """Return True if enabled and both bot_token and chat_id are present."""
        return (
            bool(self.config.enabled)
            and bool(self.config.bot_token and self.config.bot_token.strip())
            and bool(self.config.chat_id and self.config.chat_id.strip())
        )

    def reset_progress(self) -> None:
        """Reset progress notification tracker."""
        self._last_notified_percent = 0

    def should_notify_progress(self, completed: int, total: int) -> bool:
        """
        Determine if a progress notification should be dispatched based on
        completed vs total items and progress_step_percent.
        """
        if total <= 0 or not self.config.notify_build_graph or not self.is_configured:
            return False

        step = max(1, self.config.progress_step_percent)
        current_percent = int((completed / total) * 100)

        # Notify if we crossed the next step threshold (e.g. 10%, 20%, 30%...)
        if current_percent >= self._last_notified_percent + step:
            self._last_notified_percent = (current_percent // step) * step
            return True

        return False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured Telegram chat.
        Safely catches network errors, timeouts, or API exceptions to prevent interrupting jobs.
        """
        if not self.is_configured:
            return False

        token = self.config.bot_token.strip()
        url = f"https://api.telegram.org/bot{token}/sendMessage"

        # Truncate text if exceeding Telegram message length limit
        if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
            text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 15] + "\n[... truncated]"

        payload: Dict[str, Any] = {
            "chat_id": self.config.chat_id.strip(),
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if self.config.message_thread_id is not None:
            payload["message_thread_id"] = self.config.message_thread_id

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                if resp.is_success:
                    logger.debug("Telegram message sent successfully.")
                    return True
                logger.warning(
                    f"Telegram notification returned HTTP {resp.status_code}: {resp.text}"
                )
                return False
        except Exception as exc:
            logger.warning(f"Failed to deliver Telegram notification: {exc}")
            return False

    # -------------------------------------------------------------------------
    # Graph Build Notifications
    # -------------------------------------------------------------------------

    def notify_build_start(
        self,
        total_books: int,
        llm_model: str,
        embedding_model: str,
        pool_concurrency: int = 1,
        target_info: str = "",
    ) -> bool:
        """Send notification when a Knowledge Graph build job starts."""
        if not self.config.notify_build_graph or not self.is_configured:
            return False

        self.reset_progress()
        target_line = f"\n🎯 <b>Target:</b> {html.escape(target_info)}" if target_info else ""
        msg = (
            "🚀 <b>Knowledge Graph Build Started</b>\n\n"
            f"📚 <b>Books to process:</b> {total_books}\n"
            f"🧠 <b>LLM Model:</b> <code>{html.escape(llm_model)}</code>\n"
            f"⚡ <b>Embeddings:</b> <code>{html.escape(embedding_model)}</code>\n"
            f"⚙️ <b>Concurrency:</b> {pool_concurrency} worker task(s)"
            f"{target_line}"
        )
        return self.send_message(msg)

    def notify_build_progress(
        self,
        completed_books: int,
        total_books: int,
        current_book_title: str = "",
        elapsed_seconds: float = 0.0,
        eta_str: str = "--m --s",
        active_tasks: int = 0,
        assigned_chunks: int = 0,
        skipped_chunks: int = 0,
    ) -> bool:
        """Send milestone progress notification during graph building."""
        if not self.config.notify_build_graph or not self.is_configured:
            return False

        percent = int((completed_books / total_books) * 100) if total_books > 0 else 0
        elapsed_str = _format_duration(elapsed_seconds)
        title_line = (
            f"\n📖 <b>Current:</b> <i>{html.escape(current_book_title[:60])}</i>"
            if current_book_title
            else ""
        )
        msg = (
            f"📊 <b>Knowledge Graph Build Progress: {percent}%</b>\n\n"
            f"📚 <b>Books:</b> {completed_books} / {total_books}{title_line}\n"
            f"⏱ <b>Elapsed:</b> {elapsed_str} | <b>ETA:</b> {html.escape(eta_str)}\n"
            f"🧩 <b>Chunks:</b> {assigned_chunks} assigned, {skipped_chunks} skipped"
        )
        return self.send_message(msg)

    def notify_build_complete(
        self,
        total_books: int,
        completed_books: int,
        failed_books: int = 0,
        total_nodes: int = 0,
        total_edges: int = 0,
        duration_seconds: float = 0.0,
        assigned_chunks: int = 0,
        skipped_chunks: int = 0,
        graph_file: str = "",
    ) -> bool:
        """Send completion notification for a Knowledge Graph build job."""
        if not self.config.notify_build_graph or not self.is_configured:
            return False

        duration_str = _format_duration(duration_seconds)
        status_icon = "✅" if failed_books == 0 else "⚠️"
        dest_line = f"\n📁 <b>Destination:</b> <code>{html.escape(graph_file)}</code>" if graph_file else ""
        msg = (
            f"{status_icon} <b>Knowledge Graph Build Completed</b>\n\n"
            f"⏱ <b>Total Duration:</b> {duration_str}\n"
            f"📚 <b>Books Processed:</b> {completed_books} / {total_books} (Failed: {failed_books})\n"
            f"🧩 <b>Chunks:</b> {assigned_chunks} assigned | {skipped_chunks} skipped\n"
            f"🌐 <b>Graph Topology:</b> {total_nodes} nodes | {total_edges} edges"
            f"{dest_line}"
        )
        return self.send_message(msg)

    def notify_build_failure(
        self,
        book_id: Any,
        title: str,
        error: str,
    ) -> bool:
        """Send warning notification when a book fails during graph building."""
        if not self.config.notify_build_graph or not self.is_configured:
            return False

        msg = (
            "⚠️ <b>Book Processing Failed</b>\n\n"
            f"📖 <b>Book #{book_id}:</b> <i>{html.escape(title[:60])}</i>\n"
            f"❌ <b>Error:</b> <code>{html.escape(str(error)[:200])}</code>"
        )
        return self.send_message(msg)

    # -------------------------------------------------------------------------
    # Verification Notifications
    # -------------------------------------------------------------------------

    def notify_verification_start(
        self,
        mode: str,
        model: str,
        target_info: str = "",
        sample_info: str = "",
    ) -> bool:
        """Send notification when a verification audit starts."""
        if not self.config.notify_verification or not self.is_configured:
            return False

        target_line = f"\n🎯 <b>Target:</b> {html.escape(target_info)}" if target_info else ""
        sample_line = f"\n📋 <b>Sample:</b> {html.escape(sample_info)}" if sample_info else ""
        msg = (
            f"🔍 <b>Verification Started ({html.escape(mode.upper())})</b>\n\n"
            f"🎯 <b>Mode:</b> <code>{html.escape(mode)}</code>\n"
            f"🧠 <b>Model:</b> <code>{html.escape(model)}</code>"
            f"{target_line}"
            f"{sample_line}"
        )
        return self.send_message(msg)

    def notify_verification_complete(
        self,
        mode: str,
        model: str,
        duration_seconds: float,
        summary: Any,
    ) -> bool:
        """Send notification when a verification audit finishes with high-level statistics."""
        if not self.config.notify_verification or not self.is_configured:
            return False

        duration_str = _format_duration(duration_seconds)
        stats_lines = self._format_verification_stats(summary)

        msg = (
            f"✅ <b>Verification Finished ({html.escape(mode.upper())})</b>\n\n"
            f"🎯 <b>Mode:</b> <code>{html.escape(mode)}</code>\n"
            f"🧠 <b>Model:</b> <code>{html.escape(model)}</code>\n"
            f"⏱ <b>Duration:</b> {duration_str}\n\n"
            f"📊 <b>Summary Statistics:</b>\n"
            f"{stats_lines}"
        )
        return self.send_message(msg)

    def _format_verification_stats(self, summary: Any) -> str:
        """Extract and format high-level statistics from verification report objects or dicts."""
        lines = []

        # Convert Pydantic models or dict wrappers
        if hasattr(summary, "model_dump"):
            data = summary.model_dump()
        elif hasattr(summary, "stats"):
            raw_st = summary.stats
            data = raw_st.model_dump() if hasattr(raw_st, "model_dump") else (raw_st if isinstance(raw_st, dict) else {})
        elif isinstance(summary, dict):
            data = summary.get("stats") or summary.get("summary") or summary
        else:
            data = {}

        if not isinstance(data, dict):
            return "  • <i>Metrics recorded</i>"

        # 1. Calibrated Fidelity
        fidelity = data.get("mean_calibrated_fidelity") or data.get("calibrated_fidelity")
        if fidelity is not None:
            lines.append(f"  • <b>Calibrated Fidelity:</b> {float(fidelity) * 100:.2f}%")

        # 2. Grounding Recall / QA Recall
        recall = data.get("mean_grounding_recall") or data.get("mean_qa_recall") or data.get("qa_recall")
        if recall is not None:
            lines.append(f"  • <b>Factual Recall:</b> {float(recall) * 100:.2f}%")

        # 3. Net Semantic Integrity (S_net)
        net_score = data.get("mean_net_integrity_score") or data.get("mean_net_score") or data.get("net_integrity_score")
        if net_score is not None:
            lines.append(f"  • <b>Net Semantic Integrity:</b> {float(net_score):+.2f}")

        # 4. Hallucination / Contradiction rate
        hallucination = data.get("mean_hallucination_ratio") or data.get("hallucination_rate")
        if hallucination is not None:
            lines.append(f"  • <b>Hallucination Rate:</b> {float(hallucination) * 100:.2f}%")

        # 5. Graph Space Coverage / Chamfer / Vendi score (KG mode)
        cov = (
            data.get("coverage_at_tau_pct")
            or data.get("threshold_coverage_pct")
            or data.get("coverage_tau")
            or data.get("mean_coverage_ratio")
            or data.get("coverage_ratio")
        )
        if cov is not None:
            c_val = float(cov)
            if c_val <= 1.0:
                c_val *= 100.0
            tau_val = data.get("tau", 0.75)
            lines.append(f"  • <b>Corpus Coverage (τ={tau_val}):</b> {c_val:.2f}%")

        chamfer = data.get("chamfer_mean") or data.get("chamfer_mean_coverage") or data.get("mean_chamfer_coverage")
        if chamfer is not None:
            lines.append(f"  • <b>Chamfer Proximity (μ):</b> {float(chamfer):.4f}")

        vendi = data.get("vendi_diversity_score") or data.get("vendi_score") or data.get("vendi_diversity")
        if vendi is not None:
            lines.append(f"  • <b>Vendi Diversity:</b> {float(vendi):.2f}")

        nodes_cnt = data.get("nodes_count") or data.get("idea_nodes_count")
        chunks_cnt = data.get("universe_chunks_count")
        if nodes_cnt is not None and chunks_cnt is not None:
            lines.append(f"  • <b>Graph Scope:</b> {nodes_cnt:,} concepts / {chunks_cnt:,} chunks")

        drift = data.get("postprocessing_drift")
        if isinstance(drift, dict):
            f_ret = drift.get("forward_retention")
            b_fid = drift.get("backward_fidelity")
            if f_ret is not None:
                lines.append(f"  • <b>Forward Retention:</b> {float(f_ret):.4f}")
            if b_fid is not None:
                lines.append(f"  • <b>Backward Fidelity:</b> {float(b_fid):.4f}")
            c_align = drift.get("centroid_cosine_alignment")
            if c_align is not None:
                lines.append(f"  • <b>Centroid Alignment:</b> {float(c_align):.4f}")

        # 6. Sampled Chunks / Ideas count
        sampled = (
            data.get("sampled_chunks_count")
            or data.get("evaluated_count")
            or data.get("total_ideas_verified")
            or data.get("total_samples")
        )
        if sampled is not None:
            lines.append(f"  • <b>Sample Size:</b> {sampled} item(s)")

        # 7. Discrepancies
        discrepancies = (
            data.get("discrepancies_count")
            or data.get("chunks_with_hallucinations")
            or data.get("chunks_with_omissions")
        )
        if discrepancies is not None:
            lines.append(f"  • <b>Discrepancies:</b> {discrepancies}")

        if not lines:
            # Fallback to key-value pairs if standard metrics not recognized
            for k, v in list(data.items())[:5]:
                if isinstance(v, (int, float, str)) and not k.startswith("_"):
                    clean_k = k.replace("_", " ").title()
                    clean_v = f"{v:.2f}" if isinstance(v, float) else str(v)
                    lines.append(f"  • <b>{clean_k}:</b> {clean_v}")

        return "\n".join(lines) if lines else "  • <i>Metrics evaluated successfully</i>"
