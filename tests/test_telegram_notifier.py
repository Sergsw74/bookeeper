"""
Unit tests for Telegram notification subsystem, TelegramConfig, and TelegramNotifier.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from bookeeper.config import Settings, TelegramConfig
from bookeeper.notifications.telegram import TelegramNotifier, _format_duration


def test_telegram_config_defaults():
    cfg = TelegramConfig()
    assert cfg.enabled is True
    assert cfg.bot_token is None
    assert cfg.chat_id is None
    assert cfg.message_thread_id is None
    assert cfg.progress_step_percent == 10
    assert cfg.notify_build_graph is True
    assert cfg.notify_verification is True
    assert cfg.timeout_seconds == 10.0


def test_telegram_config_from_settings_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    monkeypatch.setenv("TELEGRAM_MESSAGE_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_PROGRESS_STEP", "20")
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")

    settings = Settings()
    assert settings.telegram.enabled is True
    assert settings.telegram.bot_token == "123456:TEST_TOKEN"
    assert settings.telegram.chat_id == "-1001234567890"
    assert settings.telegram.message_thread_id == 42
    assert settings.telegram.progress_step_percent == 20


def test_notifier_is_configured():
    # Not configured when credentials missing
    notifier_empty = TelegramNotifier(TelegramConfig())
    assert notifier_empty.is_configured is False

    # Configured when token and chat_id are set and enabled
    cfg_valid = TelegramConfig(bot_token="test_token", chat_id="12345")
    notifier_valid = TelegramNotifier(cfg_valid)
    assert notifier_valid.is_configured is True

    # Not configured if disabled
    cfg_disabled = TelegramConfig(bot_token="test_token", chat_id="12345", enabled=False)
    notifier_disabled = TelegramNotifier(cfg_disabled)
    assert notifier_disabled.is_configured is False


def test_format_duration():
    assert _format_duration(0) == "0s"
    assert _format_duration(45) == "45s"
    assert _format_duration(125) == "2m 5s"
    assert _format_duration(3665) == "1h 1m 5s"
    assert _format_duration(90061) == "1d 1h 1m 1s"


def test_should_notify_progress_stepping():
    cfg = TelegramConfig(
        bot_token="test_token",
        chat_id="12345",
        progress_step_percent=10,
    )
    notifier = TelegramNotifier(cfg)
    total = 100

    # 0 to 5% -> False
    assert notifier.should_notify_progress(completed=5, total=total) is False

    # 10% -> True (hits milestone 10%)
    assert notifier.should_notify_progress(completed=10, total=total) is True

    # 15% -> False (hasn't hit 20% yet)
    assert notifier.should_notify_progress(completed=15, total=total) is False

    # 22% -> True (crosses milestone 20%)
    assert notifier.should_notify_progress(completed=22, total=total) is True

    # 25% -> False
    assert notifier.should_notify_progress(completed=25, total=total) is False

    # 30% -> True
    assert notifier.should_notify_progress(completed=30, total=total) is True

    # Reset progress
    notifier.reset_progress()
    assert notifier.should_notify_progress(completed=10, total=total) is True


def test_should_notify_progress_custom_step_20():
    cfg = TelegramConfig(
        bot_token="test_token",
        chat_id="12345",
        progress_step_percent=20,
    )
    notifier = TelegramNotifier(cfg)
    total = 50

    # 5/50 = 10% -> False
    assert notifier.should_notify_progress(completed=5, total=total) is False

    # 10/50 = 20% -> True
    assert notifier.should_notify_progress(completed=10, total=total) is True

    # 15/50 = 30% -> False
    assert notifier.should_notify_progress(completed=15, total=total) is False

    # 20/50 = 40% -> True
    assert notifier.should_notify_progress(completed=20, total=total) is True


@patch("httpx.Client")
def test_send_message_success(mock_client_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value.__enter__.return_value = mock_client

    cfg = TelegramConfig(
        bot_token="bot123456:ABC-DEF",
        chat_id="987654321",
        message_thread_id=5,
    )
    notifier = TelegramNotifier(cfg)
    result = notifier.send_message("<b>Hello World</b>")

    assert result is True
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0] == "https://api.telegram.org/botbot123456:ABC-DEF/sendMessage"
    payload = kwargs["json"]
    assert payload["chat_id"] == "987654321"
    assert payload["text"] == "<b>Hello World</b>"
    assert payload["parse_mode"] == "HTML"
    assert payload["message_thread_id"] == 5


@patch("httpx.Client")
def test_send_message_error_handling(mock_client_cls):
    mock_client = MagicMock()
    mock_client.post.side_effect = Exception("Connection refused / timeout")
    mock_client_cls.return_value.__enter__.return_value = mock_client

    cfg = TelegramConfig(bot_token="bot123", chat_id="chat123")
    notifier = TelegramNotifier(cfg)

    # Must never crash or raise exception, returns False gracefully
    result = notifier.send_message("Test failure resilience")
    assert result is False


@patch.object(TelegramNotifier, "send_message")
def test_notify_build_start(mock_send):
    mock_send.return_value = True
    cfg = TelegramConfig(bot_token="tok", chat_id="chat")
    notifier = TelegramNotifier(cfg)

    success = notifier.notify_build_start(
        total_books=60,
        llm_model="qwen2.5:7b-instruct",
        embedding_model="jepa-768",
        pool_concurrency=4,
        target_info="60 books",
    )
    assert success is True
    mock_send.assert_called_once()
    msg = mock_send.call_args[0][0]
    assert "Knowledge Graph Build Started" in msg
    assert "60" in msg
    assert "qwen2.5:7b-instruct" in msg
    assert "jepa-768" in msg


@patch.object(TelegramNotifier, "send_message")
def test_notify_build_progress(mock_send):
    mock_send.return_value = True
    cfg = TelegramConfig(bot_token="tok", chat_id="chat")
    notifier = TelegramNotifier(cfg)

    success = notifier.notify_build_progress(
        completed_books=12,
        total_books=60,
        current_book_title="Reflective Formalization Evolution",
        elapsed_seconds=120,
        eta_str="8m 30s",
        assigned_chunks=150,
        skipped_chunks=25,
    )
    assert success is True
    mock_send.assert_called_once()
    msg = mock_send.call_args[0][0]
    assert "Progress: 20%" in msg
    assert "12 / 60" in msg
    assert "Reflective Formalization Evolution" in msg
    assert "150 assigned" in msg


@patch.object(TelegramNotifier, "send_message")
def test_notify_build_complete(mock_send):
    mock_send.return_value = True
    cfg = TelegramConfig(bot_token="tok", chat_id="chat")
    notifier = TelegramNotifier(cfg)

    success = notifier.notify_build_complete(
        total_books=60,
        completed_books=60,
        failed_books=0,
        total_nodes=4500,
        total_edges=9200,
        duration_seconds=3600,
        assigned_chunks=1200,
        skipped_chunks=150,
        graph_file="output/knowledge_graph.json",
    )
    assert success is True
    mock_send.assert_called_once()
    msg = mock_send.call_args[0][0]
    assert "Knowledge Graph Build Completed" in msg
    assert "60 / 60" in msg
    assert "4500 nodes" in msg
    assert "9200 edges" in msg
    assert "1h 0s" in msg or "1h" in msg


@patch.object(TelegramNotifier, "send_message")
def test_notify_build_failure(mock_send):
    mock_send.return_value = True
    cfg = TelegramConfig(bot_token="tok", chat_id="chat")
    notifier = TelegramNotifier(cfg)

    success = notifier.notify_build_failure(
        book_id=5854,
        title="Multimodal Geometry Problem Solving",
        error="ValueError: test error",
    )
    assert success is True
    mock_send.assert_called_once()
    msg = mock_send.call_args[0][0]
    assert "Book Processing Failed" in msg
    assert "5854" in msg
    assert "Multimodal Geometry Problem Solving" in msg
    assert "ValueError: test error" in msg


@patch.object(TelegramNotifier, "send_message")
def test_notify_verification_start_and_complete(mock_send):
    mock_send.return_value = True
    cfg = TelegramConfig(bot_token="tok", chat_id="chat")
    notifier = TelegramNotifier(cfg)

    # 1. Start notification
    s_ok = notifier.notify_verification_start(
        mode="ideas",
        model="qwen2.5:7b-instruct",
        target_info="knowledge_graph.json",
        sample_info="1.0% sample",
    )
    assert s_ok is True
    mock_send.assert_called_once()
    start_msg = mock_send.call_args[0][0]
    msg_lower = start_msg.lower()
    assert "verification started (ideas)" in msg_lower or "ideas" in msg_lower

    # 2. Complete notification with statistics
    mock_send.reset_mock()
    sample_stats = {
        "mean_calibrated_fidelity": 0.945,
        "mean_grounding_recall": 0.982,
        "mean_net_integrity_score": 0.92,
        "mean_hallucination_ratio": 0.018,
        "sampled_chunks_count": 25,
        "discrepancies_count": 1,
    }
    c_ok = notifier.notify_verification_complete(
        mode="ideas",
        model="qwen2.5:7b-instruct",
        duration_seconds=42.5,
        summary={"stats": sample_stats},
    )
    assert c_ok is True
    mock_send.assert_called_once()
    complete_msg = mock_send.call_args[0][0]
    assert "Verification Finished (IDEAS)" in complete_msg
    assert "Calibrated Fidelity:</b> 94.50%" in complete_msg
    assert "Factual Recall:</b> 98.20%" in complete_msg
    assert "Net Semantic Integrity:</b> +0.92" in complete_msg
    assert "Discrepancies:</b> 1" in complete_msg
