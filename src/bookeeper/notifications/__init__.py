"""
bookeeper notifications subsystem.
Supports messaging and job event notifications across external channels (e.g. Telegram).
"""

from bookeeper.notifications.telegram import TelegramNotifier

__all__ = ["TelegramNotifier"]
