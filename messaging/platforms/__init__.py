"""Messaging platform adapters (Telegram, Discord, etc.)."""

from messaging.platforms.base import (
    CLISession,
    MessagingPlatform,
    SessionManagerInterface,
)
from messaging.platforms.factory import create_messaging_platform

__all__ = [
    "CLISession",
    "MessagingPlatform",
    "SessionManagerInterface",
    "create_messaging_platform",
]
