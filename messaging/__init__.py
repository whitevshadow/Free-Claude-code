"""Platform-agnostic messaging layer."""

from messaging.event_parser import parse_cli_event
from messaging.handler import ClaudeMessageHandler
from messaging.models import IncomingMessage
from messaging.platforms.base import (
    CLISession,
    MessagingPlatform,
    SessionManagerInterface,
)
from messaging.session import SessionStore
from messaging.trees.data import MessageNode, MessageState, MessageTree
from messaging.trees.queue_manager import TreeQueueManager

__all__ = [
    "CLISession",
    "ClaudeMessageHandler",
    "IncomingMessage",
    "MessageNode",
    "MessageState",
    "MessageTree",
    "MessagingPlatform",
    "SessionManagerInterface",
    "SessionStore",
    "TreeQueueManager",
    "parse_cli_event",
]
