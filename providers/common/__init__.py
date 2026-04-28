"""Shared provider utilities used by NIM, OpenRouter, and LM Studio."""

from providers.common.error_mapping import (
    append_request_id,
    get_user_facing_error_message,
    map_error,
)
from providers.common.heuristic_tool_parser import HeuristicToolParser
from providers.common.message_converter import (
    AnthropicToOpenAIConverter,
    build_base_request_body,
    get_block_attr,
    get_block_type,
)
from providers.common.sse_builder import (
    ContentBlockManager,
    SSEBuilder,
    map_stop_reason,
)
from providers.common.think_parser import ContentChunk, ContentType, ThinkTagParser
from providers.common.utils import set_if_not_none

__all__ = [
    "AnthropicToOpenAIConverter",
    "ContentBlockManager",
    "ContentChunk",
    "ContentType",
    "HeuristicToolParser",
    "SSEBuilder",
    "ThinkTagParser",
    "append_request_id",
    "build_base_request_body",
    "get_block_attr",
    "get_block_type",
    "get_user_facing_error_message",
    "map_error",
    "map_stop_reason",
    "set_if_not_none",
]
