"""Message and tool format converters."""

import json
from typing import Any

# ── Claw Code US-008: models that reject is_error in tool results ──────────────
_MODELS_REJECTING_IS_ERROR = ("kimi", "moonshot")


def model_rejects_is_error(model: str) -> bool:
    """Return True for models that reject the is_error field in tool results.

    Based on Claw Code US-008 dogfood findings: kimi-k2 variants on NVIDIA NIM
    and DashScope reject tool result messages that contain is_error.
    """
    lower = model.lower()
    return any(token in lower for token in _MODELS_REJECTING_IS_ERROR)


# ── Claw Code US-021: request body size pre-flight ────────────────────────────
_PROVIDER_BODY_LIMITS: dict[str, int] = {
    "kimi": 6 * 1024 * 1024,  # 6 MB DashScope/Moonshot limit
    "moonshot": 6 * 1024 * 1024,
    "default": 100 * 1024 * 1024,  # 100 MB generous default
}


def check_request_body_size(body: dict[str, Any], model: str) -> None:
    """Raise ValueError if JSON-encoded request body exceeds the provider's limit.

    Based on Claw Code US-021: DashScope/kimi has a 6 MB limit that causes
    silent 400 errors when large system prompts fill up the context window.
    """
    limit = _PROVIDER_BODY_LIMITS["default"]
    lower_model = model.lower()
    for keyword, kb_limit in _PROVIDER_BODY_LIMITS.items():
        if keyword != "default" and keyword in lower_model:
            limit = kb_limit
            break

    body_bytes = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    if body_bytes > limit:
        raise ValueError(
            f"Request body {body_bytes // 1024}KB exceeds the {limit // 1024}KB "
            f"limit for model '{model}'. "
            "Start a new Claude Code session to reduce context, or switch to a "
            "model with a larger request limit (e.g. deepseek-v3.2)."
        )


def get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: Any) -> str | None:
    """Get block type from object or dict."""
    return get_block_attr(block, "type")


class AnthropicToOpenAIConverter:
    """Converts Anthropic message format to OpenAI format."""

    @staticmethod
    def convert_messages(
        messages: list[Any],
        *,
        model: str = "",
        include_thinking: bool = True,
        include_reasoning_for_openrouter: bool = False,
        include_reasoning_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert a list of Anthropic messages to OpenAI format.

        When reasoning_content preservation is enabled, assistant messages with
        thinking blocks get reasoning_content added for provider multi-turn
        reasoning continuation.

        Claw Code US-008: pass model name to strip is_error for kimi models.
        """
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_assistant_message(
                            content,
                            include_thinking=include_thinking,
                            include_reasoning_for_openrouter=include_reasoning_for_openrouter,
                            include_reasoning_content=include_reasoning_content,
                        )
                    )
                elif role == "user":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_user_message(
                            content, model=model
                        )
                    )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    @staticmethod
    def _convert_assistant_message(
        content: list[Any],
        *,
        include_thinking: bool = True,
        include_reasoning_for_openrouter: bool = False,
        include_reasoning_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert assistant message blocks, preserving interleaved thinking+text order."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        emit_reasoning_content = (
            include_reasoning_for_openrouter or include_reasoning_content
        )

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                if not include_thinking:
                    continue
                thinking = get_block_attr(block, "thinking", "")
                content_parts.append(f"<think>\n{thinking}\n</think>")
                if emit_reasoning_content:
                    thinking_parts.append(thinking)
            elif block_type == "tool_use":
                tool_input = get_block_attr(block, "input", {})
                tool_calls.append(
                    {
                        "id": get_block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": get_block_attr(block, "name"),
                            "arguments": json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input),
                        },
                    }
                )

        content_str = "\n\n".join(content_parts)

        # Ensure content is never an empty string for assistant messages
        # NIM (especially Mistral models) requires non-empty content if there are no tool calls
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if emit_reasoning_content and thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)

        return [msg]

    @staticmethod
    def _convert_user_message(
        content: list[Any],
        *,
        model: str = "",
    ) -> list[dict[str, Any]]:
        """Convert user message blocks (including tool results), preserving order.

        Claw Code US-008: strips is_error from tool result messages for kimi
        models which reject that field.
        """
        result: list[dict[str, Any]] = []
        text_parts: list[str] = []
        strip_is_error = model_rejects_is_error(model)

        def flush_text() -> None:
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "tool_result":
                flush_text()
                tool_content = get_block_attr(block, "content", "")
                if isinstance(tool_content, list):
                    tool_content = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in tool_content
                    )
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": get_block_attr(block, "tool_use_id"),
                    "content": str(tool_content) if tool_content else "",
                }
                # Claw Code US-008: kimi models reject is_error field
                if not strip_is_error:
                    is_error = get_block_attr(block, "is_error")
                    if is_error is not None:
                        tool_msg["is_error"] = is_error
                result.append(tool_msg)

        flush_text()
        return result

    @staticmethod
    def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Anthropic tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
        """Convert Anthropic system prompt to OpenAI format."""
        if isinstance(system, str):
            return {"role": "system", "content": system}
        elif isinstance(system, list):
            text_parts = [
                get_block_attr(block, "text", "")
                for block in system
                if get_block_type(block) == "text"
            ]
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None


def build_base_request_body(
    request_data: Any,
    *,
    default_max_tokens: int | None = None,
    include_thinking: bool = True,
    include_reasoning_for_openrouter: bool = False,
    include_reasoning_content: bool = False,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body.

    Handles message conversion, system prompt, max_tokens, temperature,
    top_p, stop sequences, tools, and tool_choice. Provider-specific
    parameters (extra_body, penalties, NIM settings) are added by callers.
    """
    from providers.common.utils import set_if_not_none

    model_name = getattr(request_data, "model", "")
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        model=model_name,
        include_thinking=include_thinking,
        include_reasoning_for_openrouter=include_reasoning_for_openrouter,
        include_reasoning_content=include_reasoning_content,
    )

    system = getattr(request_data, "system", None)

    from config.settings import get_settings

    settings = get_settings()

    if settings.enable_caveman:
        caveman_rules = (
            "\n\n<caveman_mode>\n"
            "CAVEMAN MODE ACTIVE — level: ultra\n\n"
            "Respond terse like smart caveman. All technical substance stay. Only fluff die.\n\n"
            "## Persistence\n"
            "ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift.\n\n"
            "## Rules\n"
            "Drop: articles (a/an/the), filler (just/really/basically/actually), pleasantries. "
            "Fragments OK. Technical terms exact. Code blocks unchanged. Errors quoted exact.\n"
            "Pattern: `[thing] [action] [reason]. [next step].`\n"
            'Not: "Sure! I\'d be happy to help you with that. The issue is likely caused by..."\n'
            'Yes: "Bug in auth. Token expiry check use `<` not `<=`. Fix:"\n\n'
            "## Claw Code Mindset\n"
            "You are operating in an autonomous Claw Code environment. "
            "You don't need to ask for permission to use tools if the goal is clear. "
            "Execute efficiently, report tersely.\n"
            "</caveman_mode>"
        )
        if system:
            if isinstance(system, list):
                system.append({"type": "text", "text": caveman_rules})
            elif isinstance(system, str):
                system += caveman_rules
        else:
            system = caveman_rules

    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = getattr(request_data, "max_tokens", None)
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", getattr(request_data, "temperature", None))
    set_if_not_none(body, "top_p", getattr(request_data, "top_p", None))

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = tool_choice

    return body
