"""FastAPI route handlers."""

import json
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.common import get_user_facing_error_message
from providers.exceptions import InvalidRequestError, ProviderError

from .dependencies import get_provider_for_type, get_settings, require_api_key
from .models.anthropic import Message, MessagesRequest, TokenCountRequest, Tool
from .models.responses import ModelResponse, ModelsListResponse, TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

router = APIRouter()


def _base_model_aliases() -> list[ModelResponse]:
    """Return Claude-compatible aliases that clients expect."""
    return [
        ModelResponse(
            id="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            created_at="2025-01-01T00:00:00Z",
        ),
        ModelResponse(
            id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            created_at="2025-01-01T00:00:00Z",
        ),
        ModelResponse(
            id="claude-3-5-haiku-20241022",
            display_name="Claude 3.5 Haiku",
            created_at="2025-01-01T00:00:00Z",
        ),
    ]


def _load_models_from_config() -> list[ModelResponse]:
    """Load Claude aliases plus provider models from models_config.json."""
    config_path = Path(__file__).parent.parent / "models_config.json"
    nvidia_catalog_path = Path(__file__).parent.parent / "nvidia_nim_models.json"
    models = _base_model_aliases()
    seen_model_ids = {model.id for model in models}

    if not config_path.exists():
        logger.warning("models_config.json not found, returning Claude model aliases")
        return models

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Load models from each tier - use real model IDs
        for tier_models in config.get("model_tiers", {}).values():
            for model_info in tier_models:
                if not isinstance(model_info, dict):
                    logger.warning(
                        "Skipping invalid model_config entry: {}", model_info
                    )
                    continue
                # Use the full model path as the ID (real NVIDIA NIM model)
                model_id = model_info.get("model")
                # Display name shows model name with tier info
                display_name = model_info.get("name") or model_id
                if not model_id:
                    logger.warning("Skipping model_config entry without model id")
                    continue
                seen_model_ids.add(model_id)

                models.append(
                    ModelResponse(
                        id=model_id,
                        display_name=display_name,
                        created_at="2025-01-01T00:00:00Z",
                    )
                )

        if nvidia_catalog_path.exists():
            with open(nvidia_catalog_path, encoding="utf-8") as f:
                nvidia_catalog = json.load(f)
            for item in nvidia_catalog.get("data", []):
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if not isinstance(model_id, str) or model_id in seen_model_ids:
                    continue
                seen_model_ids.add(model_id)
                models.append(
                    ModelResponse(
                        id=model_id,
                        display_name=model_id,
                        created_at="2025-01-01T00:00:00Z",
                    )
                )

        logger.info(f"Loaded {len(models)} Claude aliases and provider models")
        return models

    except Exception as e:
        logger.error(f"Failed to load models_config.json: {e}")
        return models


# Load models at startup
SUPPORTED_MODELS = _load_models_from_config()


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


def _safe_payload_summary(request_data: MessagesRequest) -> dict[str, object]:
    """Return request metadata without logging prompt/tool content."""
    return {
        "model": request_data.model,
        "original_model": request_data.original_model,
        "resolved_provider_model": request_data.resolved_provider_model,
        "messages": len(request_data.messages),
        "tools": len(request_data.tools or []),
        "has_system": request_data.system is not None,
        "max_tokens": request_data.max_tokens,
        "stream": request_data.stream,
    }


def _stringify_openai_content(content: Any) -> str:
    """Convert common OpenAI content shapes to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _convert_openai_tools(tools: Any) -> list[Tool] | None:
    """Convert OpenAI tool definitions to Anthropic-compatible tools."""
    if not isinstance(tools, list):
        return None

    converted: list[Tool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", {}) if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or not function.get("name"):
            continue
        converted.append(
            Tool(
                name=str(function["name"]),
                description=function.get("description"),
                input_schema=function.get("parameters") or {"type": "object"},
            )
        )
    return converted or None


def _convert_openai_messages(messages: Any) -> tuple[list[Message], str | None]:
    """Convert OpenAI chat messages to Anthropic message/system fields."""
    if not isinstance(messages, list):
        raise InvalidRequestError("messages must be a list")

    converted: list[Message] = []
    system_parts: list[str] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            raise InvalidRequestError("each message must be an object")
        role = raw_message.get("role")
        content = _stringify_openai_content(raw_message.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
        elif role in ("user", "assistant"):
            converted.append(Message(role=role, content=content))
        elif role == "tool":
            tool_call_id = raw_message.get("tool_call_id", "")
            tool_content = content or json.dumps(raw_message, default=str)
            converted.append(
                Message(
                    role="user",
                    content=f"Tool result {tool_call_id}:\n{tool_content}",
                )
            )
        else:
            raise InvalidRequestError(f"unsupported message role: {role!r}")

    if not converted:
        raise InvalidRequestError(
            "messages must include at least one user or assistant message"
        )
    return converted, "\n\n".join(system_parts) if system_parts else None


def _openai_to_messages_request(payload: dict[str, Any]) -> MessagesRequest:
    """Convert an OpenAI chat completion payload to the internal request model."""
    messages, system = _convert_openai_messages(payload.get("messages"))
    return MessagesRequest(
        model=str(payload.get("model") or ""),
        max_tokens=payload.get("max_tokens")
        or payload.get("max_completion_tokens")
        or 512,
        messages=messages,
        system=system,
        stream=bool(payload.get("stream", False)),
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        stop_sequences=payload.get("stop")
        if isinstance(payload.get("stop"), list)
        else None,
        tools=_convert_openai_tools(payload.get("tools")),
        tool_choice=payload.get("tool_choice")
        if isinstance(payload.get("tool_choice"), dict)
        else None,
        extra_body=payload.get("extra_body")
        if isinstance(payload.get("extra_body"), dict)
        else None,
    )


def _parse_sse_data(event: str) -> dict[str, Any] | None:
    """Parse the JSON data object from one SSE event string."""
    for line in event.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line.removeprefix("data:").strip()
        if not raw or raw == "[DONE]":
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    return None


async def _collect_anthropic_stream(
    provider, request_data: MessagesRequest, input_tokens: int, request_id: str
) -> tuple[str, str, int]:
    """Collect text deltas from the existing Anthropic SSE provider stream."""
    text_parts: list[str] = []
    finish_reason = "stop"
    output_tokens = 0
    async for event in provider.stream_response(
        request_data,
        input_tokens=input_tokens,
        request_id=request_id,
    ):
        data = _parse_sse_data(event)
        if not data:
            continue
        if data.get("type") == "content_block_delta":
            delta = data.get("delta", {})
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text_parts.append(str(delta.get("text", "")))
        elif data.get("type") == "message_delta":
            delta = data.get("delta", {})
            if isinstance(delta, dict):
                finish_reason = delta.get("stop_reason") or "end_turn"
            usage = data.get("usage", {})
            if isinstance(usage, dict) and isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]

    openai_finish = "length" if finish_reason == "max_tokens" else "stop"
    return "".join(text_parts), openai_finish, output_tokens


async def _stream_openai_chat_chunks(
    provider,
    request_data: MessagesRequest,
    input_tokens: int,
    request_id: str,
) -> AsyncIterator[str]:
    """Convert internal Anthropic SSE chunks to OpenAI chat completion chunks."""
    chunk_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    yield (
        "data: "
        + json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_data.original_model or request_data.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        + "\n\n"
    )

    async for event in provider.stream_response(
        request_data,
        input_tokens=input_tokens,
        request_id=request_id,
    ):
        data = _parse_sse_data(event)
        if not data:
            continue
        if data.get("type") == "content_block_delta":
            delta = data.get("delta", {})
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request_data.original_model or request_data.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": str(delta.get("text", ""))},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                )
        elif data.get("type") == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            finish_reason = "length" if stop_reason == "max_tokens" else "stop"
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request_data.original_model or request_data.model,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                    }
                )
                + "\n\n"
            )

    yield "data: [DONE]\n\n"


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""

    try:
        if not request_data.messages:
            raise InvalidRequestError("messages cannot be empty")

        optimized = try_optimizations(request_data, settings)
        if optimized is not None:
            return optimized
        logger.debug("No optimization matched, routing to provider")

        # Resolve provider from the model-aware mapping
        provider_type = Settings.parse_provider_type(
            request_data.resolved_provider_model or settings.model
        )
        provider = get_provider_for_type(provider_type)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        logger.info(
            "API_REQUEST: request_id={} model={} messages={}",
            request_id,
            request_data.model,
            len(request_data.messages),
        )
        if settings.log_full_payloads:
            logger.debug("FULL_PAYLOAD [{}]: {}", request_id, request_data.model_dump())
        else:
            logger.debug(
                "PAYLOAD_SUMMARY [{}]: {}",
                request_id,
                _safe_payload_summary(request_data),
            )

        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )
        return StreamingResponse(
            provider.stream_response(
                request_data,
                input_tokens=input_tokens,
                request_id=request_id,
            ),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except ProviderError:
        raise
    except Exception as e:
        logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=get_user_facing_error_message(e),
        ) from e


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest, _auth=Depends(require_api_key)):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            logger.info(
                "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                request_id,
                getattr(request_data, "model", "unknown"),
                len(request_data.messages),
                tokens,
            )
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            logger.error(
                "COUNT_TOKENS_ERROR: request_id={} error={}\n{}",
                request_id,
                get_user_facing_error_message(e),
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500, detail=get_user_facing_error_message(e)
            ) from e


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: dict[str, Any],
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """OpenAI-compatible chat completions endpoint backed by the provider pipeline."""
    try:
        request_data = _openai_to_messages_request(payload)
        provider_type = Settings.parse_provider_type(
            request_data.resolved_provider_model or settings.model
        )
        provider = get_provider_for_type(provider_type)
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )

        if request_data.stream:
            return StreamingResponse(
                _stream_openai_chat_chunks(
                    provider,
                    request_data,
                    input_tokens,
                    request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "x-request-id": request_id,
                },
            )

        content, finish_reason, output_tokens = await _collect_anthropic_stream(
            provider,
            request_data,
            input_tokens,
            request_id,
        )
        return {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.original_model or request_data.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    except ProviderError:
        raise
    except Exception as e:
        logger.error(f"OpenAI compatibility error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=get_user_facing_error_message(e),
        ) from e


@router.api_route("/v1/chat/completions", methods=["HEAD", "OPTIONS"])
async def probe_chat_completions(_auth=Depends(require_api_key)):
    """Respond to OpenAI compatibility probes for chat completions."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root(_auth=Depends(require_api_key)):
    """Respond to compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "auth_configured": bool(settings.anthropic_auth_token),
        "models_count": len(SUPPORTED_MODELS),
        "provider": settings.provider_type,
    }


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(_auth=Depends(require_api_key)):
    """List all available models from hierarchical configuration."""
    return ModelsListResponse(
        data=SUPPORTED_MODELS,
        first_id=SUPPORTED_MODELS[0].id if SUPPORTED_MODELS else None,
        has_more=False,
        last_id=SUPPORTED_MODELS[-1].id if SUPPORTED_MODELS else None,
    )


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
