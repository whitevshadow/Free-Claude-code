"""FastAPI route handlers."""

import json
import os
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.common import get_user_facing_error_message
from providers.exceptions import InvalidRequestError, ProviderError

from .dependencies import get_provider_for_type, get_settings, require_api_key
from .metrics import ModelMetrics
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import ModelResponse, ModelsListResponse, TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

metrics = ModelMetrics()

router = APIRouter()


def _load_models_from_config() -> list[ModelResponse]:
    """Load model list dynamically from models_config.json - returns Claude-compatible model aliases."""
    config_path = Path(__file__).parent.parent / "models_config.json"

    if not config_path.exists():
        logger.warning("models_config.json not found, returning default Claude models")
        return _get_default_claude_models()

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        claude_models = []
        tier_versions = {
            "opus": 7,
            "sonnet": 6,
            "haiku": 5,
        }  # Version numbers for aliases

        # Generate Claude model names for each tier
        for tier, models in config.get("model_tiers", {}).items():
            if not models:
                continue

            # Primary model (first in tier) - standard naming
            primary_model = models[0]
            primary_name = primary_model["name"]

            # claude-{tier}-4-20250514 (standard format)
            claude_models.append(
                ModelResponse(
                    id=f"claude-{tier}-4-20250514",
                    display_name=f"Claude {tier.capitalize()} 4 (Powered by {primary_name})",
                    created_at="2025-01-01T00:00:00Z",
                )
            )

            # claude-{tier}-4-{version} (short alias for compatibility)
            version = tier_versions.get(tier, 1)
            claude_models.append(
                ModelResponse(
                    id=f"claude-{tier}-4-{version}",
                    display_name=f"Claude {tier.capitalize()} 4.{version} (Powered by {primary_name})",
                    created_at="2025-01-01T00:00:00Z",
                )
            )

            # Legacy claude-3-{tier} format for backward compatibility
            if tier == "opus":
                date = "20240229"
                version_prefix = "3"
            elif tier == "sonnet":
                date = "20241022"
                version_prefix = "3.5"
            else:  # haiku
                date = "20240307"
                version_prefix = "3"

            legacy_id = f"claude-{version_prefix}-{tier}-{date}"
            legacy_name = f"Claude {version_prefix} {tier.capitalize()} (Powered by {primary_name})"

            claude_models.append(
                ModelResponse(
                    id=legacy_id,
                    display_name=legacy_name,
                    created_at=f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z",
                )
            )

        logger.info(
            f"Loaded {len(claude_models)} Claude-compatible model aliases from models_config.json"
        )
        return claude_models

    except Exception as e:
        logger.error(f"Failed to load models_config.json: {e}")
        return _get_default_claude_models()


def _get_default_claude_models() -> list[ModelResponse]:
    """Return hardcoded default Claude models as fallback."""
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
            id="claude-haiku-4-20250514",
            display_name="Claude Haiku 4",
            created_at="2025-01-01T00:00:00Z",
        ),
    ]


# Load models at startup
SUPPORTED_MODELS = _load_models_from_config()


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


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

    from providers.common.enhanced_error import (
        ErrorContext,
        get_enhanced_error_response,
    )

    # Track context for error reporting
    context = None
    attempt = 0
    max_fallbacks = 5  # Prevent infinite loops
    fallback_used = False
    fallback_reason = None
    response_headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    resolved_model = request_data.resolved_provider_model or settings.model
    provider_type = Settings.parse_provider_type(resolved_model)
    model_id = Settings.parse_model_name(resolved_model)
    tier = None
    for t in ("opus", "sonnet", "haiku"):
        if t in resolved_model.lower():
            tier = t
            break
    context = ErrorContext(
        provider=provider_type, model=model_id, tier=tier or "unknown"
    )

    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info(
        "API_REQUEST: request_id={} model={} messages={}",
        request_id,
        request_data.model,
        len(request_data.messages),
    )
    logger.debug("FULL_PAYLOAD [{}]: {}", request_id, request_data.model_dump())

    if not request_data.messages:
        raise InvalidRequestError("messages cannot be empty")

    optimized = try_optimizations(request_data, settings)
    if optimized is not None:
        return optimized
    logger.debug("No optimization matched, routing to provider")

    # Fallback loop
    while attempt < max_fallbacks:
        try:
            provider_type = Settings.parse_provider_type(resolved_model)
            model_id = Settings.parse_model_name(resolved_model)
            context = ErrorContext(
                provider=provider_type,
                model=model_id,
                tier=tier or "unknown",
                is_fallback=fallback_used,
            )
            provider = get_provider_for_type(provider_type)

            input_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            response_headers.update(
                {
                    "X-Model-Used": resolved_model,
                    "X-Was-Fallback": str(fallback_used),
                    "X-Requested-Tier": tier or "unknown",
                }
            )
            if fallback_used:
                response_headers["X-Fallback-Reason"] = fallback_reason or ""

            start_time = time.time()
            success = False
            try:
                response = provider.stream_response(
                    request_data,
                    input_tokens=input_tokens,
                    request_id=request_id,
                )
                success = True
            except ProviderError as e:
                logger.warning(f"ProviderError: {e!s}\n{traceback.format_exc()}")
                # Try fallback if available
                fallback_model = settings.get_fallback_model(resolved_model, attempt)
                if fallback_model and fallback_model != resolved_model:
                    fallback_used = True
                    fallback_reason = str(e)
                    attempt += 1
                    resolved_model = fallback_model
                    continue
                # No more fallbacks
                if context is None:
                    context = ErrorContext(
                        provider="unknown", model="unknown", tier="unknown"
                    )
                raise HTTPException(
                    status_code=getattr(e, "status_code", 500),
                    detail=get_enhanced_error_response(e, context),
                ) from e
            except Exception as e:
                logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
                if context is None:
                    context = ErrorContext(
                        provider="unknown", model="unknown", tier="unknown"
                    )
                raise HTTPException(
                    status_code=getattr(e, "status_code", 500),
                    detail=get_enhanced_error_response(e, context),
                ) from e
            finally:
                end_time = time.time()
                if success:
                    logger.info(
                        "API_RESPONSE: request_id={} model={} messages={} input_tokens={}",
                        request_id,
                        getattr(request_data, "model", "unknown"),
                        len(request_data.messages),
                        input_tokens,
                    )
                else:
                    logger.warning(
                        "API_RESPONSE: request_id={} model={} messages={} input_tokens={}",
                        request_id,
                        getattr(request_data, "model", "unknown"),
                        len(request_data.messages),
                        input_tokens,
                    )
                response_headers["X-Response-Time"] = f"{end_time - start_time:.3f}"
                # Record metrics
                metrics.record_request(
                    model=resolved_model,
                    success=success,
                    response_time=end_time - start_time,
                    is_fallback=fallback_used,
                )
                # Only record error if one exists
                if not success:
                    # If an exception was caught, record its type
                    exc_type = None
                    import sys

                    exc_type, _, _ = sys.exc_info()
                    if exc_type:
                        metrics.record_error(resolved_model, exc_type.__name__)
            return StreamingResponse(
                response,
                media_type="text/event-stream",
                headers=response_headers,
            )
        except ProviderError as e:
            logger.warning(f"ProviderError: {e!s}\n{traceback.format_exc()}")
            # Try fallback if available
            fallback_model = settings.get_fallback_model(resolved_model, attempt)
            if fallback_model and fallback_model != resolved_model:
                fallback_used = True
                fallback_reason = str(e)
                attempt += 1
                resolved_model = fallback_model
                continue
            # No more fallbacks
            if context is None:
                context = ErrorContext(
                    provider="unknown", model="unknown", tier="unknown"
                )
            raise HTTPException(
                status_code=getattr(e, "status_code", 500),
                detail=get_enhanced_error_response(e, context),
            ) from e
        except Exception as e:
            logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
            if context is None:
                context = ErrorContext(
                    provider="unknown", model="unknown", tier="unknown"
                )
            raise HTTPException(
                status_code=getattr(e, "status_code", 500),
                detail=get_enhanced_error_response(e, context),
            ) from e
        break


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
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health/models")
async def health_check():
    """Validate all model configurations and API keys"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": {},
    }
    settings = get_settings()
    # Check 1: API Key presence
    health["checks"]["nvidia_api_key"] = {
        "present": bool(settings.nvidia_nim_api_key),
        "status": "ok" if settings.nvidia_nim_api_key else "error",
    }
    # Check 2: Config file
    config_path = Path(__file__).parent.parent / "models_config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        health["checks"]["models_config"] = {"status": "ok", "loaded": True}
    except Exception as e:
        health["checks"]["models_config"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"
        config = None
    # Check 3: Model availability per tier
    for tier in ["opus", "sonnet", "haiku"]:
        models = []
        if config and "model_tiers" in config:
            models = config["model_tiers"].get(tier, [])
        health["checks"][f"tier_{tier}"] = {
            "status": "ok" if models else "warning",
            "models_count": len(models),
            "models": [m["model"] for m in models[:3]] if models else [],
        }
    return health


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


@router.post("/admin/reload-models")
async def reload_models(auth: str | None = None):
    """Reload models_config.json without restarting server"""
    # Simple header-based auth (use proper auth in production)
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

    # Always ensure auth is a string
    if auth is None:
        auth = ""
    if auth != ADMIN_TOKEN:
        raise HTTPException(403, "Unauthorized")
    config_path_str = str(Path(__file__).parent.parent / "models_config.json")
    config_path = Path(config_path_str)
    try:
        with open(config_path, encoding="utf-8") as f:
            new_config = json.load(f)
        # Validate tiers
        errors = []
        for tier in ["opus", "sonnet", "haiku"]:
            models = new_config.get("model_tiers", {}).get(tier, [])
            if not models:
                errors.append(
                    f"No models configured for tier '{tier}' in models_config.json"
                )
        # Optionally: update global or cached config here
        logger.info("Models config reloaded successfully")
        return {
            "status": "success" if not errors else "warning",
            "reloaded_at": datetime.now(UTC).isoformat(),
            "tiers": list(new_config.get("model_tiers", {}).keys()),
            "errors": errors,
        }
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/metrics")
async def get_metrics():
    return metrics.get_stats()
