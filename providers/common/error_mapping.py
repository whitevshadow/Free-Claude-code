"""Error mapping for OpenAI-compatible providers (NIM, OpenRouter, LM Studio)."""

import httpx
import openai

from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
)
from providers.rate_limit import GlobalRateLimiter

# Claw Code US-022: actionable hints per HTTP status code
_STATUS_HINTS: dict[int, str] = {
    400: "Bad request — the model may be temporarily degraded or unavailable. "
    "Try switching MODEL_OPUS in .env (e.g. to deepseek-v3.2).",
    401: "Authentication failed — check NVIDIA_NIM_API_KEY in your .env file.",
    403: "Access denied — your API key lacks access to this model. "
    "Try a different model in .env.",
    413: "Request too large — start a new Claude Code session to reduce context, "
    "or switch to a model with a larger context limit.",
    429: "Rate limit hit — the proxy is retrying automatically. "
    "If this persists, reduce PROVIDER_MAX_CONCURRENCY in .env.",
    500: "Provider internal error — try again or switch to a different model.",
    502: "Bad gateway — provider is temporarily unavailable. Retry shortly.",
    503: "Provider unavailable — retry shortly or switch models.",
    504: "Gateway timeout — provider took too long to respond. "
    "If using a large model, increase HTTP_READ_TIMEOUT in .env.",
}


def get_user_facing_error_message(
    e: Exception,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Return a readable, non-empty error message with actionable hints.

    Claw Code US-022: per-HTTP-code action hints help users fix issues fast.
    """
    # Check for status code first to give specific actionable hints
    status_code: int | None = None
    if isinstance(e, (openai.APIStatusError, openai.APIError)):
        status_code = getattr(e, "status_code", None)
    elif isinstance(e, httpx.HTTPStatusError):
        status_code = e.response.status_code

    if status_code and status_code in _STATUS_HINTS:
        raw = str(e).strip()
        # For 400 DEGRADED specifically, show the hint prominently
        if "degraded" in raw.lower() or not raw:
            return _STATUS_HINTS[status_code]
        return f"{raw} — {_STATUS_HINTS[status_code]}"

    message = str(e).strip()
    if message:
        return message

    if isinstance(e, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s. Increase HTTP_READ_TIMEOUT in .env."
        return "Provider request timed out."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to provider. Check your internet connection."
    if isinstance(e, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."

    if isinstance(e, (RateLimitError, openai.RateLimitError)):
        return _STATUS_HINTS[429]
    if isinstance(e, (AuthenticationError, openai.AuthenticationError)):
        return _STATUS_HINTS[401]
    if isinstance(e, (InvalidRequestError, openai.BadRequestError)):
        return _STATUS_HINTS[400]
    if isinstance(e, OverloadedError):
        return "Provider is currently overloaded. Please retry shortly."
    if isinstance(e, APIError):
        if e.status_code in (502, 503, 504):
            return _STATUS_HINTS.get(
                e.status_code, "Provider is temporarily unavailable. Please retry."
            )
        return "Provider API request failed."
    if isinstance(e, ProviderError):
        return "Provider request failed."

    return "Provider request failed unexpectedly."


def append_request_id(message: str, request_id: str | None) -> str:
    """Append request_id suffix when available."""
    base = message.strip() or "Provider request failed unexpectedly."
    if request_id:
        return f"{base} (request_id={request_id})"
    return base


def map_error(e: Exception) -> Exception:
    """Map OpenAI or HTTPX exception to specific ProviderError."""
    message = get_user_facing_error_message(e)

    # Map OpenAI Specific Errors
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(message, raw_error=str(e))
    if isinstance(e, openai.RateLimitError):
        # Trigger global rate limit block
        GlobalRateLimiter.get_instance().set_blocked(60)  # Default 60s cooldown
        return RateLimitError(message, raw_error=str(e))
    if isinstance(e, openai.BadRequestError):
        return InvalidRequestError(message, raw_error=str(e))
    if isinstance(e, openai.InternalServerError):
        raw_message = str(e)
        if "overloaded" in raw_message.lower() or "capacity" in raw_message.lower():
            return OverloadedError(message, raw_error=raw_message)
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(e, openai.APIError):
        return APIError(
            message, status_code=getattr(e, "status_code", 500), raw_error=str(e)
        )

    # Map raw HTTPX Errors
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status in (401, 403):
            return AuthenticationError(message, raw_error=str(e))
        if status == 429:
            GlobalRateLimiter.get_instance().set_blocked(60)
            return RateLimitError(message, raw_error=str(e))
        if status == 400:
            return InvalidRequestError(message, raw_error=str(e))
        if status >= 500:
            if status in (502, 503, 504):
                return OverloadedError(message, raw_error=str(e))
            return APIError(message, status_code=status, raw_error=str(e))
        return APIError(message, status_code=status, raw_error=str(e))

    return e
