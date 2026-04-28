from datetime import datetime


class ErrorContext:
    """Preserve full context when errors occur"""

    def __init__(
        self,
        provider: str,
        model: str,
        tier: str,
        is_fallback: bool = False,
        original_tier: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.tier = tier
        self.is_fallback = is_fallback
        self.original_tier = original_tier or tier
        self.timestamp = datetime.utcnow()


def get_enhanced_error_response(error: Exception, context: ErrorContext) -> dict:
    """Return structured error with full context"""
    from .error_mapping import get_user_facing_error_message

    base_message = get_user_facing_error_message(error)
    error_type = type(error).__name__
    status_code = getattr(error, "status_code", 500)
    actions = {
        401: "Check your NVIDIA_NIM_API_KEY in .env",
        403: "Verify API key permissions for this model",
        404: f"Model '{context.model}' not found - update models_config.json",
        429: "Rate limit exceeded - circuit breaker active",
        500: "Provider internal error - check provider status page",
    }
    return {
        "error": {
            "message": base_message,
            "type": error_type,
            "status_code": status_code,
            "action": actions.get(status_code, "Check logs for details"),
            "context": {
                "provider": context.provider,
                "model": context.model,
                "tier": context.tier,
                "was_fallback": context.is_fallback,
                "requested_tier": context.original_tier,
                "timestamp": context.timestamp.isoformat(),
            },
        }
    }
