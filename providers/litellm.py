"""LiteLLM provider (proxy) using OpenAI-compatible API.

This provider assumes a LiteLLM proxy running an OpenAI-compatible endpoint
at `LITELLM_BASE_URL` (default: http://localhost:4000). The proxy is expected
to forward requests to backend providers (e.g. NVIDIA NIM) and perform any
API-key rotation/load-balancing itself.
"""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider


LITELLM_BASE_URL = "http://localhost:4000"


class LiteLLMProvider(OpenAICompatibleProvider):
    """Provider for LiteLLM proxy (OpenAI-compatible API).

    This provider behaves like other OpenAI-compatible providers and simply
    points the OpenAI client at the configured LiteLLM base URL.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LITELLM",
            base_url=config.base_url or LITELLM_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Build request body using the shared request builder semantics.

        For LiteLLM we reuse the OpenAI-compatible semantics, so delegate to
        the same shape expected by OpenAI-compatible providers.
        """
        # Many models proxied by LiteLLM will be NVIDIA models; the request
        # shape is the same as other OpenAI-compatible providers so simply
        # construct the payload from the incoming request object.
        # Keep this minimal — provider-specific behavior can be added later.
        body = getattr(request, "raw_body", None)
        if isinstance(body, dict):
            return body

        # Fallback: construct a simple body with model/messages
        return {
            "model": request.model,
            "messages": getattr(request, "messages", []),
            **(
                {}
                if getattr(request, "tools", None) is None
                else {"tools": request.tools}
            ),
        }
