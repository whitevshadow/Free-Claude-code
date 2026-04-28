"""NVIDIA NIM provider implementation."""

import json
from typing import Any

import openai
from loguru import logger

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import (
    build_request_body,
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
)

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNimProvider(OpenAICompatibleProvider):
    async def stream_response(self, request, input_tokens=0, *, request_id=None):
        """Override to support multi-key retry and failover with first-token timeout."""
        import asyncio

        import httpx

        from config.settings import get_settings
        from providers.exceptions import AuthenticationError, RateLimitError

        settings = get_settings()
        keys = settings.nvidia_nim_api_keys or [settings.nvidia_nim_api_key]
        last_error = None
        max_attempts = 2  # Fail faster per key

        # First-token timeout: abort if no content received within this time
        # This is critical - API may accept request but model takes forever to start
        first_token_timeout = settings.first_token_timeout

        logger.info(
            f"NIM: Starting request with {len(keys)} API keys available (first_token_timeout={first_token_timeout}s)"
        )

        for key_index, key in enumerate(keys):
            key_suffix = key[-8:] if len(key) > 8 else key
            for attempt in range(max_attempts):
                try:
                    self._client.api_key = key
                    logger.info(
                        f"NIM: Trying key {key_index + 1}/{len(keys)} (***{key_suffix}), "
                        f"attempt {attempt + 1}/{max_attempts}"
                    )

                    # Stream with first-token timeout using anext()
                    stream = super().stream_response(
                        request, input_tokens=input_tokens, request_id=request_id
                    )

                    # Get first chunk with timeout
                    try:
                        first_chunk = await asyncio.wait_for(
                            anext(aiter(stream)), timeout=first_token_timeout
                        )
                        logger.info(
                            f"NIM: First token received within {first_token_timeout}s with key {key_index + 1}/{len(keys)}"
                        )
                        yield first_chunk
                    except TimeoutError as e:
                        raise httpx.ReadTimeout(
                            f"First token timeout after {first_token_timeout}s - model not responding"
                        ) from e

                    # Continue streaming remaining chunks (no timeout - model is responding)
                    async for chunk in stream:
                        yield chunk

                    logger.info(
                        f"NIM: Success with key {key_index + 1}/{len(keys)} (***{key_suffix})"
                    )
                    return

                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    logger.warning(
                        f"NIM: Timeout with key {key_index + 1}/{len(keys)} (***{key_suffix}), "
                        f"attempt {attempt + 1}/{max_attempts} - {type(e).__name__}: {e}"
                    )
                    last_error = e

                    # On first timeout, immediately try next key (don't retry same key)
                    if attempt == 0 and key_index < len(keys) - 1:
                        logger.info("NIM: Fast-failing to next API key due to timeout")
                        break

                    # Only wait briefly if retrying same key (last key)
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1.0)
                    continue

                except (AuthenticationError, RateLimitError) as e:
                    logger.warning(
                        f"NIM: Auth/Rate error with key {key_index + 1}/{len(keys)} (***{key_suffix}), "
                        f"switching to next key"
                    )
                    last_error = e
                    break  # Try next key immediately

                except Exception as e:
                    logger.error(
                        f"NIM: Unexpected error with key {key_index + 1}/{len(keys)} (***{key_suffix}): "
                        f"{type(e).__name__}: {e}"
                    )
                    last_error = e
                    break  # Try next key

        logger.error(
            f"NIM: All {len(keys)} API keys exhausted. Last error: {type(last_error).__name__}: {last_error}"
        )
        if last_error:
            raise last_error

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_BASE_URL,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request),
        )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if "reasoning_budget" in error_text:
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_budget after 400 error"
            )
            return retry_body

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            logger.warning("NIM_STREAM: retrying without chat_template after 400 error")
            return retry_body

        return None
