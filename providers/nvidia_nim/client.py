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
            """Override to support multi-key retry and failover."""
            from config.settings import get_settings
            import httpx
            from providers.exceptions import AuthenticationError, RateLimitError

            settings = get_settings()
            keys = settings.nvidia_nim_api_keys or [settings.nvidia_nim_api_key]
            last_error = None
            max_attempts = 3
            for key in keys:
                for attempt in range(max_attempts):
                    try:
                        self._client.api_key = key
                        # Use OpenAICompatibleProvider's streaming logic
                        async for chunk in super().stream_response(request, input_tokens=input_tokens, request_id=request_id):
                            yield chunk
                        return
                    except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                        logger.warning(f"NIM: Timeout with key {key}, attempt {attempt+1}/{max_attempts}")
                        last_error = e
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    except (AuthenticationError, RateLimitError) as e:
                        logger.warning(f"NIM: Auth/Rate error with key {key}, switching to next key")
                        last_error = e
                        break  # Try next key
                    except Exception as e:
                        last_error = e
                        break
            logger.error(f"NIM: All API keys failed or timed out. Last error: {last_error}")
            if last_error:
                raise last_error
            raise RuntimeError("NIM: All API keys failed or timed out.")


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
