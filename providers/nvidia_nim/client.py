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
    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a NIM stream, trying configured API keys before surfacing errors."""
        from config.settings import get_settings

        settings = get_settings()
        keys = settings.nvidia_nim_api_keys or [settings.nvidia_nim_api_key]
        last_error: Exception | None = None

        for key_index, key in enumerate(keys):
            self._client.api_key = key
            key_suffix = key[-8:] if len(key) > 8 else key
            try:
                logger.info(
                    "NIM: Trying API key {}/{} (***{})",
                    key_index + 1,
                    len(keys),
                    key_suffix,
                )
                return await super()._create_stream(body)
            except Exception as error:
                last_error = error
                logger.warning(
                    "NIM: API key {}/{} (***{}) failed with {}: {}",
                    key_index + 1,
                    len(keys),
                    key_suffix,
                    type(error).__name__,
                    error,
                )

        if last_error is not None:
            raise last_error
        return await super()._create_stream(body)

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
