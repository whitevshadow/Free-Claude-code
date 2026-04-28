from unittest.mock import AsyncMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.nvidia_nim.client import NVIDIA_NIM_BASE_URL, NvidiaNimProvider


@pytest.mark.asyncio
async def test_nim_stream_uses_provider_backend_not_local_messages_api():
    config = ProviderConfig(
        api_key="root",
        max_concurrency=5,
        rate_limit=100,
    )

    provider = NvidiaNimProvider(
        config=config,
        nim_settings=NimSettings(),
    )

    request = MessagesRequest(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[Message(role="user", content="Explain distributed systems")],
        stream=True,
        max_tokens=1000,
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _empty_stream()
        events = [chunk async for chunk in provider.stream_response(request)]

    assert provider._base_url == NVIDIA_NIM_BASE_URL
    assert mock_create.await_count == 1
    await_args = mock_create.await_args
    assert await_args is not None
    assert await_args.kwargs["messages"] == [
        {"role": "user", "content": "Explain distributed systems"}
    ]
    assert any("event: message_start" in chunk for chunk in events)
    assert any("event: message_stop" in chunk for chunk in events)


async def _empty_stream():
    if False:
        yield None
