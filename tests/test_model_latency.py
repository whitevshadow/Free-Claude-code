import asyncio
import time

import httpx

BASE_URL = "http://localhost:8082/v1/messages"

TEST_MODELS = [
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-122b-a10b",
    "deepseek-ai/deepseek-v3.2",
]

FIRST_TOKEN_TIMEOUT = 15


def build_large_prompt():
    # ~very large input (simulate real system prompt + history)
    base = "Explain distributed systems in depth. "
    return base * 5000  # adjust size if needed


def build_payload(model):
    return {
        "model": model,
        "max_tokens": 64000,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": build_large_prompt()}],
            }
        ],
        "system": [
            {"type": "text", "text": "You are Claude Code."},
            {"type": "text", "text": "Be detailed, structured, and correct."},
        ],
        "tools": [
            {
                "name": "test_tool",
                "description": "A dummy tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                },
            }
        ],
        "temperature": 1.0,
    }


async def run_model_latency_probe(model):
    print(f"\n🔍 Phase 3 (large context): {model}")
    start = time.time()

    payload = build_payload(model)

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", BASE_URL, json=payload) as response:
                first = True
                count = 0
                last_chunk_time = time.time()

                async for _chunk in response.aiter_text():
                    now = time.time()

                    if first:
                        latency = now - start
                        print(f"✅ First token in {latency:.2f}s")
                        first = False

                    # detect stalls
                    gap = now - last_chunk_time
                    if gap > 5:
                        print(f"⚠️ Stream stalled for {gap:.2f}s")

                    last_chunk_time = now
                    count += 1

                    # consume more chunks than before (important!)
                    if count > 100:
                        break

                if first:
                    print("❌ No tokens received")
                else:
                    print(f"📦 Received {count} chunks")

        except TimeoutError:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    for model in TEST_MODELS:
        await run_model_latency_probe(model)


if __name__ == "__main__":
    asyncio.run(main())
