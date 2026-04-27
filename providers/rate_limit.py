"""Global rate limiter for API requests."""

import asyncio
import contextlib
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar, TypeVar

import httpx
import openai
from loguru import logger

T = TypeVar("T")


class GlobalRateLimiter:
    """
    Global singleton rate limiter that blocks all requests
    when a rate limit error is encountered (reactive) and
    throttles requests (proactive) using a strict rolling window.

    Optionally enforces a max_concurrency cap: at most N provider streams
    may be open simultaneously, independent of the sliding window.

    Proactive limits - throttles requests to stay within API limits.
    Reactive limits - pauses all requests when a 429 is hit.
    Concurrency limit - caps simultaneously open streams.
    """

    _instance: ClassVar[GlobalRateLimiter | None] = None
    _instances: ClassVar[dict[str, GlobalRateLimiter]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> GlobalRateLimiter:
        return super().__new__(cls)

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        # Prevent re-initialization on singleton reuse
        if hasattr(self, "_initialized"):
            return

        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        # Monotonic timestamps of the last granted slots.
        self._request_times: deque[float] = deque()
        self._blocked_until: float = 0
        self._lock = asyncio.Lock()
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        self._initialized = True

        logger.info(
            f"GlobalRateLimiter (Provider) initialized ({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    @classmethod
    def get_instance(
        cls,
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
        namespace: str = "default",
    ) -> GlobalRateLimiter:
        """Get or create the singleton instance.

        Args:
            rate_limit: Requests per window (only used on first creation)
            rate_window: Window in seconds (only used on first creation)
            max_concurrency: Max simultaneous open streams (only used on first creation)
            namespace: Isolates limiter state for independent upstream providers.
        """
        if namespace not in cls._instances:
            cls._instances[namespace] = cls(
                rate_limit=rate_limit or 40,
                rate_window=rate_window or 60.0,
                max_concurrency=max_concurrency,
            )
            if namespace == "default":
                cls._instance = cls._instances[namespace]
        return cls._instances[namespace]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._instances = {}

    async def wait_if_blocked(self) -> bool:
        """
        Wait if currently rate limited or throttle to meet quota.

        Returns:
            True if was reactively blocked and waited, False otherwise.
        """
        # 1. Reactive check: Wait if someone hit a 429
        waited_reactively = False
        now = time.monotonic()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            logger.warning(
                f"Global provider rate limit active (reactive), waiting {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)
            waited_reactively = True

        # 2. Proactive check: strict rolling window (no bursts beyond N in last W seconds)
        await self._acquire_proactive_slot()
        return waited_reactively

    async def _acquire_proactive_slot(self) -> None:
        """
        Acquire a proactive slot enforcing a strict rolling window.

        Guarantees: at most `self._rate_limit` acquisitions in any interval of length
        `self._rate_window` (seconds).
        """
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._rate_window

                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                if len(self._request_times) < self._rate_limit:
                    self._request_times.append(now)
                    return

                oldest = self._request_times[0]
                wait_time = max(0.0, (oldest + self._rate_window) - now)

            # Sleep outside the lock so other tasks can continue to queue.
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    def set_blocked(self, seconds: float = 60) -> None:
        """
        Set global block for specified seconds (reactive).

        Args:
            seconds: How long to block (default 60s)
        """
        self._blocked_until = time.monotonic() + seconds
        logger.warning(f"Global provider rate limit set for {seconds:.1f}s (reactive)")

    def is_blocked(self) -> bool:
        """Check if currently reactively blocked."""
        return time.monotonic() < self._blocked_until

    def remaining_wait(self) -> float:
        """Get remaining reactive wait time in seconds."""
        return max(0.0, self._blocked_until - time.monotonic())

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        """Return an upstream status code when the exception carries one."""
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        """Return whether a provider call failed with a transient condition."""
        if isinstance(exc, openai.RateLimitError):
            return True

        status_code = cls._status_code(exc)
        if status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
            return True

        return isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        )

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        """Return Retry-After seconds from an upstream response, if present."""
        response = getattr(exc, "response", None)
        if response is None:
            return None
        raw = getattr(response, "headers", {}).get("retry-after")
        if not raw:
            return None
        with contextlib.suppress(ValueError, TypeError):
            return float(raw)
        return None

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        """Async context manager that holds one concurrency slot for a stream.

        Blocks until a slot is available (controlled by max_concurrency).
        """
        await self._concurrency_sem.acquire()
        try:
            yield
        finally:
            self._concurrency_sem.release()

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 6,
        base_delay: float = 5.0,
        max_delay: float = 120.0,
        jitter: float = 2.0,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with rate limiting and transient retries.

        Waits for the proactive limiter before each attempt. Retries rate limits,
        request timeouts, transport failures, and upstream 5xx/529 responses with
        exponential backoff and jitter. Reads Retry-After when available so we wait
        as long as the provider requests.

        Args:
            fn: Async callable to execute.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay cap in seconds.
            jitter: Maximum random jitter in seconds added to each delay.

        Returns:
            The result of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1 + max_retries):
            await self.wait_if_blocked()

            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not self._is_retryable(e):
                    raise

                last_exc = e
                if attempt >= max_retries:
                    logger.warning(
                        "Provider retry exhausted after {} retries for {}",
                        max_retries,
                        type(e).__name__,
                    )
                    break

                retry_after = self._retry_after(e)
                if retry_after is not None:
                    delay = min(retry_after + random.uniform(0, jitter), max_delay)
                    logger.warning(
                        "Provider retryable error {} (status={}), attempt {}/{}. "
                        "Retry-After={:.0f}s; waiting {:.1f}s...",
                        type(e).__name__,
                        self._status_code(e),
                        attempt + 1,
                        max_retries + 1,
                        retry_after,
                        delay,
                    )
                else:
                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(0, jitter)
                    logger.warning(
                        "Provider retryable error {} (status={}), attempt {}/{}. "
                        "Retrying in {:.1f}s...",
                        type(e).__name__,
                        self._status_code(e),
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )

                if self._status_code(e) == 429:
                    self.set_blocked(delay)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
