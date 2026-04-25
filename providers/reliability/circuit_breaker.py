"""Circuit breaker implementation for provider resilience."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 2  # Successes needed to close circuit
    timeout: float = 30.0  # Call timeout


class CircuitBreaker:
    """Circuit breaker for provider resilience.

    Prevents cascading failures by rejecting requests to failing providers
    and automatically recovering when the service becomes available again.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self._name = name
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            fn: Async callable to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result of fn

        Raises:
            Exception: If circuit is open or call fails
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self._name}' entering HALF_OPEN state"
                    )
                else:
                    wait_time = self._config.recovery_timeout - (
                        time.monotonic() - self._last_failure_time
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self._name}' is OPEN. "
                        f"Rejecting request. Retry in {wait_time:.1f}s"
                    )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                fn(*args, **kwargs), timeout=self._config.timeout
            )

            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._success_count += 1
                    if self._success_count >= self._config.success_threshold:
                        self._state = CircuitState.CLOSED
                        self._failure_count = 0
                        logger.info(
                            f"Circuit breaker '{self._name}' recovered to CLOSED state"
                        )
                else:
                    self._failure_count = 0

            return result

        except TimeoutError as e:
            async with self._lock:
                self._on_failure()
            raise CircuitBreakerTimeoutError(
                f"Circuit breaker '{self._name}' call timed out after {self._config.timeout}s"
            ) from e

        except Exception:
            async with self._lock:
                self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (
            time.monotonic() - self._last_failure_time >= self._config.recovery_timeout
        )

    def _on_failure(self) -> None:
        """Handle a failure."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self._name}' failed in HALF_OPEN, returning to OPEN"
            )
        elif self._failure_count >= self._config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self._name}' opened after {self._failure_count} failures"
            )

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        logger.info(f"Circuit breaker '{self._name}' manually reset to CLOSED state")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreakerTimeoutError(Exception):
    """Raised when circuit breaker call times out."""

    pass
