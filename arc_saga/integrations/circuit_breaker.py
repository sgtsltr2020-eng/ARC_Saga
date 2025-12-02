"""
Circuit Breaker Pattern Implementation.

Prevents cascading failures by stopping requests to failing services
and allowing them time to recover.

Follows: Circuit Breaker Pattern (docs/decision_catalog.md)
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from ..error_instrumentation import CircuitBreakerMetrics, log_with_context

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, service: str, state: CircuitState) -> None:
        super().__init__(f"Circuit breaker is {state.value} for service {service}")
        self.service = service
        self.state = state


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Prevents cascading failures by:
    1. Tracking failures and successes
    2. Opening circuit after failure threshold
    3. Testing recovery in half-open state
    4. Closing circuit after successful recovery

    Attributes:
        service_name: Name of the service being protected
        failure_threshold: Number of failures before opening (default: 5)
        success_threshold: Number of successes to close from half-open (default: 2)
        timeout_seconds: Time to wait before testing recovery (default: 60)
        metrics: CircuitBreakerMetrics instance for tracking

    Example:
        >>> breaker = CircuitBreaker("perplexity", failure_threshold=5)
        >>> try:
        ...     result = await breaker.call(api_function, arg1, arg2)
        ... except CircuitBreakerOpenError:
        ...     # Use fallback or cached response
        ...     result = get_cached_response()
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            service_name: Name of the service (e.g., "perplexity")
            failure_threshold: Failures before opening circuit
            success_threshold: Successes to close from half-open
            timeout_seconds: Seconds to wait before testing recovery

        Raises:
            ValueError: If thresholds are invalid
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")

        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds

        # State tracking
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # Metrics
        self.metrics = CircuitBreakerMetrics(service_name)

        log_with_context(
            "info",
            "circuit_breaker_initialized",
            service=service_name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count (in half-open state)."""
        return self._success_count

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from func (if transient/permanent)
        """
        async with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                else:
                    # Circuit still open, reject immediately
                    self.metrics.record_call(success=False)
                    log_with_context(
                        "warning",
                        "circuit_breaker_rejected",
                        service=self.service_name,
                        state=self._state.value,
                        failure_count=self._failure_count,
                    )
                    raise CircuitBreakerOpenError(self.service_name, self._state)

        # Execute function (outside lock to avoid blocking)
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to test recovery."""
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._failure_count = 0

        log_with_context(
            "info",
            "circuit_breaker_half_open",
            service=self.service_name,
        )

    async def _record_success(self) -> None:
        """Record successful call and update state."""
        async with self._lock:
            self.metrics.record_call(success=True)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    # Recovery successful, close circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self.metrics.record_recovery_attempt(success=True)

                    log_with_context(
                        "info",
                        "circuit_breaker_closed",
                        service=self.service_name,
                        success_count=self._success_count,
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call and update state."""
        async with self._lock:
            self.metrics.record_call(success=False)
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open opens circuit immediately
                self._state = CircuitState.OPEN
                self.metrics.record_circuit_open()
                self.metrics.record_recovery_attempt(success=False)

                log_with_context(
                    "warning",
                    "circuit_breaker_reopened",
                    service=self.service_name,
                    error_type=type(error).__name__,
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    # Threshold reached, open circuit
                    self._state = CircuitState.OPEN
                    self.metrics.record_circuit_open()

                    log_with_context(
                        "warning",
                        "circuit_breaker_opened",
                        service=self.service_name,
                        failure_count=self._failure_count,
                        failure_threshold=self.failure_threshold,
                    )

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Use with caution - only for testing or manual recovery.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None

        log_with_context(
            "info",
            "circuit_breaker_reset",
            service=self.service_name,
        )


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient (should retry) or permanent (fail immediately).

    Transient errors:
    - Network timeouts
    - Connection errors
    - Rate limits
    - 5xx server errors

    Permanent errors:
    - Authentication failures
    - Invalid input
    - 4xx client errors (except rate limits)

    Args:
        error: Exception to check

    Returns:
        True if error is transient, False if permanent
    """
    error_type = type(error).__name__
    error_str = str(error).lower()

    # Transient errors
    transient_patterns = [
        "timeout",
        "connection",
        "network",
        "rate limit",
        "503",
        "502",
        "504",
        "500",
    ]

    # Permanent errors
    permanent_patterns = [
        "authentication",
        "unauthorized",
        "forbidden",
        "not found",
        "invalid",
        "400",
        "401",
        "403",
        "404",
    ]

    # Check error string
    for pattern in transient_patterns:
        if pattern in error_str:
            return True

    for pattern in permanent_patterns:
        if pattern in error_str:
            return False

    # Check error type
    transient_types = (
        "TimeoutError",
        "ConnectionError",
        "ConnectionRefusedError",
        "asyncio.TimeoutError",
    )

    if error_type in transient_types:
        return True

    # Default: assume transient (safer to retry)
    return True


async def retry_with_backoff(
    func: Callable[..., Any],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Retry function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception: Last exception if all attempts fail
    """
    """
    Retry function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception: Last exception if all attempts fail
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - Intentional catch-all for retry logic
            last_error = e

            # Don't retry permanent errors
            if not is_transient_error(e):
                log_with_context(
                    "error",
                    "permanent_error_no_retry",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    attempt=attempt + 1,
                )
                raise

            # Don't retry on last attempt
            if attempt == max_attempts - 1:
                break

            # Calculate delay with exponential backoff + jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter  # nosec B311 - Not for crypto, just jitter
            total_delay = delay + jitter

            log_with_context(
                "warning",
                "retry_attempt",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=total_delay,
                error_type=type(e).__name__,
            )

            await asyncio.sleep(total_delay)

    # All attempts failed
    if last_error:
        log_with_context(
            "error",
            "retry_exhausted",
            max_attempts=max_attempts,
            error_type=type(last_error).__name__,
        )
        raise last_error

    raise RuntimeError("Retry exhausted but no error recorded")

