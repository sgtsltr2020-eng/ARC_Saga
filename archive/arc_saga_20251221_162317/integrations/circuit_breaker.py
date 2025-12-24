"""
Circuit Breaker Pattern Implementation.

Prevents cascading failures by stopping requests to failing services
and allowing them time to recover.

The circuit breaker has three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered, limited requests allowed

State transitions:
- CLOSED → OPEN: When failure count >= failure_threshold
- OPEN → HALF_OPEN: After timeout_seconds elapsed
- HALF_OPEN → CLOSED: When success_count >= success_threshold
- HALF_OPEN → OPEN: On any failure
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from ..error_instrumentation import CircuitBreakerMetrics, log_with_context

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, service: str, state: CircuitState) -> None:
        """
        Initialize CircuitBreakerOpenError.

        Args:
            service: Service name that circuit breaker protects
            state: Current circuit breaker state
        """
        self.service = service
        self.state = state
        super().__init__(
            f"Circuit breaker for {service} is {state.value}. "
            "Request rejected to prevent cascading failures."
        )


class CircuitBreaker:
    """
    Circuit breaker implementation with state management and metrics.

    Prevents cascading failures by monitoring service health and
    automatically opening the circuit when failures exceed threshold.

    Attributes:
        service_name: Name of the service being protected
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close from HALF_OPEN
        timeout_seconds: Seconds to wait before attempting recovery
        state: Current circuit breaker state (read-only)
        metrics: CircuitBreakerMetrics instance for tracking (read-only)
        failure_count: Current failure count (read-only)
        success_count: Current success count in HALF_OPEN state (read-only)

    Example:
        >>> breaker = CircuitBreaker("perplexity", failure_threshold=5)
        >>> try:
        ...     result = await breaker.call(api_function)
        ... except CircuitBreakerOpenError:
        ...     # Handle open circuit gracefully
        ...     pass
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
            service_name: Name of the service being protected
            failure_threshold: Number of consecutive failures before opening
            success_threshold: Number of successes to close from HALF_OPEN
            timeout_seconds: Seconds to wait before attempting recovery

        Raises:
            ValueError: If any threshold is invalid (< 1)
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

        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._lock = asyncio.Lock()

        # Metrics tracking
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
        """Get current success count in HALF_OPEN state."""
        return self._success_count

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func execution

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        async with self._lock:
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                # Check if we should attempt recovery
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
                    )
                    raise CircuitBreakerOpenError(self.service_name, self._state)

            # Execute function
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except Exception:
                self._record_failure()
                raise

    def _record_success(self) -> None:
        """Record successful call and update state."""
        self.metrics.record_call(success=True)

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success in CLOSED state
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record failed call and update state."""
        self.metrics.record_call(success=False)
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN immediately opens circuit
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED:
            if self._should_open():
                self._transition_to_open()

    def _should_open(self) -> bool:
        """Check if circuit should transition to OPEN."""
        return self._failure_count >= self.failure_threshold

    def _should_attempt_recovery(self) -> bool:
        """Check if circuit should attempt recovery (OPEN → HALF_OPEN)."""
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        was_half_open = self._state == CircuitState.HALF_OPEN
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
            self.metrics.record_circuit_open()

            # If transitioning from HALF_OPEN, recovery attempt failed
            if was_half_open:
                self.metrics.record_recovery_attempt(success=False)

            log_with_context(
                "warning",
                "circuit_breaker_opened",
                service=self.service_name,
                failure_count=self._failure_count,
                failure_threshold=self.failure_threshold,
            )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        if self._state != CircuitState.HALF_OPEN:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._failure_count = 0

            log_with_context(
                "info",
                "circuit_breaker_half_open",
                service=self.service_name,
                recovery_attempt=True,
            )

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self.metrics.record_recovery_attempt(success=True)

            log_with_context(
                "info",
                "circuit_breaker_closed",
                service=self.service_name,
                recovery_successful=True,
            )

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Clears all state including failure count, success count,
        and last failure time. Useful for manual recovery or testing.
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
    Determine if an error is transient (should be retried).

    Transient errors are temporary conditions that may resolve:
    - Network timeouts
    - Connection errors
    - Rate limiting
    - Server errors (5xx)

    Permanent errors should not be retried:
    - Authentication failures (401)
    - Authorization failures (403)
    - Not found (404)
    - Invalid input (400)

    Args:
        error: Exception to classify

    Returns:
        True if error is transient, False if permanent
    """
    error_str = str(error).lower()

    # Transient errors
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True

    # Rate limiting
    if "rate limit" in error_str:
        return True

    # Server errors (5xx)
    if "500" in error_str or "503" in error_str or "502" in error_str:
        return True

    # Permanent errors
    if (
        "401" in error_str
        or "unauthorized" in error_str
        or "authentication" in error_str
    ):
        return False

    if "403" in error_str or "forbidden" in error_str:
        return False

    if "404" in error_str or "not found" in error_str:
        return False

    if "400" in error_str or "bad request" in error_str or "invalid" in error_str:
        return False

    # Default to transient (safer to retry)
    return True


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Retry function with exponential backoff.

    Retries transient errors with exponential backoff and jitter.
    Permanent errors are raised immediately without retry.

    Backoff formula: delay = min(base_delay * (2 ** attempt), max_delay) + jitter
    Jitter: Random value between 0 and 0.1 * delay

    Args:
        func: Async function to retry
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func execution

    Raises:
        Exception: Last exception if all retries exhausted
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except (
            Exception
        ) as e:  # noqa: BLE001 - intentionally catching all exceptions for retry logic
            last_error = e

            # Don't retry permanent errors
            if not is_transient_error(e):
                log_with_context(
                    "info",
                    "retry_permanent_error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    attempt=attempt + 1,
                )
                raise

            # Don't retry on last attempt
            if attempt == max_attempts - 1:
                break

            # Calculate exponential backoff with jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(
                0, 0.1 * delay
            )  # nosec B311 - jitter for backoff, not cryptographic
            total_delay = delay + jitter

            log_with_context(
                "warning",
                "retry_attempt",
                error_type=type(e).__name__,
                error_message=str(e),
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=total_delay,
            )

            await asyncio.sleep(total_delay)

    # All retries exhausted
    if last_error is not None:
        log_with_context(
            "error",
            "retry_exhausted",
            error_type=type(last_error).__name__,
            error_message=str(last_error),
            max_attempts=max_attempts,
        )
        raise last_error

    # This should never happen, but satisfy type checker
    raise RuntimeError("Retry exhausted without error")
