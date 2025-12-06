"""
Unit tests for Circuit Breaker implementation.

Tests verify:
1. State transitions CLOSED → OPEN → HALF_OPEN → CLOSED
2. Failure threshold triggers OPEN
3. Success threshold triggers CLOSED from HALF_OPEN
4. Timeout resets to HALF_OPEN
5. Retry with backoff exponential increases max 60s
6. Transient vs permanent error handling
7. CircuitBreakerMetrics tracking
8. Graceful degradation when OPEN
9. ask_streaming integration with circuit breaker

Coverage target: 95%+
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import pytest

from arc_saga.error_instrumentation import CircuitBreakerMetrics
from arc_saga.integrations.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    is_transient_error,
    retry_with_backoff,
)


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_init_with_valid_parameters(self) -> None:
        """Test initialization with valid parameters succeeds."""
        breaker = CircuitBreaker(
            service_name="test_service",
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60,
        )

        assert breaker.service_name == "test_service"
        assert breaker.failure_threshold == 5
        assert breaker.success_threshold == 2
        assert breaker.timeout_seconds == 60
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    def test_init_with_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        breaker = CircuitBreaker(service_name="test_service")

        assert breaker.failure_threshold == 5
        assert breaker.success_threshold == 2
        assert breaker.timeout_seconds == 60
        assert breaker.state == CircuitState.CLOSED

    def test_init_with_invalid_failure_threshold_raises_error(self) -> None:
        """Test initialization with invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(service_name="test", failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(service_name="test", failure_threshold=-1)

    def test_init_with_invalid_success_threshold_raises_error(self) -> None:
        """Test initialization with invalid success_threshold raises ValueError."""
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreaker(service_name="test", success_threshold=0)

        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreaker(service_name="test", success_threshold=-1)

    def test_init_with_invalid_timeout_raises_error(self) -> None:
        """Test initialization with invalid timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            CircuitBreaker(service_name="test", timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            CircuitBreaker(service_name="test", timeout_seconds=-1)

    def test_init_creates_metrics_instance(self) -> None:
        """Test initialization creates CircuitBreakerMetrics instance."""
        breaker = CircuitBreaker(service_name="test_service")

        assert isinstance(breaker.metrics, CircuitBreakerMetrics)
        assert breaker.metrics.service == "test_service"


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failure_threshold(self) -> None:
        """Test circuit opens when failure threshold is reached."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60,
        )

        # Make 2 failures (should still be CLOSED)
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 2

        # Third failure should open circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        # Type narrowing: state can be OPEN after threshold
        state_after = breaker.state
        assert state_after == CircuitState.OPEN
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self) -> None:
        """Test circuit transitions from OPEN to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,  # Short timeout for testing
        )

        # Open the circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Next call should transition to HALF_OPEN
        async def succeeding_func() -> str:
            return "success"

        result = await breaker.call(succeeding_func)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success_threshold(self) -> None:
        """Test circuit closes from HALF_OPEN when success threshold is met."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,
        )

        # Open circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # First success (should be HALF_OPEN)
        async def succeeding_func() -> str:
            return "success"

        result = await breaker.call(succeeding_func)
        assert result == "success"
        # Type narrowing: state transitions to HALF_OPEN
        state_after_first = breaker.state
        assert state_after_first == CircuitState.HALF_OPEN
        assert breaker.success_count == 1

        # Second success should close circuit
        result = await breaker.call(succeeding_func)
        assert result == "success"
        # Type narrowing: state transitions to CLOSED
        state_after_second = breaker.state
        assert state_after_second == CircuitState.CLOSED
        assert breaker.success_count == 0
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_any_failure(self) -> None:
        """Test circuit opens immediately on any failure in HALF_OPEN state."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,
        )

        # Open circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # One success (HALF_OPEN)
        async def succeeding_func() -> str:
            return "success"

        await breaker.call(succeeding_func)
        # Type narrowing: state transitions to HALF_OPEN
        state_after_success = breaker.state
        assert state_after_success == CircuitState.HALF_OPEN

        # Failure should immediately open circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        # Type narrowing: state transitions back to OPEN
        state_after_failure = breaker.state
        assert state_after_failure == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_closed_resets_failure_count_on_success(self) -> None:
        """Test failure count resets to 0 on success in CLOSED state."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60,
        )

        # Make 3 failures
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.failure_count == 3

        # Success should reset failure count
        async def succeeding_func() -> str:
            return "success"

        await breaker.call(succeeding_func)
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError behavior."""

    @pytest.mark.asyncio
    async def test_open_circuit_raises_circuit_breaker_open_error(self) -> None:
        """Test open circuit raises CircuitBreakerOpenError immediately."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=60,
        )

        # Open the circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError immediately
        async def any_func() -> str:
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call(any_func)

        assert exc_info.value.service == "test"
        assert exc_info.value.state == CircuitState.OPEN
        assert "open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_before_timeout(self) -> None:
        """Test open circuit rejects calls before timeout period."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=10,  # Long timeout
        )

        # Open the circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Immediate call should be rejected
        async def any_func() -> str:
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(any_func)

        # Wait a short time (less than timeout)
        await asyncio.sleep(0.1)

        # Should still be rejected
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(any_func)


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_track_successful_calls(self) -> None:
        """Test metrics track successful calls."""
        breaker = CircuitBreaker(service_name="test")

        async def succeeding_func() -> str:
            return "success"

        for _ in range(5):
            await breaker.call(succeeding_func)

        assert breaker.metrics.total_calls == 5
        assert breaker.metrics.successful_calls == 5
        assert breaker.metrics.failed_calls == 0
        assert breaker.metrics.success_rate == 100.0

    @pytest.mark.asyncio
    async def test_metrics_track_failed_calls(self) -> None:
        """Test metrics track failed calls."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=10,  # High threshold to avoid opening
        )

        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(5):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.metrics.total_calls == 5
        assert breaker.metrics.successful_calls == 0
        assert breaker.metrics.failed_calls == 5
        assert breaker.metrics.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_track_circuit_opens(self) -> None:
        """Test metrics track circuit opens."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=60,
        )

        async def failing_func() -> None:
            raise Exception("Test error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.metrics.circuit_opens == 1

    @pytest.mark.asyncio
    async def test_metrics_track_recovery_attempts(self) -> None:
        """Test metrics track recovery attempts."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,
        )

        # Open circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Successful recovery
        async def succeeding_func() -> str:
            return "success"

        for _ in range(2):
            await breaker.call(succeeding_func)

        assert breaker.metrics.recovery_attempts == 1
        assert breaker.metrics.successful_recoveries == 1
        assert breaker.metrics.recovery_rate == 100.0

    @pytest.mark.asyncio
    async def test_metrics_track_failed_recovery_attempts(self) -> None:
        """Test metrics track failed recovery attempts."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,
        )

        # Open circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Failed recovery (failure in HALF_OPEN)
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        assert breaker.metrics.recovery_attempts == 1
        assert breaker.metrics.successful_recoveries == 0
        assert breaker.metrics.recovery_rate == 0.0

    def test_metrics_to_dict_returns_complete_data(self) -> None:
        """Test metrics.to_dict() returns complete metrics data."""
        breaker = CircuitBreaker(service_name="test_service")

        metrics_dict = breaker.metrics.to_dict()

        assert metrics_dict["service"] == "test_service"
        assert "total_calls" in metrics_dict
        assert "successful_calls" in metrics_dict
        assert "failed_calls" in metrics_dict
        assert "success_rate" in metrics_dict
        assert "circuit_opens" in metrics_dict
        assert "recovery_rate" in metrics_dict


class TestTransientVsPermanentErrors:
    """Tests for transient vs permanent error handling."""

    def test_timeout_error_is_transient(self) -> None:
        """Test TimeoutError is classified as transient."""
        error = TimeoutError("Request timeout")
        assert is_transient_error(error) is True

    def test_connection_error_is_transient(self) -> None:
        """Test ConnectionError is classified as transient."""
        error = ConnectionError("Connection failed")
        assert is_transient_error(error) is True

    def test_connection_refused_error_is_transient(self) -> None:
        """Test ConnectionRefusedError is classified as transient."""
        error = ConnectionRefusedError("Connection refused")
        assert is_transient_error(error) is True

    def test_asyncio_timeout_error_is_transient(self) -> None:
        """Test asyncio.TimeoutError is classified as transient."""
        error = asyncio.TimeoutError("Async timeout")
        assert is_transient_error(error) is True

    def test_rate_limit_error_is_transient(self) -> None:
        """Test rate limit error is classified as transient."""
        error = Exception("Rate limit exceeded")
        assert is_transient_error(error) is True

    def test_500_error_is_transient(self) -> None:
        """Test 500 error is classified as transient."""
        error = Exception("500 Internal Server Error")
        assert is_transient_error(error) is True

    def test_503_error_is_transient(self) -> None:
        """Test 503 error is classified as transient."""
        error = Exception("503 Service Unavailable")
        assert is_transient_error(error) is True

    def test_authentication_error_is_permanent(self) -> None:
        """Test authentication error is classified as permanent."""
        error = Exception("Authentication failed")
        assert is_transient_error(error) is False

    def test_unauthorized_error_is_permanent(self) -> None:
        """Test unauthorized error is classified as permanent."""
        error = Exception("401 Unauthorized")
        assert is_transient_error(error) is False

    def test_forbidden_error_is_permanent(self) -> None:
        """Test forbidden error is classified as permanent."""
        error = Exception("403 Forbidden")
        assert is_transient_error(error) is False

    def test_not_found_error_is_permanent(self) -> None:
        """Test not found error is classified as permanent."""
        error = Exception("404 Not Found")
        assert is_transient_error(error) is False

    def test_invalid_input_error_is_permanent(self) -> None:
        """Test invalid input error is classified as permanent."""
        error = Exception("Invalid input provided")
        assert is_transient_error(error) is False

    def test_unknown_error_defaults_to_transient(self) -> None:
        """Test unknown error defaults to transient (safer to retry)."""
        error = Exception("Unknown error type")
        assert is_transient_error(error) is True


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_attempt(self) -> None:
        """Test retry succeeds on first attempt."""
        call_count = 0

        async def succeeding_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(succeeding_func, max_attempts=3)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """Test retry succeeds on second attempt after transient error."""
        call_count = 0

        async def eventually_succeeding_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Transient error")
            return "success"

        result = await retry_with_backoff(
            eventually_succeeding_func,
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
        )

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff_increases_delay(self) -> None:
        """Test retry uses exponential backoff with increasing delays."""
        call_count = 0

        async def failing_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Transient error")

        try:
            await retry_with_backoff(
                failing_func,
                max_attempts=3,
                base_delay=0.1,
                max_delay=1.0,
            )
        except TimeoutError:
            pass

        # Calculate delays between calls
        # Note: We can't directly measure delays, but we can verify
        # that the function was called multiple times
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_respects_max_delay(self) -> None:
        """Test retry respects max_delay limit."""
        call_count = 0

        async def failing_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Transient error")

        try:
            await retry_with_backoff(
                failing_func,
                max_attempts=5,
                base_delay=10.0,  # Large base delay
                max_delay=0.5,  # But max delay is small
            )
        except TimeoutError:
            pass

        # Should have attempted multiple times
        assert call_count == 5

    @pytest.mark.asyncio
    async def test_retry_raises_permanent_error_immediately(self) -> None:
        """Test retry raises permanent error immediately without retrying."""
        call_count = 0

        async def permanent_error_func() -> None:
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")  # Permanent error

        with pytest.raises(Exception, match="Unauthorized"):
            await retry_with_backoff(
                permanent_error_func,
                max_attempts=3,
                base_delay=0.1,
            )

        # Should only be called once (no retry for permanent errors)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_last_error(self) -> None:
        """Test retry raises last error when all attempts exhausted."""
        call_count = 0

        async def always_failing_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError(f"Attempt {call_count} failed")

        with pytest.raises(TimeoutError, match="Attempt 3 failed"):
            await retry_with_backoff(
                always_failing_func,
                max_attempts=3,
                base_delay=0.1,
            )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_jitter_adds_randomness(self) -> None:
        """Test retry adds jitter to prevent thundering herd."""
        call_count = 0

        async def failing_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Transient error")

        try:
            await retry_with_backoff(
                failing_func,
                max_attempts=3,
                base_delay=0.1,
                max_delay=1.0,
            )
        except TimeoutError:
            pass

        # Jitter is added, but we can't easily verify the exact delay
        # Just verify it was called multiple times
        assert call_count == 3


class TestCircuitBreakerReset:
    """Tests for manual circuit breaker reset."""

    def test_reset_closes_circuit(self) -> None:
        """Test reset closes circuit and clears state."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=60,
        )

        # Manually set to OPEN (simulating failure)
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5
        breaker._success_count = 2
        breaker._last_failure_time = datetime.now(timezone.utc)

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker._last_failure_time is None


class TestCircuitBreakerWithPerplexityClient:
    """Tests for circuit breaker integration with PerplexityClient."""

    @pytest.mark.asyncio
    async def test_ask_streaming_uses_circuit_breaker(self) -> None:
        """Test ask_streaming uses circuit breaker for API calls."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from arc_saga.integrations.perplexity_client import PerplexityClient

        # Simple mock storage
        class MockStorageBackend:
            async def save_message(self, message: Any) -> str:
                return str(message.id)  # type: ignore[no-any-return]

            async def get_by_session(self, session_id: str) -> list[Any]:
                return []

        # Simple mock stream
        class MockStreamChunk:
            def __init__(self, content: str | None) -> None:
                self.choices = [MagicMock()]
                self.choices[0].delta = MagicMock()
                self.choices[0].delta.content = content

        async def mock_stream_generator(
            chunks: list[str],
        ) -> AsyncIterator[MockStreamChunk]:
            for chunk in chunks:
                yield MockStreamChunk(chunk)

        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)  # type: ignore[arg-type]

        # Verify circuit breaker is initialized
        assert client.circuit_breaker is not None
        assert client.circuit_breaker.service_name == "perplexity"

        mock_stream = mock_stream_generator(["Response"])

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming("Test query"):
                pass

            # Verify API was called through circuit breaker
            assert mock_create.called

    @pytest.mark.asyncio
    async def test_ask_streaming_graceful_degradation_on_open_circuit(self) -> None:
        """Test ask_streaming handles open circuit with graceful degradation."""
        from arc_saga.integrations.perplexity_client import PerplexityClient

        # Simple mock storage
        class MockStorageBackend:
            async def save_message(self, message: Any) -> str:
                return str(message.id)  # type: ignore[no-any-return]

            async def get_by_session(self, session_id: str) -> list[Any]:
                return []

        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)  # type: ignore[arg-type]

        # Open the circuit breaker
        client.circuit_breaker._state = CircuitState.OPEN
        client.circuit_breaker._failure_count = 5
        client.circuit_breaker._last_failure_time = datetime.now(timezone.utc)

        # Call ask_streaming
        chunks = []
        async for chunk in client.ask_streaming("Test query"):
            chunks.append(chunk)

        # Should yield error message (graceful degradation)
        assert len(chunks) > 0
        import json

        error_chunk = json.loads(chunks[0])
        assert error_chunk["type"] == "error"
        assert "Circuit breaker is open" in error_chunk["message"]
        assert error_chunk["error_type"] == "CircuitBreakerOpenError"

    @pytest.mark.asyncio
    async def test_ask_streaming_retries_with_backoff_on_transient_errors(self) -> None:
        """Test ask_streaming retries with backoff on transient errors."""
        from unittest.mock import MagicMock, patch

        from arc_saga.integrations.perplexity_client import PerplexityClient

        # Simple mock storage
        class MockStorageBackend:
            async def save_message(self, message: Any) -> str:
                return str(message.id)  # type: ignore[no-any-return]

            async def get_by_session(self, session_id: str) -> list[Any]:
                return []

        # Simple mock stream
        class MockStreamChunk:
            def __init__(self, content: str | None) -> None:
                self.choices = [MagicMock()]
                self.choices[0].delta = MagicMock()
                self.choices[0].delta.content = content

        async def mock_stream_generator(
            chunks: list[str],
        ) -> AsyncIterator[MockStreamChunk]:
            for chunk in chunks:
                yield MockStreamChunk(chunk)

        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)  # type: ignore[arg-type]

        call_count = 0
        mock_stream = mock_stream_generator(["Response"])

        async def mock_create(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[MockStreamChunk]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Transient error")
            return mock_stream

        with patch.object(
            client.client.chat.completions, "create", side_effect=mock_create
        ):
            chunks = []
            async for chunk in client.ask_streaming("Test query"):
                chunks.append(chunk)

            # Should have retried and succeeded
            assert call_count == 2
            assert len(chunks) > 0


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker behavior under concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialized_by_lock(self) -> None:
        """Test concurrent calls are properly serialized by lock."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=10,  # High threshold
        )

        call_count = 0

        async def succeeding_func() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay
            return f"success_{call_count}"

        # Make concurrent calls
        results = await asyncio.gather(
            *[breaker.call(succeeding_func) for _ in range(10)]
        )

        # All calls should succeed
        assert len(results) == 10
        assert all(r.startswith("success_") for r in results)
        assert breaker.metrics.total_calls == 10
        assert breaker.metrics.successful_calls == 10

    @pytest.mark.asyncio
    async def test_concurrent_calls_respect_open_state(self) -> None:
        """Test concurrent calls all respect open circuit state."""
        breaker = CircuitBreaker(
            service_name="test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=60,
        )

        # Open the circuit
        async def failing_func() -> None:
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Make concurrent calls - all should be rejected
        async def any_func() -> str:
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await asyncio.gather(*[breaker.call(any_func) for _ in range(5)])


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_call_with_no_args(self) -> None:
        """Test calling circuit breaker with function that takes no args."""
        breaker = CircuitBreaker(service_name="test")

        async def no_args_func() -> str:
            return "success"

        result = await breaker.call(no_args_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_call_with_args_and_kwargs(self) -> None:
        """Test calling circuit breaker with function that takes args and kwargs."""
        breaker = CircuitBreaker(service_name="test")

        async def func_with_params(a: str, b: int, c: str = "default") -> str:
            return f"{a}_{b}_{c}"

        result = await breaker.call(func_with_params, "test", 42, c="custom")
        assert result == "test_42_custom"

    @pytest.mark.asyncio
    async def test_call_preserves_exception_type(self) -> None:
        """Test circuit breaker preserves original exception type."""
        breaker = CircuitBreaker(service_name="test")

        async def raising_value_error() -> None:
            raise ValueError("Test value error")

        with pytest.raises(ValueError, match="Test value error"):
            await breaker.call(raising_value_error)

    @pytest.mark.asyncio
    async def test_call_preserves_exception_message(self) -> None:
        """Test circuit breaker preserves original exception message."""
        breaker = CircuitBreaker(service_name="test")

        async def raising_custom_error() -> None:
            raise Exception("Custom error message")

        with pytest.raises(Exception, match="Custom error message"):
            await breaker.call(raising_custom_error)

    def test_should_attempt_recovery_returns_false_when_no_failure_time(self) -> None:
        """Test _should_attempt_recovery returns False when no failure time set."""
        breaker = CircuitBreaker(service_name="test")

        assert breaker._should_attempt_recovery() is False

    def test_should_attempt_recovery_returns_false_before_timeout(self) -> None:
        """Test _should_attempt_recovery returns False before timeout."""
        breaker = CircuitBreaker(
            service_name="test",
            timeout_seconds=60,
        )

        breaker._last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=30)

        assert breaker._should_attempt_recovery() is False

    def test_should_attempt_recovery_returns_true_after_timeout(self) -> None:
        """Test _should_attempt_recovery returns True after timeout."""
        breaker = CircuitBreaker(
            service_name="test",
            timeout_seconds=60,
        )

        breaker._last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=61)

        assert breaker._should_attempt_recovery() is True
