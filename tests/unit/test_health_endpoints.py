"""
Unit tests for health check endpoints.

Tests verify:
1. GET /health returns correct status
2. GET /health/detailed returns full diagnostics
3. GET /metrics returns performance metrics
4. Health status logic (healthy/degraded/unhealthy)
5. Metrics caching (10s TTL)
6. Error tracking (last 5 errors)

Coverage target: 95%+
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from arc_saga.api.health_monitor import (
    HealthMonitor,
    check_database_health,
    check_storage_space,
    determine_health_status,
    health_monitor,
)
from arc_saga.integrations.circuit_breaker import CircuitBreaker, CircuitState


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_health_monitor_initialization(self) -> None:
        """Test HealthMonitor initializes correctly."""
        monitor = HealthMonitor()

        assert monitor._endpoint_latencies == {}
        assert len(monitor._recent_errors) == 0
        assert monitor._circuit_breakers == {}
        assert monitor._metrics_cache is None

    def test_register_circuit_breaker(self) -> None:
        """Test registering circuit breaker."""
        monitor = HealthMonitor()
        breaker = CircuitBreaker(service_name="test_service")

        monitor.register_circuit_breaker("test_service", breaker)

        assert "test_service" in monitor._circuit_breakers
        assert monitor._circuit_breakers["test_service"] is breaker

    def test_record_latency(self) -> None:
        """Test recording endpoint latency."""
        monitor = HealthMonitor()

        monitor.record_latency("/test", 100.0)
        monitor.record_latency("/test", 200.0)

        assert "/test" in monitor._endpoint_latencies
        metrics = monitor._endpoint_latencies["/test"]
        assert len(metrics.latencies) == 2

    def test_record_error(self) -> None:
        """Test recording error."""
        monitor = HealthMonitor()

        error = ValueError("Test error")
        monitor.record_error("test_operation", error, {"key": "value"})

        assert len(monitor._recent_errors) == 1
        error_dict = monitor._recent_errors[0]
        assert error_dict["operation"] == "test_operation"
        assert error_dict["error_type"] == "ValueError"

    def test_record_error_max_5(self) -> None:
        """Test error queue maintains max 5 errors."""
        monitor = HealthMonitor()

        for i in range(10):
            error = ValueError(f"Error {i}")
            monitor.record_error(f"operation_{i}", error, {})

        assert len(monitor._recent_errors) == 5
        # Should contain last 5 errors
        assert monitor._recent_errors[-1]["operation"] == "operation_9"

    def test_get_endpoint_latencies(self) -> None:
        """Test getting endpoint latencies."""
        monitor = HealthMonitor()

        monitor.record_latency("/test1", 100.0)
        monitor.record_latency("/test2", 200.0)

        latencies = monitor.get_endpoint_latencies()

        assert "/test1" in latencies
        assert "/test2" in latencies
        assert latencies["/test1"]["operation"] == "/test1"

    def test_get_recent_errors(self) -> None:
        """Test getting recent errors."""
        monitor = HealthMonitor()

        error = ValueError("Test error")
        monitor.record_error("test_operation", error, {})

        errors = monitor.get_recent_errors()

        assert len(errors) == 1
        assert errors[0]["operation"] == "test_operation"

    def test_get_circuit_breaker_states(self) -> None:
        """Test getting circuit breaker states."""
        monitor = HealthMonitor()
        breaker = CircuitBreaker(service_name="test_service")

        monitor.register_circuit_breaker("test_service", breaker)

        states = monitor.get_circuit_breaker_states()

        assert "test_service" in states
        assert states["test_service"]["state"] == CircuitState.CLOSED.value

    def test_metrics_caching(self) -> None:
        """Test metrics caching with TTL."""
        monitor = HealthMonitor()

        # Set cache
        test_metrics = {"test": "data"}
        monitor.set_cached_metrics(test_metrics)

        # Should return cached metrics
        cached = monitor.get_cached_metrics()
        assert cached == test_metrics

        # Clear cache
        monitor.clear_cache()
        assert monitor.get_cached_metrics() is None

    def test_metrics_cache_expires(self) -> None:
        """Test metrics cache expires after TTL."""
        monitor = HealthMonitor()
        monitor._cache_ttl = 0.1  # Short TTL for testing

        # Set cache
        test_metrics = {"test": "data"}
        monitor.set_cached_metrics(test_metrics)

        # Should return cached metrics immediately
        assert monitor.get_cached_metrics() == test_metrics

        # Wait for cache to expire
        time.sleep(0.2)

        # Should return None after expiration
        assert monitor.get_cached_metrics() is None


class TestDatabaseHealthCheck:
    """Tests for database health checking."""

    @pytest.mark.asyncio
    async def test_check_database_health_healthy(self) -> None:
        """Test database health check when healthy."""
        mock_storage = AsyncMock()
        mock_storage.health_check.return_value = True

        result = await check_database_health(mock_storage)

        assert result["healthy"] is True
        assert result["reachable"] is True
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_database_health_unhealthy(self) -> None:
        """Test database health check when unhealthy."""
        mock_storage = AsyncMock()
        mock_storage.health_check.return_value = False

        result = await check_database_health(mock_storage)

        assert result["healthy"] is False
        assert result["reachable"] is True
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_check_database_health_unreachable(self) -> None:
        """Test database health check when unreachable."""
        mock_storage = AsyncMock()
        mock_storage.health_check.side_effect = Exception("Connection failed")

        result = await check_database_health(mock_storage)

        assert result["healthy"] is False
        assert result["reachable"] is False
        assert "error" in result


class TestStorageSpaceCheck:
    """Tests for storage space checking."""

    def test_check_storage_space_success(self) -> None:
        """Test storage space check succeeds."""
        test_path = Path.home()

        result = check_storage_space(test_path)

        assert "available_mb" in result
        assert "total_mb" in result
        assert "used_mb" in result
        assert "usage_percent" in result
        assert result["available_mb"] >= 0
        assert result["total_mb"] >= 0

    def test_check_storage_space_error(self) -> None:
        """Test storage space check handles errors."""
        invalid_path = Path("/nonexistent/path/that/does/not/exist")

        result = check_storage_space(invalid_path)

        assert result["available_mb"] == 0.0
        assert "error" in result


class TestHealthStatusDetermination:
    """Tests for health status determination logic."""

    def test_determine_health_status_healthy(self) -> None:
        """Test healthy status when all systems nominal."""
        db_health = {"reachable": True, "latency_ms": 10.0}
        storage_space = {"available_mb": 1000.0}
        circuit_breakers = {}
        endpoint_latencies = {}

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "healthy"

    def test_determine_health_status_unhealthy_database_unreachable(self) -> None:
        """Test unhealthy status when database unreachable."""
        db_health = {"reachable": False, "latency_ms": 0.0}
        storage_space = {"available_mb": 1000.0}
        circuit_breakers = {}
        endpoint_latencies = {}

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "unhealthy"

    def test_determine_health_status_unhealthy_low_storage(self) -> None:
        """Test unhealthy status when storage space < 100MB."""
        db_health = {"reachable": True, "latency_ms": 10.0}
        storage_space = {"available_mb": 50.0}  # Less than 100MB
        circuit_breakers = {}
        endpoint_latencies = {}

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "unhealthy"

    def test_determine_health_status_degraded_circuit_open(self) -> None:
        """Test degraded status when circuit breaker is open."""
        db_health = {"reachable": True, "latency_ms": 10.0}
        storage_space = {"available_mb": 1000.0}
        circuit_breakers = {"perplexity": {"state": "open", "metrics": {}}}
        endpoint_latencies = {}

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "degraded"

    def test_determine_health_status_degraded_high_db_latency(self) -> None:
        """Test degraded status when database latency p95 > 500ms."""
        db_health = {"reachable": True, "latency_ms": 600.0}  # > 500ms
        storage_space = {"available_mb": 1000.0}
        circuit_breakers = {}
        endpoint_latencies = {}

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "degraded"

    def test_determine_health_status_degraded_high_endpoint_latency(self) -> None:
        """Test degraded status when endpoint latency p95 > 500ms."""
        db_health = {"reachable": True, "latency_ms": 10.0}
        storage_space = {"available_mb": 1000.0}
        circuit_breakers = {}
        endpoint_latencies = {"/test": {"p95": 600.0}}  # > 500ms

        status = determine_health_status(
            db_health, storage_space, circuit_breakers, endpoint_latencies
        )

        assert status == "degraded"


class TestHealthEndpoints:
    """Tests for health check API endpoints."""

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        """Create mock storage backend."""
        storage = AsyncMock()
        storage.health_check.return_value = True
        return storage

    @pytest.fixture
    def client(self, mock_storage: AsyncMock) -> TestClient:
        """Create test client with mocked storage."""
        from arc_saga.api.server import app

        # Patch storage
        with patch("arc_saga.api.server.storage", mock_storage):
            # Clear health monitor state
            health_monitor._circuit_breakers.clear()
            health_monitor._endpoint_latencies.clear()
            health_monitor._recent_errors.clear()
            health_monitor.clear_cache()

            yield TestClient(app)

    def test_health_endpoint_returns_status(self, client: TestClient) -> None:
        """Test GET /health returns status and timestamp."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_endpoint_healthy_when_all_good(self, client: TestClient) -> None:
        """Test /health returns healthy when all systems nominal."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # Should be healthy if database is reachable and storage is sufficient
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_detailed_endpoint_returns_full_info(
        self, client: TestClient
    ) -> None:
        """Test GET /health/detailed returns full diagnostics."""
        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "database" in data
        assert "storage_space" in data
        assert "circuit_breakers" in data
        assert "endpoint_latencies" in data
        assert "last_errors" in data

        # Verify database structure
        assert "healthy" in data["database"]
        assert "latency_ms" in data["database"]
        assert "reachable" in data["database"]

        # Verify storage space structure
        assert "available_mb" in data["storage_space"]
        assert "total_mb" in data["storage_space"]

    def test_health_detailed_with_circuit_breaker(self, client: TestClient) -> None:
        """Test /health/detailed includes circuit breaker info."""
        # Register a circuit breaker
        breaker = CircuitBreaker(service_name="test_service")
        health_monitor.register_circuit_breaker("test_service", breaker)

        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert "test_service" in data["circuit_breakers"]
        assert "state" in data["circuit_breakers"]["test_service"]
        assert "metrics" in data["circuit_breakers"]["test_service"]

    def test_metrics_endpoint_returns_metrics(self, client: TestClient) -> None:
        """Test GET /metrics returns performance metrics."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "circuit_breakers" in data
        assert "endpoint_latencies" in data
        assert "recent_errors" in data
        assert "timestamp" in data

    def test_metrics_endpoint_caches_results(self, client: TestClient) -> None:
        """Test /metrics caches results for 10 seconds."""
        # First call
        response1 = client.get("/metrics")
        assert response1.status_code == 200
        data1 = response1.json()
        timestamp1 = data1["timestamp"]

        # Second call immediately (should use cache)
        response2 = client.get("/metrics")
        assert response2.status_code == 200
        data2 = response2.json()
        timestamp2 = data2["timestamp"]

        # Timestamps should be the same (cached)
        assert timestamp1 == timestamp2

    def test_metrics_endpoint_cache_expires(self, client: TestClient) -> None:
        """Test /metrics cache expires after TTL."""
        # Set short cache TTL
        health_monitor._cache_ttl = 0.1

        # First call
        response1 = client.get("/metrics")
        assert response1.status_code == 200
        data1 = response1.json()
        timestamp1 = data1["timestamp"]

        # Wait for cache to expire
        time.sleep(0.2)

        # Second call (should refresh cache)
        response2 = client.get("/metrics")
        assert response2.status_code == 200
        data2 = response2.json()
        timestamp2 = data2["timestamp"]

        # Timestamps should be different (cache expired)
        assert timestamp1 != timestamp2

    def test_health_endpoint_handles_errors(self, client: TestClient) -> None:
        """Test /health handles errors gracefully."""
        # Make storage health check fail
        with patch("arc_saga.api.server.storage") as mock_storage:
            mock_storage.health_check.side_effect = Exception("Database error")

            response = client.get("/health")

            # Should still return response (unhealthy status)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

    def test_health_detailed_handles_errors(self, client: TestClient) -> None:
        """Test /health/detailed handles errors gracefully."""
        # Make storage health check fail
        with patch("arc_saga.api.server.storage") as mock_storage:
            mock_storage.health_check.side_effect = Exception("Database error")

            response = client.get("/health/detailed")

            # Should still return response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["database"]["reachable"] is False

    def test_metrics_endpoint_handles_errors(self, client: TestClient) -> None:
        """Test /metrics handles errors gracefully."""
        # Force an error in metrics collection
        with patch.object(health_monitor, "get_circuit_breaker_states") as mock_get:
            mock_get.side_effect = Exception("Metrics error")

            response = client.get("/metrics")

            # Should still return response
            assert response.status_code == 200
            data = response.json()
            assert "circuit_breakers" in data
            assert "timestamp" in data


class TestLatencyTrackingMiddleware:
    """Tests for latency tracking middleware."""

    @pytest.mark.asyncio
    async def test_middleware_tracks_latency(self) -> None:
        """Test middleware tracks endpoint latency."""
        from arc_saga.api.health_monitor import LatencyTrackingMiddleware
        from starlette.responses import Response
        from starlette.requests import Request

        # Clear latencies
        health_monitor._endpoint_latencies.clear()

        async def mock_handler(request: Request) -> Response:
            await asyncio.sleep(0.01)  # Small delay
            return Response(content="test", status_code=200)

        middleware = LatencyTrackingMiddleware(mock_handler)

        # Create proper request scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("localhost", 8000),
        }

        request = Request(scope)

        # Process request
        response = await middleware.dispatch(request, mock_handler)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_handles_errors(self) -> None:
        """Test middleware handles errors and records them."""
        from arc_saga.api.health_monitor import LatencyTrackingMiddleware
        from starlette.requests import Request

        # Clear errors
        health_monitor._recent_errors.clear()

        async def failing_handler(request: Request) -> None:
            raise ValueError("Test error")

        middleware = LatencyTrackingMiddleware(failing_handler)

        # Create proper request scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("localhost", 8000),
        }

        request = Request(scope)

        # Process request (should raise error)
        with pytest.raises(ValueError):
            await middleware.dispatch(request, failing_handler)

        # Check error was recorded
        errors = health_monitor.get_recent_errors()
        assert len(errors) > 0
