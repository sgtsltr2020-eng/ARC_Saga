"""
Health monitoring system for ARC Saga API.

Tracks API latency, errors, circuit breaker state, and system health.
Provides cached metrics for performance.
"""

from __future__ import annotations

import shutil
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..error_instrumentation import (
    ErrorContext,
    LatencyMetrics,
    create_request_context,
    log_with_context,
    request_context,
)


class HealthMonitor:
    """
    Central health monitoring system.

    Tracks:
    - API latency per endpoint
    - Recent errors (last 5)
    - Circuit breaker states
    - System metrics
    """

    def __init__(self) -> None:
        """Initialize health monitor."""
        self._endpoint_latencies: Dict[str, LatencyMetrics] = {}
        self._recent_errors: Deque[Dict[str, Any]] = deque(maxlen=5)
        self._circuit_breakers: Dict[str, Any] = {}
        self._metrics_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 10.0  # 10 seconds

    def register_circuit_breaker(self, service_name: str, breaker: Any) -> None:
        """
        Register a circuit breaker for monitoring.

        Args:
            service_name: Name of the service
            breaker: CircuitBreaker instance
        """
        self._circuit_breakers[service_name] = breaker

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """
        Record API endpoint latency.

        Args:
            endpoint: Endpoint path (e.g., "/health")
            latency_ms: Latency in milliseconds
        """
        if endpoint not in self._endpoint_latencies:
            self._endpoint_latencies[endpoint] = LatencyMetrics(operation=endpoint)

        self._endpoint_latencies[endpoint].add(latency_ms)

    def record_error(
        self,
        operation: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> None:
        """
        Record an error for health monitoring.

        Args:
            operation: Operation name
            error: Exception that occurred
            context: Additional context
        """
        error_ctx = ErrorContext(
            operation=operation,
            error=error,
            context=context,
        )

        error_dict = error_ctx.to_dict()
        self._recent_errors.append(error_dict)

        log_with_context(
            "error",
            "health_monitor_error_recorded",
            operation=operation,
            error_type=type(error).__name__,
        )

    def get_endpoint_latencies(self) -> Dict[str, Dict[str, float]]:
        """
        Get latency metrics for all endpoints.

        Returns:
            Dictionary mapping endpoint to latency metrics (p50, p95, p99)
        """
        return {
            endpoint: metrics.to_dict()
            for endpoint, metrics in self._endpoint_latencies.items()
        }

    def get_recent_errors(self) -> list[Dict[str, Any]]:
        """
        Get last 5 errors.

        Returns:
            List of error dictionaries
        """
        return list(self._recent_errors)

    def get_circuit_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get circuit breaker states and metrics.

        Returns:
            Dictionary mapping service name to circuit breaker state and metrics
        """
        states: Dict[str, Dict[str, Any]] = {}

        for service_name, breaker in self._circuit_breakers.items():
            states[service_name] = {
                "state": breaker.state.value,
                "metrics": breaker.metrics.to_dict(),
            }

        return states

    def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get cached metrics if still valid.

        Returns:
            Cached metrics dictionary or None if cache expired
        """
        current_time = time.time()
        if (
            self._metrics_cache is not None
            and (current_time - self._cache_timestamp) < self._cache_ttl
        ):
            return self._metrics_cache

        return None

    def set_cached_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Cache metrics for performance.

        Args:
            metrics: Metrics dictionary to cache
        """
        self._metrics_cache = metrics
        self._cache_timestamp = time.time()

    def clear_cache(self) -> None:
        """Clear metrics cache."""
        self._metrics_cache = None
        self._cache_timestamp = 0.0


# Global health monitor instance
health_monitor = HealthMonitor()


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API endpoint latency.

    Measures request processing time and records it in health monitor.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:  # type: ignore[no-any-return]
        """
        Process request and track latency.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from handler
        """
        # Create request context
        ctx = create_request_context(service_name="arc_saga_api")
        request_context.set(ctx)

        # Track latency
        start_time = time.time()
        endpoint = request.url.path

        try:
            response = await call_next(request)
            elapsed_ms = (time.time() - start_time) * 1000

            # Record latency (exclude health endpoints to avoid recursion)
            if not endpoint.startswith("/health") and not endpoint.startswith(
                "/metrics"
            ):
                health_monitor.record_latency(endpoint, elapsed_ms)

            return response
        except Exception as e:
            # Record error
            health_monitor.record_error(
                operation=f"{request.method} {endpoint}",
                error=e,
                context={
                    "method": request.method,
                    "path": endpoint,
                    "query_params": dict(request.query_params),
                },
            )
            raise


async def check_database_health(storage: Any) -> Dict[str, Any]:
    """
    Check database health and performance.

    Args:
        storage: StorageBackend instance

    Returns:
        Dictionary with database health information
    """
    start_time = time.time()

    try:
        is_healthy = await storage.health_check()
        latency_ms = (time.time() - start_time) * 1000

        return {
            "healthy": is_healthy,
            "latency_ms": latency_ms,
            "reachable": True,
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            "error",
            "database_health_check_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )

        return {
            "healthy": False,
            "latency_ms": latency_ms,
            "reachable": False,
            "error": str(e),
        }


def check_storage_space(storage_path: Any) -> Dict[str, Any]:
    """
    Check available storage space.

    Args:
        storage_path: Path to storage directory or file

    Returns:
        Dictionary with storage space information
    """
    try:
        # Get disk space for the path
        total, used, free = shutil.disk_usage(storage_path)

        # Convert to MB
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        used_mb = used / (1024 * 1024)

        return {
            "available_mb": round(free_mb, 2),
            "total_mb": round(total_mb, 2),
            "used_mb": round(used_mb, 2),
            "usage_percent": round((used / total) * 100, 2) if total > 0 else 0.0,
        }
    except Exception as e:
        log_with_context(
            "error",
            "storage_space_check_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )

        return {
            "available_mb": 0.0,
            "total_mb": 0.0,
            "used_mb": 0.0,
            "usage_percent": 0.0,
            "error": str(e),
        }


def determine_health_status(
    db_health: Dict[str, Any],
    storage_space: Dict[str, Any],
    circuit_breakers: Dict[str, Dict[str, Any]],
    endpoint_latencies: Dict[str, Dict[str, float]],
) -> str:
    """
    Determine overall health status.

    Health status logic:
    - healthy: all systems nominal
    - degraded: circuit breaker open OR database latency p95 > 500ms
    - unhealthy: database unreachable OR storage space < 100MB

    Args:
        db_health: Database health information
        storage_space: Storage space information
        circuit_breakers: Circuit breaker states
        endpoint_latencies: Endpoint latency metrics

    Returns:
        Health status: "healthy", "degraded", or "unhealthy"
    """
    # Check for unhealthy conditions
    if not db_health.get("reachable", False):
        return "unhealthy"

    available_mb = storage_space.get("available_mb", 0.0)
    if available_mb < 100.0:
        return "unhealthy"

    # Check for degraded conditions
    # Check circuit breakers
    for service_name, breaker_info in circuit_breakers.items():
        if breaker_info.get("state") == "open":
            return "degraded"

    # Check database latency
    db_latency_p95 = db_health.get("latency_ms", 0.0)
    if db_latency_p95 > 500.0:
        return "degraded"

    # Check endpoint latencies
    for endpoint, metrics in endpoint_latencies.items():
        p95 = metrics.get("p95", 0.0)
        if p95 > 500.0:
            return "degraded"

    return "healthy"
