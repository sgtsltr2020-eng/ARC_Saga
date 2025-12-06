"""
End-to-end orchestration flow coverage (capture → search → reason/fallback).

Non-invasive: uses FastAPI ASGI app directly with in-memory SQLite path to avoid
touching user data. Focus is verifying current behavior, not changing logic.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest
import pytest_asyncio
from httpx import AsyncClient

from arc_saga.api import server
from arc_saga.api.health_monitor import health_monitor
from arc_saga.integrations.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture(autouse=True)
def reset_health_monitor():
    """Reset health monitor state before and after each test."""
    health_monitor._circuit_breakers.clear()
    health_monitor._endpoint_latencies.clear()
    health_monitor._recent_errors.clear()
    health_monitor.clear_cache()
    yield
    health_monitor._circuit_breakers.clear()
    health_monitor._endpoint_latencies.clear()
    health_monitor._recent_errors.clear()
    health_monitor.clear_cache()


@pytest_asyncio.fixture
async def test_client(tmp_path) -> AsyncIterator[AsyncClient]:
    """Spin up ASGI test client with isolated SQLite DB path."""
    # Point storage to a temp file and reset connection
    server.storage.db_path = tmp_path / "memory.db"  # type: ignore[attr-defined]
    if getattr(server.storage, "_connection", None) is not None:
        try:
            server.storage._connection.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        server.storage._connection = None  # type: ignore[attr-defined]

    await server.storage.initialize()

    async with AsyncClient(app=server.app, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_capture_search_thread_flow(test_client: AsyncClient) -> None:
    """Capture a message, search it, then retrieve the thread."""
    capture_payload = {
        "source": "perplexity",
        "role": "user",
        "content": "Hello world from E2E",
        "thread_id": "thread-123",
        "metadata": {"k": "v"},
    }

    capture_resp = await test_client.post("/capture", json=capture_payload)
    assert capture_resp.status_code == 200
    capture_json = capture_resp.json()
    thread_id = capture_json["thread_id"]
    assert capture_json["status"] == "stored"

    search_resp = await test_client.post(
        "/search", json={"query": "Hello", "limit": 10}
    )
    assert search_resp.status_code == 200
    search_json = search_resp.json()
    assert search_json["count"] >= 1
    assert any("Hello world" in r["content"] for r in search_json["results"])

    thread_resp = await test_client.get(f"/thread/{thread_id}")
    assert thread_resp.status_code == 200
    thread_json = thread_resp.json()
    assert thread_json["thread_id"] == thread_id
    assert thread_json["count"] == 1
    assert thread_json["messages"][0]["content"] == capture_payload["content"]


@pytest.mark.asyncio
async def test_perplexity_endpoint_graceful_unavailable(
    test_client: AsyncClient,
) -> None:
    """Perplexity endpoint should respond 503 when integration is absent (fallback)."""
    ask_resp = await test_client.post(
        "/perplexity/ask",
        json={"query": "test", "thread_id": "thread-xyz", "inject_context": False},
    )
    assert ask_resp.status_code == 503
    assert "not available" in ask_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_metrics_reflect_circuit_breaker_state(test_client: AsyncClient) -> None:
    """Metrics endpoint should surface registered circuit breaker state/metrics."""
    # Register a dummy circuit breaker and force OPEN state
    breaker = CircuitBreaker(
        service_name="dummy", failure_threshold=1, timeout_seconds=1
    )
    breaker._state = CircuitState.OPEN  # type: ignore[attr-defined]
    breaker.metrics.record_circuit_open()
    health_monitor.register_circuit_breaker("dummy", breaker)

    metrics_resp = await test_client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_json = metrics_resp.json()
    assert "dummy" in metrics_json["circuit_breakers"]
    assert metrics_json["circuit_breakers"]["dummy"]["total_calls"] >= 0
