"""
Lightweight observability helpers: Prometheus metrics + OTEL spans.

Non-invasive wrappers only — no business logic changes. Safe to import even if
optional dependencies are missing (falls back gracefully).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .error_instrumentation import (
    create_request_context,
    get_request_context,
    log_with_context,
    request_context,
)

try:
    from prometheus_client import Counter, Histogram  # type: ignore
except Exception:  # pragma: no cover - optional dep guard
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import Span  # type: ignore
except Exception:  # pragma: no cover - optional dep guard
    trace = None  # type: ignore
    Span = None  # type: ignore

# Prometheus metrics (labels kept small to avoid high cardinality)
REQUEST_LATENCY = (
    Histogram(
        "arc_saga_request_latency_ms",
        "Request latency in milliseconds",
        ["path", "method", "status"],
    )
    if Histogram
    else None
)

REQUEST_COUNT = (
    Counter(
        "arc_saga_request_total",
        "Total requests by path/method/status",
        ["path", "method", "status"],
    )
    if Counter
    else None
)

ERROR_COUNT = (
    Counter(
        "arc_saga_errors_total",
        "Total errors by path/method/status",
        ["path", "method", "status"],
    )
    if Counter
    else None
)


def _safe_label(value: str) -> str:
    """Normalize label values to avoid explosions."""
    return value.split("?")[0][:120]


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware to capture request latency, counts, and optional OTEL spans."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # type: ignore[override]
        start = time.perf_counter()
        path = _safe_label(request.url.path)
        method = request.method

        # Ensure request context exists for correlation IDs
        ctx = create_request_context(service_name="arc_saga_api")
        request_context.set(ctx)

        span_cm = _noop_span()
        if trace:
            tracer = trace.get_tracer("arc_saga")
            span_cm = tracer.start_as_current_span(
                name=f"http.{method.lower()}",
                attributes={
                    "http.method": method,
                    "http.target": path,
                },
            )

        try:
            with span_cm:
                response = await call_next(request)

            status = str(response.status_code)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if REQUEST_LATENCY:
                REQUEST_LATENCY.labels(path=path, method=method, status=status).observe(
                    elapsed_ms
                )
            if REQUEST_COUNT:
                REQUEST_COUNT.labels(path=path, method=method, status=status).inc()

            return response
        except Exception as exc:  # pragma: no cover - pass-through to existing handlers
            status = "500"
            if ERROR_COUNT:
                ERROR_COUNT.labels(path=path, method=method, status=status).inc()
            log_with_context(
                "error",
                "request_failed",
                path=path,
                method=method,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise


def _noop_span():
    """Fallback context manager when OTEL is unavailable."""

    @contextmanager
    def _cm():
        yield

    return _cm()


@asynccontextmanager
async def instrument_async_operation(name: str, **attrs: Any):
    """
    Async context manager for instrumenting background operations.

    Falls back to a no-op if OTEL is not installed.
    """
    ctx = get_request_context()
    start = time.perf_counter()
    span_cm = _noop_span()

    if trace:
        tracer = trace.get_tracer("arc_saga")
        span_cm = tracer.start_as_current_span(
            name=name,
            attributes={**attrs, "request_id": ctx.get("request_id", "n/a")},
        )

    try:
        with span_cm:
            yield
        duration_ms = (time.perf_counter() - start) * 1000
        log_with_context("info", f"{name}_complete", duration_ms=duration_ms)
    except Exception as exc:
        log_with_context(
            "error",
            f"{name}_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def init_observability(app: Any) -> None:
    """
    Initialize observability on a FastAPI app.

    Safe to call multiple times; middleware will only be added once.
    """
    middleware_names = {mw.cls for mw in app.user_middleware}
    if ObservabilityMiddleware not in middleware_names:
        app.add_middleware(ObservabilityMiddleware)

