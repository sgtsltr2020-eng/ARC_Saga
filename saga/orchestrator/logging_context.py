"""
Logging Context Integration.

Injects Arbitration Context trace IDs into structured logs and optionally into
OpenTelemetry (if available).
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from .arbitration_context import (
    ArbitrationContext,
    reset_arbitration_context,
    set_arbitration_context,
)

# Fallback for OpenTelemetry to avoid hard dependency breakage
try:
    from opentelemetry import trace
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    trace = None # type: ignore

def _get_tracer():
    if API_AVAILABLE and trace:
        return trace.get_tracer(__name__)
    return None

@contextlib.contextmanager
def with_trace_logging(ctx: ArbitrationContext) -> Iterator[None]:
    """
    Context manager to activate tracing and structured logging context.

    Features:
    1. Sets the ContextVar for internal access.
    2. Injects [TRACE:...] [SPAN:...] prefixes (conceptually) or attributes into logging.
    3. Starts an OpenTelemetry span if available.

    Args:
        ctx: The ArbitrationContext to activate.
    """
    token = set_arbitration_context(ctx)
    tracer = _get_tracer()
    
    # Update logging adapter context if we had one, or just ensure
    # subsequent log_with_context calls pick it up via get_arbitration_context().
    # Note: Our error_instrumentation.py likely needs to read get_arbitration_context()
    # to inject fields. For now, this sets the stage.
    
    span = None
    if tracer:
        # Start OTel span
        span = tracer.start_span(
            name=f"saga.arbitration.{ctx.span_id}",
            attributes={
                "saga.trace_id": str(ctx.trace_id),
                "saga.span_id": ctx.span_id,
                "saga.workflow_id": ctx.workflow_id,
                "saga.ag_manager_id": str(ctx.ag_manager_id or ""),
            }
        )
    
    try:
        # Log entry (optional, but good for debug)
        # instrumentation_logger.info(f"[TRACE:{ctx.trace_id}] Entering span {ctx.span_id}")
        if span:
            with trace.use_span(span, end_on_exit=True):
                yield
        else:
            yield
    finally:
        reset_arbitration_context(token)
