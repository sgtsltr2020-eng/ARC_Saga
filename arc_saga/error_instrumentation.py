"""
Error Instrumentation Module
Comprehensive logging and error tracking for ARC SAGA.
Makes debugging trivial by capturing complete context.
"""

from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import traceback
import json
import logging
from contextvars import ContextVar
from enum import Enum

# Context storage (thread-safe)
request_context: ContextVar[Dict[str, Any]] = ContextVar(
    'request_context',
    default={}
)

def create_request_context(
    user_id: Optional[str] = None,
    service_name: str = "arc_saga"
) -> Dict[str, Any]:
    """
    Create context with unique IDs for entire request lifecycle.

    Use this at the start of every request/operation.
    All logs will automatically include these IDs.
    """
    return {
        "request_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "span_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "service_name": service_name,
        "user_id": user_id,
    }

def get_request_context() -> Dict[str, Any]:
    """
    Get current request context (use in all logging).

    If no context exists, creates one automatically.
    """
    ctx = request_context.get({})
    if not ctx:
        ctx = create_request_context()
        request_context.set(ctx)
    return ctx

def get_correlation_id() -> str:
    """Get correlation ID for this request."""
    return get_request_context().get("request_id", "unknown")

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def log_with_context(
    level: str,
    message: str,
    **kwargs: Any
) -> None:
    """
    Log with full context and structured data.
    Every log includes correlation IDs, timestamp, context, and any extra kwargs.
    """
    ctx = get_request_context()
    log_entry = {
        **ctx,  # Include correlation IDs
        "message": message,
        "level": level.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs  # Additional context
    }
    logger = logging.getLogger(__name__)
    if level.upper() == "ERROR":
        logger.error(json.dumps(log_entry, default=str))
    elif level.upper() == "WARNING":
        logger.warning(json.dumps(log_entry, default=str))
    elif level.upper() == "CRITICAL":
        logger.critical(json.dumps(log_entry, default=str))
    else:
        logger.info(json.dumps(log_entry, default=str))

class LatencyMetrics:
    """Track and analyze operation latency."""

    def __init__(self, operation: str):
        self.operation = operation
        self.latencies: list[float] = []

    @property
    def p50(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = max(0, int(len(sorted_latencies) * 0.95) - 1)
        return sorted_latencies[index]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = max(0, int(len(sorted_latencies) * 0.99) - 1)
        return sorted_latencies[index]

    def add(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)
        # Alert if slow
        if latency_ms > 1000:
            log_with_context(
                "warning",
                f"{self.operation} slow",
                latency_ms=latency_ms,
                p95=self.p95,
                p99=self.p99
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "operation": self.operation,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "count": len(self.latencies),
        }

class ErrorContext:
    """
    Capture complete context when error occurs.
    """

    def __init__(
        self,
        operation: str,
        error: Exception,
        context: Dict[str, Any],
    ):
        self.operation = operation
        self.error = error
        self.context = context
        self.stack_trace = traceback.format_exc()
        self.timestamp = datetime.utcnow().isoformat()

    def log(self) -> None:
        log_with_context(
            "error",
            f"{self.operation}_failed",
            error_type=type(self.error).__name__,
            error_message=str(self.error),
            stack_trace=self.stack_trace,
            operation_context=self.context,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "stack_trace": self.stack_trace,
            "context": self.context,
            "timestamp": self.timestamp,
            "correlation_id": get_correlation_id(),
        }

class CircuitBreakerMetrics:
    """Track circuit breaker state and effectiveness."""

    def __init__(self, service: str):
        self.service = service
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opens = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0

    def record_call(self, success: bool) -> None:
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

    def record_circuit_open(self) -> None:
        self.circuit_opens += 1
        log_with_context(
            "warning",
            f"{self.service}_circuit_open",
            service=self.service,
            circuit_opens=self.circuit_opens,
            success_rate=self.success_rate
        )

    def record_recovery_attempt(self, success: bool) -> None:
        self.recovery_attempts += 1
        if success:
            self.successful_recoveries += 1
            log_with_context(
                "info",
                f"{self.service}_recovered",
                service=self.service,
                recovery_success_rate=self.recovery_rate
            )

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def recovery_rate(self) -> float:
        if self.recovery_attempts == 0:
            return 0
        return (self.successful_recoveries / self.recovery_attempts) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "circuit_opens": self.circuit_opens,
            "recovery_rate": self.recovery_rate,
        }
