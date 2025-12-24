"""
Arbitration Context for Distributed Tracing and Guardrails.

Provides immutable context propagation for reasoning traces, including integration
with Antigravity dashboard identifiers.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Module-level context variable for thread-safe propagation
_current_context: ContextVar[Optional[ArbitrationContext]] = ContextVar(
    "current_arbitration_context", default=None
)


@dataclass(frozen=True)
class ArbitrationContext:
    """
    Immutable context for an arbitration workflow.

    Attributes:
        trace_id: Canonical UUIDv4 correlation for the entire arbitration flow.
        span_id: Unique identifier for the current operation/span.
        parent_span_id: Identifier of the parent span, if any.
        workflow_id: High-level workflow identifier (e.g., from Orchestrator).
        ag_manager_id: Antigravity Manager ID for dashboard linking.
        ag_agent_ids: Tuple of active Antigravity Agent IDs in this scope.
    """

    trace_id: uuid.UUID
    span_id: str
    workflow_id: str
    parent_span_id: Optional[str] = None
    ag_manager_id: Optional[str] = None
    ag_agent_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate context fields."""
        if not self.span_id:
            raise ValueError("span_id cannot be empty.")
        if not self.workflow_id:
            raise ValueError("workflow_id cannot be empty.")
        
        # Ensure ag_agent_ids is a tuple (immutability)
        # Note: field(default_factory=tuple) helps, but explicit check implies strictness
        if not isinstance(self.ag_agent_ids, tuple):
             object.__setattr__(self, "ag_agent_ids", tuple(self.ag_agent_ids))

    def child_span(self, span_id: str) -> ArbitrationContext:
        """
        Create a child context with the current span as parent.

        Args:
            span_id: New span identifier.

        Returns:
            New ArbitrationContext instance linking to this parent.
        """
        return ArbitrationContext(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self.span_id,
            workflow_id=self.workflow_id,
            ag_manager_id=self.ag_manager_id,
            ag_agent_ids=self.ag_agent_ids,
        )


def get_arbitration_context() -> Optional[ArbitrationContext]:
    """Get the current active arbitration context."""
    return _current_context.get()


def set_arbitration_context(ctx: ArbitrationContext) -> Token:
    """
    Set the current arbitration context.

    Args:
        ctx: The context to activate.

    Returns:
        Token to reset the context later.
    """
    return _current_context.set(ctx)


def reset_arbitration_context(token: Token) -> None:
    """
    Reset the arbitration context to a previous state.

    Args:
        token: Token returned by set_arbitration_context.
    """
    _current_context.reset(token)
