"""
Metrics Hooks for Orchestration Optimization.

Defines immutable event structures for telemetry and simple aggregation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
from typing import Iterable, Union

from .core import WorkflowPattern
from .judgement import VerdictStatus


@dataclass(frozen=True)
class MetricsEvent:
    """
    Immutable recording of a unit of work's cost and outcome.

    Attributes:
        latency_ms: Execution duration in milliseconds.
        tokens_estimated: Pre-flight token estimate.
        tokens_actual: Actual tokens consumed.
        cost_usd: Calculated cost in USD.
        outcome: VerdictStatus or similar outcome identifier.
        agent_type: Classification of agent (e.g., 'judge', 'reviewer').
        task_type: Classification of task (e.g., 'plan', 'code').
        pattern: Workflow pattern used.
        timestamp: Creation time (UTC).
    """

    latency_ms: int
    tokens_estimated: int
    tokens_actual: int
    cost_usd: float
    outcome: Union[VerdictStatus, str]
    agent_type: str
    task_type: str
    pattern: Union[WorkflowPattern, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate metric fields."""
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.tokens_estimated < 0:
            raise ValueError("tokens_estimated cannot be negative")
        if self.tokens_actual < 0:
            raise ValueError("tokens_actual cannot be negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")


class MetricsAggregator:
    """Simple in-memory aggregator for SAGA metrics."""

    def __init__(self, events: Iterable[MetricsEvent]) -> None:
        """Initialize with a collection of events."""
        self._events = tuple(events)  # Snapshot

    def calculate_p95_latency(self, pattern: Union[WorkflowPattern, str]) -> int:
        """
        Calculate 95th percentile latency for a specific pattern.

        Args:
            pattern: The workflow pattern to filter by.

        Returns:
            p95 latency in ms.

        Raises:
            ValueError: If no events match the pattern.
        """
        # Filter
        matches = [e.latency_ms for e in self._events if str(e.pattern) == str(pattern)]
        if not matches:
             raise ValueError(f"No events found for pattern: {pattern}")
        
        # Sort and pick
        matches.sort()
        count = len(matches)
        index = ceil(0.95 * count) - 1
        return matches[index]

    def calculate_rejection_rate(self, pattern: Union[WorkflowPattern, str]) -> float:
        """
        Calculate rejection rate (REJECT/BUDGET_EXHAUSTED) for a pattern.

        Args:
             pattern: The workflow pattern to filter by.

        Returns:
            Fraction between 0.0 and 1.0.

        Raises:
             ValueError: If no events match the pattern.
        """
        matches = [e for e in self._events if str(e.pattern) == str(pattern)]
        if not matches:
             raise ValueError(f"No events found for pattern: {pattern}")
        
        # Count rejections
        # We consider REJECT and BUDGET_EXHAUSTED as "Rejections"
        rejected_count = 0
        for e in matches:
             outcome_str = str(e.outcome).lower()
             # string matching against VerdictStatus values or enum members
             if "reject" in outcome_str or "budget_exhausted" in outcome_str:
                 rejected_count += 1
        
        return float(rejected_count) / len(matches)
