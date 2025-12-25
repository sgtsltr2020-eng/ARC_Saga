"""
Governor - Turn-Control & Agent Efficiency Management
======================================================

The Governor is the "nervous system" that monitors and controls
agent execution efficiency. It implements:

1. Turn Tracking: Monitors turns per task (max 5 before MASS trigger)
2. Information-Gain Decay: Detects stagnant progress
3. Burn Rate Tracking: tokens/time for cost optimization
4. MASS Trigger: Multi-Agent System Search when efficiency drops

Design Principles:
- Every "turn" is a compute expense - track aggressively
- If 3 consecutive turns don't improve compliance, trigger dropout
- Log specific MASS trigger reasons for divergent model selection

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class MASSTriggeredReason(str, Enum):
    """Reasons for triggering Multi-Agent System Search."""
    TURN_LIMIT_EXCEEDED = "TURN_LIMIT_EXCEEDED"
    STAGNANT_GAIN = "STAGNANT_GAIN"
    UTILITY_BELOW_THRESHOLD = "UTILITY_BELOW_THRESHOLD"
    COMPLIANCE_REGRESSION = "COMPLIANCE_REGRESSION"
    USER_ESCALATION = "USER_ESCALATION"
    SIMULATION_FAILURE = "SIMULATION_FAILURE"  # Phase 8 Stage D: Docker trial failure


class SimulationFailureType(str, Enum):
    """Types of simulation failure for graduated penalties."""
    COMPLIANCE = "COMPLIANCE"  # FQL validation failed
    STATIC = "STATIC"  # Ruff/linting failed
    DYNAMIC = "DYNAMIC"  # Pytest/runtime failed


class AgentMode(str, Enum):
    """Agent operational modes for swarm topology."""
    SOLO = "SOLO"  # Single agent handling task
    SWARM = "SWARM"  # Multiple specialists active


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class TurnMetrics:
    """
    Metrics for a single turn in agent execution.

    A "turn" is a compute expense - each interaction between
    agents or between agent and Mimiry counts as a turn.

    Attributes:
        turn_id: Unique identifier for this turn
        task_id: Parent task identifier
        agent_id: Agent that executed this turn
        start_time: When the turn began
        end_time: When the turn completed
        tokens_used: Total tokens consumed
        compliance_before: Compliance score before this turn
        compliance_after: Compliance score after this turn
        compliance_delta: Change in compliance (can be negative)
    """
    turn_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    agent_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    tokens_used: int = 0
    compliance_before: float = 0.0
    compliance_after: float = 0.0
    compliance_delta: float = 0.0

    def complete(self, tokens: int, compliance_after: float) -> None:
        """Mark the turn as complete with final metrics."""
        self.end_time = datetime.utcnow()
        self.tokens_used = tokens
        self.compliance_after = compliance_after
        self.compliance_delta = compliance_after - self.compliance_before

    @property
    def duration_seconds(self) -> float:
        """Calculate turn duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def is_productive(self) -> bool:
        """Check if turn improved compliance."""
        return self.compliance_delta > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tokens_used": self.tokens_used,
            "compliance_before": self.compliance_before,
            "compliance_after": self.compliance_after,
            "compliance_delta": self.compliance_delta,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AgentUtilityScore:
    """
    Utility score for an agent in the swarm.

    Used for dropout decisions - agents with utility below
    threshold (0.15) are candidates for pruning.

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type of agent (Coder, Reviewer, etc.)
        success_rate: Ratio of successful turns (0.0-1.0)
        contribution_to_fql: FQL success contribution
        tokens_per_success: Efficiency metric (lower is better)
        total_turns: Number of turns executed
        productive_turns: Number of turns with positive compliance delta
    """
    agent_id: str
    agent_type: str = "generic"
    success_rate: float = 0.0
    contribution_to_fql: float = 0.0
    tokens_per_success: float = 0.0
    total_turns: int = 0
    productive_turns: int = 0
    last_active: datetime = field(default_factory=datetime.utcnow)

    @property
    def utility(self) -> float:
        """
        Calculate overall utility score.

        Weighted combination of success rate and FQL contribution.
        """
        # Weight: 60% success rate, 40% FQL contribution
        return (self.success_rate * 0.6) + (self.contribution_to_fql * 0.4)

    def update_from_turn(self, turn: TurnMetrics) -> None:
        """Update utility based on a completed turn."""
        self.total_turns += 1
        self.last_active = datetime.utcnow()

        if turn.is_productive:
            self.productive_turns += 1

        # Recalculate success rate
        self.success_rate = self.productive_turns / self.total_turns if self.total_turns > 0 else 0.0

        # Update tokens per success
        if self.productive_turns > 0:
            # This is a running average approximation
            self.tokens_per_success = (
                (self.tokens_per_success * (self.productive_turns - 1) + turn.tokens_used)
                / self.productive_turns
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "success_rate": self.success_rate,
            "contribution_to_fql": self.contribution_to_fql,
            "tokens_per_success": self.tokens_per_success,
            "utility": self.utility,
            "total_turns": self.total_turns,
            "productive_turns": self.productive_turns,
        }


@dataclass
class TaskTurnRecord:
    """
    Aggregate record of turns for a specific task.

    Tracks all turns and enables MASS trigger decisions.
    """
    task_id: str
    turns: list[TurnMetrics] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    mass_triggered: bool = False
    mass_trigger_reason: MASSTriggeredReason | None = None

    @property
    def turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used across all turns."""
        return sum(t.tokens_used for t in self.turns)

    @property
    def total_duration_seconds(self) -> float:
        """Get total duration of all turns."""
        return sum(t.duration_seconds for t in self.turns)

    @property
    def final_compliance(self) -> float:
        """Get compliance score from last turn."""
        if not self.turns:
            return 0.0
        return self.turns[-1].compliance_after

    def get_recent_deltas(self, count: int = 3) -> list[float]:
        """Get the most recent compliance deltas."""
        return [t.compliance_delta for t in self.turns[-count:]]


# ============================================================
# GOVERNOR CLASS
# ============================================================

class Governor:
    """
    Turn-Control Governor for agent orchestration.

    The Governor monitors agent efficiency and triggers MASS
    (Multi-Agent System Search) when efficiency drops below
    acceptable thresholds.

    Key Features:
    - Max 5 turns per task before MASS trigger
    - Information-Gain Decay detection (3 stagnant turns)
    - Burn rate tracking (tokens/time)
    - Specific MASS trigger reason logging

    Usage:
        ```python
        governor = Governor()

        # Track a turn
        metrics = governor.start_turn("task-123", "agent-alpha", 50.0)
        # ... agent does work ...
        governor.complete_turn(metrics, tokens=1500, compliance_after=75.0)

        # Check if MASS should be triggered
        should_mass, reason = governor.should_trigger_mass("task-123")
        if should_mass:
            # Reconfigure swarm with divergent reasoning
            pass
        ```
    """

    # Thresholds (Advisory: These are tuned for Minisforum local hardware)
    MAX_TURNS_PER_TASK = 5
    UTILITY_DROPOUT_THRESHOLD = 0.15
    STAGNANT_TURN_THRESHOLD = 3  # Consecutive non-improving turns
    MIN_COMPLIANCE_GAIN = 0.5  # Minimum gain to be considered "progress"

    def __init__(self) -> None:
        """Initialize the Governor."""
        self._task_records: dict[str, TaskTurnRecord] = {}
        self._agent_scores: dict[str, AgentUtilityScore] = {}
        self._current_mode = AgentMode.SOLO

        logger.info(
            "Governor initialized",
            extra={
                "max_turns": self.MAX_TURNS_PER_TASK,
                "dropout_threshold": self.UTILITY_DROPOUT_THRESHOLD,
                "stagnant_threshold": self.STAGNANT_TURN_THRESHOLD,
            }
        )

    def start_turn(
        self,
        task_id: str,
        agent_id: str,
        compliance_before: float
    ) -> TurnMetrics:
        """
        Start tracking a new turn.

        Args:
            task_id: Task being worked on
            agent_id: Agent executing the turn
            compliance_before: Compliance score at turn start

        Returns:
            TurnMetrics object to track this turn
        """
        # Ensure task record exists
        if task_id not in self._task_records:
            self._task_records[task_id] = TaskTurnRecord(task_id=task_id)

        # Ensure agent score exists
        if agent_id not in self._agent_scores:
            self._agent_scores[agent_id] = AgentUtilityScore(agent_id=agent_id)

        turn = TurnMetrics(
            task_id=task_id,
            agent_id=agent_id,
            compliance_before=compliance_before
        )

        logger.debug(
            "Turn started",
            extra={
                "turn_id": turn.turn_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "compliance_before": compliance_before,
            }
        )

        return turn

    def complete_turn(
        self,
        turn: TurnMetrics,
        tokens: int,
        compliance_after: float
    ) -> TurnMetrics:
        """
        Complete a turn and update all metrics.

        Args:
            turn: The TurnMetrics object from start_turn
            tokens: Tokens consumed in this turn
            compliance_after: Compliance score after the turn

        Returns:
            Updated TurnMetrics
        """
        turn.complete(tokens, compliance_after)

        # Update task record
        task_record = self._task_records.get(turn.task_id)
        if task_record:
            task_record.turns.append(turn)

        # Update agent score
        agent_score = self._agent_scores.get(turn.agent_id)
        if agent_score:
            agent_score.update_from_turn(turn)

        logger.info(
            "Turn completed",
            extra={
                "turn_id": turn.turn_id,
                "task_id": turn.task_id,
                "agent_id": turn.agent_id,
                "tokens_used": tokens,
                "compliance_delta": turn.compliance_delta,
                "duration_seconds": turn.duration_seconds,
                "is_productive": turn.is_productive,
            }
        )

        return turn

    def track_turn(
        self,
        task_id: str,
        agent_id: str,
        tokens: int,
        compliance_before: float,
        compliance_after: float
    ) -> TurnMetrics:
        """
        Convenience method to track a complete turn in one call.

        Args:
            task_id: Task being worked on
            agent_id: Agent executing the turn
            tokens: Tokens consumed
            compliance_before: Compliance score at turn start
            compliance_after: Compliance score after turn

        Returns:
            Completed TurnMetrics
        """
        turn = self.start_turn(task_id, agent_id, compliance_before)
        return self.complete_turn(turn, tokens, compliance_after)

    def evaluate_efficiency(self, agent_id: str) -> float:
        """
        Evaluate an agent's efficiency.

        Returns utility score (0.0-1.0). Agents below UTILITY_DROPOUT_THRESHOLD
        are candidates for pruning.

        Args:
            agent_id: Agent to evaluate

        Returns:
            Utility score (0.0-1.0)
        """
        agent_score = self._agent_scores.get(agent_id)
        if agent_score is None:
            return 0.0
        return agent_score.utility

    def should_trigger_mass(self, task_id: str) -> tuple[bool, MASSTriggeredReason | None]:
        """
        Check if MASS (Multi-Agent System Search) should be triggered.

        MASS triggers under these conditions:
        1. Task exceeds MAX_TURNS_PER_TASK (5 turns)
        2. STAGNANT_TURN_THRESHOLD (3) consecutive non-improving turns
        3. Compliance regressed from initial state

        Args:
            task_id: Task to check

        Returns:
            (should_trigger, reason) tuple
        """
        task_record = self._task_records.get(task_id)
        if task_record is None:
            return False, None

        # Already triggered?
        if task_record.mass_triggered:
            return False, None

        # Check turn limit
        if task_record.turn_count >= self.MAX_TURNS_PER_TASK:
            task_record.mass_triggered = True
            task_record.mass_trigger_reason = MASSTriggeredReason.TURN_LIMIT_EXCEEDED
            logger.warning(
                "MASS triggered: Turn limit exceeded",
                extra={
                    "task_id": task_id,
                    "turn_count": task_record.turn_count,
                    "reason": "TURN_LIMIT_EXCEEDED",
                    "recommendation": "Use divergent reasoning model for next swarm",
                }
            )
            return True, MASSTriggeredReason.TURN_LIMIT_EXCEEDED

        # Check for stagnant progress (Information-Gain Decay)
        recent_deltas = task_record.get_recent_deltas(self.STAGNANT_TURN_THRESHOLD)
        if len(recent_deltas) >= self.STAGNANT_TURN_THRESHOLD:
            if all(delta < self.MIN_COMPLIANCE_GAIN for delta in recent_deltas):
                task_record.mass_triggered = True
                task_record.mass_trigger_reason = MASSTriggeredReason.STAGNANT_GAIN
                logger.warning(
                    "MASS triggered: Stagnant gain detected",
                    extra={
                        "task_id": task_id,
                        "recent_deltas": recent_deltas,
                        "reason": "STAGNANT_GAIN",
                        "recommendation": "Use divergent reasoning model for architectural reconsideration",
                    }
                )
                return True, MASSTriggeredReason.STAGNANT_GAIN

        # Check for regression
        if task_record.turns:
            initial_compliance = task_record.turns[0].compliance_before
            current_compliance = task_record.final_compliance
            if current_compliance < initial_compliance and task_record.turn_count >= 2:
                task_record.mass_triggered = True
                task_record.mass_trigger_reason = MASSTriggeredReason.COMPLIANCE_REGRESSION
                logger.warning(
                    "MASS triggered: Compliance regression",
                    extra={
                        "task_id": task_id,
                        "initial_compliance": initial_compliance,
                        "current_compliance": current_compliance,
                        "reason": "COMPLIANCE_REGRESSION",
                        "recommendation": "Reset to initial state and use fresh approach",
                    }
                )
                return True, MASSTriggeredReason.COMPLIANCE_REGRESSION

        return False, None

    def get_burn_rate(self, task_id: str) -> dict[str, float]:
        """
        Calculate burn rate metrics for cost optimization.

        Burn rate = tokens/time, useful for $0/local hardware constraints.

        Args:
            task_id: Task to calculate burn rate for

        Returns:
            Dict with burn rate metrics
        """
        task_record = self._task_records.get(task_id)
        if task_record is None or not task_record.turns:
            return {
                "tokens_per_second": 0.0,
                "tokens_per_turn": 0.0,
                "seconds_per_turn": 0.0,
                "total_tokens": 0,
                "total_seconds": 0.0,
            }

        total_tokens = task_record.total_tokens
        total_seconds = max(task_record.total_duration_seconds, 0.001)  # Avoid div by zero
        turn_count = max(task_record.turn_count, 1)

        return {
            "tokens_per_second": total_tokens / total_seconds,
            "tokens_per_turn": total_tokens / turn_count,
            "seconds_per_turn": total_seconds / turn_count,
            "total_tokens": total_tokens,
            "total_seconds": total_seconds,
        }

    def get_agent_utility(self, agent_id: str) -> AgentUtilityScore | None:
        """Get utility score for an agent."""
        return self._agent_scores.get(agent_id)

    def get_low_utility_agents(self) -> list[AgentUtilityScore]:
        """Get all agents below the dropout threshold."""
        return [
            score for score in self._agent_scores.values()
            if score.utility < self.UTILITY_DROPOUT_THRESHOLD and score.total_turns > 0
        ]

    def get_task_summary(self, task_id: str) -> dict[str, Any]:
        """Get a summary of task execution metrics."""
        task_record = self._task_records.get(task_id)
        if task_record is None:
            return {"error": "Task not found"}

        return {
            "task_id": task_id,
            "turn_count": task_record.turn_count,
            "total_tokens": task_record.total_tokens,
            "total_duration_seconds": task_record.total_duration_seconds,
            "final_compliance": task_record.final_compliance,
            "mass_triggered": task_record.mass_triggered,
            "mass_trigger_reason": task_record.mass_trigger_reason.value if task_record.mass_trigger_reason else None,
            "burn_rate": self.get_burn_rate(task_id),
        }

    def record_simulation_failure(
        self,
        agent_id: str,
        failure_type: SimulationFailureType,
        spark_id: str | None = None
    ) -> tuple[bool, MASSTriggeredReason | None]:
        """
        Apply utility penalty for simulation failure.

        Graduated penalties based on failure type:
        - COMPLIANCE (FQL): Minor penalty (0.1) - "bad protocol"
        - STATIC (Ruff): Moderate penalty (0.2) - "bad grammar"
        - DYNAMIC (Pytest): Major penalty (0.3) - "bad logic"

        Args:
            agent_id: Agent responsible for the failed synthesis
            failure_type: Type of failure for penalty calculation
            spark_id: Optional Spark ID for provenance tracking

        Returns:
            (should_swap, reason) tuple - True if agent should be swapped
        """
        # Graduated penalties
        penalties = {
            SimulationFailureType.COMPLIANCE: 0.1,
            SimulationFailureType.STATIC: 0.2,
            SimulationFailureType.DYNAMIC: 0.3,
        }
        penalty = penalties.get(failure_type, 0.2)

        agent_score = self._agent_scores.get(agent_id)
        if agent_score is None:
            # Create score if doesn't exist
            agent_score = AgentUtilityScore(agent_id=agent_id)
            self._agent_scores[agent_id] = agent_score

        # Apply penalty to success rate
        agent_score.success_rate = max(0.0, agent_score.success_rate - penalty)

        logger.warning(
            "Simulation failure recorded",
            extra={
                "agent_id": agent_id,
                "failure_type": failure_type.value,
                "penalty": penalty,
                "new_success_rate": agent_score.success_rate,
                "utility": agent_score.utility,
                "spark_id": spark_id,
            }
        )

        # Check if agent should be swapped (below dropout threshold)
        if agent_score.utility < self.UTILITY_DROPOUT_THRESHOLD:
            logger.warning(
                "MASS triggered: Simulation failure pushed agent below threshold",
                extra={
                    "agent_id": agent_id,
                    "utility": agent_score.utility,
                    "threshold": self.UTILITY_DROPOUT_THRESHOLD,
                    "reason": "SIMULATION_FAILURE",
                    "recommendation": "Swap to REFACTORING_CHAMELEON for fresh approach",
                }
            )
            return True, MASSTriggeredReason.SIMULATION_FAILURE

        return False, None

    def reset_task(self, task_id: str) -> None:
        """Reset tracking for a task (for retries)."""
        if task_id in self._task_records:
            del self._task_records[task_id]
            logger.info(f"Task record reset: {task_id}")

    @property
    def current_mode(self) -> AgentMode:
        """Get current agent operational mode."""
        return self._current_mode

    def set_mode(self, mode: AgentMode) -> None:
        """Set agent operational mode."""
        if self._current_mode != mode:
            logger.info(
                "Agent mode changed",
                extra={
                    "from_mode": self._current_mode.value,
                    "to_mode": mode.value,
                }
            )
            self._current_mode = mode

    def get_stats(self) -> dict[str, Any]:
        """Get overall Governor statistics."""
        return {
            "total_tasks_tracked": len(self._task_records),
            "total_agents_tracked": len(self._agent_scores),
            "current_mode": self._current_mode.value,
            "low_utility_agent_count": len(self.get_low_utility_agents()),
            "mass_triggered_tasks": sum(
                1 for r in self._task_records.values() if r.mass_triggered
            ),
        }


# Export
__all__ = [
    "Governor",
    "TurnMetrics",
    "AgentUtilityScore",
    "TaskTurnRecord",
    "MASSTriggeredReason",
    "AgentMode",
]
