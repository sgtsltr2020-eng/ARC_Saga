"""
Swarm - Agent Dropout & Coordination
=====================================

The Swarm module manages the active agent pool and implements
RL-based pruning logic for underperforming agents.

Design Principles:
- Start "Lean" (1 agent) - only spawn specialists when needed
- Agents with utility < 0.15 are pruned
- "Refactoring Chameleon" approaches problems from architectural angle

Key Components:
- AgentDropout: RL-based pruning system
- SwarmCoordinator: Manages active agent pool

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

from saga.core.mae.governor import AgentMode, Governor, TurnMetrics

logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class AgentType(str, Enum):
    """Types of agents in the swarm."""
    CODER = "CODER"
    REVIEWER = "REVIEWER"
    TESTING = "TESTING"
    DOCS = "DOCS"
    SECURITY = "SECURITY"
    REFACTORING_CHAMELEON = "REFACTORING_CHAMELEON"
    ASYNC_SPECIALIST = "ASYNC_SPECIALIST"


class DropoutReason(str, Enum):
    """Reasons for agent dropout."""
    LOW_UTILITY = "LOW_UTILITY"
    REDUNDANT = "REDUNDANT"
    SUBSTANDARD_OUTPUT = "SUBSTANDARD_OUTPUT"
    TASK_COMPLETE = "TASK_COMPLETE"
    USER_REMOVED = "USER_REMOVED"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class SwarmAgent:
    """
    Represents an agent in the swarm.

    Tracks state, specialization, and activity metrics.
    """
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    agent_type: AgentType = AgentType.CODER
    is_active: bool = True
    spawned_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    tasks_completed: int = 0
    dropout_reason: DropoutReason | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "spawned_at": self.spawned_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "tasks_completed": self.tasks_completed,
            "dropout_reason": self.dropout_reason.value if self.dropout_reason else None,
        }


@dataclass
class DropoutDecision:
    """
    Record of a dropout decision for an agent.
    """
    agent_id: str
    agent_type: AgentType
    reason: DropoutReason
    utility_at_dropout: float
    decided_at: datetime = field(default_factory=datetime.utcnow)
    replacement_spawned: bool = False
    replacement_type: AgentType | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "reason": self.reason.value,
            "utility_at_dropout": self.utility_at_dropout,
            "decided_at": self.decided_at.isoformat(),
            "replacement_spawned": self.replacement_spawned,
            "replacement_type": self.replacement_type.value if self.replacement_type else None,
        }


# ============================================================
# AGENT DROPOUT CLASS
# ============================================================

class AgentDropout:
    """
    RL-based agent pruning system.

    Monitors agent utility scores and prunes agents that
    fall below the threshold. Uses a simple RL approach
    based on cumulative success/failure signals.

    Key Features:
    - Dropout threshold: 0.15
    - Redundancy detection (multiple agents doing same work)
    - Chameleon spawning for architectural approaches

    Usage:
        ```python
        dropout = AgentDropout(governor)

        # Check if agent should be dropped
        should_drop, reason = dropout.should_dropout("agent-123")
        if should_drop:
            dropout.prune_agent("agent-123", reason)

            # Spawn replacement with different approach
            new_agent = dropout.spawn_chameleon({"task": "refactor auth"})
        ```
    """

    DROPOUT_THRESHOLD = 0.15
    REDUNDANCY_SIMILARITY_THRESHOLD = 0.85

    def __init__(self, governor: Governor) -> None:
        """Initialize the AgentDropout system."""
        self._governor = governor
        self._dropout_history: list[DropoutDecision] = []
        self._agent_type_registry: dict[str, AgentType] = {}

    def calculate_utility(
        self,
        agent_id: str,
        metrics: list[TurnMetrics] | None = None
    ) -> float:
        """
        Calculate utility score for an agent.

        Uses Governor's tracked metrics if available, otherwise
        calculates from provided metrics.

        Args:
            agent_id: Agent to calculate utility for
            metrics: Optional list of turn metrics

        Returns:
            Utility score (0.0-1.0)
        """
        # First try Governor's tracked score
        utility_score = self._governor.get_agent_utility(agent_id)
        if utility_score is not None:
            return utility_score.utility

        # Calculate from provided metrics
        if not metrics:
            return 0.0

        productive_count = sum(1 for m in metrics if m.is_productive)
        total_count = len(metrics)

        if total_count == 0:
            return 0.0

        return productive_count / total_count

    def should_dropout(self, agent_id: str) -> tuple[bool, DropoutReason | None]:
        """
        Determine if an agent should be dropped from the swarm.

        Args:
            agent_id: Agent to evaluate

        Returns:
            (should_drop, reason) tuple
        """
        utility = self.calculate_utility(agent_id)

        # Check utility threshold
        if utility < self.DROPOUT_THRESHOLD:
            logger.info(
                "Agent below dropout threshold",
                extra={
                    "agent_id": agent_id,
                    "utility": utility,
                    "threshold": self.DROPOUT_THRESHOLD,
                }
            )
            return True, DropoutReason.LOW_UTILITY

        return False, None

    def detect_redundancy(
        self,
        agent_id: str,
        other_agents: list[str]
    ) -> tuple[bool, str | None]:
        """
        Detect if an agent is redundant (doing same work as another).

        Redundancy is detected by comparing output patterns and
        compliance improvements across agents.

        Args:
            agent_id: Agent to check
            other_agents: List of other active agent IDs

        Returns:
            (is_redundant, redundant_with_agent_id)
        """
        agent_score = self._governor.get_agent_utility(agent_id)
        if agent_score is None:
            return False, None

        for other_id in other_agents:
            if other_id == agent_id:
                continue

            other_score = self._governor.get_agent_utility(other_id)
            if other_score is None:
                continue

            # Check if scores are very similar (redundant work)
            if agent_score.total_turns > 0 and other_score.total_turns > 0:
                score_diff = abs(agent_score.success_rate - other_score.success_rate)
                if score_diff < 0.1:  # Very similar performance
                    # Keep the one with higher utility
                    if agent_score.utility < other_score.utility:
                        logger.info(
                            "Redundant agent detected",
                            extra={
                                "agent_id": agent_id,
                                "redundant_with": other_id,
                                "agent_utility": agent_score.utility,
                                "other_utility": other_score.utility,
                            }
                        )
                        return True, other_id

        return False, None

    def prune_agent(
        self,
        agent_id: str,
        reason: DropoutReason = DropoutReason.LOW_UTILITY
    ) -> DropoutDecision:
        """
        Prune an agent from the swarm.

        Args:
            agent_id: Agent to prune
            reason: Why the agent is being pruned

        Returns:
            DropoutDecision record
        """
        utility = self.calculate_utility(agent_id)
        agent_type = self._agent_type_registry.get(agent_id, AgentType.CODER)

        decision = DropoutDecision(
            agent_id=agent_id,
            agent_type=agent_type,
            reason=reason,
            utility_at_dropout=utility,
        )

        self._dropout_history.append(decision)

        logger.warning(
            "Agent pruned from swarm",
            extra={
                "agent_id": agent_id,
                "agent_type": agent_type.value,
                "reason": reason.value,
                "utility": utility,
            }
        )

        return decision

    def spawn_chameleon(
        self,
        task_context: dict[str, Any]
    ) -> SwarmAgent:
        """
        Spawn a Refactoring Chameleon agent.

        The Chameleon approaches problems from an architectural angle,
        used when standard agents fail to make progress.

        Args:
            task_context: Context about the failing task

        Returns:
            New SwarmAgent of REFACTORING_CHAMELEON type
        """
        agent = SwarmAgent(
            agent_type=AgentType.REFACTORING_CHAMELEON,
        )

        self._agent_type_registry[agent.agent_id] = AgentType.REFACTORING_CHAMELEON

        logger.info(
            "Refactoring Chameleon spawned",
            extra={
                "agent_id": agent.agent_id,
                "task_context": task_context.get("task", "unknown"),
                "reason": "previous_agents_stagnant",
            }
        )

        return agent

    def spawn_specialist(
        self,
        agent_type: AgentType,
        task_context: dict[str, Any] | None = None
    ) -> SwarmAgent:
        """
        Spawn a specialist agent.

        Args:
            agent_type: Type of specialist to spawn
            task_context: Optional context about the task

        Returns:
            New SwarmAgent
        """
        agent = SwarmAgent(agent_type=agent_type)
        self._agent_type_registry[agent.agent_id] = agent_type

        logger.info(
            "Specialist agent spawned",
            extra={
                "agent_id": agent.agent_id,
                "agent_type": agent_type.value,
                "task": task_context.get("task", "unknown") if task_context else "unknown",
            }
        )

        return agent

    def get_dropout_history(self) -> list[DropoutDecision]:
        """Get history of dropout decisions."""
        return self._dropout_history.copy()

    def get_dropout_stats(self) -> dict[str, Any]:
        """Get dropout statistics."""
        by_reason: dict[str, int] = {}
        for decision in self._dropout_history:
            reason = decision.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1

        return {
            "total_dropouts": len(self._dropout_history),
            "by_reason": by_reason,
            "chameleons_spawned": sum(
                1 for d in self._dropout_history
                if d.replacement_type == AgentType.REFACTORING_CHAMELEON
            ),
        }


# ============================================================
# SWARM COORDINATOR CLASS
# ============================================================

class SwarmCoordinator:
    """
    Manages the active agent swarm.

    Starts "Lean" (1 agent) and only spawns specialists when
    Warden flags high complexity. This prevents the "too many
    cooks" problem that plagues most agentic frameworks.

    Key Features:
    - Solo → Swarm mode transition
    - Complexity-based specialist spawning
    - Agent routing based on task type
    - Swarm Heat indicator for UI

    Usage:
        ```python
        coordinator = SwarmCoordinator(governor)

        # Add initial agent
        primary = coordinator.add_agent("primary-coder", AgentType.CODER)

        # Check if more agents needed
        if coordinator.should_scale_up(complexity_score=0.8):
            reviewer = coordinator.spawn_for_complexity(0.8)

        # Route task to best agent
        best_agent = coordinator.route_to_best_agent(task)
        ```
    """

    COMPLEXITY_SCALE_THRESHOLD = 0.65
    MAX_SWARM_SIZE = 5

    def __init__(self, governor: Governor) -> None:
        """Initialize the SwarmCoordinator."""
        self._governor = governor
        self._agents: dict[str, SwarmAgent] = {}
        self._dropout = AgentDropout(governor)
        self._task_assignments: dict[str, str] = {}  # task_id -> agent_id

    def add_agent(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.CODER
    ) -> SwarmAgent:
        """
        Add an agent to the swarm.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent

        Returns:
            SwarmAgent instance
        """
        agent = SwarmAgent(
            agent_id=agent_id,
            agent_type=agent_type,
        )

        self._agents[agent_id] = agent
        self._dropout._agent_type_registry[agent_id] = agent_type

        # Update mode if swarm grows
        if len(self.get_active_agents()) > 1:
            self._governor.set_mode(AgentMode.SWARM)

        logger.info(
            "Agent added to swarm",
            extra={
                "agent_id": agent_id,
                "agent_type": agent_type.value,
                "swarm_size": len(self._agents),
            }
        )

        return agent

    def remove_agent(self, agent_id: str, reason: DropoutReason = DropoutReason.TASK_COMPLETE) -> bool:
        """
        Remove an agent from the swarm.

        Args:
            agent_id: Agent to remove
            reason: Why agent is being removed

        Returns:
            True if agent was removed
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        agent.is_active = False
        agent.dropout_reason = reason

        # Update mode if swarm shrinks to 1
        active = self.get_active_agents()
        if len(active) <= 1:
            self._governor.set_mode(AgentMode.SOLO)

        logger.info(
            "Agent removed from swarm",
            extra={
                "agent_id": agent_id,
                "reason": reason.value,
                "remaining_active": len(active),
            }
        )

        return True

    def get_active_agents(self) -> list[SwarmAgent]:
        """Get all active agents in the swarm."""
        return [a for a in self._agents.values() if a.is_active]

    def get_agent(self, agent_id: str) -> SwarmAgent | None:
        """Get a specific agent."""
        return self._agents.get(agent_id)

    def should_scale_up(self, complexity_score: float) -> bool:
        """
        Determine if swarm should scale up based on complexity.

        Args:
            complexity_score: Task complexity (0.0-1.0)

        Returns:
            True if more agents should be spawned
        """
        active_count = len(self.get_active_agents())

        if active_count >= self.MAX_SWARM_SIZE:
            return False

        return complexity_score >= self.COMPLEXITY_SCALE_THRESHOLD

    def spawn_for_complexity(
        self,
        complexity_score: float,
        task_context: dict[str, Any] | None = None
    ) -> SwarmAgent | None:
        """
        Spawn appropriate specialist based on complexity.

        Args:
            complexity_score: Task complexity (0.0-1.0)
            task_context: Context about the task

        Returns:
            New SwarmAgent or None if not needed
        """
        if not self.should_scale_up(complexity_score):
            return None

        # Determine type based on context
        agent_type = self._determine_specialist_type(task_context)

        agent = self._dropout.spawn_specialist(agent_type, task_context)
        self._agents[agent.agent_id] = agent

        # Update to SWARM mode
        self._governor.set_mode(AgentMode.SWARM)

        return agent

    def _determine_specialist_type(
        self,
        task_context: dict[str, Any] | None
    ) -> AgentType:
        """Determine which specialist type to spawn."""
        if task_context is None:
            return AgentType.REVIEWER

        task_desc = task_context.get("task", "").lower()

        if "security" in task_desc or "auth" in task_desc:
            return AgentType.SECURITY
        elif "test" in task_desc or "coverage" in task_desc:
            return AgentType.TESTING
        elif "async" in task_desc or "concurrent" in task_desc:
            return AgentType.ASYNC_SPECIALIST
        elif "doc" in task_desc or "readme" in task_desc:
            return AgentType.DOCS
        else:
            return AgentType.REVIEWER

    def route_to_best_agent(
        self,
        task_context: dict[str, Any]
    ) -> str | None:
        """
        Route a task to the best available agent.

        Uses agent utility and specialization to find optimal match.

        Args:
            task_context: Context about the task

        Returns:
            agent_id of best agent, or None if no agents available
        """
        active = self.get_active_agents()
        if not active:
            return None

        if len(active) == 1:
            return active[0].agent_id

        # Find best match by type and utility
        task_type = self._determine_specialist_type(task_context)

        # Prefer matching specialists
        specialists = [a for a in active if a.agent_type == task_type]
        if specialists:
            # Pick highest utility among specialists
            best = max(
                specialists,
                key=lambda a: self._dropout.calculate_utility(a.agent_id)
            )
            return best.agent_id

        # Fall back to highest utility overall
        best = max(
            active,
            key=lambda a: self._dropout.calculate_utility(a.agent_id)
        )
        return best.agent_id

    def prune_low_utility_agents(self) -> list[DropoutDecision]:
        """
        Prune all agents below utility threshold.

        Returns:
            List of dropout decisions made
        """
        decisions = []
        active = self.get_active_agents()

        for agent in active:
            should_drop, reason = self._dropout.should_dropout(agent.agent_id)
            if should_drop and reason:
                decision = self._dropout.prune_agent(agent.agent_id, reason)
                self.remove_agent(agent.agent_id, DropoutReason.LOW_UTILITY)
                decisions.append(decision)

        return decisions

    def get_swarm_heat(self) -> float:
        """
        Calculate "Swarm Heat" for UI indicator.

        Heat represents activity level and complexity handling.
        0.0 = cold (solo mode), 1.0 = max heat (all slots filled)

        Returns:
            Heat score (0.0-1.0)
        """
        active_count = len(self.get_active_agents())
        return min(1.0, active_count / self.MAX_SWARM_SIZE)

    def get_swarm_status(self) -> dict[str, Any]:
        """Get comprehensive swarm status."""
        active = self.get_active_agents()

        return {
            "mode": self._governor.current_mode.value,
            "active_agent_count": len(active),
            "max_swarm_size": self.MAX_SWARM_SIZE,
            "swarm_heat": self.get_swarm_heat(),
            "agents": [a.to_dict() for a in active],
            "agent_types": {t.value: sum(1 for a in active if a.agent_type == t) for t in AgentType},
            "dropout_stats": self._dropout.get_dropout_stats(),
        }


# Export
__all__ = [
    "AgentDropout",
    "SwarmCoordinator",
    "SwarmAgent",
    "AgentType",
    "DropoutReason",
    "DropoutDecision",
]
