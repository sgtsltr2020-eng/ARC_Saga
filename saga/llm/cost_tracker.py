"""
Cost Tracking - Budget Enforcement

Tracks LLM costs per task, agent, and session.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3B - Agent Execution Framework
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TaskCost:
    """Cost record for a single task."""
    task_id: str
    agent_name: str
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class CostTracker:
    """
    Tracks costs across tasks and enforces budget limits.

    Usage:
        tracker = CostTracker()
        tracker.record_task_cost("task-1", 0.05, "CodingAgent")
        total = tracker.get_total_cost()
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.task_costs: list[TaskCost] = []
        self.budget_limit: float = 100.0  # Default $100
        logger.info("CostTracker initialized", extra={"budget": self.budget_limit})

    def record_task_cost(
        self,
        task_id: str,
        cost: float,
        agent_name: str = "unknown"
    ) -> None:
        """
        Record cost for a task.

        Args:
            task_id: Task identifier
            cost: Cost in USD
            agent_name: Agent that executed task
        """
        record = TaskCost(
            task_id=task_id,
            agent_name=agent_name,
            cost_usd=cost
        )
        self.task_costs.append(record)

        logger.info(
            "Cost recorded",
            extra={
                "task_id": task_id,
                "cost": cost,
                "agent": agent_name,
                "total": self.get_total_cost()
            }
        )

    def get_task_cost(self, task_id: str) -> float:
        """Get total cost for specific task."""
        return sum(
            record.cost_usd
            for record in self.task_costs
            if record.task_id == task_id
        )

    def get_total_cost(self) -> float:
        """Get total cost across all tasks."""
        return sum(record.cost_usd for record in self.task_costs)

    def check_budget(self) -> tuple[bool, float]:
        """
        Check if budget exceeded.

        Returns:
            (within_budget, remaining)
        """
        total = self.get_total_cost()
        remaining = self.budget_limit - total
        within_budget = remaining >= 0

        if not within_budget:
            logger.warning(
                "Budget exceeded",
                extra={"total": total, "limit": self.budget_limit}
            )

        return within_budget, remaining

    def reset(self) -> None:
        """Reset all tracked costs."""
        self.task_costs.clear()
        logger.info("Cost tracker reset")
