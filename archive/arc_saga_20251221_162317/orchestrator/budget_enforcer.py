"""
Hard Budget Ceilings and Enforcement.

Provides a precise decision engine for budget compliance, surfacing distinct outcomes
for Soft Cap warnings and Hard Cap termination.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from ..error_instrumentation import log_with_context


class BudgetDecision(str, Enum):
    """Explicit decision outcomes from budget checks."""
    OK = "ok"
    SOFT_CAP_WARNING = "soft_cap_warning"
    HARD_CAP_EXCEEDED = "hard_cap_exceeded"

# Protocol for TokenBudget compatibility (avoiding tight coupling if actual class moves)
@runtime_checkable
class TokenBudgetLike(Protocol):
    remaining: int
    total: int

class BudgetEnforcer:
    """
    Enforces budget constraints with a multi-layer safety net.
    """

    def __init__(self, soft_cap_ratio: float = 0.8) -> None:
        """
        Initialize enforcer.

        Args:
           soft_cap_ratio: Percentage of budget usage that triggers a warning (0.0 - 1.0).
        """
        self.soft_cap_ratio = soft_cap_ratio

    def preflight_check(self, budget: TokenBudgetLike, estimated_next: int) -> BudgetDecision:
        """
        Check if an estimated operation would breach the budget.

        Args:
            budget: Current budget state.
            estimated_next: Expected token cost of the next operation.

        Returns:
            BudgetDecision (OK, Warning, or Hard Cap Exceeded).
        """
        # Hard Cap Check: Can we afford this?
        if estimated_next > budget.remaining:
            log_with_context(
                "warning", 
                "budget_preflight_hard_cap", 
                estimated=estimated_next, 
                remaining=budget.remaining
            )
            return BudgetDecision.HARD_CAP_EXCEEDED

        # Soft Cap Check: Are we getting low?
        # Threshold: Remaining drops below (1 - ratio) * total
        # e.g. ratio 0.8 -> warn when remaining < 0.2 * total
        # Or simpler: Usage > 80% means remaining < 20%
        
        warn_threshold = budget.total * (1.0 - self.soft_cap_ratio)
        remaining_after = budget.remaining - estimated_next
        
        if remaining_after <= warn_threshold:
            # Only warn if we aren't already dead (handled by Hard Cap)
            # and if we are actually crossing into danger zone
            log_with_context(
                "warning", 
                "budget_soft_cap_warning", 
                remaining_after=remaining_after, 
                threshold=warn_threshold
            )
            return BudgetDecision.SOFT_CAP_WARNING
        
        return BudgetDecision.OK

    def runtime_check(self, budget: TokenBudgetLike) -> BudgetDecision:
        """
        Check budget state after an operation.

        Args:
             budget: Current budget state (already updated with actuals).

        Returns:
             BudgetDecision (OK, Warning, or Hard Cap Exceeded).
        """
        if budget.remaining <= 0:
             log_with_context("error", "budget_hard_cap_breach", remaining=budget.remaining)
             return BudgetDecision.HARD_CAP_EXCEEDED
        
        warn_threshold = budget.total * (1.0 - self.soft_cap_ratio)
        if budget.remaining <= warn_threshold:
             # Just a warning context, simple check
             return BudgetDecision.SOFT_CAP_WARNING
        
        return BudgetDecision.OK
