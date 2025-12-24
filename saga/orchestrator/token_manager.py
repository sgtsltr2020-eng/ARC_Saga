"""
Token Budget Manager for Orchestrator.

Manages token estimation, tracking, and budget enforcement for AI operations.
Follows Phase 2 patterns from decision_catalog.md.

This module provides:
- Token usage estimation based on memory tiers
- Actual token tracking from provider responses
- Budget management with alert thresholds
- Event-driven usage history tracking

Example:
    >>> from saga.orchestrator.token_manager import (
    ...     TokenBudgetManager, LocalTokenEstimator, TokenBudget
    ... )
    >>> from saga.orchestrator.events import InMemoryEventStore
    >>> from saga.orchestrator.types import AITask, AITaskInput, AIProvider
    >>>
    >>> estimator = LocalTokenEstimator()
    >>> budget = TokenBudget(total=50000, remaining=50000)
    >>> event_store = InMemoryEventStore()
    >>> manager = TokenBudgetManager(estimator, budget, event_store)
    >>>
    >>> task = AITask(
    ...     operation="generate",
    ...     input_data=AITaskInput(
    ...         prompt="Hello world",
    ...         model="gpt-4",
    ...         provider=AIProvider.OPENAI,
    ...     ),
    ... )
    >>> allocations = await manager.allocate_tokens([task])
    >>> print(allocations)  # {'task-id': 5002}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..error_instrumentation import log_with_context
from .events import IEventStore, OrchestratorEvent
from .types import AIProvider, AIResult, AITask


@dataclass(frozen=True)
class TokenUsage(OrchestratorEvent):
    """
    Immutable record of token usage for an AI operation.

    Tracks both estimated and actual token consumption along with cost
    and provider information for budget analysis and optimization.

    Extends OrchestratorEvent for event store compatibility.
    Uses aggregate_id for request_id and created_at for timestamp.

    Attributes:
        aggregate_id: Request identifier (inherited from OrchestratorEvent)
        estimated: Estimated tokens before execution
        actual: Actual tokens consumed
        cost_usd: Cost in USD for this operation
        provider: AI provider used
        created_at: When the usage occurred (inherited from OrchestratorEvent)
        event_type: Event type identifier (default: "TokenUsage")

    Example:
        >>> usage = TokenUsage(
        ...     aggregate_id="req-123",
        ...     estimated=5000,
        ...     actual=4800,
        ...     cost_usd=0.096,
        ...     provider=AIProvider.OPENAI,
        ...     created_at=datetime.now(timezone.utc),
        ... )

    Raises:
        ValueError: If any numeric field is negative
    """

    event_type: str = field(default="TokenUsage")
    estimated: int = 0
    actual: int = 0
    cost_usd: float = 0.0
    provider: AIProvider = AIProvider.OPENAI

    def __post_init__(self) -> None:
        """Validate token usage data after initialization."""
        if self.estimated < 0:
            raise ValueError("TokenUsage estimated cannot be negative")
        if self.actual < 0:
            raise ValueError("TokenUsage actual cannot be negative")
        if self.cost_usd < 0.0:
            raise ValueError("TokenUsage cost_usd cannot be negative")

    @property
    def request_id(self) -> str:
        """Get request ID (alias for aggregate_id)."""
        return self.aggregate_id

    @property
    def timestamp(self) -> datetime:
        """Get timestamp (alias for created_at)."""
        return self.created_at


@dataclass(frozen=True)
class TokenBudget:
    """
    Immutable token budget configuration.

    Defines total budget, remaining tokens, and alert threshold
    for budget monitoring and enforcement.

    Attributes:
        total: Total token budget allocated
        remaining: Remaining tokens available
        alert_threshold: Fraction of total budget that triggers alerts (default: 0.2)

    Example:
        >>> budget = TokenBudget(
        ...     total=50000,
        ...     remaining=45000,
        ...     alert_threshold=0.2,
        ... )
        >>> print(f"Budget: {budget.remaining}/{budget.total}")

    Raises:
        ValueError: If total <= 0, remaining < 0, or alert_threshold out of range
    """

    total: int
    remaining: int
    alert_threshold: float = 0.2

    def __post_init__(self) -> None:
        """Validate budget configuration after initialization."""
        if self.total <= 0:
            raise ValueError("TokenBudget total must be positive")
        if self.remaining < 0:
            raise ValueError("TokenBudget remaining cannot be negative")
        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("TokenBudget alert_threshold must be between 0.0 and 1.0")


class ITokenEstimator(Protocol):
    """
    Protocol for token estimation implementations.

    Defines the contract for estimating token usage before execution.
    Implementations can use various strategies (heuristics, ML models, etc.).
    """

    async def estimate(self, prompt: str, memory_tier: str) -> int:
        """
        Estimate token usage for a prompt and memory tier.

        Args:
            prompt: Input prompt text
            memory_tier: Memory tier identifier (minimal, standard, enhanced, etc.)

        Returns:
            Estimated token count

        Raises:
            ValueError: If memory_tier is invalid
        """
        ...


class LocalTokenEstimator:
    """
    Local token estimator using memory tier costs and prompt length.

    Uses predefined base costs per memory tier plus prompt length estimation
    (4 characters ≈ 1 token heuristic).

    Attributes:
        MEMORY_TIER_COSTS: Base token costs per memory tier

    Example:
        >>> estimator = LocalTokenEstimator()
        >>> tokens = await estimator.estimate("Hello world", "standard")
        >>> print(tokens)  # 5002 (5000 base + 2 prompt tokens)
    """

    MEMORY_TIER_COSTS: dict[str, int] = {
        "minimal": 2000,
        "standard": 5000,
        "enhanced": 8000,
        "complete": 10000,
        "unlimited": 15000,
    }

    async def estimate(self, prompt: str, memory_tier: str) -> int:
        """
        Estimate tokens for prompt and memory tier.

        Args:
            prompt: Input prompt text
            memory_tier: Memory tier (minimal, standard, enhanced, complete, unlimited)

        Returns:
            Estimated token count (base cost + prompt tokens)

        Raises:
            ValueError: If memory_tier is not in MEMORY_TIER_COSTS

        Example:
            >>> estimator = LocalTokenEstimator()
            >>> tokens = await estimator.estimate("Test", "standard")
            >>> print(tokens)  # 5001
        """
        if memory_tier not in self.MEMORY_TIER_COSTS:
            raise ValueError(
                f"Invalid memory_tier: {memory_tier}. "
                f"Must be one of: {list(self.MEMORY_TIER_COSTS.keys())}"
            )

        base_cost = self.MEMORY_TIER_COSTS[memory_tier]
        prompt_tokens = len(prompt) // 4  # 4 chars ≈ 1 token heuristic
        estimated = base_cost + prompt_tokens

        log_with_context(
            "info",
            "token_estimation_completed",
            memory_tier=memory_tier,
            prompt_length=len(prompt),
            estimated_tokens=estimated,
            base_cost=base_cost,
            prompt_tokens=prompt_tokens,
        )

        return estimated


class ProviderTokenTracker:
    """
    Tracks actual token usage from AI provider responses.

    Placeholder implementation for Phase 2.1.
    Future: Parse actual usage from provider API headers.

    Example:
        >>> tracker = ProviderTokenTracker()
        >>> tokens = await tracker.track_actual(result)
        >>> print(tokens)  # 4800
    """

    async def track_actual(self, result: AIResult) -> int:
        """
        Track actual tokens used from AIResult.

        Args:
            result: AI task result containing token usage

        Returns:
            Actual tokens used

        Raises:
            ValueError: If result.output_data is None (failed task)

        Example:
            >>> tracker = ProviderTokenTracker()
            >>> tokens = await tracker.track_actual(successful_result)
            >>> print(tokens)  # 4800
        """
        if result.output_data is None:
            raise ValueError(
                f"Cannot track tokens for task {result.task_id}: "
                "output_data is None (task failed)"
            )

        return result.output_data.tokens_used


class TokenBudgetManager:
    """
    Manages token budget allocation, tracking, and enforcement.

    Coordinates token estimation, usage recording, and budget monitoring
    with event-driven history tracking.

    Attributes:
        estimator: Token estimator implementation
        _budget: Internal mutable budget state (TokenBudget is frozen)
        event_store: Event store for usage history

    Example:
        >>> estimator = LocalTokenEstimator()
        >>> budget = TokenBudget(total=50000, remaining=50000)
        >>> event_store = InMemoryEventStore()
        >>> manager = TokenBudgetManager(estimator, budget, event_store)
        >>>
        >>> allocations = await manager.allocate_tokens([task1, task2])
        >>> usage = TokenUsage(...)
        >>> await manager.record_usage(usage)
        >>> current_budget = await manager.check_budget()
    """

    def __init__(
        self,
        estimator: ITokenEstimator,
        budget: TokenBudget,
        event_store: IEventStore[TokenUsage],
    ) -> None:
        """
        Initialize token budget manager.

        Args:
            estimator: Token estimator implementation
            budget: Initial token budget
            event_store: Event store for usage history
        """
        self.estimator = estimator
        self._budget = budget
        self.event_store = event_store

        log_with_context(
            "info",
            "token_budget_manager_initialized",
            total_budget=budget.total,
            remaining_budget=budget.remaining,
            alert_threshold=budget.alert_threshold,
        )

    async def allocate_tokens(
        self, tasks: list[AITask], memory_tier: str = "standard"
    ) -> dict[str, int]:
        """
        Allocate token estimates for a list of tasks.

        Estimates token usage for each task based on prompt and memory tier,
        returning a mapping of task IDs to estimated tokens.

        Args:
            tasks: List of AI tasks to estimate
            memory_tier: Memory tier for estimation (default: "standard")

        Returns:
            Dictionary mapping task IDs to estimated token counts

        Example:
            >>> allocations = await manager.allocate_tokens([task1, task2])
            >>> print(allocations)  # {'task-1': 5002, 'task-2': 5005}
        """
        log_with_context(
            "info",
            "token_allocation_started",
            task_count=len(tasks),
            memory_tier=memory_tier,
        )

        allocations: dict[str, int] = {}

        for task in tasks:
            prompt = task.input_data.prompt
            estimated = await self.estimator.estimate(prompt, memory_tier)
            allocations[task.id] = estimated

        log_with_context(
            "info",
            "token_allocation_completed",
            task_count=len(tasks),
            total_estimated=sum(allocations.values()),
            allocations=allocations,
        )

        return allocations

    async def record_usage(self, usage: TokenUsage) -> None:
        """
        Record token usage and update budget.

        Appends usage to event store, updates remaining budget,
        and logs warnings if budget falls below threshold.

        Args:
            usage: Token usage record to record

        Raises:
            EventStoreError: If event store append fails (logged but not raised)

        Example:
            >>> usage = TokenUsage(
            ...     request_id="req-1",
            ...     estimated=5000,
            ...     actual=4800,
            ...     cost_usd=0.096,
            ...     provider=AIProvider.OPENAI,
            ...     timestamp=datetime.now(timezone.utc),
            ... )
            >>> await manager.record_usage(usage)
        """
        log_with_context(
            "info",
            "token_usage_recording_started",
            request_id=usage.request_id,
            estimated=usage.estimated,
            actual=usage.actual,
            provider=usage.provider.value,
        )

        # Append to event store (handle failures gracefully)
        try:
            await self.event_store.append(usage)
        except Exception as e:
            log_with_context(
                "error",
                "token_usage_event_store_failed",
                request_id=usage.request_id,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            # Continue with budget update even if event store fails

        # Update budget (create new instance since TokenBudget is frozen)
        new_remaining = self._budget.remaining - usage.actual
        if new_remaining < 0:
            new_remaining = 0  # Prevent negative remaining

        self._budget = TokenBudget(
            total=self._budget.total,
            remaining=new_remaining,
            alert_threshold=self._budget.alert_threshold,
        )

        # Check if budget is below threshold
        threshold_amount = self._budget.total * self._budget.alert_threshold
        if self._budget.remaining < threshold_amount:
            log_with_context(
                "warning",
                "token_budget_threshold_exceeded",
                remaining=self._budget.remaining,
                total=self._budget.total,
                threshold=threshold_amount,
                percentage_remaining=(
                    self._budget.remaining / self._budget.total * 100
                ),
            )

        log_with_context(
            "info",
            "token_usage_recorded",
            request_id=usage.request_id,
            actual=usage.actual,
            remaining_budget=self._budget.remaining,
            total_budget=self._budget.total,
        )

    async def check_budget(self) -> TokenBudget:
        """
        Get current budget state.

        Returns:
            Current token budget (frozen copy)

        Example:
            >>> budget = await manager.check_budget()
            >>> print(f"Remaining: {budget.remaining}/{budget.total}")
        """
        return self._budget

    async def get_usage_history(self, since: datetime) -> list[TokenUsage]:
        """
        Get token usage history since a timestamp.

        Queries event store for all usage events after the specified time.

        Args:
            since: Timestamp to query from (exclusive)

        Returns:
            List of TokenUsage events in chronological order

        Example:
            >>> since = datetime.now(timezone.utc) - timedelta(hours=1)
            >>> history = await manager.get_usage_history(since)
            >>> print(f"Found {len(history)} usage records")
        """
        log_with_context(
            "info",
            "token_usage_history_query",
            since=since.isoformat(),
        )

        events = await self.event_store.get_events_since(since)

        # Type cast: IEventStore returns list[E] where E is TokenUsage
        usage_history: list[TokenUsage] = [
            event for event in events if isinstance(event, TokenUsage)
        ]

        log_with_context(
            "info",
            "token_usage_history_retrieved",
            since=since.isoformat(),
            count=len(usage_history),
        )

        return usage_history
