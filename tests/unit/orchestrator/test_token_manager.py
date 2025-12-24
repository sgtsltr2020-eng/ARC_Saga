"""
Unit tests for token manager implementations.

Tests verify:
1. TokenUsage creation and validation
2. TokenBudget creation and validation
3. LocalTokenEstimator estimation logic
4. ProviderTokenTracker tracking
5. TokenBudgetManager operations
6. Concurrent access handling
7. Error handling

Coverage target: 98%+
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from saga.orchestrator.events import InMemoryEventStore
from saga.orchestrator.token_manager import (
    LocalTokenEstimator,
    ProviderTokenTracker,
    TokenBudget,
    TokenBudgetManager,
    TokenUsage,
)
from saga.orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    AITaskInput,
    TaskStatus,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_creation(self) -> None:
        """Test creating token usage with valid data."""
        now = datetime.now(timezone.utc)
        usage = TokenUsage(
            aggregate_id="req-123",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
            created_at=now,
        )

        assert usage.aggregate_id == "req-123"
        assert usage.request_id == "req-123"  # Property alias
        assert usage.estimated == 5000
        assert usage.actual == 4800
        assert usage.cost_usd == 0.096
        assert usage.provider == AIProvider.OPENAI
        assert usage.created_at == now
        assert usage.timestamp == now  # Property alias
        assert usage.event_type == "TokenUsage"

    def test_token_usage_immutability(self) -> None:
        """Test token usage is frozen (immutable)."""
        usage = TokenUsage(
            aggregate_id="req-123",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

        with pytest.raises(AttributeError):
            usage.estimated = 6000  # type: ignore[misc]

    def test_token_usage_validation_negative_estimated(self) -> None:
        """Test validation fails for negative estimated tokens."""
        with pytest.raises(ValueError, match="estimated cannot be negative"):
            TokenUsage(
                aggregate_id="req-123",
                estimated=-100,
                actual=4800,
                cost_usd=0.096,
                provider=AIProvider.OPENAI,
            )

    def test_token_usage_validation_negative_actual(self) -> None:
        """Test validation fails for negative actual tokens."""
        with pytest.raises(ValueError, match="actual cannot be negative"):
            TokenUsage(
                aggregate_id="req-123",
                estimated=5000,
                actual=-100,
                cost_usd=0.096,
                provider=AIProvider.OPENAI,
            )

    def test_token_usage_validation_negative_cost(self) -> None:
        """Test validation fails for negative cost."""
        with pytest.raises(ValueError, match="cost_usd cannot be negative"):
            TokenUsage(
                aggregate_id="req-123",
                estimated=5000,
                actual=4800,
                cost_usd=-0.01,
                provider=AIProvider.OPENAI,
            )


class TestTokenBudget:
    """Tests for TokenBudget dataclass."""

    def test_token_budget_creation(self) -> None:
        """Test creating token budget with valid data."""
        budget = TokenBudget(
            total=50000,
            remaining=45000,
            alert_threshold=0.2,
        )

        assert budget.total == 50000
        assert budget.remaining == 45000
        assert budget.alert_threshold == 0.2

    def test_token_budget_default_threshold(self) -> None:
        """Test default alert threshold is 0.2."""
        budget = TokenBudget(total=50000, remaining=50000)

        assert budget.alert_threshold == 0.2

    def test_token_budget_immutability(self) -> None:
        """Test token budget is frozen (immutable)."""
        budget = TokenBudget(total=50000, remaining=50000)

        with pytest.raises(AttributeError):
            budget.remaining = 40000  # type: ignore[misc]

    def test_token_budget_validation_zero_total(self) -> None:
        """Test validation fails for zero total."""
        with pytest.raises(ValueError, match="total must be positive"):
            TokenBudget(total=0, remaining=0)

    def test_token_budget_validation_negative_total(self) -> None:
        """Test validation fails for negative total."""
        with pytest.raises(ValueError, match="total must be positive"):
            TokenBudget(total=-1000, remaining=0)

    def test_token_budget_validation_negative_remaining(self) -> None:
        """Test validation fails for negative remaining."""
        with pytest.raises(ValueError, match="remaining cannot be negative"):
            TokenBudget(total=50000, remaining=-100)

    def test_token_budget_validation_threshold_too_high(self) -> None:
        """Test validation fails for threshold > 1.0."""
        with pytest.raises(ValueError, match="alert_threshold must be between"):
            TokenBudget(total=50000, remaining=50000, alert_threshold=1.5)

    def test_token_budget_validation_threshold_negative(self) -> None:
        """Test validation fails for negative threshold."""
        with pytest.raises(ValueError, match="alert_threshold must be between"):
            TokenBudget(total=50000, remaining=50000, alert_threshold=-0.1)

    def test_token_budget_threshold_boundary_values(self) -> None:
        """Test threshold at boundary values (0.0 and 1.0)."""
        budget_zero = TokenBudget(total=50000, remaining=50000, alert_threshold=0.0)
        budget_one = TokenBudget(total=50000, remaining=50000, alert_threshold=1.0)

        assert budget_zero.alert_threshold == 0.0
        assert budget_one.alert_threshold == 1.0


class TestLocalTokenEstimator:
    """Tests for LocalTokenEstimator."""

    @pytest.fixture
    def estimator(self) -> LocalTokenEstimator:
        """Create estimator for each test."""
        return LocalTokenEstimator()

    @pytest.mark.asyncio
    async def test_estimate_minimal_tier(self, estimator: LocalTokenEstimator) -> None:
        """Test estimation for minimal tier."""
        tokens = await estimator.estimate("test", "minimal")

        # 2000 base + 1 prompt token (4 chars // 4)
        assert tokens == 2001

    @pytest.mark.asyncio
    async def test_estimate_standard_tier(self, estimator: LocalTokenEstimator) -> None:
        """Test estimation for standard tier."""
        tokens = await estimator.estimate("test", "standard")

        # 5000 base + 1 prompt token
        assert tokens == 5001

    @pytest.mark.asyncio
    async def test_estimate_enhanced_tier(self, estimator: LocalTokenEstimator) -> None:
        """Test estimation for enhanced tier."""
        tokens = await estimator.estimate("test", "enhanced")

        # 8000 base + 1 prompt token
        assert tokens == 8001

    @pytest.mark.asyncio
    async def test_estimate_complete_tier(self, estimator: LocalTokenEstimator) -> None:
        """Test estimation for complete tier."""
        tokens = await estimator.estimate("test", "complete")

        # 10000 base + 1 prompt token
        assert tokens == 10001

    @pytest.mark.asyncio
    async def test_estimate_unlimited_tier(
        self, estimator: LocalTokenEstimator
    ) -> None:
        """Test estimation for unlimited tier."""
        tokens = await estimator.estimate("test", "unlimited")

        # 15000 base + 1 prompt token
        assert tokens == 15001

    @pytest.mark.asyncio
    async def test_estimate_adds_prompt_tokens(
        self, estimator: LocalTokenEstimator
    ) -> None:
        """Test prompt tokens are added correctly (4 chars = 1 token)."""
        # 40 chars = 10 tokens
        prompt = "a" * 40
        tokens = await estimator.estimate(prompt, "standard")

        # 5000 base + 10 prompt tokens
        assert tokens == 5010

    @pytest.mark.asyncio
    async def test_estimate_raises_on_invalid_tier(
        self, estimator: LocalTokenEstimator
    ) -> None:
        """Test ValueError for invalid memory tier."""
        with pytest.raises(ValueError, match="Invalid memory_tier"):
            await estimator.estimate("test", "invalid_tier")

    @pytest.mark.asyncio
    async def test_estimate_empty_prompt(self, estimator: LocalTokenEstimator) -> None:
        """Test estimation with empty prompt returns base cost only."""
        tokens = await estimator.estimate("", "standard")

        # 5000 base + 0 prompt tokens
        assert tokens == 5000

    @pytest.mark.asyncio
    async def test_memory_tier_costs_values(
        self, estimator: LocalTokenEstimator
    ) -> None:
        """Test all memory tier costs are correctly defined."""
        assert estimator.MEMORY_TIER_COSTS == {
            "minimal": 2000,
            "standard": 5000,
            "enhanced": 8000,
            "complete": 10000,
            "unlimited": 15000,
        }


class TestProviderTokenTracker:
    """Tests for ProviderTokenTracker."""

    @pytest.fixture
    def tracker(self) -> ProviderTokenTracker:
        """Create tracker for each test."""
        return ProviderTokenTracker()

    @pytest.mark.asyncio
    async def test_track_actual_from_result(
        self, tracker: ProviderTokenTracker
    ) -> None:
        """Test tracking actual tokens from AIResult."""
        output = AIResultOutput(
            response="Test response",
            tokens_used=100,
            prompt_tokens=40,
            completion_tokens=60,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )
        result = AIResult(
            task_id="task-123",
            success=True,
            output_data=output,
            duration_ms=150,
        )

        tokens = await tracker.track_actual(result)

        assert tokens == 100

    @pytest.mark.asyncio
    async def test_track_actual_raises_on_none_output_data(
        self, tracker: ProviderTokenTracker
    ) -> None:
        """Test ValueError when output_data is None."""
        result = AIResult(
            task_id="task-123",
            success=False,
            output_data=None,
            error="Task failed",
            status=TaskStatus.FAILED,
        )

        with pytest.raises(ValueError, match="output_data is None"):
            await tracker.track_actual(result)


class TestTokenBudgetManager:
    """Tests for TokenBudgetManager."""

    @pytest_asyncio.fixture
    async def event_store(self) -> InMemoryEventStore:
        """Create in-memory event store for each test."""
        return InMemoryEventStore()

    @pytest_asyncio.fixture
    async def estimator(self) -> LocalTokenEstimator:
        """Create token estimator for each test."""
        return LocalTokenEstimator()

    @pytest_asyncio.fixture
    async def manager(
        self,
        event_store: InMemoryEventStore,
        estimator: LocalTokenEstimator,
    ) -> TokenBudgetManager:
        """Create token budget manager for each test."""
        budget = TokenBudget(total=50000, remaining=50000, alert_threshold=0.2)
        return TokenBudgetManager(estimator, budget, event_store)

    @pytest.fixture
    def sample_task(self) -> AITask:
        """Create sample AI task."""
        return AITask(
            id="test-task-1",
            operation="generate",
            input_data=AITaskInput(
                prompt="Test prompt for token estimation",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=1000,
                temperature=0.7,
            ),
            timeout_ms=30000,
        )

    @pytest.fixture
    def sample_usage(self) -> TokenUsage:
        """Create sample token usage."""
        return TokenUsage(
            aggregate_id="test-req-1",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

    @pytest.mark.asyncio
    async def test_allocate_tokens_single_task(
        self, manager: TokenBudgetManager, sample_task: AITask
    ) -> None:
        """Test allocating tokens for a single task."""
        allocations = await manager.allocate_tokens([sample_task])

        assert len(allocations) == 1
        assert sample_task.id in allocations
        # 5000 base + prompt tokens (35 chars // 4 = 8)
        expected = 5000 + len(sample_task.input_data.prompt) // 4
        assert allocations[sample_task.id] == expected

    @pytest.mark.asyncio
    async def test_allocate_tokens_multiple_tasks(
        self, manager: TokenBudgetManager
    ) -> None:
        """Test allocating tokens for multiple tasks."""
        tasks = [
            AITask(
                id=f"task-{i}",
                operation="generate",
                input_data=AITaskInput(
                    prompt=f"Prompt {i}" * 10,
                    model="gpt-4",
                    provider=AIProvider.OPENAI,
                ),
            )
            for i in range(3)
        ]

        allocations = await manager.allocate_tokens(tasks)

        assert len(allocations) == 3
        for task in tasks:
            assert task.id in allocations
            assert allocations[task.id] > 5000  # Base cost + prompt

    @pytest.mark.asyncio
    async def test_allocate_tokens_with_memory_tier(
        self, manager: TokenBudgetManager, sample_task: AITask
    ) -> None:
        """Test allocation with different memory tiers."""
        tiers = ["minimal", "standard", "enhanced", "complete", "unlimited"]
        expected_bases = [2000, 5000, 8000, 10000, 15000]

        for tier, expected_base in zip(tiers, expected_bases):
            allocations = await manager.allocate_tokens([sample_task], memory_tier=tier)
            prompt_tokens = len(sample_task.input_data.prompt) // 4
            expected = expected_base + prompt_tokens
            assert allocations[sample_task.id] == expected

    @pytest.mark.asyncio
    async def test_allocate_tokens_empty_list(
        self, manager: TokenBudgetManager
    ) -> None:
        """Test allocating tokens for empty task list."""
        allocations = await manager.allocate_tokens([])

        assert allocations == {}

    @pytest.mark.asyncio
    async def test_record_usage_updates_remaining(
        self, manager: TokenBudgetManager, sample_usage: TokenUsage
    ) -> None:
        """Test recording usage updates remaining budget."""
        initial_budget = await manager.check_budget()
        initial_remaining = initial_budget.remaining

        await manager.record_usage(sample_usage)

        updated_budget = await manager.check_budget()
        assert updated_budget.remaining == initial_remaining - sample_usage.actual

    @pytest.mark.asyncio
    async def test_record_usage_appends_to_event_store(
        self,
        manager: TokenBudgetManager,
        sample_usage: TokenUsage,
        event_store: InMemoryEventStore,
    ) -> None:
        """Test recording usage appends to event store."""
        await manager.record_usage(sample_usage)

        assert event_store.event_count == 1
        events = await event_store.get_events(sample_usage.aggregate_id)
        assert len(events) == 1
        assert events[0].aggregate_id == sample_usage.aggregate_id

    @pytest.mark.asyncio
    async def test_record_usage_logs_warning_at_threshold(
        self, estimator: LocalTokenEstimator, event_store: InMemoryEventStore
    ) -> None:
        """Test warning is logged exactly once when budget crosses below threshold."""
        # Create manager with low remaining budget
        budget = TokenBudget(total=10000, remaining=2500, alert_threshold=0.2)
        manager = TokenBudgetManager(estimator, budget, event_store)

        # Usage that brings remaining below threshold (10000 * 0.2 = 2000)
        usage = TokenUsage(
            aggregate_id="req-1",
            estimated=1000,
            actual=1000,  # Remaining will be 1500 < 2000
            cost_usd=0.02,
            provider=AIProvider.OPENAI,
        )

        with patch("saga.orchestrator.token_manager.log_with_context") as mock_log:
            await manager.record_usage(usage)

            # Verify warning was logged exactly once
            warning_calls = [
                call for call in mock_log.call_args_list if call[0][0] == "warning"
            ]
            assert len(warning_calls) == 1

    @pytest.mark.asyncio
    async def test_record_usage_below_threshold_no_warning(
        self, manager: TokenBudgetManager
    ) -> None:
        """Test no warning when budget always stays above threshold."""
        # Manager starts with 50000 remaining, threshold is 0.2 (10000)
        # Using 100 tokens keeps remaining at 49900, well above 10000
        usage = TokenUsage(
            aggregate_id="req-1",
            estimated=100,
            actual=100,
            cost_usd=0.002,
            provider=AIProvider.OPENAI,
        )

        with patch("saga.orchestrator.token_manager.log_with_context") as mock_log:
            await manager.record_usage(usage)

            # Verify no warning was logged (remaining stays above threshold)
            warning_calls = [
                call for call in mock_log.call_args_list if call[0][0] == "warning"
            ]
            assert len(warning_calls) == 0

    @pytest.mark.asyncio
    async def test_check_budget_returns_current_state(
        self, manager: TokenBudgetManager
    ) -> None:
        """Test check_budget returns current budget state."""
        budget = await manager.check_budget()

        assert budget.total == 50000
        assert budget.remaining == 50000
        assert budget.alert_threshold == 0.2

    @pytest.mark.asyncio
    async def test_get_usage_history_queries_event_store(
        self, manager: TokenBudgetManager
    ) -> None:
        """Test get_usage_history returns events from store."""
        now = datetime.now(timezone.utc)
        usage1 = TokenUsage(
            aggregate_id="req-1",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
            created_at=now + timedelta(seconds=1),
        )
        usage2 = TokenUsage(
            aggregate_id="req-2",
            estimated=3000,
            actual=2800,
            cost_usd=0.056,
            provider=AIProvider.ANTHROPIC,
            created_at=now + timedelta(seconds=2),
        )

        await manager.record_usage(usage1)
        await manager.record_usage(usage2)

        history = await manager.get_usage_history(now)

        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_usage_history_empty(self, manager: TokenBudgetManager) -> None:
        """Test get_usage_history returns empty list when no events."""
        now = datetime.now(timezone.utc)
        history = await manager.get_usage_history(now)

        assert history == []

    @pytest.mark.asyncio
    async def test_concurrent_usage_recording(
        self,
        estimator: LocalTokenEstimator,
        event_store: InMemoryEventStore,
    ) -> None:
        """Test concurrent usage recording is handled correctly."""
        budget = TokenBudget(total=100000, remaining=100000)
        manager = TokenBudgetManager(estimator, budget, event_store)

        usages = [
            TokenUsage(
                aggregate_id=f"req-{i}",
                estimated=100,
                actual=100,
                cost_usd=0.002,
                provider=AIProvider.OPENAI,
            )
            for i in range(10)
        ]

        # Record all usages concurrently
        await asyncio.gather(*[manager.record_usage(usage) for usage in usages])

        # All events should be stored
        assert event_store.event_count == 10

        # Budget should be updated
        final_budget = await manager.check_budget()
        assert final_budget.remaining == 100000 - (100 * 10)

    @pytest.mark.asyncio
    async def test_record_usage_handles_event_store_failure(
        self, estimator: LocalTokenEstimator
    ) -> None:
        """Test record_usage handles event store failure gracefully."""
        # Create mock event store that raises on append
        mock_store: Any = AsyncMock()
        mock_store.append = AsyncMock(side_effect=Exception("Store failed"))

        budget = TokenBudget(total=50000, remaining=50000)
        manager = TokenBudgetManager(estimator, budget, mock_store)

        usage = TokenUsage(
            aggregate_id="req-1",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

        # Should not raise, but should log error
        with patch("saga.orchestrator.token_manager.log_with_context") as mock_log:
            await manager.record_usage(usage)

            # Verify error was logged
            error_calls = [
                call for call in mock_log.call_args_list if call[0][0] == "error"
            ]
            assert len(error_calls) >= 1

        # Budget should still be updated
        final_budget = await manager.check_budget()
        assert final_budget.remaining == 50000 - 4800

    @pytest.mark.asyncio
    async def test_record_usage_prevents_negative_remaining(
        self, estimator: LocalTokenEstimator, event_store: InMemoryEventStore
    ) -> None:
        """Test remaining budget doesn't go negative."""
        budget = TokenBudget(total=1000, remaining=500)
        manager = TokenBudgetManager(estimator, budget, event_store)

        usage = TokenUsage(
            aggregate_id="req-1",
            estimated=1000,
            actual=1000,  # More than remaining
            cost_usd=0.02,
            provider=AIProvider.OPENAI,
        )

        await manager.record_usage(usage)

        final_budget = await manager.check_budget()
        assert final_budget.remaining == 0  # Should be 0, not negative

    @pytest.mark.asyncio
    async def test_manager_initialization_logging(
        self, estimator: LocalTokenEstimator, event_store: InMemoryEventStore
    ) -> None:
        """Test manager logs on initialization."""
        budget = TokenBudget(total=50000, remaining=50000)

        with patch("saga.orchestrator.token_manager.log_with_context") as mock_log:
            TokenBudgetManager(estimator, budget, event_store)

            # Verify initialization was logged
            init_calls = [
                call for call in mock_log.call_args_list if "initialized" in str(call)
            ]
            assert len(init_calls) >= 1


class TestITokenEstimatorProtocol:
    """Tests for ITokenEstimator protocol compliance."""

    def test_local_estimator_implements_protocol(self) -> None:
        """Test LocalTokenEstimator implements ITokenEstimator."""
        estimator = LocalTokenEstimator()

        # Check protocol compliance
        assert hasattr(estimator, "estimate")
        assert callable(estimator.estimate)


class TestTokenUsageEventIntegration:
    """Tests for TokenUsage as OrchestratorEvent."""

    def test_token_usage_inherits_event_fields(self) -> None:
        """Test TokenUsage has all OrchestratorEvent fields."""
        usage = TokenUsage(
            aggregate_id="req-123",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

        # Event fields
        assert hasattr(usage, "id")
        assert hasattr(usage, "aggregate_id")
        assert hasattr(usage, "event_type")
        assert hasattr(usage, "created_at")
        assert hasattr(usage, "correlation_id")
        assert hasattr(usage, "source")
        assert hasattr(usage, "version")

    def test_token_usage_to_dict(self) -> None:
        """Test TokenUsage can be serialized to dict."""
        usage = TokenUsage(
            aggregate_id="req-123",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

        data = usage.to_dict()

        assert data["aggregate_id"] == "req-123"
        assert data["estimated"] == 5000
        assert data["actual"] == 4800
        assert data["event_type"] == "TokenUsage"

    @pytest.mark.asyncio
    async def test_token_usage_stored_in_event_store(self) -> None:
        """Test TokenUsage can be stored in InMemoryEventStore."""
        store = InMemoryEventStore()
        usage = TokenUsage(
            aggregate_id="req-123",
            estimated=5000,
            actual=4800,
            cost_usd=0.096,
            provider=AIProvider.OPENAI,
        )

        event_id = await store.append(usage)

        assert event_id == usage.id
        events = await store.get_events("req-123")
        assert len(events) == 1
        assert events[0].aggregate_id == "req-123"
