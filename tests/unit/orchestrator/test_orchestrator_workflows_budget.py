"""
Unit tests for orchestrator token budget integration.

Tests verify:
1. Pre-flight budget checks before workflow execution
2. Budget exceeded error handling
3. Post-execution usage recording
4. Failed task exclusion from usage tracking
5. Backward compatibility (orchestrator without budget manager)
6. Memory tier parameter handling
7. Logging and event emission for budget operations

Coverage target: 95%+ for budget integration paths
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from saga.orchestrator.core import (
    BudgetExceededError,
    Orchestrator,
    WorkflowPattern,
)
from saga.orchestrator.events import InMemoryEventStore, WorkflowCompletedEvent
from saga.orchestrator.token_manager import (
    LocalTokenEstimator,
    TokenBudget,
    TokenBudgetManager,
)
from saga.orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    AITaskInput,
    Result,
    Task,
    TaskStatus,
)


@pytest.fixture
def token_event_store() -> InMemoryEventStore:
    """Create event store for token usage events."""
    return InMemoryEventStore()


@pytest.fixture
def token_budget_manager(
    token_event_store: InMemoryEventStore,
) -> TokenBudgetManager:
    """Create token budget manager with sufficient budget."""
    estimator = LocalTokenEstimator()
    budget = TokenBudget(total=50000, remaining=50000, alert_threshold=0.2)
    return TokenBudgetManager(estimator, budget, token_event_store)


@pytest.fixture
def token_budget_manager_insufficient(
    token_event_store: InMemoryEventStore,
) -> TokenBudgetManager:
    """Create token budget manager with insufficient budget."""
    estimator = LocalTokenEstimator()
    budget = TokenBudget(total=10000, remaining=1000, alert_threshold=0.2)
    return TokenBudgetManager(estimator, budget, token_event_store)


@pytest.fixture
def ai_tasks() -> list[AITask]:
    """Create sample AI tasks for testing."""
    return [
        AITask(
            operation="chat_completion",
            input_data=AITaskInput(
                prompt="What is the capital of France?",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=100,
            ),
        ),
        AITask(
            operation="chat_completion",
            input_data=AITaskInput(
                prompt="Explain quantum computing",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=200,
            ),
        ),
    ]


@pytest.fixture
def orchestrator_with_budget(
    token_budget_manager: TokenBudgetManager,
) -> Orchestrator:
    """Create orchestrator with token budget manager."""
    event_store = InMemoryEventStore()
    return Orchestrator(
        event_store=event_store,
        token_budget_manager=token_budget_manager,
    )


@pytest.fixture
def orchestrator_without_budget() -> Orchestrator:
    """Create orchestrator without token budget manager (backward compatibility)."""
    event_store = InMemoryEventStore()
    return Orchestrator(event_store=event_store)


class TestWorkflowBudgetIntegration:
    """Tests for workflow budget integration."""

    @pytest.mark.asyncio
    async def test_workflow_executes_with_sufficient_budget(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test workflow executes successfully when budget is sufficient."""

        # Mock task executor to return successful AI results
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test response",
                        tokens_used=100,
                        prompt_tokens=50,
                        completion_tokens=50,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                        cost_usd=Decimal("0.001"),
                    ),
                    duration_ms=100,
                    status=TaskStatus.COMPLETED,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Execute workflow
        results = await orchestrator_with_budget.execute_workflow(
            WorkflowPattern.SEQUENTIAL, ai_tasks
        )

        # Assert workflow completed successfully
        assert len(results) == 2
        assert all(r.success for r in results)

        # Verify budget manager was used
        budget = await token_budget_manager.check_budget()
        assert budget.remaining < 50000  # Budget was consumed

    @pytest.mark.asyncio
    async def test_workflow_blocked_when_budget_insufficient(
        self,
        token_budget_manager_insufficient: TokenBudgetManager,
        ai_tasks: list[AITask],
    ) -> None:
        """Test workflow is blocked when budget is insufficient."""
        event_store = InMemoryEventStore()
        orchestrator = Orchestrator(
            event_store=event_store,
            token_budget_manager=token_budget_manager_insufficient,
        )

        # Execute workflow - should raise BudgetExceededError
        with pytest.raises(BudgetExceededError) as exc_info:
            await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, ai_tasks)

        # Verify error details
        assert exc_info.value.workflow_id is not None
        assert exc_info.value.requested_tokens > exc_info.value.remaining_tokens

        # Verify workflow did not execute (no WorkflowCompletedEvent with success)
        events = await event_store.get_events(exc_info.value.workflow_id)
        completed_events = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        # Should have a failed completion event, not a successful one
        if completed_events:
            assert not completed_events[0].success

    @pytest.mark.asyncio
    async def test_workflow_without_budget_manager_executes_normally(
        self,
        orchestrator_without_budget: Orchestrator,
    ) -> None:
        """Test orchestrator without budget manager works (backward compatibility)."""
        tasks = [
            Task(
                operation="test",
                input_data={"key": "value"},
            )
        ]

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            return Result(
                task_id=task.id,
                success=True,
                output_data=task.input_data,
                duration_ms=100,
            )

        orchestrator_without_budget._task_executor = mock_executor

        # Execute workflow - should work normally
        results = await orchestrator_without_budget.execute_workflow(
            WorkflowPattern.SEQUENTIAL, tasks
        )

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_budget_check_logs_correctly(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
    ) -> None:
        """Test budget check logs correctly."""

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test",
                        tokens_used=100,
                        prompt_tokens=50,
                        completion_tokens=50,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                    ),
                    duration_ms=100,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        with patch("saga.orchestrator.core.log_with_context") as mock_log:
            await orchestrator_with_budget.execute_workflow(
                WorkflowPattern.SEQUENTIAL, ai_tasks
            )

            # Verify budget check was logged
            budget_check_calls = [
                call
                for call in mock_log.call_args_list
                if len(call[0]) > 1 and call[0][1] == "workflow_budget_check_passed"
            ]
            assert len(budget_check_calls) > 0

            # Verify log contains required fields
            call_kwargs = budget_check_calls[0][1]
            assert "workflow_id" in call_kwargs
            assert "estimated_tokens" in call_kwargs
            assert "remaining_tokens" in call_kwargs
            assert "total_budget" in call_kwargs
            assert "memory_tier" in call_kwargs

    @pytest.mark.asyncio
    async def test_budget_exceeded_logs_correctly(
        self,
        token_budget_manager_insufficient: TokenBudgetManager,
        ai_tasks: list[AITask],
    ) -> None:
        """Test budget exceeded error logs correctly."""
        event_store = InMemoryEventStore()
        orchestrator = Orchestrator(
            event_store=event_store,
            token_budget_manager=token_budget_manager_insufficient,
        )

        with patch("saga.orchestrator.core.log_with_context") as mock_log:
            with pytest.raises(BudgetExceededError):
                await orchestrator.execute_workflow(
                    WorkflowPattern.SEQUENTIAL, ai_tasks
                )

            # Verify error was logged before exception
            error_calls = [
                call
                for call in mock_log.call_args_list
                if len(call[0]) > 1 and call[0][1] == "workflow_budget_exceeded"
            ]
            assert len(error_calls) > 0

            # Verify log contains required context
            call_kwargs = error_calls[0][1]
            assert "workflow_id" in call_kwargs
            assert "requested_tokens" in call_kwargs
            assert "remaining_tokens" in call_kwargs
            assert "total_budget" in call_kwargs

    @pytest.mark.asyncio
    async def test_usage_recorded_after_workflow_completion(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test usage is recorded after workflow completion."""

        # Mock task executor with known token usage
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test response",
                        tokens_used=150,  # Known value
                        prompt_tokens=75,
                        completion_tokens=75,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                        cost_usd=Decimal("0.001"),
                    ),
                    duration_ms=100,
                    status=TaskStatus.COMPLETED,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Get initial budget
        initial_budget = await token_budget_manager.check_budget()
        initial_remaining = initial_budget.remaining

        # Execute workflow
        await orchestrator_with_budget.execute_workflow(
            WorkflowPattern.SEQUENTIAL, ai_tasks
        )

        # Verify budget was updated (usage recorded)
        final_budget = await token_budget_manager.check_budget()
        # Should have consumed tokens (2 tasks * 150 tokens = 300)
        assert final_budget.remaining < initial_remaining

        # Verify usage history contains the workflow usage
        since = datetime.now(timezone.utc).replace(year=2000)  # Get all recent events
        history = await token_budget_manager.get_usage_history(since)
        assert len(history) > 0
        # Last usage should match our workflow
        last_usage = history[-1]
        assert last_usage.actual > 0  # Actual tokens were recorded

    @pytest.mark.asyncio
    async def test_usage_recording_handles_event_store_failure(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
    ) -> None:
        """Test usage recording handles event store failures gracefully."""

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test",
                        tokens_used=100,
                        prompt_tokens=50,
                        completion_tokens=50,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                    ),
                    duration_ms=100,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Mock event store to raise exception
        with patch.object(
            orchestrator_with_budget._token_budget_manager.event_store,
            "append",
            side_effect=Exception("Event store failure"),
        ):
            with patch(
                "saga.orchestrator.token_manager.log_with_context"
            ) as mock_log:
                # Workflow should still complete successfully
                results = await orchestrator_with_budget.execute_workflow(
                    WorkflowPattern.SEQUENTIAL, ai_tasks
                )

                assert len(results) == 2
                assert all(r.success for r in results)

                # Verify error was logged but not raised
                # The error is logged from token_manager.py, not core.py
                error_calls = [
                    call
                    for call in mock_log.call_args_list
                    if len(call[0]) > 1
                    and call[0][1] == "token_usage_event_store_failed"
                ]
                assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_non_ai_tasks_skip_budget_check(
        self,
        orchestrator_with_budget: Orchestrator,
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test non-AI tasks skip budget check."""
        # Create non-AI tasks
        tasks = [
            Task(operation="process", input_data={"key": "value"}),
            Task(operation="transform", input_data={"data": "test"}),
        ]

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            return Result(
                task_id=task.id,
                success=True,
                output_data=task.input_data,
                duration_ms=100,
            )

        orchestrator_with_budget._task_executor = mock_executor

        # Get initial budget
        initial_budget = await token_budget_manager.check_budget()

        # Execute workflow
        results = await orchestrator_with_budget.execute_workflow(
            WorkflowPattern.SEQUENTIAL, tasks
        )

        assert len(results) == 2
        assert all(r.success for r in results)

        # Verify budget was NOT consumed (non-AI tasks don't use tokens)
        final_budget = await token_budget_manager.check_budget()
        assert final_budget.remaining == initial_budget.remaining

    @pytest.mark.asyncio
    async def test_memory_tier_parameter_respected(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test memory tier parameter is respected."""

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test",
                        tokens_used=100,
                        prompt_tokens=50,
                        completion_tokens=50,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                    ),
                    duration_ms=100,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Mock allocate_tokens to verify it's called with correct tier
        original_allocate = token_budget_manager.allocate_tokens

        async def mock_allocate(
            tasks: list[AITask], memory_tier: str
        ) -> dict[str, int]:
            # Call original to get real allocations
            return await original_allocate(tasks, memory_tier)

        with patch.object(
            token_budget_manager, "allocate_tokens", side_effect=mock_allocate
        ) as mock_allocate_patched:
            await orchestrator_with_budget.execute_workflow(
                WorkflowPattern.SEQUENTIAL, ai_tasks, memory_tier="enhanced"
            )

            # Verify allocate_tokens was called with "enhanced" tier
            mock_allocate_patched.assert_called_once()
            call_args = mock_allocate_patched.call_args
            assert call_args[0][1] == "enhanced"  # memory_tier parameter

    @pytest.mark.asyncio
    async def test_parallel_workflow_budget_check(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test parallel workflow budget check sums all task estimates correctly."""

        # Mock task executor
        async def mock_executor(task: Task[Any]) -> Result[Any]:
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=AIResultOutput(
                        response="Test",
                        tokens_used=100,
                        prompt_tokens=50,
                        completion_tokens=50,
                        provider=task.input_data.provider,
                        model=task.input_data.model,
                    ),
                    duration_ms=100,
                )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Execute parallel workflow
        results = await orchestrator_with_budget.execute_workflow(
            WorkflowPattern.PARALLEL, ai_tasks
        )

        assert len(results) == 2
        assert all(r.success for r in results)

        # Verify budget was consumed (both tasks executed)
        budget = await token_budget_manager.check_budget()
        assert budget.remaining < 50000

    @pytest.mark.asyncio
    async def test_failed_tasks_excluded_from_usage_recording(
        self,
        orchestrator_with_budget: Orchestrator,
        ai_tasks: list[AITask],
        token_budget_manager: TokenBudgetManager,
    ) -> None:
        """Test failed tasks are excluded from usage recording."""
        # Mock task executor - first task succeeds, second fails
        call_count = 0

        async def mock_executor(task: Task[Any]) -> Result[Any]:
            nonlocal call_count
            call_count += 1
            if isinstance(task, Task) and isinstance(task.input_data, AITaskInput):
                if call_count == 1:
                    # First task succeeds
                    return AIResult(
                        task_id=task.id,
                        success=True,
                        output_data=AIResultOutput(
                            response="Success",
                            tokens_used=150,
                            prompt_tokens=75,
                            completion_tokens=75,
                            provider=task.input_data.provider,
                            model=task.input_data.model,
                        ),
                        duration_ms=100,
                        status=TaskStatus.COMPLETED,
                    )
                else:
                    # Second task fails (no output_data)
                    return AIResult(
                        task_id=task.id,
                        success=False,
                        error="Task failed",
                        error_type="TestError",
                        duration_ms=50,
                        status=TaskStatus.FAILED,
                    )
            return Result(task_id=task.id, success=True, duration_ms=100)

        orchestrator_with_budget._task_executor = mock_executor

        # Get initial budget
        initial_budget = await token_budget_manager.check_budget()
        initial_remaining = initial_budget.remaining

        # Execute workflow
        results = await orchestrator_with_budget.execute_workflow(
            WorkflowPattern.SEQUENTIAL, ai_tasks
        )

        assert len(results) == 2
        assert results[0].success
        assert not results[1].success

        # Verify only successful task's tokens were recorded
        final_budget = await token_budget_manager.check_budget()
        tokens_consumed = initial_remaining - final_budget.remaining

        # Should only consume tokens from first (successful) task (150 tokens)
        # Not from second (failed) task
        assert tokens_consumed == 150

        # Verify usage history
        since = datetime.now(timezone.utc).replace(year=2000)
        history = await token_budget_manager.get_usage_history(since)
        if history:
            last_usage = history[-1]
            # Actual should only include successful task (150), not failed task
            assert last_usage.actual == 150
