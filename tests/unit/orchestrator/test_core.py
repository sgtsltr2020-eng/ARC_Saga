"""
Unit tests for orchestrator core module.

Tests verify:
1. Orchestrator initialization
2. Workflow execution patterns (sequential, parallel, dynamic)
3. Policy enforcement
4. Operation logging
5. Event emission
6. Circuit breaker integration
7. Error handling

Coverage target: 98%+
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from arc_saga.orchestrator.core import (
    Command,
    OperationContext,
    Orchestrator,
    OrchestratorError,
    Policy,
    PolicyResult,
    PolicyViolationError,
    WorkflowError,
    WorkflowPattern,
)
from arc_saga.orchestrator.events import (
    InMemoryEventStore,
    OrchestratorEvent,
)
from arc_saga.orchestrator.types import Result, Task, TaskStatus


class TestWorkflowPattern:
    """Tests for WorkflowPattern enum."""

    def test_all_patterns_have_string_values(self) -> None:
        """Test all patterns have lowercase string values."""
        assert WorkflowPattern.SEQUENTIAL.value == "sequential"
        assert WorkflowPattern.PARALLEL.value == "parallel"
        assert WorkflowPattern.DYNAMIC.value == "dynamic"

    def test_pattern_is_string_subclass(self) -> None:
        """Test WorkflowPattern inherits from str."""
        assert isinstance(WorkflowPattern.SEQUENTIAL, str)


class TestPolicy:
    """Tests for Policy dataclass."""

    def test_create_policy(self) -> None:
        """Test creating a policy."""
        policy = Policy(
            name="rate_limit",
            description="Limits request rate",
            check_func_name="check_rate_limit",
            parameters={"max_requests": 100},
        )

        assert policy.name == "rate_limit"
        assert policy.description == "Limits request rate"
        assert policy.check_func_name == "check_rate_limit"
        assert policy.parameters == {"max_requests": 100}
        assert policy.enabled is True

    def test_policy_defaults(self) -> None:
        """Test policy default values."""
        policy = Policy(name="test")

        assert policy.description == ""
        assert policy.check_func_name == ""
        assert policy.parameters == {}
        assert policy.enabled is True

    def test_policy_disabled(self) -> None:
        """Test creating disabled policy."""
        policy = Policy(name="test", enabled=False)

        assert policy.enabled is False


class TestCommand:
    """Tests for Command dataclass."""

    def test_create_command(self) -> None:
        """Test creating a command."""
        command = Command(
            type="execute_task",
            payload={"task_id": "123"},
            user_id="user-456",
        )

        assert command.type == "execute_task"
        assert command.payload == {"task_id": "123"}
        assert command.user_id == "user-456"
        assert len(command.correlation_id) == 36  # UUID

    def test_command_defaults(self) -> None:
        """Test command default values."""
        command = Command(type="test")

        assert command.payload == {}
        assert command.user_id is None

    def test_command_custom_correlation_id(self) -> None:
        """Test command with custom correlation ID."""
        command = Command(
            type="test",
            correlation_id="custom-corr-id",
        )

        assert command.correlation_id == "custom-corr-id"


class TestPolicyResult:
    """Tests for PolicyResult dataclass."""

    def test_create_allowed_result(self) -> None:
        """Test creating allowed policy result."""
        result = PolicyResult(
            policy_name="rate_limit",
            allowed=True,
            reason="Under rate limit",
        )

        assert result.policy_name == "rate_limit"
        assert result.allowed is True
        assert result.reason == "Under rate limit"
        assert result.evaluated_at.tzinfo == timezone.utc

    def test_create_denied_result(self) -> None:
        """Test creating denied policy result."""
        result = PolicyResult(
            policy_name="auth_check",
            allowed=False,
            reason="Insufficient permissions",
        )

        assert result.allowed is False


class TestOperationContext:
    """Tests for OperationContext dataclass."""

    def test_create_operation_context(self) -> None:
        """Test creating operation context."""
        context = OperationContext(
            operation_name="user_login",
            user_id="user-123",
            parameters={"ip": "192.168.1.1"},
            level="INFO",
        )

        assert context.operation_name == "user_login"
        assert context.user_id == "user-123"
        assert context.parameters == {"ip": "192.168.1.1"}
        assert context.level == "INFO"
        assert len(context.correlation_id) == 36

    def test_context_defaults(self) -> None:
        """Test operation context default values."""
        context = OperationContext(operation_name="test")

        assert context.user_id is None
        assert context.parameters == {}
        assert context.level == "INFO"
        assert context.additional_data == {}


class TestOrchestratorErrors:
    """Tests for orchestrator error classes."""

    def test_orchestrator_error(self) -> None:
        """Test OrchestratorError."""
        error = OrchestratorError("Something failed", "execute_workflow")

        assert "Orchestrator execute_workflow failed" in str(error)
        assert "Something failed" in str(error)
        assert error.operation == "execute_workflow"

    def test_workflow_error(self) -> None:
        """Test WorkflowError."""
        error = WorkflowError(
            "Task execution failed",
            workflow_id="wf-123",
            failed_tasks=["task-1", "task-2"],
        )

        assert error.workflow_id == "wf-123"
        assert error.failed_tasks == ["task-1", "task-2"]
        assert error.operation == "execute_workflow"

    def test_policy_violation_error(self) -> None:
        """Test PolicyViolationError."""
        error = PolicyViolationError("rate_limit", "Rate exceeded")

        assert error.policy_name == "rate_limit"
        assert error.reason == "Rate exceeded"
        assert "rate_limit" in str(error)


class TestOrchestratorInitialization:
    """Tests for Orchestrator initialization."""

    def test_init_with_event_store(self) -> None:
        """Test initialization with event store."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        assert orchestrator.event_store is store
        assert orchestrator.circuit_breaker is None

    def test_init_with_circuit_breaker(self) -> None:
        """Test initialization with circuit breaker."""
        from arc_saga.integrations.circuit_breaker import CircuitBreaker

        store = InMemoryEventStore()
        breaker = CircuitBreaker("test-service")
        orchestrator = Orchestrator(store, circuit_breaker=breaker)

        assert orchestrator.circuit_breaker is breaker

    def test_init_with_custom_executor(self) -> None:
        """Test initialization with custom task executor."""
        store = InMemoryEventStore()

        async def custom_executor(task: Task[Any]) -> Result[Any]:
            return Result(
                task_id=task.id,
                success=True,
                output_data="custom",
            )

        orchestrator = Orchestrator(store, task_executor=custom_executor)

        assert orchestrator._task_executor is custom_executor


class TestOrchestratorWorkflow:
    """Tests for Orchestrator workflow execution."""

    @pytest_asyncio.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator with in-memory event store."""
        store = InMemoryEventStore()
        return Orchestrator(store)

    @pytest.mark.asyncio
    async def test_execute_sequential_workflow(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test sequential workflow execution."""
        tasks = [
            Task(operation="task1", input_data={"id": 1}),
            Task(operation="task2", input_data={"id": 2}),
            Task(operation="task3", input_data={"id": 3}),
        ]

        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify events were emitted
        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)
        assert store.event_count >= 4  # 1 started + 3 tasks + 1 completed

    @pytest.mark.asyncio
    async def test_execute_parallel_workflow(self, orchestrator: Orchestrator) -> None:
        """Test parallel workflow execution."""
        tasks = [
            Task(operation="task1", input_data={"id": 1}),
            Task(operation="task2", input_data={"id": 2}),
            Task(operation="task3", input_data={"id": 3}),
        ]

        results = await orchestrator.execute_workflow(WorkflowPattern.PARALLEL, tasks)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_dynamic_workflow(self, orchestrator: Orchestrator) -> None:
        """Test dynamic workflow execution."""
        tasks = [
            Task(operation="task1", input_data={"id": 1}),
            Task(operation="task2", input_data={"id": 2}),
        ]

        results = await orchestrator.execute_workflow(WorkflowPattern.DYNAMIC, tasks)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dynamic_workflow_stops_on_failure(self) -> None:
        """Test dynamic workflow stops on task failure."""
        store = InMemoryEventStore()
        call_count = 0

        async def failing_executor(task: Task[Any]) -> Result[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return Result(
                    task_id=task.id,
                    success=False,
                    error="Task failed",
                    status=TaskStatus.FAILED,
                )
            return Result(
                task_id=task.id,
                success=True,
                output_data="ok",
            )

        orchestrator = Orchestrator(store, task_executor=failing_executor)

        tasks = [
            Task(operation="task1", input_data={}),
            Task(operation="task2", input_data={}),  # This will fail
            Task(operation="task3", input_data={}),  # Should not execute
        ]

        results = await orchestrator.execute_workflow(WorkflowPattern.DYNAMIC, tasks)

        # Should stop after failure
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False

    @pytest.mark.asyncio
    async def test_empty_tasks_raises_error(self, orchestrator: Orchestrator) -> None:
        """Test empty tasks list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, [])

    @pytest.mark.asyncio
    async def test_workflow_with_correlation_id(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test workflow with custom correlation ID."""
        tasks = [Task(operation="test", input_data={})]

        await orchestrator.execute_workflow(
            WorkflowPattern.SEQUENTIAL,
            tasks,
            correlation_id="custom-corr-123",
        )

        # Verify correlation ID in events
        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)
        # Note: get_events returns by aggregate_id which is workflow_id
        # So we check all events
        all_events = await store.get_events_since(
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        assert any(e.correlation_id == "custom-corr-123" for e in all_events)

    @pytest.mark.asyncio
    async def test_workflow_emits_started_and_completed_events(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test workflow emits WorkflowStartedEvent and WorkflowCompletedEvent."""
        tasks = [Task(operation="test", input_data={})]

        await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)
        all_events = await store.get_events_since(
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        event_types = [e.event_type for e in all_events]
        assert "WorkflowStartedEvent" in event_types
        assert "WorkflowCompletedEvent" in event_types


class TestOrchestratorTaskExecution:
    """Tests for task execution with various scenarios."""

    @pytest.mark.asyncio
    async def test_task_execution_with_circuit_breaker(self) -> None:
        """Test task execution through circuit breaker."""
        from arc_saga.integrations.circuit_breaker import CircuitBreaker

        store = InMemoryEventStore()
        breaker = CircuitBreaker("test-service", failure_threshold=5)

        async def executor(task: Task[Any]) -> Result[Any]:
            return Result(
                task_id=task.id,
                success=True,
                output_data="executed",
            )

        orchestrator = Orchestrator(
            store,
            circuit_breaker=breaker,
            task_executor=executor,
        )

        tasks = [Task(operation="test", input_data={})]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert results[0].success is True
        assert results[0].output_data == "executed"

    @pytest.mark.asyncio
    async def test_task_execution_with_open_circuit_breaker(self) -> None:
        """Test task execution when circuit breaker is open."""
        from arc_saga.integrations.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
        )

        store = InMemoryEventStore()
        breaker = CircuitBreaker("test-service", failure_threshold=2)

        # Open the circuit breaker
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5
        breaker._last_failure_time = datetime.now(timezone.utc)

        orchestrator = Orchestrator(store, circuit_breaker=breaker)

        tasks = [Task(operation="test", input_data={})]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert results[0].success is False
        assert "CircuitBreakerOpenError" in str(results[0].error_type)

    @pytest.mark.asyncio
    async def test_task_execution_exception_handling(self) -> None:
        """Test task execution handles exceptions gracefully."""
        store = InMemoryEventStore()

        async def failing_executor(task: Task[Any]) -> Result[Any]:
            raise RuntimeError("Task crashed")

        orchestrator = Orchestrator(store, task_executor=failing_executor)

        tasks = [Task(operation="test", input_data={})]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert results[0].success is False
        assert "Task crashed" in str(results[0].error)
        assert results[0].error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_parallel_tasks_execute_concurrently(self) -> None:
        """Test parallel tasks actually execute concurrently."""
        store = InMemoryEventStore()
        execution_times: list[float] = []

        async def slow_executor(task: Task[Any]) -> Result[Any]:
            import time

            start = time.perf_counter()
            await asyncio.sleep(0.1)  # 100ms
            execution_times.append(time.perf_counter() - start)
            return Result(
                task_id=task.id,
                success=True,
                output_data="done",
            )

        orchestrator = Orchestrator(store, task_executor=slow_executor)

        tasks = [Task(operation=f"task{i}", input_data={}) for i in range(5)]

        import time

        start = time.perf_counter()
        results = await orchestrator.execute_workflow(WorkflowPattern.PARALLEL, tasks)
        total_time = time.perf_counter() - start

        assert len(results) == 5
        assert all(r.success for r in results)
        # Parallel execution should be faster than 5 * 100ms = 500ms
        # Allow some overhead but should be around 100-200ms
        assert total_time < 0.4  # Less than 400ms


class TestOrchestratorPolicy:
    """Tests for policy enforcement."""

    @pytest_asyncio.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator with in-memory event store."""
        store = InMemoryEventStore()
        return Orchestrator(store)

    @pytest.mark.asyncio
    async def test_enforce_policy_allowed(self, orchestrator: Orchestrator) -> None:
        """Test enforcing policy that allows command."""
        policy = Policy(
            name="allow_all",
            check_func_name="check_allow",
        )

        orchestrator.register_policy_check("check_allow", lambda cmd: True)

        command = Command(type="test_command")
        result = await orchestrator.enforce_policy(policy, command)

        assert result.allowed is True
        assert result.policy_name == "allow_all"

    @pytest.mark.asyncio
    async def test_enforce_policy_denied(self, orchestrator: Orchestrator) -> None:
        """Test enforcing policy that denies command."""
        policy = Policy(
            name="deny_all",
            check_func_name="check_deny",
        )

        orchestrator.register_policy_check("check_deny", lambda cmd: False)

        command = Command(type="test_command")
        result = await orchestrator.enforce_policy(policy, command)

        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_enforce_disabled_policy(self, orchestrator: Orchestrator) -> None:
        """Test enforcing disabled policy always allows."""
        policy = Policy(
            name="disabled_policy",
            enabled=False,
        )

        command = Command(type="test_command")
        result = await orchestrator.enforce_policy(policy, command)

        assert result.allowed is True
        assert "disabled" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_enforce_policy_no_check_function(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test enforcing policy without registered check function."""
        policy = Policy(
            name="unregistered",
            check_func_name="nonexistent_check",
        )

        command = Command(type="test_command")
        result = await orchestrator.enforce_policy(policy, command)

        assert result.allowed is True
        assert "No check function" in result.reason

    @pytest.mark.asyncio
    async def test_enforce_policy_check_exception(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test policy check that raises exception."""
        policy = Policy(
            name="error_policy",
            check_func_name="check_error",
        )

        def failing_check(cmd: Command) -> bool:
            raise ValueError("Check failed")

        orchestrator.register_policy_check("check_error", failing_check)

        command = Command(type="test_command")
        result = await orchestrator.enforce_policy(policy, command)

        assert result.allowed is False
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_enforce_policy_emits_event(self, orchestrator: Orchestrator) -> None:
        """Test policy enforcement emits PolicyEnforcedEvent."""
        policy = Policy(
            name="test_policy",
            check_func_name="check_test",
        )

        orchestrator.register_policy_check("check_test", lambda cmd: True)

        command = Command(type="test_command")
        await orchestrator.enforce_policy(policy, command)

        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)

        all_events = await store.get_events_since(
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        policy_events = [e for e in all_events if e.event_type == "PolicyEnforcedEvent"]

        assert len(policy_events) == 1


class TestOrchestratorLogging:
    """Tests for operation logging."""

    @pytest_asyncio.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator with in-memory event store."""
        store = InMemoryEventStore()
        return Orchestrator(store)

    @pytest.mark.asyncio
    async def test_log_operation(self, orchestrator: Orchestrator) -> None:
        """Test logging an operation."""
        context = OperationContext(
            operation_name="user_login",
            user_id="user-123",
            parameters={"ip": "192.168.1.1"},
            level="INFO",
        )

        await orchestrator.log_operation(context)

        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)

        all_events = await store.get_events_since(
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        log_events = [e for e in all_events if e.event_type == "OperationLoggedEvent"]

        assert len(log_events) == 1

    @pytest.mark.asyncio
    async def test_log_operation_with_additional_data(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test logging operation with additional data."""
        context = OperationContext(
            operation_name="api_call",
            parameters={"endpoint": "/api/test"},
            additional_data={"response_code": 200},
        )

        await orchestrator.log_operation(context)

        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)
        assert store.event_count == 1

    @pytest.mark.asyncio
    async def test_log_operation_different_levels(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test logging operations at different levels."""
        for level in ["INFO", "WARNING", "ERROR"]:
            context = OperationContext(
                operation_name=f"operation_{level.lower()}",
                level=level,
            )
            await orchestrator.log_operation(context)

        store = orchestrator.event_store
        assert isinstance(store, InMemoryEventStore)
        assert store.event_count == 3


class TestOrchestratorWorkflowErrors:
    """Tests for workflow error handling."""

    @pytest.mark.asyncio
    async def test_workflow_executor_exception(self) -> None:
        """Test workflow handles executor exceptions."""
        store = InMemoryEventStore()

        async def error_executor(task: Task[Any]) -> Result[Any]:
            raise RuntimeError("Fatal error")

        orchestrator = Orchestrator(store, task_executor=error_executor)

        tasks = [Task(operation="test", input_data={})]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        # Should have captured the error
        assert results[0].success is False
        assert "Fatal error" in str(results[0].error)

    @pytest.mark.asyncio
    async def test_parallel_workflow_with_mixed_results(self) -> None:
        """Test parallel workflow with some failing tasks."""
        store = InMemoryEventStore()
        call_count = 0

        async def mixed_executor(task: Task[Any]) -> Result[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Task 2 failed")
            return Result(
                task_id=task.id,
                success=True,
                output_data="ok",
            )

        orchestrator = Orchestrator(store, task_executor=mixed_executor)

        tasks = [Task(operation=f"task{i}", input_data={}) for i in range(3)]

        results = await orchestrator.execute_workflow(WorkflowPattern.PARALLEL, tasks)

        assert len(results) == 3
        success_count = sum(1 for r in results if r.success)
        assert success_count == 2


class TestOrchestratorWorkflowFailure:
    """Tests for workflow-level failure handling."""

    @pytest.mark.asyncio
    async def test_workflow_raises_workflow_error_on_event_store_failure(
        self,
    ) -> None:
        """Test workflow raises WorkflowError when event store fails."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        # Make the event store fail during workflow execution
        original_append = store.append
        call_count = 0

        async def failing_append(event: OrchestratorEvent) -> str:
            nonlocal call_count
            call_count += 1
            # Let first few events succeed, then fail
            if call_count <= 2:  # WorkflowStarted + TaskExecuted
                return await original_append(event)
            # Fail on WorkflowCompleted (triggers error path)
            raise RuntimeError("Event store failure")

        store.append = failing_append  # type: ignore[method-assign]

        tasks = [Task(operation="test", input_data={})]

        # The error should propagate up
        with pytest.raises(RuntimeError, match="Event store failure"):
            await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)


class TestOrchestratorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_task_workflow(self) -> None:
        """Test workflow with single task."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        tasks = [Task(operation="single", input_data={"x": 1})]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_large_workflow(self) -> None:
        """Test workflow with many tasks."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        tasks = [Task(operation=f"task{i}", input_data={"id": i}) for i in range(100)]

        results = await orchestrator.execute_workflow(WorkflowPattern.PARALLEL, tasks)

        assert len(results) == 100
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_workflow_with_none_input_data(self) -> None:
        """Test workflow with tasks that have None input data."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        tasks = [Task(operation="test", input_data=None)]
        results = await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_register_multiple_policy_checks(self) -> None:
        """Test registering multiple policy checks."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        orchestrator.register_policy_check("check1", lambda cmd: True)
        orchestrator.register_policy_check("check2", lambda cmd: False)

        assert "check1" in orchestrator._policy_registry
        assert "check2" in orchestrator._policy_registry

    @pytest.mark.asyncio
    async def test_workflow_events_have_correct_aggregate_id(self) -> None:
        """Test all workflow events share the same aggregate_id."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        tasks = [
            Task(operation="task1", input_data={}),
            Task(operation="task2", input_data={}),
        ]

        await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

        all_events = await store.get_events_since(
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        # All events should have the same workflow_id as aggregate_id
        aggregate_ids = {e.aggregate_id for e in all_events}
        assert len(aggregate_ids) == 1

    @pytest.mark.asyncio
    async def test_workflow_exception_handler_logs_and_raises_workflow_error(
        self,
    ) -> None:
        """Test workflow exception handler logs error and raises WorkflowError."""
        store = InMemoryEventStore()
        orchestrator = Orchestrator(store)

        tasks = [Task(operation="test", input_data={})]

        # Mock event_store.append to raise exception during first WorkflowCompletedEvent
        # This happens inside the try block and will trigger the workflow-level exception handler
        original_append = store.append
        first_completed = True

        async def failing_append(event: OrchestratorEvent) -> str:
            nonlocal first_completed
            # Fail on first WorkflowCompletedEvent (success path, inside try block)
            # Allow the exception handler's WorkflowCompletedEvent to succeed
            if event.event_type == "WorkflowCompletedEvent" and first_completed:
                first_completed = False
                raise RuntimeError("Workflow execution failed")
            return await original_append(event)

        store.append = failing_append  # type: ignore[method-assign]

        with patch("arc_saga.orchestrator.core.log_with_context") as mock_log:
            with pytest.raises(WorkflowError) as exc_info:
                await orchestrator.execute_workflow(WorkflowPattern.SEQUENTIAL, tasks)

            # Verify WorkflowError has correct workflow_id
            assert exc_info.value.workflow_id is not None
            assert len(exc_info.value.failed_tasks) == 1

            # Verify error was logged
            error_calls = [
                call
                for call in mock_log.call_args_list
                if len(call[0]) > 1 and call[0][1] == "workflow_failed"
            ]
            assert len(error_calls) > 0

            # Verify log contains required fields
            call_kwargs = error_calls[0][1]
            assert "workflow_id" in call_kwargs
            assert "error_type" in call_kwargs
            assert "error_message" in call_kwargs
            assert "duration_ms" in call_kwargs
