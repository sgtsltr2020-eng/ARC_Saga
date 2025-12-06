"""
Orchestrator Core Module.

Implements the main Orchestrator class for workflow execution using
event-driven CQRS pattern from decision_catalog.md.

Key responsibilities:
- Execute workflows (sequential, parallel, dynamic patterns)
- Enforce policies on commands before execution
- Log operations with structured context
- Emit events for all state changes

Example:
    >>> from arc_saga.orchestrator import Orchestrator, InMemoryEventStore
    >>> from arc_saga.orchestrator import WorkflowPattern, Task
    >>>
    >>> store = InMemoryEventStore()
    >>> orchestrator = Orchestrator(store)
    >>> tasks = [Task(operation="process", input_data={"id": 1})]
    >>> results = await orchestrator.execute_workflow(
    ...     WorkflowPattern.SEQUENTIAL, tasks
    ... )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar
from uuid import uuid4

from ..error_instrumentation import log_with_context
from ..integrations.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
)
from .cost_models import CostSettings
from .cost_optimizer import CostOptimizer
from .provider_router import ProviderRouter, RoutingRule
from .events import (
    IEventStore,
    OperationLoggedEvent,
    OrchestratorEvent,
    PolicyEnforcedEvent,
    TaskExecutedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from .token_manager import TokenBudgetManager, TokenUsage
from .types import (
    AIProvider,
    AIResultOutput,
    AITaskInput,
    Result,
    Task,
    TaskStatus,
)

# Type variables
T = TypeVar("T")  # Task input type
R = TypeVar("R")  # Result output type


def create_cost_aware_router(
    rules: list[RoutingRule],
    default_order: list[AIProvider] | None = None,
) -> ProviderRouter:
    """
    Build a ProviderRouter with optional CostOptimizer (opt-out via env SAGA_COST_DISABLE).

    This is additive and preserves existing behavior when disable flag is set.
    """

    settings = CostSettings()
    optimizer = None if settings.disable else CostOptimizer(settings)
    return ProviderRouter(
        rules=rules,
        default_order=default_order or [],
        optimizer=optimizer,
    )


class WorkflowPattern(str, Enum):
    """
    Workflow execution patterns.

    Attributes:
        SEQUENTIAL: Execute tasks one after another in order
        PARALLEL: Execute all tasks concurrently
        DYNAMIC: Adapt execution based on task results
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class Policy:
    """
    Policy definition for command validation.

    Policies define rules that must be satisfied before
    a command can be executed.

    Attributes:
        name: Policy identifier
        description: Human-readable description
        check_func_name: Name of the check function
        parameters: Policy parameters
        enabled: Whether the policy is active
    """

    name: str
    description: str = ""
    check_func_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class Command:
    """
    Command to be executed through the orchestrator.

    Commands represent user intents that need policy validation.

    Attributes:
        type: Command type identifier
        payload: Command data
        user_id: User initiating the command
        correlation_id: ID for tracing related operations
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    correlation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class PolicyResult:
    """
    Result of policy enforcement.

    Attributes:
        policy_name: Name of the evaluated policy
        allowed: Whether the command is allowed
        reason: Explanation for the decision
        evaluated_at: Timestamp of evaluation
    """

    policy_name: str
    allowed: bool
    reason: str
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class OperationContext:
    """
    Context for operation logging.

    Captures all relevant information about an operation
    for auditing and debugging.

    Attributes:
        operation_name: Name of the operation
        user_id: User performing the operation
        correlation_id: ID for tracing
        parameters: Operation parameters (sanitized)
        level: Log level (INFO, WARNING, ERROR)
        additional_data: Extra context data
    """

    operation_name: str
    user_id: str | None = None
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    parameters: dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"
    additional_data: dict[str, Any] = field(default_factory=dict)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    def __init__(self, message: str, operation: str = "") -> None:
        """
        Initialize OrchestratorError.

        Args:
            message: Error description
            operation: Operation that failed
        """
        self.operation = operation
        super().__init__(f"Orchestrator {operation} failed: {message}")


class WorkflowError(OrchestratorError):
    """Exception raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_id: str,
        failed_tasks: list[str] | None = None,
    ) -> None:
        """
        Initialize WorkflowError.

        Args:
            message: Error description
            workflow_id: ID of the failed workflow
            failed_tasks: List of failed task IDs
        """
        self.workflow_id = workflow_id
        self.failed_tasks = failed_tasks or []
        super().__init__(message, "execute_workflow")


class BudgetExceededError(OrchestratorError):
    """Exception raised when workflow exceeds token budget."""

    def __init__(
        self,
        workflow_id: str,
        requested_tokens: int,
        remaining_tokens: int,
        total_budget: int,
    ) -> None:
        """
        Initialize BudgetExceededError.

        Args:
            workflow_id: ID of the workflow that exceeded budget
            requested_tokens: Tokens requested for the workflow
            remaining_tokens: Tokens remaining in budget
            total_budget: Total token budget
        """
        self.workflow_id = workflow_id
        self.requested_tokens = requested_tokens
        self.remaining_tokens = remaining_tokens
        self.total_budget = total_budget
        message = (
            f"Workflow {workflow_id} requires {requested_tokens} tokens, "
            f"but only {remaining_tokens}/{total_budget} tokens remaining"
        )
        super().__init__(message, "execute_workflow")


class PolicyViolationError(OrchestratorError):
    """Exception raised when policy enforcement fails."""

    def __init__(self, policy_name: str, reason: str) -> None:
        """
        Initialize PolicyViolationError.

        Args:
            policy_name: Name of the violated policy
            reason: Reason for violation
        """
        self.policy_name = policy_name
        self.reason = reason
        super().__init__(
            f"Policy '{policy_name}' violated: {reason}",
            "enforce_policy",
        )


# Task executor type: async function that takes Task and returns Result
TaskExecutor = Callable[[Task[Any]], Awaitable[Result[Any]]]


class Orchestrator:
    """
    Main orchestrator for workflow execution.

    Implements event-driven CQRS pattern with:
    - Event emission on every state change
    - Optional circuit breaker for external task execution
    - Comprehensive structured logging

    Attributes:
        event_store: Event store for persisting events
        circuit_breaker: Optional circuit breaker for resilience

    Example:
        >>> store = InMemoryEventStore()
        >>> orchestrator = Orchestrator(store)
        >>> tasks = [Task(operation="process", input_data={"id": 1})]
        >>> results = await orchestrator.execute_workflow(
        ...     WorkflowPattern.SEQUENTIAL, tasks
        ... )
    """

    def __init__(
        self,
        event_store: IEventStore[OrchestratorEvent],
        circuit_breaker: CircuitBreaker | None = None,
        task_executor: TaskExecutor | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            event_store: Event store for persisting events
            circuit_breaker: Optional circuit breaker for resilience
            task_executor: Optional custom task executor function
            token_budget_manager: Optional token budget manager for budget enforcement
        """
        self._event_store = event_store
        self._circuit_breaker = circuit_breaker
        self._task_executor = task_executor or self._default_task_executor
        self._token_budget_manager = token_budget_manager
        self._policy_registry: dict[str, Callable[[Command], bool]] = {}

        log_with_context(
            "info",
            "orchestrator_initialized",
            has_circuit_breaker=circuit_breaker is not None,
            has_custom_executor=task_executor is not None,
            has_token_budget_manager=token_budget_manager is not None,
        )

    async def execute_workflow(
        self,
        pattern: WorkflowPattern,
        tasks: list[Task[Any]],
        correlation_id: str | None = None,
        memory_tier: str = "standard",
    ) -> list[Result[Any]]:
        """
        Execute a workflow with the specified pattern.

        Args:
            pattern: Execution pattern (SEQUENTIAL, PARALLEL, DYNAMIC)
            tasks: List of tasks to execute
            correlation_id: Optional ID for tracing related operations
            memory_tier: Memory tier for token estimation (default: "standard")

        Returns:
            List of results in the same order as input tasks

        Raises:
            WorkflowError: If workflow execution fails
            BudgetExceededError: If workflow exceeds token budget
            ValueError: If tasks list is empty
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty")

        workflow_id = str(uuid4())
        correlation_id = correlation_id or str(uuid4())
        start_time = time.perf_counter()

        # Cache budget allocation data for post-execution usage recording
        allocations: dict[str, int] | None = None
        total_estimated: int = 0

        # Pre-flight budget check (before workflow execution)
        if self._token_budget_manager is not None:
            # Filter to only AI tasks for budget checking
            # AITask is a type alias, so check if input_data is AITaskInput
            ai_tasks = [
                task
                for task in tasks
                if isinstance(task, Task) and isinstance(task.input_data, AITaskInput)
            ]

            if ai_tasks:
                # Allocate tokens for all AI tasks
                allocations = await self._token_budget_manager.allocate_tokens(
                    ai_tasks, memory_tier
                )
                total_estimated = sum(allocations.values())

                # Check if budget is sufficient
                budget = await self._token_budget_manager.check_budget()
                if total_estimated > budget.remaining:
                    log_with_context(
                        "error",
                        "workflow_budget_exceeded",
                        workflow_id=workflow_id,
                        requested_tokens=total_estimated,
                        remaining_tokens=budget.remaining,
                        total_budget=budget.total,
                        memory_tier=memory_tier,
                    )
                    raise BudgetExceededError(
                        workflow_id=workflow_id,
                        requested_tokens=total_estimated,
                        remaining_tokens=budget.remaining,
                        total_budget=budget.total,
                    )

                # Log successful budget check
                log_with_context(
                    "info",
                    "workflow_budget_check_passed",
                    workflow_id=workflow_id,
                    estimated_tokens=total_estimated,
                    remaining_tokens=budget.remaining,
                    total_budget=budget.total,
                    memory_tier=memory_tier,
                )

        log_with_context(
            "info",
            "workflow_started",
            workflow_id=workflow_id,
            pattern=pattern.value,
            task_count=len(tasks),
            correlation_id=correlation_id,
        )

        # Emit workflow started event
        await self._event_store.append(
            WorkflowStartedEvent(
                aggregate_id=workflow_id,
                pattern=pattern.value,
                task_count=len(tasks),
                task_ids=tuple(t.id for t in tasks),
                correlation_id=correlation_id,
            )
        )

        try:
            # Execute based on pattern
            if pattern == WorkflowPattern.SEQUENTIAL:
                results = await self._execute_sequential(
                    workflow_id, tasks, correlation_id
                )
            elif pattern == WorkflowPattern.PARALLEL:
                results = await self._execute_parallel(
                    workflow_id, tasks, correlation_id
                )
            else:  # DYNAMIC
                results = await self._execute_dynamic(
                    workflow_id, tasks, correlation_id
                )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            success_count = sum(1 for r in results if r.success)
            failed_count = len(results) - success_count
            all_success = failed_count == 0

            # Emit workflow completed event
            await self._event_store.append(
                WorkflowCompletedEvent(
                    aggregate_id=workflow_id,
                    success=all_success,
                    duration_ms=duration_ms,
                    completed_count=success_count,
                    failed_count=failed_count,
                    correlation_id=correlation_id,
                )
            )

            log_with_context(
                "info",
                "workflow_completed",
                workflow_id=workflow_id,
                success=all_success,
                duration_ms=duration_ms,
                completed_count=success_count,
                failed_count=failed_count,
            )

            # Post-execution usage recording (after workflow completes)
            if self._token_budget_manager is not None and allocations is not None:
                # Filter results to AI results only
                # AIResult is a type alias, so check if output_data is AIResultOutput
                ai_results = [
                    r
                    for r in results
                    if isinstance(r, Result)
                    and isinstance(r.output_data, AIResultOutput)
                ]

                # Calculate total actual tokens from successful tasks only
                # Only count tokens from successful tasks with valid output_data.
                # Failed tasks or tasks without output_data are excluded to prevent
                # double-counting and ensure accurate budget tracking.
                total_actual = 0
                primary_provider = AIProvider.OPENAI

                for result in ai_results:
                    # Skip results where output_data is None (failed tasks)
                    if result.output_data is None:
                        continue
                    # Skip results where success is False (explicit failures)
                    if not result.success:
                        continue
                    # Count tokens from successful tasks with valid output_data
                    total_actual += result.output_data.tokens_used
                    # Use first successful task's provider as primary
                    if primary_provider == AIProvider.OPENAI:
                        # Find corresponding task to get provider
                        for task in tasks:
                            if (
                                isinstance(task, Task)
                                and isinstance(task.input_data, AITaskInput)
                                and task.id == result.task_id
                            ):
                                primary_provider = task.input_data.provider
                                break

                # Create TokenUsage event
                usage = TokenUsage(
                    aggregate_id=workflow_id,
                    estimated=total_estimated,  # From cached pre-flight value
                    actual=total_actual,  # From filtered successful results
                    cost_usd=0.0,  # Placeholder for future cost computation
                    provider=primary_provider,
                    created_at=datetime.now(timezone.utc),
                    correlation_id=correlation_id,
                )

                # Record usage (handle event store failures gracefully)
                try:
                    await self._token_budget_manager.record_usage(usage)
                except Exception as e:
                    log_with_context(
                        "error",
                        "token_usage_recording_failed",
                        workflow_id=workflow_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )
                    # Continue - don't crash workflow if usage recording fails

            return results

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Emit failed workflow event
            await self._event_store.append(
                WorkflowCompletedEvent(
                    aggregate_id=workflow_id,
                    success=False,
                    duration_ms=duration_ms,
                    completed_count=0,
                    failed_count=len(tasks),
                    correlation_id=correlation_id,
                )
            )

            log_with_context(
                "error",
                "workflow_failed",
                workflow_id=workflow_id,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            raise WorkflowError(
                str(e),
                workflow_id=workflow_id,
                failed_tasks=[t.id for t in tasks],
            ) from e

    async def _execute_sequential(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        correlation_id: str,
    ) -> list[Result[Any]]:
        """
        Execute tasks sequentially.

        Args:
            workflow_id: Workflow identifier
            tasks: Tasks to execute
            correlation_id: Correlation ID for tracing

        Returns:
            List of results in task order
        """
        results: list[Result[Any]] = []

        for task in tasks:
            result = await self._execute_single_task(workflow_id, task, correlation_id)
            results.append(result)

        return results

    async def _execute_parallel(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        correlation_id: str,
    ) -> list[Result[Any]]:
        """
        Execute tasks in parallel.

        Args:
            workflow_id: Workflow identifier
            tasks: Tasks to execute
            correlation_id: Correlation ID for tracing

        Returns:
            List of results in task order
        """
        # Create coroutines for all tasks
        coroutines = [
            self._execute_single_task(workflow_id, task, correlation_id)
            for task in tasks
        ]

        # Execute all in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=False)

        return list(results)

    async def _execute_dynamic(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        correlation_id: str,
    ) -> list[Result[Any]]:
        """
        Execute tasks dynamically based on results.

        Executes tasks sequentially but stops early if a task fails.

        Args:
            workflow_id: Workflow identifier
            tasks: Tasks to execute
            correlation_id: Correlation ID for tracing

        Returns:
            List of results (may be shorter than tasks if early termination)
        """
        results: list[Result[Any]] = []

        for task in tasks:
            result = await self._execute_single_task(workflow_id, task, correlation_id)
            results.append(result)

            # Stop on failure in dynamic mode
            if not result.success:
                log_with_context(
                    "warning",
                    "dynamic_workflow_early_termination",
                    workflow_id=workflow_id,
                    failed_task_id=task.id,
                    completed_count=len(results),
                    remaining_count=len(tasks) - len(results),
                )
                break

        return results

    async def _execute_single_task(
        self,
        workflow_id: str,
        task: Task[Any],
        correlation_id: str,
    ) -> Result[Any]:
        """
        Execute a single task with circuit breaker if available.

        Args:
            workflow_id: Workflow identifier
            task: Task to execute
            correlation_id: Correlation ID for tracing

        Returns:
            Task result
        """
        start_time = time.perf_counter()

        log_with_context(
            "info",
            "task_execution_started",
            workflow_id=workflow_id,
            task_id=task.id,
            operation=task.operation,
            correlation_id=correlation_id,
        )

        try:
            # Execute through circuit breaker if available
            if self._circuit_breaker:
                try:
                    result = await self._circuit_breaker.call(self._task_executor, task)
                except CircuitBreakerOpenError as e:
                    # Circuit is open, create failure result
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    result = Result[Any](
                        task_id=task.id,
                        success=False,
                        error=str(e),
                        error_type="CircuitBreakerOpenError",
                        duration_ms=duration_ms,
                        status=TaskStatus.FAILED,
                    )
            else:
                result = await self._task_executor(task)

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Emit task executed event
            await self._event_store.append(
                TaskExecutedEvent(
                    aggregate_id=workflow_id,
                    task_id=task.id,
                    operation=task.operation,
                    success=result.success,
                    duration_ms=duration_ms,
                    error=result.error,
                    correlation_id=correlation_id,
                )
            )

            log_with_context(
                "info" if result.success else "warning",
                "task_execution_completed",
                workflow_id=workflow_id,
                task_id=task.id,
                success=result.success,
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Emit failed task event
            await self._event_store.append(
                TaskExecutedEvent(
                    aggregate_id=workflow_id,
                    task_id=task.id,
                    operation=task.operation,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e),
                    correlation_id=correlation_id,
                )
            )

            log_with_context(
                "error",
                "task_execution_failed",
                workflow_id=workflow_id,
                task_id=task.id,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return Result[Any](
                task_id=task.id,
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                status=TaskStatus.FAILED,
            )

    async def _default_task_executor(self, task: Task[Any]) -> Result[Any]:
        """
        Default task executor (placeholder).

        Override with custom executor for actual task execution.

        Args:
            task: Task to execute

        Returns:
            Successful result with input data echoed
        """
        # Default implementation just echoes the input
        return Result[Any](
            task_id=task.id,
            success=True,
            output_data=task.input_data,
            duration_ms=0,
            status=TaskStatus.COMPLETED,
        )

    def register_policy_check(
        self,
        policy_name: str,
        check_func: Callable[[Command], bool],
    ) -> None:
        """
        Register a policy check function.

        Args:
            policy_name: Name of the policy
            check_func: Function that returns True if command is allowed
        """
        self._policy_registry[policy_name] = check_func

        log_with_context(
            "info",
            "policy_registered",
            policy_name=policy_name,
        )

    async def enforce_policy(
        self,
        policy: Policy,
        command: Command,
    ) -> PolicyResult:
        """
        Enforce a policy on a command.

        Args:
            policy: Policy to enforce
            command: Command to validate

        Returns:
            PolicyResult indicating whether command is allowed

        Raises:
            PolicyViolationError: If policy is violated and blocking is enabled
        """
        log_with_context(
            "info",
            "policy_enforcement_started",
            policy_name=policy.name,
            command_type=command.type,
            correlation_id=command.correlation_id,
        )

        # Check if policy is enabled
        if not policy.enabled:
            result = PolicyResult(
                policy_name=policy.name,
                allowed=True,
                reason="Policy is disabled",
            )

            await self._event_store.append(
                PolicyEnforcedEvent(
                    aggregate_id=command.correlation_id,
                    policy_name=policy.name,
                    command_type=command.type,
                    allowed=True,
                    reason="Policy is disabled",
                    correlation_id=command.correlation_id,
                )
            )

            return result

        # Look up check function
        check_func = self._policy_registry.get(policy.check_func_name)

        if check_func is None:
            # No check function registered, allow by default
            result = PolicyResult(
                policy_name=policy.name,
                allowed=True,
                reason="No check function registered",
            )
        else:
            # Execute check function
            try:
                allowed = check_func(command)
                reason = "Policy check passed" if allowed else "Policy check failed"
                result = PolicyResult(
                    policy_name=policy.name,
                    allowed=allowed,
                    reason=reason,
                )
            except Exception as e:
                # Check function failed, deny by default
                result = PolicyResult(
                    policy_name=policy.name,
                    allowed=False,
                    reason=f"Policy check error: {e}",
                )

        # Emit policy enforced event
        await self._event_store.append(
            PolicyEnforcedEvent(
                aggregate_id=command.correlation_id,
                policy_name=policy.name,
                command_type=command.type,
                allowed=result.allowed,
                reason=result.reason,
                correlation_id=command.correlation_id,
            )
        )

        log_with_context(
            "info" if result.allowed else "warning",
            "policy_enforcement_completed",
            policy_name=policy.name,
            command_type=command.type,
            allowed=result.allowed,
            reason=result.reason,
        )

        return result

    async def log_operation(self, context: OperationContext) -> None:
        """
        Log an operation with full context.

        Args:
            context: Operation context to log
        """
        # Log to structured logger
        log_with_context(
            context.level.lower(),
            f"operation_{context.operation_name}",
            operation_name=context.operation_name,
            user_id=context.user_id,
            correlation_id=context.correlation_id,
            parameters=context.parameters,
            **context.additional_data,
        )

        # Emit operation logged event
        import json

        context_data = json.dumps(
            {
                "parameters": context.parameters,
                "additional_data": context.additional_data,
            },
            default=str,
        )

        await self._event_store.append(
            OperationLoggedEvent(
                aggregate_id=context.correlation_id,
                operation_name=context.operation_name,
                context_data=context_data,
                level=context.level,
                correlation_id=context.correlation_id,
            )
        )

    @property
    def event_store(self) -> IEventStore[OrchestratorEvent]:
        """Get the event store."""
        return self._event_store

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Get the circuit breaker."""
        return self._circuit_breaker
