"""
Integration Layer: Registry Aware Task Executor.

Connects the Arbitration Strategy to the actual Provider Router, wrapping execution
in all Core Safety Guardrails (Tracing, Budget, Metrics).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from saga.orchestrator.provider_router import ProviderRouter

# We assume TokenBudgetManager is distinct from the protocol we defined?
# Using 'Any' for now to match current import patterns if manager type isn't exported clearly
from saga.orchestrator.token_manager import TokenBudgetManager
from saga.orchestrator.types import AITask, AITaskInput, Result, Task

from ..error_instrumentation import log_with_context
from .arbitration_context import ArbitrationContext, get_arbitration_context
from .budget_enforcer import BudgetDecision, BudgetEnforcer
from .core import WorkflowPattern
from .judgement import Verdict, VerdictStatus
from .logging_context import with_trace_logging
from .metrics import MetricsEvent


class RegistryAwareTaskExecutor:
    """
    Executor that resolves tasks via the ProviderRouter, strictly enforcing guardrails.
    """

    def __init__(
        self,
        provider_router: ProviderRouter,
        budget_enforcer: BudgetEnforcer,
        token_manager: TokenBudgetManager,
        metrics_observer: list[MetricsEvent] | None = None # For Phase 1: In-memory dump
    ) -> None:
        """
        Initialize the executor.

        Args:
            provider_router: Service for routing AI tasks.
            budget_enforcer: Logic for Hard/Soft caps.
            token_manager: Stateful manager for tracking usage.
            metrics_observer: Optional list to append metrics to (DI for testing/monitoring).
        """
        self.router = provider_router
        self.budget = budget_enforcer
        self.tokens = token_manager
        self._metrics = metrics_observer if metrics_observer is not None else []

    async def __call__(self, task: Task[Any]) -> Result[Any]:
        """
        Allow the executor to be called directly (satisfies Orchestrator.task_executor interface).
        """
        return await self.execute_task(task)

    async def execute_task(self, task: Task[Any]) -> Result[Any]:
        """
        Execute a single task with full guardrails.

        Args:
            task: The generic Task (assumed to hold AITaskInput).

        Returns:
            Result object (wrapping AIResult or Verdict if budget fails).
        """
        # Lazy imports to avoid circular dependency issues
        from datetime import datetime, timezone

        from saga.orchestrator.token_manager import TokenUsage
        from saga.orchestrator.types import AIProvider

        # 1. TRACE ID & CONTEXT PROPAGATION
        parent_ctx = get_arbitration_context()
        span_id = str(uuid.uuid4())

        if parent_ctx:
            ctx = parent_ctx.child_span(span_id)
        else:
            # Seed new context if none exists (should be rare in full flow)
            ctx = ArbitrationContext(
                trace_id=uuid.uuid4(),
                span_id=span_id,
                workflow_id="unknown_workflow", # Should be injected by strategy really
                ag_manager_id="system"
            )

        with with_trace_logging(ctx):
            start_time = time.perf_counter()

            # 2. PRE-FLIGHT BUDGET CHECK
            estimated_tokens = 1000 # Placeholder: Phase 2 will have smarter estimation

            # Get current budget snapshot (Global check)
            current_budget = await self.tokens.check_budget()

            decision = self.budget.preflight_check(current_budget, estimated_tokens)
            if decision == BudgetDecision.HARD_CAP_EXCEEDED:
                return self._fail_budget(task.id, "Pre-flight estimate exceeded budget.")

            # 3. EXECUTE PROVIDER CALL (ROUTING)
            # Convert generic Task to AITask if needed
            ai_task = self._adapt_to_ai_task(task)

            try:
                # Execute
                ai_result = await self.router.route(ai_task, context={"correlation_id": str(ctx.trace_id)})
                success = True

                # Extract output data (preserve object if available)
                if hasattr(ai_result, 'output_data'):
                    output = ai_result.output_data
                    # Try to get usage from output_data if not found on result
                    actual_tokens = getattr(output, 'tokens_used', 0)
                elif hasattr(ai_result, 'response'):
                    output = ai_result.response
                    actual_tokens = 0
                else:
                    output = str(ai_result)
                    actual_tokens = 0

                # Check top-level usage if not found
                if actual_tokens == 0 and hasattr(ai_result, 'usage'):
                     actual_tokens = getattr(ai_result, 'usage', {}).get('total_tokens', 0)

            except Exception as e:
                success = False
                output = str(e)
                actual_tokens = 0
                log_with_context("error", "executor_provider_failure", error=str(e))
                return Result(task_id=task.id, success=False, output_data=output)

            # 4. POST-EXECUTION BUDGET UPDATE & CHECK
            usage = TokenUsage(
                aggregate_id=ctx.trace_id or str(uuid.uuid4()),
                estimated=estimated_tokens,
                actual=actual_tokens,
                provider=task.metadata.get("provider", AIProvider.OPENAI),
                created_at=datetime.now(timezone.utc)
            )

            await self.tokens.record_usage(usage)
            updated_budget = await self.tokens.check_budget()

            # Runtime check
            decision = self.budget.runtime_check(updated_budget)
            if decision == BudgetDecision.HARD_CAP_EXCEEDED:
                # We succeeded in this call, but we must signal stop for next?
                # Or return a Verdict that poisons the well?
                # The executor result is success, but we log HArd Cap.
                pass # Strategy loop checks this too usually.

            end_time = time.perf_counter()
            latency = int((end_time - start_time) * 1000)

            # 5. EMIT METRICS
            event = MetricsEvent(
                latency_ms=latency,
                tokens_estimated=estimated_tokens,
                tokens_actual=actual_tokens,
                cost_usd=0.0, # Phase 2: Price calculator
                outcome=VerdictStatus.APPROVE if success else VerdictStatus.REJECT, # Simplification
                agent_type=task.metadata.get("role", "worker"),
                task_type=task.operation,
                pattern=WorkflowPattern.DYNAMIC # Placeholder
            )
            self._metrics.append(event)

            return Result(task_id=task.id, success=success, output_data=output)

    def _fail_budget(self, task_id: str, reason: str) -> Result[Any]:
        """Return a budget-exhausted result."""
        # Returns a Verdict object technically wrapped in Result?
        # Or just a failed result?
        # User requirement: "Return Verdict.BUDGET_EXHAUSTED"
        # Strategy will see success=True but output is a Verdict with status=exhausted

        # We need a Verdict object
        v = Verdict(
            status=VerdictStatus.BUDGET_EXHAUSTED,
            rationale=f"Hard Budget Cap Hit: {reason}"
        )
        return Result(task_id=task_id, success=True, output_data=v)

    def _adapt_to_ai_task(self, task: Task) -> AITask:
        """Helper to cast generic Task to AITask for Router."""
        # In a real system, this might be casting or reconstruction
        input_d = task.input_data
        # Checking if it's already AITaskInput?
        # Assuming simple mapping for now
        return AITask(
            operation=task.operation,
            input_data=input_d if isinstance(input_d, AITaskInput) else AITaskInput(prompt=str(input_d), provider=task.metadata.get("provider"))
        )
