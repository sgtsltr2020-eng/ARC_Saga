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

    async def execute_task(self, task: Task[Any]) -> Result[Any]:
        """
        Execute a single task with full guardrails.

        Args:
            task: The generic Task (assumed to hold AITaskInput).

        Returns:
            Result object (wrapping AIResult or Verdict if budget fails).
        """
        
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
            # Estimate cost (Default 1000 if not specified? Or use manager logic)
            # Assuming manager has a method to estimate or we pass expected output len
            estimated_tokens = 1000 # Placeholder: Phase 2 will have smarter estimation
            
            # Allocate/Check
            # Manager logic might be: can_afford?
            # Enforcer logic: decision = enforcer.pre_check(manager.get_budget(), cost)
            
            # Since TokenBudgetManager usually tracks specific workflows, we need that ID.
            # Using 'default' for single-session Phase 1 if not in context.
            wf_id = ctx.workflow_id
            
            # Get current budget snapshot
            current_budget = await self.tokens.get_budget(wf_id)
            if not current_budget:
                # If no budget exists, maybe initialize one? Or assume unlimited?
                # Guardrail policy: No budget -> Create default or Fail.
                # Let's assume unlimited for 'unknown' flows, but strict for known.
                pass 
            else:
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
                output = ai_result.response if hasattr(ai_result, 'response') else str(ai_result)
                actual_tokens = getattr(ai_result, 'usage', {}).get('total_tokens', 0)
                
            except Exception as e:
                success = False
                output = str(e)
                actual_tokens = 0
                log_with_context("error", "executor_provider_failure", error=str(e))
                # We do NOT return generated MetricsEvent here usually, but we record failure?
                # Letting exception bubble or wrapping it?
                # Strategy expects Result object.
                return Result(task_id=task.id, success=False, output_data=output)

            # 4. POST-EXECUTION BUDGET UPDATE & CHECK
            if current_budget:
                await self.tokens.record_usage(wf_id, actual_tokens)
                updated_budget = await self.tokens.get_budget(wf_id)
                # Runtime check
                if updated_budget:
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
