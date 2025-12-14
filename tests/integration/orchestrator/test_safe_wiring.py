
"""
Integration Test: Safe Execution Wiring.

Verifies that the RegistryAwareTaskExecutor can be injected into the Orchestrator
and that it correctly enforces all Phase 1 Guardrails (Tracing, Budget, Metrics).
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from arc_saga.orchestrator.budget_enforcer import BudgetEnforcer
from arc_saga.orchestrator.core import Orchestrator, Task, WorkflowPattern
from arc_saga.orchestrator.events import IEventStore
from arc_saga.orchestrator.executor import RegistryAwareTaskExecutor
from arc_saga.orchestrator.judgement import VerdictStatus
from arc_saga.orchestrator.metrics import MetricsEvent
from arc_saga.orchestrator.token_manager import TokenBudgetManager
from arc_saga.orchestrator.types import AIProvider, AITaskInput


# Mock Event Store
class MockEventStore(IEventStore):
    async def append(self, event): pass
    async def get_events(self, aggregate_id: str) -> list[Any]: return []
    async def get_events_since(self, timestamp: datetime) -> list[Any]: return []

@pytest.mark.asyncio
async def test_safe_execution_flow():
    # 1. SETUP DEPENDENCIES
    event_store = MockEventStore()
    
    # Provider Router (Mocked)
    router = Mock()
    router.route = AsyncMock(return_value=MagicMock(
        response="Mock AI Response",
        usage={"total_tokens": 50},
        success=True
    ))

    # Budget Enforcer & Manager
    budget_enforcer = BudgetEnforcer(soft_cap_ratio=0.8)
    token_manager = Mock(spec=TokenBudgetManager)
    # Orchestrator pre-flight check mock
    token_manager.allocate_tokens = AsyncMock(return_value={"task-1": 100})
    token_manager.check_budget = AsyncMock(return_value=Mock(remaining=5000, total=10000))
    # Executor runtime usage recording
    token_manager.record_usage = AsyncMock()
    # Executor budget fetch (Executor uses get_budget, we mocked check_budget for orchestrator? 
    # Let's check executor code: it calls self.tokens.get_budget(wf_id))
    token_manager.get_budget = AsyncMock(return_value=Mock(remaining=5000, total=10000))

    # Metrics Observer (to capture emitted metrics)
    metrics_log: list[MetricsEvent] = []

    # Safe Executor
    safe_executor = RegistryAwareTaskExecutor(
        provider_router=router,
        budget_enforcer=budget_enforcer,
        token_manager=token_manager,
        metrics_observer=metrics_log
    )

    # 2. INJECT INTO ORCHESTRATOR
    orchestrator = Orchestrator(
        event_store=event_store,
        token_budget_manager=token_manager,
        task_executor=safe_executor.execute_task  # <--- WIRING HERE
    )

    # 3. DEFINE WORKFLOW
    task_input = AITaskInput(
        prompt="Test Prompt",
        model="gpt-4",
        provider=AIProvider.OPENAI
    )
    task = Task[AITaskInput](
        id="task-1",
        operation="generate",
        input_data=task_input,
        metadata={"role": "worker"}
    )
    
    # 4. EXECUTE
    # Using DYNAMIC pattern (simple execution) or ARBITRATION
    # Arbitration strategy also uses the executor.
    results = await orchestrator.execute_workflow(
        pattern=WorkflowPattern.DYNAMIC,
        tasks=[task]
    )

    # 5. VERIFY
    
    # A. Result Success
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].output_data == "Mock AI Response"
    
    # B. Trace Context (Guardrail 1)
    # The router should have been called with a context containing a trace_id
    # Executor: await self.router.route(ai_task, context={"correlation_id": str(ctx.trace_id)})
    router.route.assert_called_once()
    call_args = router.route.call_args
    # call_args[1] is kwargs. Check 'context' key.
    # Note: route(task, context=...)
    passed_context = call_args.kwargs.get('context') or (call_args[0][1] if len(call_args[0]) > 1 else {})
    assert "correlation_id" in passed_context
    trace_id = passed_context["correlation_id"]
    # Check it look like a UUID
    assert len(trace_id) > 10 
    
    # C. Budget Checks (Guardrail 3)
    # Orchestrator pre-flight
    token_manager.allocate_tokens.assert_called()
    # Executor pre-flight
    # token_manager.get_budget call count should be >= 1 (Executor calls it)
    assert token_manager.get_budget.call_count >= 1
    # Usage recording
    token_manager.record_usage.assert_called()
    
    # D. Metrics Emission (Guardrail 2)
    assert len(metrics_log) == 1
    metric = metrics_log[0]
    assert metric.tokens_actual == 50 # From mock router
    assert metric.outcome == VerdictStatus.APPROVE # Success mapped to Approve
    assert metric.agent_type == "worker"
