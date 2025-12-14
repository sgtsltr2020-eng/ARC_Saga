"""
Verification Tests for Phase 1 Core Safety Guardrails.
"""

import uuid
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, Mock

import pytest

from arc_saga.orchestrator.arbitration_context import (
    ArbitrationContext,
    get_arbitration_context,
    set_arbitration_context,
)
from arc_saga.orchestrator.budget_enforcer import BudgetDecision, BudgetEnforcer
from arc_saga.orchestrator.judgement import VerdictStatus
from arc_saga.orchestrator.logging_context import with_trace_logging
from arc_saga.orchestrator.metrics import MetricsAggregator, MetricsEvent

# --- GUARDRAIL 1: ARBITRATION CONTEXT ---

def test_arbitration_context_immutability():
    ctx = ArbitrationContext(uuid.uuid4(), "span-1", "wf-1")
    with pytest.raises(FrozenInstanceError):
        ctx.span_id = "span-2" # type: ignore

def test_arbitration_context_tuple_normalization():
    ctx = ArbitrationContext(uuid.uuid4(), "span-1", "wf-1", ag_agent_ids=["a", "b"]) # type: ignore
    assert isinstance(ctx.ag_agent_ids, tuple)
    assert ctx.ag_agent_ids == ("a", "b")

@pytest.mark.asyncio
async def test_context_propagation():
    # Setup context
    root_id = uuid.uuid4()
    ctx = ArbitrationContext(root_id, "root", "wf-1")
    set_arbitration_context(ctx)
    
    assert get_arbitration_context() == ctx
    
    # Nested async call
    async def inner():
        current = get_arbitration_context()
        assert current == ctx
        assert current.trace_id == root_id
        
        # Create child
        child = current.child_span("child")
        set_arbitration_context(child)
        assert get_arbitration_context().span_id == "child"
        
    await inner()
    # Ensure context restoration (if we used contextvars directly it flows down, 
    # but modifications in inner don't affect outer unless we use tokens correctly)
    # The inner function didn't reset, so in this simple test flow it might stick 
    # if running in same task context. Good test of contextvar hygiene.
    # In real usage we use `with_trace_logging` which handles reset.

def test_logging_context_manager():
    ctx = ArbitrationContext(uuid.uuid4(), "span-1", "wf-1")
    
    # Verify manager sets/resets
    assert get_arbitration_context() is None
    with with_trace_logging(ctx):
        assert get_arbitration_context() == ctx
    assert get_arbitration_context() is None

# --- GUARDRAIL 2: METRICS HOOKS ---

def test_metrics_immutability():
    event = MetricsEvent(100, 10, 10, 0.0, VerdictStatus.APPROVE, "test", "test", "pat")
    with pytest.raises(FrozenInstanceError):
        event.latency_ms = 200 # type: ignore

def test_metrics_aggregation_p95():
    # 20 events: 1..20 ms
    events = [
        MetricsEvent(i, 0, 0, 0.0, "ok", "a", "t", "pattern-A") 
        for i in range(1, 21)
    ]
    agg = MetricsAggregator(events)
    p95 = agg.calculate_p95_latency("pattern-A")
    # 95% of 20 = 19. Index 18 (one less than 19th item? No, ceil formula).
    # ceil(0.95 * 20) = 19. Index 19? 
    # Implementation: matches[ceil(0.95 * n) - 1]
    # ceil(19) - 1 = 18. Match sorted[18] -> 19
    # Wait, 1..20. Index 0 is 1. Index 19 is 20.
    # 95th percentile of 1..20 is 19. (Top 5% is 20).
    assert p95 == 19

def test_metrics_aggregation_rejection_rate():
    events = [
        MetricsEvent(0, 0, 0, 0, VerdictStatus.APPROVE, "a", "t", "pattern-B"),
        MetricsEvent(0, 0, 0, 0, VerdictStatus.REJECT, "a", "t", "pattern-B"),
        MetricsEvent(0, 0, 0, 0, VerdictStatus.BUDGET_EXHAUSTED, "a", "t", "pattern-B"),
        MetricsEvent(0, 0, 0, 0, VerdictStatus.APPROVE, "a", "t", "pattern-B"),
    ]
    agg = MetricsAggregator(events)
    rate = agg.calculate_rejection_rate("pattern-B")
    # 2 rejections / 4 total = 0.5
    assert rate == 0.5

# --- GUARDRAIL 3: BUDGET ENFORCER ---

def test_budget_enforcement_soft_cap():
    enforcer = BudgetEnforcer(soft_cap_ratio=0.8)
    budget = Mock()
    budget.remaining = 150
    budget.total = 1000 # 80% used = remaining 200. We are at 150 (85% used).
    
    # Pre-flight
    decision = enforcer.preflight_check(budget, 10) # 150 - 10 = 140 < 200
    assert decision == BudgetDecision.SOFT_CAP_WARNING
    
    # Runtime
    decision_rt = enforcer.runtime_check(budget)
    assert decision_rt == BudgetDecision.SOFT_CAP_WARNING

def test_budget_enforcement_hard_cap():
    enforcer = BudgetEnforcer()
    budget = Mock()
    budget.remaining = 50
    budget.total = 1000
    
    # Pre-flight cost > remaining
    decision = enforcer.preflight_check(budget, 100)
    assert decision == BudgetDecision.HARD_CAP_EXCEEDED
    
    # Runtime <= 0
    budget.remaining = 0
    decision_rt = enforcer.runtime_check(budget)
    assert decision_rt == BudgetDecision.HARD_CAP_EXCEEDED

# --- INTEGRATION: EXECUTOR ---

@pytest.mark.asyncio
async def test_executor_blocks_on_hard_cap():
    from arc_saga.orchestrator.executor import RegistryAwareTaskExecutor
    from arc_saga.orchestrator.types import Task
    
    router = Mock()
    budget_logic = BudgetEnforcer()
    token_manager = Mock()
    token_manager.get_budget = AsyncMock(return_value=Mock(remaining=50, total=100))
    token_manager.allocatetokens = AsyncMock() # Not used directly in executor yet, just get_budget logic
    
    executor = RegistryAwareTaskExecutor(router, budget_logic, token_manager)
    
    # Assume preflight estimate is 1000 (default in executor placeholder)
    task = Task("test_op", input_data="test")
    
    # This should fail hard cap (1000 > 50)
    result = await executor.execute_task(task)
    
    assert result.success is True # It returns a Result wrapper
    # But output should be Verdict with BUDGET_EXHAUSTED
    assert result.output_data.status == VerdictStatus.BUDGET_EXHAUSTED
    assert "Hard Budget Cap Hit" in result.output_data.rationale
    
    # Ensure router was NOT called
    router.route.assert_not_called()
