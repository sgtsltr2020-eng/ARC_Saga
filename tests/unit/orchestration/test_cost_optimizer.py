from decimal import Decimal

import pytest

from arc_saga.orchestrator.cost_models import CostProfile, CostProfileRegistry
from arc_saga.orchestrator.cost_optimizer import CostOptimizer
from arc_saga.orchestrator.errors import BudgetExceededRoutingError
from arc_saga.orchestrator.types import AIProvider, AITask, AITaskInput


@pytest.fixture(autouse=True)
def reset_optimizer_singleton(monkeypatch):
    from arc_saga.orchestrator import cost_optimizer

    cost_optimizer.CostOptimizer._instance = None
    yield
    cost_optimizer.CostOptimizer._instance = None


@pytest.fixture
def task():
    return AITask(
        operation="chat_completion",
        input_data=AITaskInput(
            prompt="hello world",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        ),
    )


@pytest.fixture
def restore_profiles():
    original = CostProfileRegistry.all()
    yield
    for provider, profile in original.items():
        CostProfileRegistry.update(provider, profile)


def test_rank_cheapest_prefers_lower_cost(task, monkeypatch, restore_profiles):
    cheap_profile = CostProfile(
        provider=AIProvider.GROQ,
        cost_per_1k=Decimal("0.0005"),
        latency_p95_ms=120.0,
        quality=0.9,
    )
    costly_profile = CostProfile(
        provider=AIProvider.OPENAI,
        cost_per_1k=Decimal("0.01"),
        latency_p95_ms=120.0,
        quality=0.9,
    )
    CostProfileRegistry.update(AIProvider.GROQ, cheap_profile)
    CostProfileRegistry.update(AIProvider.OPENAI, costly_profile)
    monkeypatch.setenv("SAGA_COST_STRATEGY", "CHEAPEST")

    optimizer = CostOptimizer()
    ordered = optimizer.rank_providers(task, [AIProvider.OPENAI, AIProvider.GROQ])

    assert ordered[0] == AIProvider.GROQ


def test_budget_exceeded_raises(task, monkeypatch):
    # Force a tiny budget to trigger BudgetExceededRoutingError
    long_prompt = "x" * 2000
    task_long = AITask(
        operation="chat_completion",
        input_data=AITaskInput(
            prompt=long_prompt,
            model="gpt-4",
            provider=AIProvider.OPENAI,
        ),
    )
    monkeypatch.setenv("SAGA_COST_MAX_USD", "0.00001")
    monkeypatch.setenv("SAGA_COST_ENFORCE_HARD_LIMITS", "true")
    optimizer = CostOptimizer()

    with pytest.raises(BudgetExceededRoutingError):
        optimizer.rank_providers(task_long, [AIProvider.OPENAI, AIProvider.GROQ])


def test_disabled_returns_original_order(task, monkeypatch):
    monkeypatch.setenv("SAGA_COST_DISABLE", "true")
    optimizer = CostOptimizer()

    ordered = optimizer.rank_providers(task, [AIProvider.OPENAI, AIProvider.GROQ])

    assert ordered == [AIProvider.OPENAI, AIProvider.GROQ]


def test_fastest_prefers_low_latency(task, monkeypatch, restore_profiles):
    slow = CostProfile(
        provider=AIProvider.PERPLEXITY,
        cost_per_1k=Decimal("0.000"),
        latency_p95_ms=900.0,
        quality=0.85,
    )
    fast = CostProfile(
        provider=AIProvider.GROQ,
        cost_per_1k=Decimal("0.001"),
        latency_p95_ms=120.0,
        quality=0.84,
    )
    CostProfileRegistry.update(AIProvider.PERPLEXITY, slow)
    CostProfileRegistry.update(AIProvider.GROQ, fast)
    monkeypatch.setenv("SAGA_COST_STRATEGY", "FASTEST")

    optimizer = CostOptimizer()
    ordered = optimizer.rank_providers(task, [AIProvider.PERPLEXITY, AIProvider.GROQ])

    assert ordered[0] == AIProvider.GROQ
