from decimal import Decimal

import pytest

from arc_saga.orchestrator.cost_models import CostProfile, CostProfileRegistry
from arc_saga.orchestrator.cost_optimizer import CostOptimizer
from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry
from arc_saga.orchestrator.errors import BudgetExceededRoutingError
from arc_saga.orchestrator.provider_router import ProviderRouter
from arc_saga.orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    AITaskInput,
)


class StubEngine:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    async def reason(self, task: AITask) -> AIResult:
        output = AIResultOutput(
            response="ok",
            tokens_used=20,
            prompt_tokens=10,
            completion_tokens=10,
            provider=self.provider,
            model=task.input_data.model,
            cost_usd=Decimal("0.0"),
        )
        return AIResult(task_id=task.id, success=True, output_data=output)


@pytest.fixture(autouse=True)
def reset_registry():
    ReasoningEngineRegistry.clear()
    yield
    ReasoningEngineRegistry.clear()


@pytest.fixture(autouse=True)
def reset_optimizer(monkeypatch):
    from arc_saga.orchestrator import cost_optimizer

    cost_optimizer.CostOptimizer._instance = None
    monkeypatch.delenv("SAGA_COST_DISABLE", raising=False)
    monkeypatch.delenv("SAGA_COST_MAX_USD", raising=False)
    yield
    cost_optimizer.CostOptimizer._instance = None


@pytest.fixture
def restore_profiles():
    original = CostProfileRegistry.all()
    yield
    for provider, profile in original.items():
        CostProfileRegistry.update(provider, profile)


@pytest.mark.asyncio
async def test_router_with_optimizer_selects_cheapest(monkeypatch, restore_profiles):
    cheap = CostProfile(
        provider=AIProvider.GROQ,
        cost_per_1k=Decimal("0.0005"),
        latency_p95_ms=120.0,
        quality=0.9,
    )
    pricey = CostProfile(
        provider=AIProvider.OPENAI,
        cost_per_1k=Decimal("0.01"),
        latency_p95_ms=120.0,
        quality=0.9,
    )
    CostProfileRegistry.update(AIProvider.GROQ, cheap)
    CostProfileRegistry.update(AIProvider.OPENAI, pricey)
    monkeypatch.setenv("SAGA_COST_STRATEGY", "CHEAPEST")

    ReasoningEngineRegistry.register(AIProvider.GROQ, StubEngine(AIProvider.GROQ))
    ReasoningEngineRegistry.register(AIProvider.OPENAI, StubEngine(AIProvider.OPENAI))

    router = ProviderRouter(
        rules=[],
        default_order=[AIProvider.OPENAI, AIProvider.GROQ],
        optimizer=CostOptimizer(),
    )
    task = AITask(
        operation="chat_completion",
        input_data=AITaskInput(
            prompt="pick cheapest",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        ),
    )

    result = await router.route(task)

    assert result.success is True
    assert result.output_data is not None
    assert result.output_data.provider == AIProvider.GROQ


@pytest.mark.asyncio
async def test_router_budget_exceeded(monkeypatch):
    long_prompt = "x" * 2000
    task = AITask(
        operation="chat_completion",
        input_data=AITaskInput(
            prompt=long_prompt,
            model="gpt-4",
            provider=AIProvider.OPENAI,
        ),
    )
    monkeypatch.setenv("SAGA_COST_MAX_USD", "0.00001")
    monkeypatch.setenv("SAGA_COST_ENFORCE_HARD_LIMITS", "true")

    router = ProviderRouter(
        rules=[],
        default_order=[AIProvider.OPENAI, AIProvider.GROQ],
        optimizer=CostOptimizer(),
    )

    with pytest.raises(BudgetExceededRoutingError):
        await router.route(task)
