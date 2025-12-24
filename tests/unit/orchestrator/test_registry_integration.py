"""
Integration test for Registry-Aware Orchestration.

Verifies:
Orchestrator -> RegistryAwareTaskExecutor -> ProviderRouter -> EngineRegistry -> IReasoningEngine
"""
import pytest

from saga.orchestrator.budget_enforcer import BudgetEnforcer
from saga.orchestrator.core import Orchestrator, WorkflowPattern
from saga.orchestrator.engine_registry import EngineRegistry
from saga.orchestrator.events import InMemoryEventStore
from saga.orchestrator.executor import RegistryAwareTaskExecutor
from saga.orchestrator.protocols import IReasoningEngine
from saga.orchestrator.provider_router import ProviderRouter, RoutingRule
from saga.orchestrator.token_manager import (
    LocalTokenEstimator,
    TokenBudget,
    TokenBudgetManager,
)
from saga.orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITaskInput,
    Task,
)


class MockEngine(IReasoningEngine):
    async def reason(self, task):
        return AIResult(
            task_id=task.id,
            success=True,
            output_data=AIResultOutput(
                response="Mock response",
                tokens_used=10,
                prompt_tokens=5,
                completion_tokens=5,
                provider=task.input_data.provider,
                model="mock-model"
            )
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_registry_aware_orchestration():
    # 1. Setup Registry
    registry = EngineRegistry()
    mock_engine = MockEngine()
    registry.register(AIProvider.OPENAI, mock_engine)

    # 2. Setup Router
    rules = [
        RoutingRule(
            task_types={"ai_completion"},
            ordered_providers=[AIProvider.OPENAI]
        )
    ]
    router = ProviderRouter(rules=rules, registry=registry)

    # 3. Setup Budget Enforcer and Token Manager
    budget_enforcer = BudgetEnforcer(soft_cap_ratio=0.8)
    token_event_store = InMemoryEventStore()
    estimator = LocalTokenEstimator()
    budget = TokenBudget(total=10000, remaining=10000)
    token_manager = TokenBudgetManager(
        estimator=estimator,
        budget=budget,
        event_store=token_event_store
    )

    # 4. Setup Executor with required dependencies
    executor = RegistryAwareTaskExecutor(
        provider_router=router,
        budget_enforcer=budget_enforcer,
        token_manager=token_manager
    )
    print(f"Is executor callable? {callable(executor)}")
    print(f"Executor type: {type(executor)}")
    print(f"Has __call__? {callable(executor)}")

    # 5. Setup Orchestrator
    event_store = InMemoryEventStore()
    orchestrator = Orchestrator(
        event_store=event_store,
        task_executor=executor,
    )

    # 6. Create Task
    task_input = AITaskInput(
        prompt="Test",
        model="gpt-4",
        provider=AIProvider.OPENAI
    )
    task = Task(
        operation="ai_completion",
        input_data=task_input
    )

    # 7. Execute
    results = await orchestrator.execute_workflow(
        WorkflowPattern.SEQUENTIAL,
        [task]
    )

    # 8. Verify
    print(f"Results: {results}")
    if results:
        print(f"Result 0 success: {results[0].success}")
        print(f"Result 0 output: {results[0].output_data}")
    assert len(results) == 1
    assert results[0].success
    assert results[0].output_data.response == "Mock response"
    assert results[0].output_data.provider == AIProvider.OPENAI

