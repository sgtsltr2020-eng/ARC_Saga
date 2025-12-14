"""
Integration test for Registry-Aware Orchestration.

Verifies:
Orchestrator -> RegistryAwareTaskExecutor -> ProviderRouter -> EngineRegistry -> IReasoningEngine
"""
import pytest

from arc_saga.orchestrator.core import Orchestrator, WorkflowPattern
from arc_saga.orchestrator.engine_registry import EngineRegistry
from arc_saga.orchestrator.events import InMemoryEventStore
from arc_saga.orchestrator.executor import RegistryAwareTaskExecutor
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.provider_router import ProviderRouter, RoutingRule
from arc_saga.orchestrator.types import (
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
    
    # 3. Setup Executor
    executor = RegistryAwareTaskExecutor(router)
    
    # 4. Setup Orchestrator
    event_store = InMemoryEventStore()
    orchestrator = Orchestrator(
        event_store=event_store,
        task_executor=executor,
    )
    
    # 5. Create Task
    task_input = AITaskInput(
        prompt="Test",
        model="gpt-4",
        provider=AIProvider.OPENAI
    )
    task = Task(
        operation="ai_completion",
        input_data=task_input
    )
    
    # 6. Execute
    results = await orchestrator.execute_workflow(
        WorkflowPattern.SEQUENTIAL,
        [task]
    )
    
    # 7. Verify
    assert len(results) == 1
    assert results[0].success
    assert results[0].output_data.response == "Mock response"
    assert results[0].output_data.provider == AIProvider.OPENAI
