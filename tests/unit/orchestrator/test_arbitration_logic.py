from unittest.mock import AsyncMock, Mock

import pytest

from arc_saga.orchestrator.errors import BudgetExceededError
from arc_saga.orchestrator.judgement import Verdict, VerdictStatus
from arc_saga.orchestrator.patterns import ArbitrationStrategy
from arc_saga.orchestrator.types import AIProvider, AITaskInput, Result, Task


@pytest.fixture
def mock_budget_manager():
    manager = Mock()
    manager.allocate_tokens = AsyncMock(return_value={"task-1": 100})
    # Mock budget object
    budget = Mock()
    budget.remaining = 500
    budget.total = 1000
    manager.check_budget = AsyncMock(return_value=budget)
    return manager

def test_verdict_structure():
    v = Verdict(
        status=VerdictStatus.REJECT,
        rationale="Logic flaw",
        discrepancies=["Review A says X, Review B says Y"],
        required_changes=["Fix loop"]
    )
    assert v.is_blocking()
    assert v.status == "reject"

@pytest.mark.asyncio
async def test_arbitration_budget_pass(mock_budget_manager):
    strategy = ArbitrationStrategy(token_budget_manager=mock_budget_manager)
    task = Task("op", input_data=AITaskInput("p", "m", AIProvider.OPENAI))
    
    # Execute with sufficient budget (100 cost < 500 remaining)
    async def executor(t): return Result(t.id, True)
    
    results = await strategy.execute("wf-1", [task], executor, "corr-1")
    assert len(results) == 1

@pytest.mark.asyncio
async def test_arbitration_budget_exceeded(mock_budget_manager):
    # Setup low budget
    budget = Mock()
    budget.remaining = 50 
    budget.total = 1000
    mock_budget_manager.check_budget = AsyncMock(return_value=budget)
    # Allocation returns 100
    
    strategy = ArbitrationStrategy(token_budget_manager=mock_budget_manager)
    task = Task("op", input_data=AITaskInput("p", "m", AIProvider.OPENAI))

    async def executor(t): return Result(t.id, True)

    with pytest.raises(BudgetExceededError) as exc:
        await strategy.execute("wf-fail", [task], executor, "corr-2")
    
    assert "Workflow wf-fail requires 100 tokens" in str(exc.value)
