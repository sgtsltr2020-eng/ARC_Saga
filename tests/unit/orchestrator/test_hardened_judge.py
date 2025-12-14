from unittest.mock import AsyncMock, Mock

import pytest

from arc_saga.orchestrator.judgement import (
    VerdictParser,
    VerdictStatus,
    validate_artifacts,
)
from arc_saga.orchestrator.patterns import ArbitrationStrategy
from arc_saga.orchestrator.roles import AgentRole
from arc_saga.orchestrator.types import AIProvider, AITaskInput, Result, Task


# --- Fixtures ---
@pytest.fixture
def mock_budget_manager():
    manager = Mock()
    manager.allocate_tokens = AsyncMock(return_value={"task-1": 10})
    budget = Mock()
    budget.remaining = 1000
    budget.total = 1000
    manager.check_budget = AsyncMock(return_value=budget)
    return manager

@pytest.fixture
def strategy(mock_budget_manager):
    return ArbitrationStrategy(mock_budget_manager, strict_mode=True)

# --- Tests ---

def test_verdict_parser_valid():
    output = '```json\n{"status": "approve", "rationale": "LGTM"}\n```'
    v = VerdictParser.parse(output)
    assert v.status == VerdictStatus.APPROVE
    assert v.rationale == "LGTM"

def test_verdict_parser_malformed_resilient():
    # Bad JSON (trailing comma or similar? standard json fail).
    # Our parser currently just fails safe to REJECT on bad json.
    output = 'Not JSON at all'
    v = VerdictParser.parse(output)
    assert v.status == VerdictStatus.REJECT
    assert "Failed to parse" in v.rationale

def test_input_guard_valid():
    validate_artifacts(["some context"]) # Should not raise

def test_input_guard_empty():
    with pytest.raises(ValueError):
        validate_artifacts([])
    with pytest.raises(ValueError):
        validate_artifacts([""]) # Empty string artifact

@pytest.mark.asyncio
async def test_strict_mode_halt(strategy):
    # Setup: 2 tasks. Task 1 is Judge (returns REJECT). Task 2 is dependent (should be skipped).
    # But wait, our Arbitrator runs in batches. Task 2 isn't started if it depends on 1.
    # To test HALT, we need Task 1 (Judge) to run, and check that *subsequent* independent tasks are cancelled?
    # Or just that the loop breaks.
    
    # Let's say we have Task A (Judge) and Task B (Independent).
    # If Judge fails strict mode, we expect Task B to NOT run if it was pending.
    
    judge_task = Task(
        "judge_op", 
        input_data=AITaskInput("p", "m", AIProvider.OPENAI),
        metadata={"role": AgentRole.JUDGE.value}
    )
    
    other_task = Task("other", input_data={"x": 1})
    
    # Executor mocks
    async def executor(t):
        if t.id == judge_task.id:
            # Return REJECT JSON
            return Result(t.id, True, output_data='{"status": "reject", "rationale": "No Good"}')
        return Result(t.id, True, output_data="ok")

    results = await strategy.execute("wf-strict", [judge_task, other_task], executor, "corr-1")
    
    # Expectation: Only Judge ran. Other task skipped due to halt.
    assert len(results) == 1
    assert results[0].task_id == judge_task.id
    # Assert result output is what we gave
    assert "reject" in str(results[0].output_data)

@pytest.mark.asyncio
async def test_strict_mode_approve_continues(strategy):
    judge_task = Task(
        "judge_op", 
        input_data=AITaskInput("p", "m", AIProvider.OPENAI),
        metadata={"role": AgentRole.JUDGE.value}
    )
    other_task = Task("other", input_data={"x": 1})
    
    async def executor(t):
        if t.id == judge_task.id:
            return Result(t.id, True, output_data='{"status": "approve", "rationale": "Good"}')
        return Result(t.id, True, output_data="ok")

    results = await strategy.execute("wf-ok", [judge_task, other_task], executor, "corr-2")
    
    # Expectation: Both ran
    assert len(results) == 2
