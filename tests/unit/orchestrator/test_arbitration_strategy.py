import pytest

from saga.orchestrator.core import Task
from saga.orchestrator.patterns import ArbitrationStrategy
from saga.orchestrator.types import AIProvider, AITaskInput, Result


# Mock Executor
async def mock_executor(task: Task) -> Result:
    return Result(
        task_id=task.id,
        success=True,
        output_data=f"Output from {task.operation}",
        duration_ms=10
    )

@pytest.mark.asyncio
async def test_arbitration_strategy_simple_sequence():
    """Test standard execution without dependencies (parallel behaves like batch)."""
    strategy = ArbitrationStrategy()
    tasks = [
        Task(operation="task_a", input_data={"val": 1}),
        Task(operation="task_b", input_data={"val": 2}),
    ]
    
    results = await strategy.execute("wf-1", tasks, mock_executor, "corr-1")
    
    assert len(results) == 2
    assert results[0].success
    assert results[1].success
    assert results[0].output_data == "Output from task_a"

@pytest.mark.asyncio
async def test_arbitration_strategy_dependencies():
    """Test correct ordering and context injection."""
    strategy = ArbitrationStrategy()
    
    # Task A
    task_a = Task(operation="A", input_data={"prompt": "init"})
    
    # Task B depends on A
    # To test context injection, we need input_data to have system_prompt (AITaskInput mimics)
    # But Task is generic. Let's use a dict that won't fail usage, or proper dataclass
    # The code checks `hasattr(task.input_data, "system_prompt")`
    
    input_b = AITaskInput(
        prompt="Review A", 
        model="gpt-4", 
        provider=AIProvider.OPENAI, 
        system_prompt="BaseSys"
    )
    task_b = Task(
        operation="B", 
        input_data=input_b,
        metadata={"context_dependencies": [task_a.id]}
    )
    
    tasks = [task_b, task_a] # Reverse order input to prove topological definition works?
    # Our impl iterates `pending_tasks`. 
    # Pass 1: Task B has dep A (not done). Task A has no deps. -> Run A.
    # Pass 2: A done. Task B has dep A (done). -> Run B.
    
    # We need a custom executor that captures the input to verify context injection
    captured_inputs = {}
    async def capturing_executor(task: Task) -> Result:
        captured_inputs[task.id] = task.input_data
        out = "OutputA" if task.id == task_a.id else "OutputB"
        return Result(task_id=task.id, success=True, output_data=out)

    results = await strategy.execute("wf-dep", tasks, capturing_executor, "corr-dep")
    
    assert len(results) == 2
    
    # Verify Order: A must have run before B (implied by logic, but explicit check?)
    # The results are re-ordered to match input [B, A], so results[0] is B, results[1] is A.
    assert results[1].task_id == task_a.id
    assert results[0].task_id == task_b.id
    
    # Verify Injection
    input_data_b = captured_inputs[task_b.id]
    assert "=== UPSTREAM CONTEXT ===" in input_data_b.system_prompt
    assert "OutputA" in input_data_b.system_prompt
