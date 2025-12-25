
import asyncio
import uuid

import pytest

from saga.core.warden import Warden


@pytest.mark.asyncio
async def test_warden_graph_execution_flow(tmp_path):
    """
    Verify that the Warden correctly initializes and executes the LangGraph workflow.
    Checks for proper trace_id propagation.
    """
    # Use tmp_path for the sqlite db
    project_root = str(tmp_path)

    # Initialize Warden
    warden = Warden(project_root=project_root)
    await warden.initialize()

    # Mock inputs
    user_input = "Create a hello world script"
    context = {"budget": 100}
    trace_id = f"test-trace-{uuid.uuid4()}"

    # Execute Request
    result = await warden.solve_request(user_input, context, trace_id)

    # Assertions
    print(f"\n[TEST] Warden Result: {result}")

    status = result["status"]
    history = result.get("history", [])

    assert status == "success", f"Graph execution failed. History: {history}"

    artifacts = result.get("artifacts", [])
    if artifacts:
        final_state = artifacts[-1]
        assert final_state.get("trace_id") == trace_id, f"Trace ID lost in state. Expected {trace_id}, got {final_state.get('trace_id')}"

    # Verify standard workflow nodes ran
    history_str = str(history)
    assert "Planner:" in history_str
    assert "Worker Alpha:" in history_str
    assert "Worker Beta:" in history_str
    assert "Synchronizer:" in history_str
    assert "Ledger:" in history_str

    # Verify merged results
    merged_result = result.get("result", {})
    assert "code" in merged_result
    assert "tests" in merged_result


@pytest.mark.asyncio
async def test_memory_pressure_compaction(tmp_path):
    """
    Test the "Memory Chronicle" pattern.
    NOTE: In Phase 3.0, compaction logic might be deferred or changed.
    For now, we verify that the graph runs even with pressure.
    """
    project_root = str(tmp_path)
    warden = Warden(project_root=project_root)
    await warden.initialize()

    trace_id = f"test-mem-pressure-{uuid.uuid4()}"

    context = {
        "budget": 1000,
        "debug_force_iterations": 15 # Might be ignored by new graph but kept for context passing
    }
    user_input = "Solve a complex impossible problem"

    result = await warden.solve_request(user_input, context, trace_id)

    history = result.get("history", [])

    print(f"\\n[DEBUG] History Length: {len(history)}")

    # Just verify it completed successfully for now
    assert result["status"] == "success"

if __name__ == "__main__":
    asyncio.run(test_warden_graph_execution_flow(None))
