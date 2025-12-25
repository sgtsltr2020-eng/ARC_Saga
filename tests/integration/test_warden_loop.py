from unittest.mock import AsyncMock, patch

import pytest

from saga.core.mimiry import OracleResponse
from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.warden import Warden, WardenProposal
from saga.orchestrator.core import InMemoryEventStore, Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_delegates_to_warden_facade():
    """
    Verify that Orchestrator.process_natural_language_command calls Warden.solve_request
    and returns the result.
    """
    # Setup
    event_store = InMemoryEventStore()
    orchestrator = Orchestrator(event_store)

    # Mock the Warden instance inside orchestrator
    mock_warden = AsyncMock(spec=Warden)
    orchestrator.warden = mock_warden

    # Setup mock return
    expected_result = {
        "status": "success",
        "tasks_completed": 1,
        "artifacts": [{"code": "print('hello')"}]
    }
    mock_warden.solve_request.return_value = expected_result
    mock_warden.initialize.return_value = None

    # Execute
    result = await orchestrator.process_natural_language_command(
        command="Create a hello world script",
        user_context={"budget": 10},
        trace_id="test-trace-123"
    )

    # Verify
    assert result == expected_result
    mock_warden.initialize.assert_awaited_once()
    mock_warden.solve_request.assert_awaited_once()

    # Check arguments passed to solve_request
    call_args = mock_warden.solve_request.call_args
    assert "User Request: Create a hello world script" in call_args.kwargs["user_input"]
    assert "Optimization Hints" in call_args.kwargs["user_input"] # Verifies optimization step
    assert call_args.kwargs["trace_id"] == "test-trace-123"

@pytest.mark.asyncio
async def test_warden_solve_request_flow():
    """
    Verify Warden.solve_request executes the full Plan-Execute loop.
    Mocking Mimiry and Agent execution to isolate the flow logic.
    """
    # Setup Warden with mocks
    with patch("saga.core.warden.Mimiry") as MockMimiry, \
         patch("saga.core.warden.CodingAgent") as MockAgent, \
         patch("saga.core.warden.CodexIndexClient") as MockCodex:

        warden = Warden()
        warden.mimiry = AsyncMock() # Mock the instance
        warden.llm_client = AsyncMock() # Mock LLM
        warden.task_store = AsyncMock() # Mock Store

        # 1. Mock receive_proposal (Planning Phase)
        # We'll rely on the real `receive_proposal` but mock the internal calls (mimiry, decompose)
        # Actually, let's mock receive_proposal to focus on the loop logic
        mock_graph = TaskGraph()
        task = Task(id="t1", description="do it", weight="simple")
        mock_graph.add_task(task)

        proposal = WardenProposal(
            decision="approved",
            task_graph=mock_graph,
            mimiry_guidance=OracleResponse(
                canonical_answer="Go ahead",
                severity="INFO",
                violations_detected=[],
                confidence_score=1.0
            )
        )
        warden.receive_proposal = AsyncMock(return_value=proposal)

        # 2. Mock execute_task (Execution Phase)
        warden.execute_task = AsyncMock(return_value=[{"status": "success"}])

        # Action behavior: execute_task should mark task as done (simulating side effect)
        async def side_effect(t):
            t.status = "done"
            return [{"status": "success"}]
        warden.execute_task.side_effect = side_effect

        # Execute
        result = await warden.solve_request(
            user_input="Do something",
            context={},
            trace_id="trace-abc"
        )

        # Verify
        assert result["status"] == "success"
        assert result["tasks_completed"] == 1
        warden.receive_proposal.assert_awaited_once()
        warden.execute_task.assert_awaited()
