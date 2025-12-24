"""
End-to-End Integration Tests (Mocked Mimiry)
============================================
"""

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_crud_decomposition_to_approval(integration_warden, conflicting_agents):
    """Full flow: Proposal → Decompose → Conflict → Resolve → Approve."""
    # Step 1: Receive proposal
    proposal = await integration_warden.receive_proposal(
        saga_request="Create user CRUD endpoints",
        context={"budget": 100},
        trace_id="int-test-001"
    )

    assert proposal.decision == "approved"
    # Using 'task_graph' instead of 'todo_list' as per new implementation
    assert proposal.task_graph is not None
    tasks = proposal.task_graph.get_all_tasks()
    assert len(tasks) == 4  # POST/GET/PATCH/DELETE

    # Step 2: Mock agent conflict on first task
    task = tasks[0]  # POST /users/
    resolution = await integration_warden.resolve_via_mimiry(conflicting_agents, task)

    assert resolution.agents_in_alignment == ["AgentB"]
    assert "structured logging" in resolution.canonical_approach

    # Step 3: Enforce canonical
    await integration_warden.enforce_canonical(task, resolution)

    assert task.status == "done"
    assert task.mimiry_measurement["canonical_approach"] == resolution.canonical_approach

@pytest.mark.asyncio
@pytest.mark.integration
async def test_budget_escalation(integration_warden):
    """Budget exceed → warning but proceed (user decides)."""
    proposal = await integration_warden.receive_proposal(
        saga_request="Create user CRUD endpoints",
        context={"budget": 50},  # Cost is 75, so 75 > 50 triggers warning
        trace_id="int-test-002"
    )

    assert proposal.decision == "approved"  # Warns but doesn't block
    assert proposal.estimated_cost > 50  # Triggers log warning
