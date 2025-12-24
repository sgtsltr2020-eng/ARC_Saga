"""
Comprehensive Tests for The Warden
===================================

Tests delegation, task decomposition, TaskGraph management, Mimiry integration,
quality gate enforcement.

Author: ARC SAGA Development Team
Date: December 14, 2025
Status: Phase 2 Week 1 - Comprehensive Testing
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from saga.core.mimiry import OracleResponse
from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.warden import (
    Warden,
    WardenProposal,
)


class TestTaskDataclass:
    """Test Task dataclass."""

    def test_task_creation(self) -> None:
        """Test creating a Task."""
        task = Task(
            id="task-001",
            description="Create user endpoint",
            weight="simple",
            budget_allocation=15.0,
            checklist=["Item 1", "Item 2"],
        )

        assert task.id == "task-001"
        assert task.weight == "simple"
        assert task.status == "pending"
        assert len(task.checklist) == 2

    def test_task_default_values(self) -> None:
        """Test Task default values."""
        task = Task(
            id="task-001",
            description="Test task",
            weight="simple",
        )

        assert task.status == "pending"
        assert task.assigned_agent is None
        assert task.budget_allocation == 0.0
        assert task.dependencies == []
        assert task.completed_at is None

    def test_task_to_dict(self, simple_task: Task) -> None:
        """Test Task serialization."""
        task_dict = simple_task.to_dict()

        assert isinstance(task_dict, dict)
        assert task_dict["id"] == simple_task.id
        assert task_dict["weight"] == "simple"
        assert task_dict["status"] == "pending"


class TestTaskGraph:
    """Test TaskGraph functionality (replacing TODOList)."""

    @pytest.fixture
    def empty_graph(self) -> TaskGraph:
        """Provide fresh TaskGraph."""
        return TaskGraph()

    def test_graph_initialization(self, empty_graph: TaskGraph) -> None:
        """Test graph initializes empty."""
        assert empty_graph.get_all_tasks() == []

    def test_add_task(self, empty_graph: TaskGraph) -> None:
        """Test adding task to graph."""
        task = Task(id="t1", description="Test", weight="simple")
        empty_graph.add_task(task)

        assert len(empty_graph.get_all_tasks()) == 1
        assert empty_graph.get_task("t1") == task

    def test_add_dependency(self, empty_graph: TaskGraph) -> None:
        """Test adding dependencies."""
        t1 = Task(id="t1", description="Base", weight="simple")
        t2 = Task(id="t2", description="Dependent", weight="simple", dependencies=["t1"])

        empty_graph.add_task(t1)
        # Should auto-add dependency edge
        empty_graph.add_task(t2)

        assert "t1" in empty_graph.graph.pred["t2"]

    def test_get_ready_tasks(self, empty_graph: TaskGraph) -> None:
        """Test getting tasks ready for execution."""
        t1 = Task(id="t1", description="Base", weight="simple") # No deps
        t2 = Task(id="t2", description="Dependent", weight="simple", dependencies=["t1"])

        empty_graph.add_task(t1)
        empty_graph.add_task(t2)

        ready = empty_graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

        # Mark t1 done
        t1.status = "done"

        ready_next = empty_graph.get_ready_tasks()
        assert len(ready_next) == 1
        assert ready_next[0].id == "t2"

    def test_cycle_detection(self, empty_graph: TaskGraph) -> None:
        """Test detecting circular dependencies."""
        t1 = Task(id="t1", description="A", weight="simple", dependencies=["t2"])
        t2 = Task(id="t2", description="B", weight="simple", dependencies=["t1"])

        empty_graph.add_task(t1)
        empty_graph.add_task(t2)

        assert empty_graph.is_cyclic() is True


class TestWardenInitialization:
    """Test Warden initialization."""

    def test_warden_initialization(self, warden: Warden) -> None:
        """Test Warden initializes correctly."""
        assert warden is not None
        assert hasattr(warden, 'constitution')
        assert hasattr(warden, 'codex_manager')
        assert hasattr(warden, 'mimiry')

    def test_warden_has_mimiry_instance(self, warden: Warden) -> None:
        """Test Warden has Mimiry oracle instance."""
        assert warden.mimiry is not None
        from saga.core.mimiry import Mimiry
        assert isinstance(warden.mimiry, Mimiry)

    def test_warden_has_constitution(self, warden: Warden) -> None:
        """Test Warden has SagaConstitution."""
        assert warden.constitution is not None


class TestReceiveProposal:
    """Test Warden.receive_proposal()"""

    @pytest.mark.asyncio
    async def test_receive_simple_proposal(self, warden: Warden) -> None:
        """Test receiving a simple proposal from SAGA."""
        # Mock Mimiry to avoid external calls and ensure deterministic behavior
        with patch.object(warden.mimiry, 'consult_on_discrepancy', new_callable=AsyncMock) as mock_consult:
            mock_consult.return_value = OracleResponse(
                question="test",
                canonical_answer="Aligned",
                cited_rules=[],
                severity="ACCEPTABLE"
            )

            proposal = await warden.receive_proposal(
                saga_request="Create GET /users endpoint",
                context={"budget": 100},
                trace_id="trace-001"
            )

            assert proposal.decision in ["approved", "modified", "rejected"]
            assert proposal.rationale is not None

    @pytest.mark.asyncio
    async def test_proposal_creates_task_graph(self, warden: Warden) -> None:
        """Test proposal creates TaskGraph."""
        with patch.object(warden.mimiry, 'consult_on_discrepancy', new_callable=AsyncMock) as mock_consult:
            mock_consult.return_value = OracleResponse(
                question="test",
                canonical_answer="Aligned",
                cited_rules=[],
                severity="ACCEPTABLE"
            )

            proposal = await warden.receive_proposal(
                saga_request="Create user CRUD endpoints",
                context={"budget": 100},
                trace_id="trace-002"
            )

            if proposal.decision == "approved":
                assert proposal.task_graph is not None
                assert isinstance(proposal.task_graph, TaskGraph)
                assert len(proposal.task_graph.get_all_tasks()) > 0

    @pytest.mark.asyncio
    async def test_proposal_rejected_on_critical_violations(self, warden: Warden) -> None:
        """Test proposal rejected when Mimiry finds CRITICAL violations."""
        with patch.object(warden.mimiry, 'consult_on_discrepancy', new_callable=AsyncMock) as mock_consult:
            mock_consult.return_value = OracleResponse(
                question="test",
                canonical_answer="Critical violation detected",
                cited_rules=[1, 2],
                violations_detected=["Critical error"],
                severity="CRITICAL",
            )

            proposal = await warden.receive_proposal(
                saga_request="Create insecure endpoint",
                context={},
                trace_id="trace-004"
            )

            assert proposal.decision == "rejected"
            assert "Critical" in proposal.rationale or "violation" in proposal.rationale.lower()


class TestDecomposeIntoTasks:
    """Test Warden.decompose_into_tasks()"""

    @pytest.mark.asyncio
    async def test_decompose_into_tasks_returns_list(self, warden: Warden) -> None:
        """Test decomposing returns a list of tasks."""
        tasks = await warden.decompose_into_tasks(
            request="Create user CRUD endpoints",
            context={},
            trace_id="trace-007"
        )

        assert isinstance(tasks, list)
        if len(tasks) > 0:
            assert isinstance(tasks[0], Task)


class TestResolveViaMimiry:
    """Test Warden.resolve_via_mimiry() - CRITICAL INTEGRATION"""

    @pytest.mark.asyncio
    async def test_resolve_conflicting_outputs(
        self,
        warden: Warden,
        simple_task: Task,
        agent_outputs_conflicting: list[dict[str, Any]]
    ) -> None:
        """Test resolving conflicting agent outputs via Mimiry."""
        # Mock Mimiry to isolate Warden logic
        with patch.object(warden.mimiry, 'resolve_conflict', new_callable=AsyncMock) as mock_resolve:
            from saga.core.mimiry import ConflictResolution
            mock_resolve.return_value = ConflictResolution(
                conflict_description="Mock",
                conflicting_approaches=["A", "B"],
                canonical_approach="B",
                rationale="Test",
                cited_rules=[1],
                agents_in_alignment=["AgentB"],
                agents_in_violation=["AgentA"]
            )

            resolution = await warden.resolve_via_mimiry(
                agents_outputs=agent_outputs_conflicting,
                task=simple_task
            )

            assert resolution is not None
            assert resolution.canonical_approach is not None
            assert len(resolution.agents_in_alignment) > 0 or len(resolution.agents_in_violation) > 0


class TestWardenProposal:
    """Test WardenProposal dataclass."""

    def test_proposal_creation(self) -> None:
        """Test creating a WardenProposal."""
        proposal = WardenProposal(
            decision="approved",
            rationale="All checks passed",
            estimated_cost=50.0,
        )

        assert proposal.decision == "approved"
        assert proposal.estimated_cost == 50.0


class TestExecuteTask:
    """Test Warden.execute_task()"""

    @pytest.mark.asyncio
    async def test_execute_task_success(self, warden: Warden) -> None:
        """Test executing a task successfully."""
        task = Task(id="t1", description="Implement auth", weight="complex", budget_allocation=100)

        # Mock CodingAgent and dependencies
        from saga.agents.coder import AgentOutput

        mock_output = AgentOutput(
            task_id="t1",
            status="completed",
            production_code="def auth(): pass",
            test_code="def test(): pass",
            file_path="auth.py",
            test_path="test_auth.py",
            rationale="Done",
            confidence=0.9,
            cost_usd=0.01,
            tokens_used=100,
            llm_model="gpt-4"
        )

        with patch("saga.core.warden.CodingAgent") as MockAgentClass:
            mock_agent_instance = MockAgentClass.return_value
            mock_agent_instance.solve_task = AsyncMock(return_value=mock_output)

            # Mock LoreBook (warden.lorebook may be None from fixture)
            if warden.lorebook is None:
                warden.lorebook = AsyncMock()
            warden.lorebook.record_outcome = AsyncMock()

            outputs = await warden.execute_task(task)

            assert len(outputs) == 1
            result = outputs[0]
            assert result["agent"] == "CodingAgent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
