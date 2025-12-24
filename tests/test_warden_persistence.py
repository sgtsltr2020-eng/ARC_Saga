"""
Warden Persistence Integration Tests
=====================================

End-to-end test for save → restart → resume workflow.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.warden import Warden
from saga.storage.task_store import TaskStore


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


def test_warden_save_and_resume_workflow(tmp_path: Path):
    """
    Test complete workflow: Create → Save → Exit → Resume → Verify

    Simulates user creating work, laptop crashing, then resuming.
    """
    async def _test():
        project_root = str(tmp_path / "test_project")

        # Create project directories
        (tmp_path / "test_project").mkdir()
        (tmp_path / "test_project" / "saga" / "api").mkdir(parents=True)
        (tmp_path / "test_project" / "tests").mkdir()

        # ========== SESSION 1: Create and save work ==========
        warden1 = Warden(project_root=project_root)

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.initialize = AsyncMock()
        warden1.llm_client = mock_llm

        # Mock LoreBook
        mock_lorebook = AsyncMock()
        mock_lorebook.initialize = AsyncMock()
        mock_lorebook.get_project_patterns = AsyncMock(return_value=[])
        mock_lorebook.record_outcome = AsyncMock()
        mock_lorebook.get_relevant_decisions = AsyncMock(return_value=[])
        warden1.lorebook = mock_lorebook

        # Mock Mimiry
        mock_mimiry_response = MagicMock()
        mock_mimiry_response.severity = "OK"
        mock_mimiry_response.violations_detected = []
        mock_mimiry_response.canonical_answer = "Approved"
        warden1.mimiry.consult_on_discrepancy = AsyncMock(return_value=mock_mimiry_response)

        await warden1.initialize()

        # Verify TaskStore initialized
        assert warden1.task_store is not None

        # Create a simple graph directly
        graph = TaskGraph()
        task1 = Task(
            id="task-001",
            description="Create User model",
            weight="simple",
            status="done",
            trace_id="req-test-001"
        )
        task2 = Task(
            id="task-002",
            description="Create login endpoint",
            weight="complex",
            status="pending",
            dependencies=["task-001"],
            trace_id="req-test-001"
        )
        graph.add_task(task1)
        graph.add_task(task2)

        # Save graph
        await warden1.task_store.save_task_graph(
            graph=graph,
            request_id="req-test-001",
            estimated_cost=50.0
        )

        # Write actual files for verification
        (tmp_path / "test_project" / "saga" / "api" / "users.py").write_text(
            "def get_user(user_id: int):\n    return {'id': user_id}"
        )

        # Mark task1 as done in DB
        await warden1.task_store.update_task_status(
            task_id="task-001",
            status="done",
            warden_verification="approved"
        )

        # Verify saved to database
        loaded_graph = await warden1.task_store.load_task_graph("req-test-001")
        assert loaded_graph is not None

        # Close session 1
        await warden1.task_store.close()

        # ========== SESSION 2: Resume work (simulate restart) ==========
        warden2 = Warden(project_root=project_root)
        warden2.llm_client = mock_llm
        warden2.lorebook = mock_lorebook
        await warden2.initialize()

        # Resume previous work
        resumed_graph = await warden2.resume_work("req-test-001")

        assert resumed_graph is not None

        # Verify first task still exists
        t1 = resumed_graph.get_task("task-001")
        assert t1 is not None

        # Get remaining work
        pending = resumed_graph.get_ready_tasks()
        # task-002 should be ready since task-001 is done
        assert len(pending) >= 1

        # Clean up
        await warden2.task_store.close()

    run_async(_test())


def test_warden_resume_nonexistent(tmp_path: Path):
    """Test resuming non-existent work returns None."""
    async def _test():
        project_root = str(tmp_path / "test_project")
        (tmp_path / "test_project").mkdir()

        warden = Warden(project_root=project_root)

        # Mock dependencies
        warden.llm_client = AsyncMock()
        warden.llm_client.initialize = AsyncMock()
        warden.lorebook = AsyncMock()
        warden.lorebook.initialize = AsyncMock()

        await warden.initialize()

        # Try to resume non-existent work
        result = await warden.resume_work("nonexistent-request")

        assert result is None

        await warden.task_store.close()

    run_async(_test())


def test_task_store_persistence_across_instances(tmp_path: Path):
    """Test that data persists across TaskStore instances."""
    async def _test():
        db_path = str(tmp_path / "tasks.db")

        # Instance 1: Save data
        store1 = TaskStore(db_path=db_path)
        await store1.initialize()

        graph = TaskGraph()
        task = Task(id="task-001", description="Persistent task", weight="simple")
        graph.add_task(task)

        await store1.save_task_graph(graph, "req-persist-001")
        await store1.close()

        # Instance 2: Load data (new connection)
        store2 = TaskStore(db_path=db_path)
        await store2.initialize()

        loaded = await store2.load_task_graph("req-persist-001")

        assert loaded is not None
        t1 = loaded.get_task("task-001")
        assert t1 is not None
        assert t1.description == "Persistent task"

        await store2.close()

    run_async(_test())
