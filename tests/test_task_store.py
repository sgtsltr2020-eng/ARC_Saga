"""
TaskStore Tests
===============

Comprehensive tests for SQLite-backed TaskGraph persistence.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from pathlib import Path

from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.storage.task_store import TaskStore


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


def test_save_and_load_task_graph(tmp_path: Path):
    """Test saving and loading TaskGraph."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create TaskGraph with 2 tasks
        graph = TaskGraph()
        task1 = Task(
            id="task-001",
            description="Create User model",
            weight="simple",
            status="pending",
            trace_id="trace-001"
        )
        task2 = Task(
            id="task-002",
            description="Create login endpoint",
            weight="complex",
            status="pending",
            dependencies=["task-001"],
            trace_id="trace-001"
        )
        graph.add_task(task1)
        graph.add_task(task2)

        # Save
        await store.save_task_graph(
            graph=graph,
            request_id="req-test-001",
            estimated_cost=50.0
        )

        # Load
        loaded_graph = await store.load_task_graph("req-test-001")

        # Assertions
        assert loaded_graph is not None

        t1 = loaded_graph.get_task("task-001")
        t2 = loaded_graph.get_task("task-002")

        assert t1 is not None
        assert t2 is not None
        assert t1.description == "Create User model"
        assert t2.dependencies == ["task-001"]

        await store.close()

    run_async(_test())


def test_load_nonexistent_graph(tmp_path: Path):
    """Test loading non-existent graph returns None."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        result = await store.load_task_graph("nonexistent")
        assert result is None

        await store.close()

    run_async(_test())


def test_update_task_status(tmp_path: Path):
    """Test updating task status."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Setup
        graph = TaskGraph()
        task = Task(id="task-001", description="Test", weight="simple", status="pending")
        graph.add_task(task)
        await store.save_task_graph(graph, "req-test-002")

        # Update status
        await store.update_task_status(
            task_id="task-001",
            status="done",
            warden_verification="approved"
        )

        # Verify
        loaded = await store.load_task_graph("req-test-002")
        t1 = loaded.get_task("task-001")

        assert t1 is not None
        assert t1.status == "done"
        assert t1.warden_verification == "approved"
        assert t1.completed_at is not None

        await store.close()

    run_async(_test())


def test_get_pending_tasks(tmp_path: Path):
    """Test getting pending tasks."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create graph with dependency chain
        graph = TaskGraph()
        task1 = Task(id="task-001", description="First", weight="simple", status="done")
        task2 = Task(id="task-002", description="Second", weight="simple", status="pending", dependencies=["task-001"])
        task3 = Task(id="task-003", description="Third", weight="simple", status="pending", dependencies=["task-002"])
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)

        await store.save_task_graph(graph, "req-test-003")

        # Get pending (should only return task2 since task1 is done and task3 depends on task2)
        pending = await store.get_pending_tasks("req-test-003")

        assert len(pending) == 1
        assert pending[0].id == "task-002"

        await store.close()

    run_async(_test())


def test_save_overwrites_existing(tmp_path: Path):
    """Test that saving same request_id updates existing data."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        graph1 = TaskGraph()
        task1 = Task(id="task-001", description="Original", weight="simple")
        graph1.add_task(task1)

        await store.save_task_graph(graph1, "req-test-004")

        # Save again with updated description
        graph2 = TaskGraph()
        task2 = Task(id="task-001", description="Updated", weight="simple")
        graph2.add_task(task2)

        await store.save_task_graph(graph2, "req-test-004")

        # Verify updated
        loaded = await store.load_task_graph("req-test-004")
        t1 = loaded.get_task("task-001")

        assert t1 is not None
        assert t1.description == "Updated"

        await store.close()

    run_async(_test())
