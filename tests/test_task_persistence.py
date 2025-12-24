"""
Tests for Task Persistence (TaskStore & TaskVerifier)
=====================================================

Verifies SQLite persistence and filesystem verification logic.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from pathlib import Path

from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.task_verifier import TaskVerifier
from saga.storage.task_store import TaskStore


# Helper
def run_async(coro):
    return asyncio.run(coro)

# --- TaskStore Tests ---

def test_task_store_initialization(tmp_path: Path):
    """Test database creation."""
    async def _test():
        store = TaskStore(str(tmp_path / "tasks.db"))
        await store.initialize()

        assert (tmp_path / "tasks.db").exists()

        # Verify tables
        cursor = await store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in await cursor.fetchall()]
        assert "task_graphs" in tables
        assert "tasks" in tables

        await store.close()

    run_async(_test())

def test_save_and_load_task_graph(tmp_path: Path):
    """Test saving and loading a graph."""
    async def _test():
        store = TaskStore(str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create Graph
        graph = TaskGraph()
        task1 = Task(id="t1", description="Task 1", weight="simple", trace_id="trace-1")
        task2 = Task(id="t2", description="Task 2", weight="complex", trace_id="trace-1")
        task2.dependencies = ["t1"]
        graph.add_task(task1)
        graph.add_task(task2)

        # Save
        await store.save_task_graph(graph, request_id="req-1", estimated_cost=10.0)

        # Load
        loaded_graph = await store.load_task_graph("req-1")
        assert loaded_graph is not None

        # FIX: Use get_task instead of tasks[]
        t1 = loaded_graph.get_task("t1")
        t2 = loaded_graph.get_task("t2")
        assert t1 is not None and t2 is not None
        assert t1.description == "Task 1"
        assert t2.dependencies == ["t1"]

        await store.close()

    run_async(_test())

def test_update_task_status(tmp_path: Path):
    """Test status update."""
    async def _test():
        store = TaskStore(str(tmp_path / "tasks.db"))
        await store.initialize()

        graph = TaskGraph()
        task = Task(id="t1", description="Task 1", weight="simple")
        graph.add_task(task)
        await store.save_task_graph(graph, "req-1")

        # Update
        await store.update_task_status("t1", "done", "approved")

        # Verify
        loaded_graph = await store.load_task_graph("req-1")

        # FIX: Use get_task instead of tasks[]
        t1 = loaded_graph.get_task("t1")
        assert t1 is not None
        assert t1.status == "done"
        assert t1.warden_verification == "approved"
        assert t1.completed_at is not None

        await store.close()

    run_async(_test())

# --- TaskVerifier Tests ---

def test_verify_file_existence(tmp_path: Path):
    """Test existence check."""
    async def _test():
        verifier = TaskVerifier(str(tmp_path))
        task = Task(id="t1", description="test", weight="simple", status="done")

        # Create file
        (tmp_path / "test.py").write_text("print('hello')")

        # Test pass
        result = await verifier.verify_task(
            task,
            level="exists",
            code_files={"test.py": ""}
        )
        assert result.verified is True

        # Test fail (missing)
        result = await verifier.verify_task(
            task,
            level="exists",
            code_files={"missing.py": ""}
        )
        assert result.verified is False
        assert "File does not exist" in result.issues[0]

    run_async(_test())

def test_verify_syntax(tmp_path: Path):
    """Test syntax check."""
    async def _test():
        verifier = TaskVerifier(str(tmp_path))
        task = Task(id="t1", description="test", weight="simple", status="done")

        # Valid code
        (tmp_path / "valid.py").write_text("def foo(): pass")
        result = await verifier.verify_task(
            task,
            level="syntax",
            code_files={"valid.py": "def foo(): pass"}
        )
        assert result.verified is True

        # Invalid code
        (tmp_path / "invalid.py").write_text("def foo( pass")
        result = await verifier.verify_task(
            task,
            level="syntax",
            code_files={"invalid.py": "def foo( pass"}
        )
        assert result.verified is False
        assert "Syntax error" in result.issues[0]

    run_async(_test())
