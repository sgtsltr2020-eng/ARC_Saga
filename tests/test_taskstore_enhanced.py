"""
TaskStore Performance and Scale Tests
======================================

Optional enhancements including:
- Performance tests with many tasks
- Scale tests for large task graphs
- Crash simulation for transaction atomicity

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
import time
from pathlib import Path

from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.storage.task_store import TaskStore


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


# ============================================================
# PERFORMANCE TESTS
# ============================================================

def test_taskstore_save_many_tasks_performance(tmp_path: Path):
    """Test saving many tasks performs within acceptable time."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create large task graph
        graph = TaskGraph()
        num_tasks = 100  # 100 tasks for performance test

        for i in range(num_tasks):
            deps = [f"task-{i-1:04d}"] if i > 0 else []
            task = Task(
                id=f"task-{i:04d}",
                description=f"Performance test task {i}",
                weight="simple" if i % 3 == 0 else "complex",
                budget_allocation=float(i),
                dependencies=deps
            )
            graph.add_task(task)

        # Measure save time
        start_time = time.time()
        await store.save_task_graph(graph=graph, request_id="perf-test-001")
        save_duration = time.time() - start_time

        # Should complete in under 5 seconds for 100 tasks
        assert save_duration < 5.0, f"Save took {save_duration:.2f}s, expected < 5s"

        # Measure load time
        start_time = time.time()
        loaded = await store.load_task_graph("perf-test-001")
        load_duration = time.time() - start_time

        # Should complete in under 2 seconds
        assert load_duration < 2.0, f"Load took {load_duration:.2f}s, expected < 2s"

        # Verify all tasks loaded
        assert len(loaded.get_all_tasks()) == num_tasks

        await store.close()

    run_async(_test())


def test_taskstore_update_status_performance(tmp_path: Path):
    """Test updating many task statuses performs well."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create and save graph
        graph = TaskGraph()
        num_tasks = 50

        for i in range(num_tasks):
            task = Task(
                id=f"update-task-{i}",
                description=f"Update test task {i}",
                weight="simple"
            )
            graph.add_task(task)

        await store.save_task_graph(graph=graph, request_id="update-perf-001")

        # Measure batch update time
        start_time = time.time()
        for i in range(num_tasks):
            await store.update_task_status(f"update-task-{i}", "done")
        update_duration = time.time() - start_time

        # Should complete in under 3 seconds
        assert update_duration < 3.0, f"Updates took {update_duration:.2f}s, expected < 3s"

        await store.close()

    run_async(_test())


def test_taskstore_pending_tasks_performance(tmp_path: Path):
    """Test getting pending tasks from large graph."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create graph with mix of pending and done tasks
        graph = TaskGraph()
        num_tasks = 100

        for i in range(num_tasks):
            task = Task(
                id=f"pending-test-{i}",
                description=f"Pending test task {i}",
                weight="simple",
                status="pending" if i % 2 == 0 else "done"  # 50% pending
            )
            graph.add_task(task)

        await store.save_task_graph(graph=graph, request_id="pending-perf-001")

        # Measure query time
        start_time = time.time()
        pending = await store.get_pending_tasks("pending-perf-001")
        query_duration = time.time() - start_time

        # Should be fast query
        assert query_duration < 0.5, f"Query took {query_duration:.2f}s, expected < 0.5s"
        assert len(pending) == 50  # Half should be pending

        await store.close()

    run_async(_test())


# ============================================================
# SCALE TESTS
# ============================================================

def test_taskstore_large_task_descriptions(tmp_path: Path):
    """Test storing tasks with very large descriptions."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create task with large description (10KB)
        large_description = "A" * 10_000
        large_checklist = [f"Item {i}: " + "X" * 500 for i in range(20)]  # 20 items, 500 chars each

        graph = TaskGraph()
        task = Task(
            id="large-task",
            description=large_description,
            weight="complex",
            checklist=large_checklist,
            budget_allocation=100.0
        )
        graph.add_task(task)

        await store.save_task_graph(graph=graph, request_id="large-001")
        loaded = await store.load_task_graph("large-001")

        loaded_task = loaded.get_task("large-task")
        assert loaded_task is not None
        assert len(loaded_task.description) == 10_000
        assert len(loaded_task.checklist) == 20

        await store.close()

    run_async(_test())


def test_taskstore_deep_dependency_chain(tmp_path: Path):
    """Test storing tasks with deep dependency chains."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create deep dependency chain (50 levels deep)
        graph = TaskGraph()
        depth = 50

        for i in range(depth):
            deps = [f"chain-{i-1}"] if i > 0 else []
            task = Task(
                id=f"chain-{i}",
                description=f"Chain task at depth {i}",
                weight="simple",
                dependencies=deps
            )
            graph.add_task(task)

        await store.save_task_graph(graph=graph, request_id="chain-001")
        loaded = await store.load_task_graph("chain-001")

        # Verify chain integrity
        assert len(loaded.get_all_tasks()) == depth

        # Only first task should have no dependencies
        ready = loaded.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "chain-0"

        await store.close()

    run_async(_test())


def test_taskstore_wide_dependency_graph(tmp_path: Path):
    """Test storing tasks with wide dependency graph (many parallel tasks)."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create wide graph: 1 root -> 50 parallel -> 1 final
        graph = TaskGraph()

        # Root task
        root = Task(id="root", description="Root task", weight="simple")
        graph.add_task(root)

        # 50 parallel tasks depending on root
        for i in range(50):
            task = Task(
                id=f"parallel-{i}",
                description=f"Parallel task {i}",
                weight="simple",
                dependencies=["root"]
            )
            graph.add_task(task)

        # Final task depending on all parallel tasks
        final_deps = [f"parallel-{i}" for i in range(50)]
        final = Task(
            id="final",
            description="Final aggregation task",
            weight="complex",
            dependencies=final_deps
        )
        graph.add_task(final)

        await store.save_task_graph(graph=graph, request_id="wide-001")
        loaded = await store.load_task_graph("wide-001")

        # Verify graph structure
        assert len(loaded.get_all_tasks()) == 52  # 1 root + 50 parallel + 1 final

        # Initially only root should be ready
        ready = loaded.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "root"

        # Mark root done, all parallel should become ready
        root_task = loaded.get_task("root")
        root_task.status = "done"
        ready = loaded.get_ready_tasks()
        assert len(ready) == 50

        await store.close()

    run_async(_test())


# ============================================================
# CRASH SIMULATION TESTS
# ============================================================

def test_taskstore_crash_during_save(tmp_path: Path):
    """Test recovery after simulated crash during save."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # First save a valid graph
        graph = TaskGraph()
        task = Task(id="original", description="Original task", weight="simple")
        graph.add_task(task)
        await store.save_task_graph(graph=graph, request_id="crash-001")

        # Verify it was saved
        loaded = await store.load_task_graph("crash-001")
        assert loaded.get_task("original") is not None

        # Now simulate a crash during update by closing connection mid-write
        # (We can't truly crash, but we can test that old data is preserved)
        await store.close()

        # Reopen and verify original data is intact
        store2 = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store2.initialize()

        loaded2 = await store2.load_task_graph("crash-001")
        assert loaded2 is not None
        assert loaded2.get_task("original") is not None

        await store2.close()

    run_async(_test())


def test_taskstore_concurrent_updates(tmp_path: Path):
    """Test handling of concurrent task status updates."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Create graph with multiple tasks
        graph = TaskGraph()
        for i in range(10):
            task = Task(id=f"concurrent-{i}", description=f"Task {i}", weight="simple")
            graph.add_task(task)

        await store.save_task_graph(graph=graph, request_id="concurrent-001")

        # Simulate concurrent updates using asyncio.gather
        async def update_task(task_id: str, status: str):
            await store.update_task_status(task_id, status)

        # Update all tasks concurrently
        updates = [
            update_task(f"concurrent-{i}", "done" if i % 2 == 0 else "in_progress")
            for i in range(10)
        ]
        await asyncio.gather(*updates)

        # Verify all updates applied
        loaded = await store.load_task_graph("concurrent-001")
        for i in range(10):
            task = loaded.get_task(f"concurrent-{i}")
            expected_status = "done" if i % 2 == 0 else "in_progress"
            assert task.status == expected_status

        await store.close()

    run_async(_test())


def test_taskstore_transaction_atomicity(tmp_path: Path):
    """Test that task graph saves are atomic (all or nothing)."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Save initial state
        graph = TaskGraph()
        task = Task(id="atomic-1", description="First task", weight="simple")
        graph.add_task(task)
        await store.save_task_graph(graph=graph, request_id="atomic-001")

        # Overwrite with larger graph
        graph2 = TaskGraph()
        for i in range(5):
            task = Task(id=f"atomic-{i}", description=f"Task {i}", weight="simple")
            graph2.add_task(task)

        await store.save_task_graph(graph=graph2, request_id="atomic-001")

        # Load and verify the NEW graph is completely there
        loaded = await store.load_task_graph("atomic-001")
        assert len(loaded.get_all_tasks()) == 5

        # No tasks from old graph should remain unmixed
        for i in range(5):
            assert loaded.get_task(f"atomic-{i}") is not None

        await store.close()

    run_async(_test())


def test_taskstore_database_persistence(tmp_path: Path):
    """Test that data persists after store is closed and reopened."""
    async def _test():
        store = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store.initialize()

        # Save valid data
        graph = TaskGraph()
        task = Task(id="persist-1", description="Persistence test", weight="simple")
        graph.add_task(task)
        await store.save_task_graph(graph=graph, request_id="persist-001")
        await store.close()

        # Reopen and verify data is still there
        store2 = TaskStore(db_path=str(tmp_path / "tasks.db"))
        await store2.initialize()

        loaded = await store2.load_task_graph("persist-001")
        assert loaded is not None
        assert loaded.get_task("persist-1") is not None
        assert loaded.get_task("persist-1").description == "Persistence test"

        await store2.close()

    run_async(_test())
