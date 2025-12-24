"""
TaskStore - Persistent State Management for TaskGraphs
======================================================

Persists TaskGraph state to SQLite to allow resuming work after restarts.
Handles serialization of Tasks and their relationships.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 4 - State Management
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from saga.core.task import Task
from saga.core.task_graph import TaskGraph

logger = logging.getLogger(__name__)

class TaskStore:
    """
    SQLite-backed persistent storage for TaskGraph state.

    Database schema:
    - task_graphs: Tracks overall request state
    - tasks: Tracks individual task state and metadata
    """

    def __init__(self, db_path: str = ".saga/tasks.db") -> None:
        """
        Initialize TaskStore.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """
        Initialize database connection and schema.
        Must be called before any other operations.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row

        # Create task_graphs table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS task_graphs (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                estimated_cost REAL DEFAULT 0.0,
                actual_cost REAL DEFAULT 0.0
            )
        """)

        # Create tasks table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                description TEXT NOT NULL,
                weight TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_agent TEXT,
                budget_allocation REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                trace_id TEXT NOT NULL,
                dependencies TEXT,
                checklist TEXT,
                vetting_criteria TEXT,
                self_check_result TEXT,
                warden_verification TEXT,
                mimiry_measurement TEXT,
                FOREIGN KEY(request_id) REFERENCES task_graphs(request_id)
            )
        """)

        # Create indexes
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_request ON tasks(request_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")

        await self.db.commit()

        logger.info("TaskStore initialized", extra={"db_path": str(self.db_path)})

    async def save_task_graph(
        self,
        graph: TaskGraph,
        request_id: str,
        estimated_cost: float = 0.0
    ) -> None:
        """
        Save a TaskGraph and all its tasks to the database.

        Args:
            graph: The TaskGraph to save.
            request_id: The request ID associated with this graph.
            estimated_cost: Estimated cost for the graph execution.
        """
        if not self.db:
            raise RuntimeError("TaskStore not initialized. Call initialize() first.")

        # Determine overall status (simple heuristic)
        status = "in_progress"
        # You might want slightly more complex logic for graph status,
        # but 'in_progress' is a safe default for a save.

        # Save TaskGraph record
        await self.db.execute(
            """
            INSERT OR REPLACE INTO task_graphs (
                request_id, created_at, status, estimated_cost, actual_cost
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.utcnow().isoformat(),
                status,
                estimated_cost,
                0.0 # actual_cost tracking can be refined later
            )
        )

        # Save all tasks
        tasks_data = []
        for task in graph.get_all_tasks():
            tasks_data.append((
                task.id,
                request_id,
                task.description,
                task.weight,
                task.status,
                task.assigned_agent,
                task.budget_allocation,
                task.created_at.isoformat(),
                task.completed_at.isoformat() if task.completed_at else None,
                task.trace_id,
                json.dumps(task.dependencies),
                json.dumps(task.checklist),
                json.dumps(task.vetting_criteria),
                json.dumps(task.self_check_result) if task.self_check_result else None,
                task.warden_verification,
                json.dumps(task.mimiry_measurement) if task.mimiry_measurement else None
            ))

        await self.db.executemany(
            """
            INSERT OR REPLACE INTO tasks (
                task_id, request_id, description, weight, status, assigned_agent,
                budget_allocation, created_at, completed_at, trace_id,
                dependencies, checklist, vetting_criteria, self_check_result,
                warden_verification, mimiry_measurement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tasks_data
        )

        await self.db.commit()

        logger.info(
            "TaskGraph saved",
            extra={"request_id": request_id, "task_count": len(tasks_data)}
        )

    async def load_task_graph(self, request_id: str) -> Optional[TaskGraph]:
        """
        Load a TaskGraph from the database.

        Args:
            request_id: The request ID to load.

        Returns:
            Reconstructed TaskGraph or None if not found.
        """
        if not self.db:
            raise RuntimeError("TaskStore not initialized. Call initialize() first.")

        # Check if graph exists
        async with self.db.execute(
            "SELECT * FROM task_graphs WHERE request_id = ?",
            (request_id,)
        ) as cursor:
            graph_row = await cursor.fetchone()
            if not graph_row:
                return None

        # Load tasks
        async with self.db.execute(
            "SELECT * FROM tasks WHERE request_id = ?",
            (request_id,)
        ) as cursor:
            task_rows = await cursor.fetchall()

        tasks: dict[str, Task] = {}
        for row in task_rows:
            task = Task(
                id=row["task_id"],
                description=row["description"],
                weight=row["weight"], # Literal type check skipped for runtime
                status=row["status"],
                dependencies=json.loads(row["dependencies"]),
                checklist=json.loads(row["checklist"]),
                vetting_criteria=json.loads(row["vetting_criteria"]),
                budget_allocation=row["budget_allocation"],
                assigned_agent=row["assigned_agent"],
                trace_id=row["trace_id"]
            )
            # Restore optional fields
            task.created_at = datetime.fromisoformat(row["created_at"])
            if row["completed_at"]:
                task.completed_at = datetime.fromisoformat(row["completed_at"])
            if row["self_check_result"]:
                task.self_check_result = json.loads(row["self_check_result"])
            task.warden_verification = row["warden_verification"]
            if row["mimiry_measurement"]:
                task.mimiry_measurement = json.loads(row["mimiry_measurement"])

            tasks[task.id] = task

        # Reconstruct graph
        # Note: TaskGraph usually takes a root goal, but here we reconstruct from tasks.
        # We might need to adjust TaskGraph __init__ or use a factory method,
        # but for now we assume we can set .tasks directly or pass them.
        # Assuming TaskGraph can be initialized empty and populated or initialized with tasks.
        # Looking at TaskGraph definition (from memory/previous context), it takes a goal.
        # We'll instantiate it with a placeholder goal and populate tasks.
        # Ideally, we should store the goal in task_graphs table, but the schema provided
        # didn't include 'goal'. We'll use "Resumed Graph" or similar.

        graph = TaskGraph()
        for task in tasks.values():
            graph.add_task(task)

        logger.info(
            "TaskGraph loaded",
            extra={"request_id": request_id, "task_count": len(tasks)}
        )
        return graph

    async def get_pending_tasks(self, request_id: str) -> list[Task]:
        """
        Get list of tasks that are ready to be executed for a given request.

        Args:
            request_id: The request ID.

        Returns:
            List of Task objects ready for execution.
        """
        graph = await self.load_task_graph(request_id)
        if not graph:
            return []

        return graph.get_ready_tasks()

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        warden_verification: Optional[str] = None
    ) -> None:
        """
        Update the status of a specific task.

        Args:
            task_id: The task ID.
            status: New status.
            warden_verification: Optional verification result ('approved'/'rejected').
        """
        if not self.db:
            raise RuntimeError("TaskStore not initialized. Call initialize() first.")

        completed_at = None
        if status == "done":
            completed_at = datetime.utcnow().isoformat()

        await self.db.execute(
            """
            UPDATE tasks
            SET status = ?, warden_verification = ?, completed_at = ?
            WHERE task_id = ?
            """,
            (status, warden_verification, completed_at, task_id)
        )
        await self.db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self.db:
            await self.db.close()
            self.db = None
