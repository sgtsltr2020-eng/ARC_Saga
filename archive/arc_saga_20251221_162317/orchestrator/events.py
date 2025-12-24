"""
Event Store Implementation for Orchestrator.

Provides event sourcing infrastructure with protocol-based abstraction.
Supports both SQLite (persistent) and InMemory (testing) implementations.

This module follows the event-driven CQRS pattern from decision_catalog.md:
- Events are immutable facts that represent state changes
- Event store is append-only (no updates or deletes)
- Events enable replay, auditing, and temporal queries

Example:
    >>> from saga.orchestrator.events import InMemoryEventStore
    >>> from saga.orchestrator.events import WorkflowStartedEvent
    >>>
    >>> store = InMemoryEventStore()
    >>> event = WorkflowStartedEvent(
    ...     aggregate_id="workflow-123",
    ...     pattern="SEQUENTIAL",
    ...     task_count=3,
    ... )
    >>> event_id = await store.append(event)
    >>> events = await store.get_events("workflow-123")
"""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import aiosqlite

from ..error_instrumentation import log_with_context

# Type variable for generic event store
E = TypeVar("E", bound="OrchestratorEvent")


@dataclass(frozen=True)
class OrchestratorEvent:
    """
    Base class for all orchestrator events.

    All events must be immutable (frozen dataclass) and contain:
    - id: Unique event identifier
    - aggregate_id: ID of the aggregate this event belongs to
    - event_type: String identifier for the event type
    - created_at: UTC timestamp when event was created
    - correlation_id: Optional ID linking related events
    - source: Component that emitted the event

    Subclasses should add domain-specific fields.

    Example:
        >>> @dataclass(frozen=True)
        ... class MyEvent(OrchestratorEvent):
        ...     custom_field: str = ""
    """

    aggregate_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = field(default="OrchestratorEvent")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    source: str = "orchestrator"
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """
        Convert event to dictionary for serialization.

        Returns:
            Dictionary representation with datetime as ISO string.
        """
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestratorEvent:
        """
        Create event from dictionary.

        Args:
            data: Dictionary with event data.

        Returns:
            Reconstructed event instance.
        """
        # Convert ISO string back to datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass(frozen=True)
class WorkflowStartedEvent(OrchestratorEvent):
    """
    Event emitted when a workflow execution begins.

    Attributes:
        pattern: Workflow pattern (SEQUENTIAL, PARALLEL, DYNAMIC)
        task_count: Number of tasks in the workflow
        task_ids: List of task IDs in execution order
    """

    event_type: str = field(default="WorkflowStartedEvent")
    pattern: str = ""
    task_count: int = 0
    task_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WorkflowCompletedEvent(OrchestratorEvent):
    """
    Event emitted when a workflow execution completes.

    Attributes:
        success: Whether all tasks completed successfully
        duration_ms: Total workflow duration in milliseconds
        completed_count: Number of successfully completed tasks
        failed_count: Number of failed tasks
    """

    event_type: str = field(default="WorkflowCompletedEvent")
    success: bool = True
    duration_ms: int = 0
    completed_count: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class TaskExecutedEvent(OrchestratorEvent):
    """
    Event emitted when a task execution completes.

    Attributes:
        task_id: ID of the executed task
        operation: Task operation name
        success: Whether the task succeeded
        duration_ms: Task execution duration
        error: Error message if failed
    """

    event_type: str = field(default="TaskExecutedEvent")
    task_id: str = ""
    operation: str = ""
    success: bool = True
    duration_ms: int = 0
    error: str | None = None


@dataclass(frozen=True)
class PolicyEnforcedEvent(OrchestratorEvent):
    """
    Event emitted when a policy is enforced.

    Attributes:
        policy_name: Name of the enforced policy
        command_type: Type of command being validated
        allowed: Whether the command was allowed
        reason: Reason for the decision
    """

    event_type: str = field(default="PolicyEnforcedEvent")
    policy_name: str = ""
    command_type: str = ""
    allowed: bool = True
    reason: str = ""


@dataclass(frozen=True)
class OperationLoggedEvent(OrchestratorEvent):
    """
    Event emitted for operation logging/auditing.

    Attributes:
        operation_name: Name of the operation
        context_data: Serialized operation context
        level: Log level (INFO, WARNING, ERROR)
    """

    event_type: str = field(default="OperationLoggedEvent")
    operation_name: str = ""
    context_data: str = ""
    level: str = "INFO"


# Event type registry for deserialization
EVENT_TYPE_REGISTRY: dict[str, type[OrchestratorEvent]] = {
    "OrchestratorEvent": OrchestratorEvent,
    "WorkflowStartedEvent": WorkflowStartedEvent,
    "WorkflowCompletedEvent": WorkflowCompletedEvent,
    "TaskExecutedEvent": TaskExecutedEvent,
    "PolicyEnforcedEvent": PolicyEnforcedEvent,
    "OperationLoggedEvent": OperationLoggedEvent,
}


def deserialize_event(event_type: str, data: dict[str, Any]) -> OrchestratorEvent:
    """
    Deserialize event from stored data.

    Args:
        event_type: Event type string identifier
        data: Event data dictionary

    Returns:
        Reconstructed event instance

    Raises:
        ValueError: If event_type is unknown
    """
    event_class = EVENT_TYPE_REGISTRY.get(event_type)
    if event_class is None:
        raise ValueError(f"Unknown event type: {event_type}")

    # Convert created_at string to datetime
    if isinstance(data.get("created_at"), str):
        data["created_at"] = datetime.fromisoformat(data["created_at"])

    # Handle tuple fields stored as lists
    if "task_ids" in data and isinstance(data["task_ids"], list):
        data["task_ids"] = tuple(data["task_ids"])

    return event_class(**data)


@runtime_checkable
class IEventStore(Protocol[E]):
    """
    Protocol for event store implementations.

    Defines the contract for event persistence with append-only semantics.
    All implementations must be async for compatibility with I/O operations.

    Type Parameters:
        E: Event type (must extend OrchestratorEvent)
    """

    @abstractmethod
    async def append(self, event: E) -> str:
        """
        Append an event to the store.

        Args:
            event: Event to append

        Returns:
            Event ID

        Raises:
            EventStoreError: If append fails
        """
        ...

    @abstractmethod
    async def get_events(self, aggregate_id: str) -> list[E]:
        """
        Get all events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            List of events in chronological order
        """
        ...

    @abstractmethod
    async def get_events_since(self, timestamp: datetime) -> list[E]:
        """
        Get all events since a timestamp.

        Args:
            timestamp: Cutoff timestamp (exclusive)

        Returns:
            List of events after timestamp in chronological order
        """
        ...


class EventStoreError(Exception):
    """Base exception for event store errors."""

    def __init__(self, message: str, operation: str = "") -> None:
        """
        Initialize EventStoreError.

        Args:
            message: Error description
            operation: Operation that failed
        """
        self.operation = operation
        super().__init__(f"EventStore {operation} failed: {message}")


class InMemoryEventStore(IEventStore[OrchestratorEvent]):
    """
    In-memory event store for testing.

    Thread-safe implementation using asyncio.Lock for concurrency control.
    Events are stored in a dictionary keyed by aggregate_id.

    Example:
        >>> store = InMemoryEventStore()
        >>> event = WorkflowStartedEvent(aggregate_id="wf-1", pattern="SEQ")
        >>> event_id = await store.append(event)
        >>> events = await store.get_events("wf-1")
    """

    def __init__(self) -> None:
        """Initialize empty in-memory event store."""
        self._events: dict[str, list[OrchestratorEvent]] = {}
        self._all_events: list[OrchestratorEvent] = []
        self._lock = asyncio.Lock()

        log_with_context(
            "info",
            "event_store_initialized",
            store_type="InMemoryEventStore",
        )

    async def append(self, event: OrchestratorEvent) -> str:
        """
        Append an event to the in-memory store.

        Args:
            event: Event to append

        Returns:
            Event ID
        """
        async with self._lock:
            aggregate_id = event.aggregate_id

            if aggregate_id not in self._events:
                self._events[aggregate_id] = []

            self._events[aggregate_id].append(event)
            self._all_events.append(event)

            log_with_context(
                "info",
                "event_appended",
                event_id=event.id,
                event_type=event.event_type,
                aggregate_id=aggregate_id,
            )

            return event.id

    async def get_events(self, aggregate_id: str) -> list[OrchestratorEvent]:
        """
        Get all events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            List of events in chronological order (empty if none)
        """
        async with self._lock:
            events = self._events.get(aggregate_id, [])
            # Return sorted copy to prevent mutation
            return sorted(events, key=lambda e: e.created_at)

    async def get_events_since(self, timestamp: datetime) -> list[OrchestratorEvent]:
        """
        Get all events since a timestamp.

        Args:
            timestamp: Cutoff timestamp (exclusive)

        Returns:
            List of events after timestamp in chronological order
        """
        async with self._lock:
            filtered = [e for e in self._all_events if e.created_at > timestamp]
            return sorted(filtered, key=lambda e: e.created_at)

    async def clear(self) -> None:
        """
        Clear all events from the store.

        Useful for test cleanup.
        """
        async with self._lock:
            self._events.clear()
            self._all_events.clear()

            log_with_context(
                "info",
                "event_store_cleared",
                store_type="InMemoryEventStore",
            )

    @property
    def event_count(self) -> int:
        """Get total number of events in store."""
        return len(self._all_events)


class SQLiteEventStore(IEventStore[OrchestratorEvent]):
    """
    SQLite-based persistent event store.

    Uses aiosqlite for async database operations. Schema is created
    automatically on initialization if not exists.

    Attributes:
        db_path: Path to SQLite database file

    Example:
        >>> store = SQLiteEventStore("events.db")
        >>> await store.initialize()
        >>> event = WorkflowStartedEvent(aggregate_id="wf-1", pattern="SEQ")
        >>> event_id = await store.append(event)
    """

    def __init__(self, db_path: str | Path = "orchestrator_events.db") -> None:
        """
        Initialize SQLite event store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._initialized = False
        self._lock = asyncio.Lock()

        log_with_context(
            "info",
            "event_store_initialized",
            store_type="SQLiteEventStore",
            db_path=str(self.db_path),
        )

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates tables and indexes if they don't exist.
        Must be called before using the store.

        Raises:
            EventStoreError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Read schema from file
            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                schema_sql = schema_path.read_text(encoding="utf-8")
            else:
                # Fallback inline schema
                schema_sql = """
                CREATE TABLE IF NOT EXISTS orchestrator_events (
                    id TEXT PRIMARY KEY,
                    aggregate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    correlation_id TEXT,
                    sequence_number INTEGER,
                    source TEXT,
                    version INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_events_aggregate_created 
                    ON orchestrator_events(aggregate_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_created_at 
                    ON orchestrator_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_events_correlation_id 
                    ON orchestrator_events(correlation_id);
                """

            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(schema_sql)
                await db.commit()

            self._initialized = True

            log_with_context(
                "info",
                "event_store_schema_created",
                db_path=str(self.db_path),
            )

        except Exception as e:
            log_with_context(
                "error",
                "event_store_initialization_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise EventStoreError(str(e), "initialize") from e

    async def append(self, event: OrchestratorEvent) -> str:
        """
        Append an event to the SQLite store.

        Args:
            event: Event to append

        Returns:
            Event ID

        Raises:
            EventStoreError: If append fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                async with aiosqlite.connect(self.db_path) as db:
                    # Serialize event data
                    event_dict = event.to_dict()
                    event_data = json.dumps(event_dict, default=str)

                    await db.execute(
                        """
                        INSERT INTO orchestrator_events 
                        (id, aggregate_id, event_type, event_data, created_at,
                         correlation_id, source, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.id,
                            event.aggregate_id,
                            event.event_type,
                            event_data,
                            event.created_at.isoformat(),
                            event.correlation_id,
                            event.source,
                            event.version,
                        ),
                    )
                    await db.commit()

            log_with_context(
                "info",
                "event_appended",
                event_id=event.id,
                event_type=event.event_type,
                aggregate_id=event.aggregate_id,
                store_type="SQLiteEventStore",
            )

            return event.id

        except Exception as e:
            log_with_context(
                "error",
                "event_append_failed",
                event_type=event.event_type,
                aggregate_id=event.aggregate_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise EventStoreError(str(e), "append") from e

    async def get_events(self, aggregate_id: str) -> list[OrchestratorEvent]:
        """
        Get all events for an aggregate from SQLite.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            List of events in chronological order

        Raises:
            EventStoreError: If query fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT event_type, event_data 
                    FROM orchestrator_events 
                    WHERE aggregate_id = ?
                    ORDER BY created_at ASC
                    """,
                    (aggregate_id,),
                )
                rows = await cursor.fetchall()

            events: list[OrchestratorEvent] = []
            for row in rows:
                event_type = row["event_type"]
                event_data = json.loads(row["event_data"])
                event = deserialize_event(event_type, event_data)
                events.append(event)

            return events

        except Exception as e:
            log_with_context(
                "error",
                "event_query_failed",
                aggregate_id=aggregate_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise EventStoreError(str(e), "get_events") from e

    async def get_events_since(self, timestamp: datetime) -> list[OrchestratorEvent]:
        """
        Get all events since a timestamp from SQLite.

        Args:
            timestamp: Cutoff timestamp (exclusive)

        Returns:
            List of events after timestamp in chronological order

        Raises:
            EventStoreError: If query fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT event_type, event_data 
                    FROM orchestrator_events 
                    WHERE created_at > ?
                    ORDER BY created_at ASC
                    """,
                    (timestamp.isoformat(),),
                )
                rows = await cursor.fetchall()

            events: list[OrchestratorEvent] = []
            for row in rows:
                event_type = row["event_type"]
                event_data = json.loads(row["event_data"])
                event = deserialize_event(event_type, event_data)
                events.append(event)

            return events

        except Exception as e:
            log_with_context(
                "error",
                "event_query_failed",
                operation="get_events_since",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise EventStoreError(str(e), "get_events_since") from e

    async def close(self) -> None:
        """
        Close the event store.

        Note: aiosqlite manages connections per operation,
        so this is mainly for cleanup logging.
        """
        log_with_context(
            "info",
            "event_store_closed",
            store_type="SQLiteEventStore",
            db_path=str(self.db_path),
        )
