"""
Unit tests for orchestrator event store implementations.

Tests verify:
1. Event creation and serialization
2. InMemoryEventStore operations
3. SQLiteEventStore operations
4. Concurrent access handling
5. Event deserialization
6. Error handling

Coverage target: 98%+
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from saga.orchestrator.events import (
    EVENT_TYPE_REGISTRY,
    EventStoreError,
    InMemoryEventStore,
    OrchestratorEvent,
    OperationLoggedEvent,
    PolicyEnforcedEvent,
    SQLiteEventStore,
    TaskExecutedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
    deserialize_event,
)


class TestOrchestratorEventBase:
    """Tests for base OrchestratorEvent class."""

    def test_create_base_event(self) -> None:
        """Test creating base orchestrator event."""
        event = OrchestratorEvent(aggregate_id="agg-123")

        assert event.aggregate_id == "agg-123"
        assert len(event.id) == 36  # UUID format
        assert event.event_type == "OrchestratorEvent"
        assert event.source == "orchestrator"
        assert event.version == 1
        assert event.correlation_id is None

    def test_event_created_at_is_utc(self) -> None:
        """Test event created_at is UTC timezone."""
        event = OrchestratorEvent(aggregate_id="agg-123")

        assert event.created_at.tzinfo == timezone.utc

    def test_event_with_custom_id(self) -> None:
        """Test creating event with custom ID."""
        event = OrchestratorEvent(
            aggregate_id="agg-123",
            id="custom-event-id",
        )

        assert event.id == "custom-event-id"

    def test_event_with_correlation_id(self) -> None:
        """Test creating event with correlation ID."""
        event = OrchestratorEvent(
            aggregate_id="agg-123",
            correlation_id="corr-456",
        )

        assert event.correlation_id == "corr-456"

    def test_event_to_dict(self) -> None:
        """Test event serialization to dictionary."""
        event = OrchestratorEvent(
            aggregate_id="agg-123",
            id="event-id",
            correlation_id="corr-456",
        )

        data = event.to_dict()

        assert data["aggregate_id"] == "agg-123"
        assert data["id"] == "event-id"
        assert data["event_type"] == "OrchestratorEvent"
        assert data["correlation_id"] == "corr-456"
        assert isinstance(data["created_at"], str)  # ISO format

    def test_event_from_dict(self) -> None:
        """Test event deserialization from dictionary."""
        data = {
            "aggregate_id": "agg-123",
            "id": "event-id",
            "event_type": "OrchestratorEvent",
            "created_at": "2024-01-01T12:00:00+00:00",
            "correlation_id": None,
            "source": "orchestrator",
            "version": 1,
        }

        event = OrchestratorEvent.from_dict(data)

        assert event.aggregate_id == "agg-123"
        assert event.id == "event-id"
        assert isinstance(event.created_at, datetime)

    def test_event_is_immutable(self) -> None:
        """Test event is frozen (immutable)."""
        event = OrchestratorEvent(aggregate_id="agg-123")

        with pytest.raises(AttributeError):
            event.aggregate_id = "modified"  # type: ignore[misc]


class TestWorkflowStartedEvent:
    """Tests for WorkflowStartedEvent."""

    def test_create_workflow_started_event(self) -> None:
        """Test creating workflow started event."""
        event = WorkflowStartedEvent(
            aggregate_id="workflow-123",
            pattern="SEQUENTIAL",
            task_count=5,
            task_ids=("task-1", "task-2", "task-3"),
        )

        assert event.aggregate_id == "workflow-123"
        assert event.event_type == "WorkflowStartedEvent"
        assert event.pattern == "SEQUENTIAL"
        assert event.task_count == 5
        assert event.task_ids == ("task-1", "task-2", "task-3")

    def test_workflow_started_event_defaults(self) -> None:
        """Test workflow started event default values."""
        event = WorkflowStartedEvent(aggregate_id="workflow-123")

        assert event.pattern == ""
        assert event.task_count == 0
        assert event.task_ids == ()

    def test_workflow_started_event_to_dict(self) -> None:
        """Test workflow started event serialization."""
        event = WorkflowStartedEvent(
            aggregate_id="workflow-123",
            pattern="PARALLEL",
            task_count=3,
            task_ids=("t1", "t2", "t3"),
        )

        data = event.to_dict()

        assert data["pattern"] == "PARALLEL"
        assert data["task_count"] == 3
        assert data["task_ids"] == ("t1", "t2", "t3")


class TestWorkflowCompletedEvent:
    """Tests for WorkflowCompletedEvent."""

    def test_create_workflow_completed_event(self) -> None:
        """Test creating workflow completed event."""
        event = WorkflowCompletedEvent(
            aggregate_id="workflow-123",
            success=True,
            duration_ms=5000,
            completed_count=5,
            failed_count=0,
        )

        assert event.event_type == "WorkflowCompletedEvent"
        assert event.success is True
        assert event.duration_ms == 5000
        assert event.completed_count == 5
        assert event.failed_count == 0

    def test_workflow_completed_event_failure(self) -> None:
        """Test workflow completed event for failed workflow."""
        event = WorkflowCompletedEvent(
            aggregate_id="workflow-123",
            success=False,
            duration_ms=1000,
            completed_count=2,
            failed_count=3,
        )

        assert event.success is False
        assert event.failed_count == 3


class TestTaskExecutedEvent:
    """Tests for TaskExecutedEvent."""

    def test_create_task_executed_event(self) -> None:
        """Test creating task executed event."""
        event = TaskExecutedEvent(
            aggregate_id="workflow-123",
            task_id="task-456",
            operation="chat_completion",
            success=True,
            duration_ms=150,
        )

        assert event.event_type == "TaskExecutedEvent"
        assert event.task_id == "task-456"
        assert event.operation == "chat_completion"
        assert event.success is True
        assert event.duration_ms == 150
        assert event.error is None

    def test_task_executed_event_with_error(self) -> None:
        """Test task executed event with error."""
        event = TaskExecutedEvent(
            aggregate_id="workflow-123",
            task_id="task-456",
            operation="chat_completion",
            success=False,
            duration_ms=1000,
            error="Connection timeout",
        )

        assert event.success is False
        assert event.error == "Connection timeout"


class TestPolicyEnforcedEvent:
    """Tests for PolicyEnforcedEvent."""

    def test_create_policy_enforced_event(self) -> None:
        """Test creating policy enforced event."""
        event = PolicyEnforcedEvent(
            aggregate_id="policy-123",
            policy_name="rate_limit",
            command_type="ExecuteTask",
            allowed=True,
            reason="Under rate limit",
        )

        assert event.event_type == "PolicyEnforcedEvent"
        assert event.policy_name == "rate_limit"
        assert event.command_type == "ExecuteTask"
        assert event.allowed is True
        assert event.reason == "Under rate limit"

    def test_policy_enforced_event_denied(self) -> None:
        """Test policy enforced event for denied command."""
        event = PolicyEnforcedEvent(
            aggregate_id="policy-123",
            policy_name="auth_check",
            command_type="DeleteWorkflow",
            allowed=False,
            reason="Insufficient permissions",
        )

        assert event.allowed is False
        assert event.reason == "Insufficient permissions"


class TestOperationLoggedEvent:
    """Tests for OperationLoggedEvent."""

    def test_create_operation_logged_event(self) -> None:
        """Test creating operation logged event."""
        event = OperationLoggedEvent(
            aggregate_id="ops-123",
            operation_name="user_login",
            context_data='{"user_id": "user-456"}',
            level="INFO",
        )

        assert event.event_type == "OperationLoggedEvent"
        assert event.operation_name == "user_login"
        assert event.context_data == '{"user_id": "user-456"}'
        assert event.level == "INFO"


class TestEventTypeRegistry:
    """Tests for event type registry and deserialization."""

    def test_all_events_registered(self) -> None:
        """Test all event types are in registry."""
        assert "OrchestratorEvent" in EVENT_TYPE_REGISTRY
        assert "WorkflowStartedEvent" in EVENT_TYPE_REGISTRY
        assert "WorkflowCompletedEvent" in EVENT_TYPE_REGISTRY
        assert "TaskExecutedEvent" in EVENT_TYPE_REGISTRY
        assert "PolicyEnforcedEvent" in EVENT_TYPE_REGISTRY
        assert "OperationLoggedEvent" in EVENT_TYPE_REGISTRY

    def test_deserialize_workflow_started_event(self) -> None:
        """Test deserializing workflow started event."""
        data = {
            "aggregate_id": "wf-123",
            "id": "event-123",
            "event_type": "WorkflowStartedEvent",
            "created_at": "2024-01-01T12:00:00+00:00",
            "correlation_id": None,
            "source": "orchestrator",
            "version": 1,
            "pattern": "SEQUENTIAL",
            "task_count": 3,
            "task_ids": ["t1", "t2", "t3"],  # List from JSON
        }

        event = deserialize_event("WorkflowStartedEvent", data)

        assert isinstance(event, WorkflowStartedEvent)
        assert event.pattern == "SEQUENTIAL"
        assert event.task_ids == ("t1", "t2", "t3")  # Converted to tuple

    def test_deserialize_unknown_event_raises_error(self) -> None:
        """Test deserializing unknown event type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown event type"):
            deserialize_event("UnknownEvent", {"aggregate_id": "agg-123"})


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore implementation."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        """Create fresh in-memory event store for each test."""
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, store: InMemoryEventStore) -> None:
        """Test appending and retrieving events."""
        event1 = WorkflowStartedEvent(
            aggregate_id="wf-123",
            pattern="SEQUENTIAL",
        )
        event2 = TaskExecutedEvent(
            aggregate_id="wf-123",
            task_id="task-1",
            success=True,
        )

        await store.append(event1)
        await store.append(event2)

        events = await store.get_events("wf-123")

        assert len(events) == 2
        assert events[0].event_type == "WorkflowStartedEvent"
        assert events[1].event_type == "TaskExecutedEvent"

    @pytest.mark.asyncio
    async def test_get_events_empty_aggregate(self, store: InMemoryEventStore) -> None:
        """Test getting events for non-existent aggregate returns empty list."""
        events = await store.get_events("non-existent")

        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_returns_chronological_order(
        self, store: InMemoryEventStore
    ) -> None:
        """Test events are returned in chronological order."""
        # Create events with explicit timestamps
        now = datetime.now(timezone.utc)
        event1 = WorkflowStartedEvent(
            aggregate_id="wf-123",
            created_at=now + timedelta(seconds=2),
        )
        event2 = TaskExecutedEvent(
            aggregate_id="wf-123",
            task_id="task-1",
            created_at=now + timedelta(seconds=1),
        )
        event3 = WorkflowCompletedEvent(
            aggregate_id="wf-123",
            created_at=now + timedelta(seconds=3),
        )

        # Append in non-chronological order
        await store.append(event1)
        await store.append(event2)
        await store.append(event3)

        events = await store.get_events("wf-123")

        # Should be sorted by created_at
        assert events[0].event_type == "TaskExecutedEvent"
        assert events[1].event_type == "WorkflowStartedEvent"
        assert events[2].event_type == "WorkflowCompletedEvent"

    @pytest.mark.asyncio
    async def test_get_events_since(self, store: InMemoryEventStore) -> None:
        """Test getting events since a timestamp."""
        now = datetime.now(timezone.utc)
        old_event = WorkflowStartedEvent(
            aggregate_id="wf-123",
            created_at=now - timedelta(hours=1),
        )
        new_event = TaskExecutedEvent(
            aggregate_id="wf-123",
            task_id="task-1",
            created_at=now + timedelta(seconds=1),
        )

        await store.append(old_event)
        await store.append(new_event)

        events = await store.get_events_since(now)

        assert len(events) == 1
        assert events[0].event_type == "TaskExecutedEvent"

    @pytest.mark.asyncio
    async def test_get_events_since_empty(self, store: InMemoryEventStore) -> None:
        """Test getting events since future timestamp returns empty."""
        event = WorkflowStartedEvent(aggregate_id="wf-123")
        await store.append(event)

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        events = await store.get_events_since(future)

        assert events == []

    @pytest.mark.asyncio
    async def test_clear_removes_all_events(self, store: InMemoryEventStore) -> None:
        """Test clearing the event store."""
        await store.append(WorkflowStartedEvent(aggregate_id="wf-1"))
        await store.append(WorkflowStartedEvent(aggregate_id="wf-2"))

        assert store.event_count == 2

        await store.clear()

        assert store.event_count == 0
        assert await store.get_events("wf-1") == []
        assert await store.get_events("wf-2") == []

    @pytest.mark.asyncio
    async def test_event_count_property(self, store: InMemoryEventStore) -> None:
        """Test event_count property."""
        assert store.event_count == 0

        await store.append(WorkflowStartedEvent(aggregate_id="wf-1"))
        assert store.event_count == 1

        await store.append(TaskExecutedEvent(aggregate_id="wf-1", task_id="t1"))
        assert store.event_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, store: InMemoryEventStore) -> None:
        """Test concurrent event appends are handled correctly."""
        events = [
            WorkflowStartedEvent(aggregate_id="wf-123", id=f"event-{i}")
            for i in range(100)
        ]

        # Append all events concurrently
        await asyncio.gather(*[store.append(event) for event in events])

        # All events should be stored
        assert store.event_count == 100
        stored_events = await store.get_events("wf-123")
        assert len(stored_events) == 100

    @pytest.mark.asyncio
    async def test_multiple_aggregates(self, store: InMemoryEventStore) -> None:
        """Test storing events for multiple aggregates."""
        await store.append(WorkflowStartedEvent(aggregate_id="wf-1"))
        await store.append(WorkflowStartedEvent(aggregate_id="wf-2"))
        await store.append(TaskExecutedEvent(aggregate_id="wf-1", task_id="t1"))

        events_wf1 = await store.get_events("wf-1")
        events_wf2 = await store.get_events("wf-2")

        assert len(events_wf1) == 2
        assert len(events_wf2) == 1

    @pytest.mark.asyncio
    async def test_append_returns_event_id(self, store: InMemoryEventStore) -> None:
        """Test append returns the event ID."""
        event = WorkflowStartedEvent(
            aggregate_id="wf-123",
            id="custom-id-123",
        )

        event_id = await store.append(event)

        assert event_id == "custom-id-123"


class TestSQLiteEventStore:
    """Tests for SQLiteEventStore implementation."""

    @pytest.fixture
    def temp_db(self) -> Path:
        """Create temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield Path(path)
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest_asyncio.fixture
    async def store(self, temp_db: Path) -> SQLiteEventStore:
        """Create and initialize SQLite event store."""
        store = SQLiteEventStore(temp_db)
        await store.initialize()
        return store

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self, temp_db: Path) -> None:
        """Test initialize creates database schema."""
        store = SQLiteEventStore(temp_db)

        await store.initialize()

        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, temp_db: Path) -> None:
        """Test initialize can be called multiple times safely."""
        store = SQLiteEventStore(temp_db)

        await store.initialize()
        await store.initialize()  # Should not raise

        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, store: SQLiteEventStore) -> None:
        """Test appending and retrieving events."""
        event1 = WorkflowStartedEvent(
            aggregate_id="wf-123",
            pattern="SEQUENTIAL",
            task_count=3,
        )
        event2 = TaskExecutedEvent(
            aggregate_id="wf-123",
            task_id="task-1",
            success=True,
            duration_ms=100,
        )

        await store.append(event1)
        await store.append(event2)

        events = await store.get_events("wf-123")

        assert len(events) == 2
        assert events[0].event_type == "WorkflowStartedEvent"
        assert events[1].event_type == "TaskExecutedEvent"

    @pytest.mark.asyncio
    async def test_get_events_preserves_data(self, store: SQLiteEventStore) -> None:
        """Test retrieved events have correct data."""
        original = WorkflowStartedEvent(
            aggregate_id="wf-123",
            pattern="PARALLEL",
            task_count=5,
            task_ids=("t1", "t2", "t3", "t4", "t5"),
            correlation_id="corr-456",
        )

        await store.append(original)
        events = await store.get_events("wf-123")

        retrieved = events[0]
        assert isinstance(retrieved, WorkflowStartedEvent)
        assert retrieved.pattern == "PARALLEL"
        assert retrieved.task_count == 5
        assert retrieved.task_ids == ("t1", "t2", "t3", "t4", "t5")
        assert retrieved.correlation_id == "corr-456"

    @pytest.mark.asyncio
    async def test_get_events_empty_aggregate(self, store: SQLiteEventStore) -> None:
        """Test getting events for non-existent aggregate."""
        events = await store.get_events("non-existent")

        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_since(self, store: SQLiteEventStore) -> None:
        """Test getting events since a timestamp."""
        now = datetime.now(timezone.utc)
        old_event = WorkflowStartedEvent(
            aggregate_id="wf-123",
            created_at=now - timedelta(hours=1),
        )
        new_event = TaskExecutedEvent(
            aggregate_id="wf-123",
            task_id="task-1",
            created_at=now + timedelta(seconds=1),
        )

        await store.append(old_event)
        await store.append(new_event)

        events = await store.get_events_since(now)

        assert len(events) == 1
        assert events[0].event_type == "TaskExecutedEvent"

    @pytest.mark.asyncio
    async def test_append_auto_initializes(self, temp_db: Path) -> None:
        """Test append automatically initializes if needed."""
        store = SQLiteEventStore(temp_db)
        event = WorkflowStartedEvent(aggregate_id="wf-123")

        # Should auto-initialize
        event_id = await store.append(event)

        assert event_id == event.id
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_get_events_auto_initializes(self, temp_db: Path) -> None:
        """Test get_events automatically initializes if needed."""
        store = SQLiteEventStore(temp_db)

        # Should auto-initialize and return empty
        events = await store.get_events("wf-123")

        assert events == []
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_get_events_since_auto_initializes(self, temp_db: Path) -> None:
        """Test get_events_since automatically initializes if needed."""
        store = SQLiteEventStore(temp_db)

        # Should auto-initialize and return empty
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        events = await store.get_events_since(past)

        assert events == []
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, store: SQLiteEventStore) -> None:
        """Test concurrent appends are handled correctly."""
        events = [
            WorkflowStartedEvent(aggregate_id="wf-123", id=f"event-{i}")
            for i in range(20)
        ]

        # Append all events concurrently
        await asyncio.gather(*[store.append(event) for event in events])

        # All events should be stored
        stored_events = await store.get_events("wf-123")
        assert len(stored_events) == 20

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_db: Path) -> None:
        """Test events persist across store instances."""
        # First instance writes
        store1 = SQLiteEventStore(temp_db)
        await store1.append(
            WorkflowStartedEvent(
                aggregate_id="wf-123",
                id="event-1",
                pattern="SEQUENTIAL",
            )
        )
        await store1.close()

        # Second instance reads
        store2 = SQLiteEventStore(temp_db)
        events = await store2.get_events("wf-123")

        assert len(events) == 1
        assert events[0].id == "event-1"

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, store: SQLiteEventStore) -> None:
        """Test close method does not raise."""
        await store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_all_event_types_persist(self, store: SQLiteEventStore) -> None:
        """Test all event types can be persisted and retrieved."""
        events_to_store = [
            WorkflowStartedEvent(
                aggregate_id="test",
                pattern="DYNAMIC",
                task_count=10,
            ),
            WorkflowCompletedEvent(
                aggregate_id="test",
                success=True,
                duration_ms=5000,
            ),
            TaskExecutedEvent(
                aggregate_id="test",
                task_id="t1",
                operation="process",
                success=True,
            ),
            PolicyEnforcedEvent(
                aggregate_id="test",
                policy_name="rate_limit",
                allowed=True,
            ),
            OperationLoggedEvent(
                aggregate_id="test",
                operation_name="user_action",
                level="INFO",
            ),
        ]

        for event in events_to_store:
            await store.append(event)

        retrieved = await store.get_events("test")

        assert len(retrieved) == 5
        event_types = [e.event_type for e in retrieved]
        assert "WorkflowStartedEvent" in event_types
        assert "WorkflowCompletedEvent" in event_types
        assert "TaskExecutedEvent" in event_types
        assert "PolicyEnforcedEvent" in event_types
        assert "OperationLoggedEvent" in event_types


class TestEventStoreError:
    """Tests for EventStoreError exception."""

    def test_error_message_format(self) -> None:
        """Test error message includes operation."""
        error = EventStoreError("Connection failed", "append")

        assert "EventStore append failed" in str(error)
        assert "Connection failed" in str(error)
        assert error.operation == "append"

    def test_error_without_operation(self) -> None:
        """Test error message without operation."""
        error = EventStoreError("Unknown error")

        assert "EventStore" in str(error)
        assert "Unknown error" in str(error)


class TestSQLiteEventStoreErrors:
    """Tests for SQLiteEventStore error handling."""

    @pytest.mark.asyncio
    async def test_initialize_without_schema_file(self) -> None:
        """Test initialize uses fallback when schema.sql is missing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = SQLiteEventStore(path)
            # Temporarily rename schema.sql to simulate missing file
            schema_path = (
                Path(__file__).parent.parent.parent.parent
                / "saga"
                / "orchestrator"
                / "schema.sql"
            )
            backup_path = schema_path.with_suffix(".sql.bak")

            schema_existed = schema_path.exists()
            if schema_existed:
                schema_path.rename(backup_path)

            try:
                await store.initialize()
                # Should work with fallback schema
                event = WorkflowStartedEvent(aggregate_id="test")
                await store.append(event)
                events = await store.get_events("test")
                assert len(events) == 1
            finally:
                if schema_existed:
                    backup_path.rename(schema_path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_initialize_with_invalid_db_path(self) -> None:
        """Test initialize fails with invalid database path."""
        # Try to create database in non-existent directory
        store = SQLiteEventStore("/nonexistent/path/events.db")

        with pytest.raises(EventStoreError, match="initialize"):
            await store.initialize()

    @pytest.mark.asyncio
    async def test_append_duplicate_id_raises_error(self) -> None:
        """Test appending event with duplicate ID raises error."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = SQLiteEventStore(path)
            await store.initialize()

            event1 = WorkflowStartedEvent(
                aggregate_id="test",
                id="duplicate-id",
            )
            event2 = WorkflowStartedEvent(
                aggregate_id="test",
                id="duplicate-id",  # Same ID
            )

            await store.append(event1)

            with pytest.raises(EventStoreError, match="append"):
                await store.append(event2)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_get_events_with_corrupted_json(self) -> None:
        """Test get_events handles corrupted JSON gracefully."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = SQLiteEventStore(path)
            await store.initialize()

            # Insert corrupted data directly
            import aiosqlite

            async with aiosqlite.connect(path) as db:
                await db.execute(
                    """
                    INSERT INTO orchestrator_events 
                    (id, aggregate_id, event_type, event_data, created_at, source, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "bad-event",
                        "test",
                        "WorkflowStartedEvent",
                        "not valid json{{{",  # Corrupted JSON
                        "2024-01-01T00:00:00+00:00",
                        "test",
                        1,
                    ),
                )
                await db.commit()

            with pytest.raises(EventStoreError, match="get_events"):
                await store.get_events("test")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_get_events_since_with_corrupted_json(self) -> None:
        """Test get_events_since handles corrupted JSON gracefully."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = SQLiteEventStore(path)
            await store.initialize()

            # Insert corrupted data directly
            import aiosqlite

            async with aiosqlite.connect(path) as db:
                await db.execute(
                    """
                    INSERT INTO orchestrator_events 
                    (id, aggregate_id, event_type, event_data, created_at, source, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "bad-event",
                        "test",
                        "WorkflowStartedEvent",
                        "{invalid}",  # Corrupted JSON
                        datetime.now(timezone.utc).isoformat(),
                        "test",
                        1,
                    ),
                )
                await db.commit()

            past = datetime.now(timezone.utc) - timedelta(hours=1)
            with pytest.raises(EventStoreError, match="get_events_since"):
                await store.get_events_since(past)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_inmemory_store_empty_aggregate_id(self) -> None:
        """Test handling empty aggregate_id in InMemoryEventStore."""
        store = InMemoryEventStore()
        event = OrchestratorEvent(aggregate_id="")

        # Should still work (no validation on aggregate_id in store)
        event_id = await store.append(event)
        events = await store.get_events("")

        assert len(events) == 1
        assert events[0].id == event_id

    @pytest.mark.asyncio
    async def test_event_with_special_characters(self) -> None:
        """Test events with special characters in data."""
        store = InMemoryEventStore()
        event = OperationLoggedEvent(
            aggregate_id="test",
            operation_name="test<>&\"'",
            context_data='{"key": "value with \\"quotes\\""}',
        )

        await store.append(event)
        events = await store.get_events("test")

        assert len(events) == 1
        assert "test<>&\"'" in events[0].operation_name  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_sqlite_with_unicode(self) -> None:
        """Test SQLite store handles unicode properly."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = SQLiteEventStore(path)
            event = OperationLoggedEvent(
                aggregate_id="test",
                operation_name="处理 🚀 Обработка",
                context_data="Unicode: 日本語 العربية",
            )

            await store.append(event)
            events = await store.get_events("test")

            assert len(events) == 1
            retrieved = events[0]
            assert isinstance(retrieved, OperationLoggedEvent)
            assert "处理" in retrieved.operation_name
            assert "🚀" in retrieved.operation_name
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_large_event_data(self) -> None:
        """Test handling large event data."""
        store = InMemoryEventStore()
        large_context = "x" * 100000  # 100KB of data
        event = OperationLoggedEvent(
            aggregate_id="test",
            operation_name="large_operation",
            context_data=large_context,
        )

        await store.append(event)
        events = await store.get_events("test")

        assert len(events) == 1
        assert len(events[0].context_data) == 100000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_get_events_since_exact_timestamp(self) -> None:
        """Test get_events_since with exact event timestamp."""
        store = InMemoryEventStore()
        now = datetime.now(timezone.utc)
        event = WorkflowStartedEvent(
            aggregate_id="wf-123",
            created_at=now,
        )

        await store.append(event)

        # Event at exact timestamp should NOT be included (exclusive)
        events = await store.get_events_since(now)
        assert len(events) == 0

        # Event before timestamp should be included
        events = await store.get_events_since(now - timedelta(milliseconds=1))
        assert len(events) == 1
