"""
arc_saga/tests/unit/test_storage.py
Unit tests for SQLite storage backend.

Covers: CRUD operations, FTS search, session grouping, error handling
"""

import pytest
import asyncio
from pathlib import Path
import tempfile

from arc_saga.arc_saga.storage.sqlite import SQLiteStorage
from arc_saga.arc_saga.models import Message, Provider, MessageRole, File, FileType
from arc_saga.arc_saga.exceptions import StorageError


@pytest.fixture
async def storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SQLiteStorage(db_path=db_path)
        await store.initialize()
        yield store
        # Close connection before cleanup
        if store._connection:
            store._connection.close()
            store._connection = None


@pytest.mark.asyncio
async def test_save_and_retrieve_message(storage: SQLiteStorage) -> None:
    """Test saving and retrieving a message."""
    msg = Message(
        provider=Provider.PERPLEXITY,
        role=MessageRole.USER,
        content="Test message",
        tags=["test", "unit"]
    )
    
    msg_id = await storage.save_message(msg)
    assert msg_id == msg.id
    
    retrieved = await storage.get_message_by_id(msg_id)
    assert retrieved is not None
    assert retrieved.content == "Test message"
    assert retrieved.tags == ["test", "unit"]


@pytest.mark.asyncio
async def test_search_messages(storage: SQLiteStorage) -> None:
    """Test full-text search."""
    msg1 = Message(
        provider=Provider.OPENAI,
        role=MessageRole.ASSISTANT,
        content="This is a test message about Python",
        tags=["python", "testing"]
    )
    msg2 = Message(
        provider=Provider.ANTHROPIC,
        role=MessageRole.USER,
        content="Another message about Java",
        tags=["java"]
    )
    
    await storage.save_message(msg1)
    await storage.save_message(msg2)
    
    # Search for Python
    results = await storage.search_messages("Python")
    assert len(results) >= 1
    assert any(msg1.id in r.entity_id for r in results)
    
    # Search with tag filter
    results = await storage.search_messages("message", tags=["python"])
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_session_grouping(storage: SQLiteStorage) -> None:
    """Test session grouping functionality."""
    session_id = "session-123"
    
    msg1 = Message(
        provider=Provider.PERPLEXITY,
        role=MessageRole.USER,
        content="First message",
        session_id=session_id
    )
    msg2 = Message(
        provider=Provider.PERPLEXITY,
        role=MessageRole.ASSISTANT,
        content="Second message",
        session_id=session_id
    )
    
    await storage.save_message(msg1)
    await storage.save_message(msg2)
    
    # Retrieve session
    messages = await storage.get_by_session(session_id)
    assert len(messages) == 2
    assert messages[0].content == "First message"
    assert messages[1].content == "Second message"


@pytest.mark.asyncio
async def test_save_file(storage: SQLiteStorage) -> None:
    """Test saving files."""
    file = File(
        filename="test.py",
        filepath="files/test.py",
        file_type=FileType.CODE,
        extracted_text="print('hello')",
        tags=["python"]
    )
    
    file_id = await storage.save_file(file)
    assert file_id == file.id
    
    retrieved = await storage.get_file_by_id(file_id)
    assert retrieved is not None
    assert retrieved.filename == "test.py"


@pytest.mark.asyncio
async def test_health_check(storage: SQLiteStorage) -> None:
    """Test health check."""
    is_healthy = await storage.health_check()
    assert is_healthy is True
