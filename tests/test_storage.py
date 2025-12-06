"""Test SQLite storage"""

import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
from arc_saga.storage.sqlite import SQLiteStorage
from arc_saga.models import Message, MessageRole, Provider


@pytest_asyncio.fixture
async def storage():
    """Create temporary storage for testing"""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))
        await stor.initialize()
        yield stor
    finally:
        # Simple cleanup - just remove directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_save_message(storage):
    """Test saving a message"""
    msg = Message(
        provider=Provider.OPENAI, role=MessageRole.USER, content="Test message"
    )

    message_id = await storage.save_message(msg)

    assert message_id == msg.id


@pytest.mark.asyncio
async def test_get_message_by_id(storage):
    """Test retrieving a message"""
    msg = Message(
        provider=Provider.OPENAI, role=MessageRole.USER, content="Test message"
    )

    await storage.save_message(msg)
    retrieved = await storage.get_message_by_id(msg.id)

    assert retrieved is not None
    assert retrieved.content == "Test message"


@pytest.mark.asyncio
async def test_search_messages(storage):
    """Test searching messages"""
    msg1 = Message(
        provider=Provider.OPENAI, role=MessageRole.USER, content="Python programming"
    )
    msg2 = Message(
        provider=Provider.OPENAI,
        role=MessageRole.USER,
        content="JavaScript development",
    )

    await storage.save_message(msg1)
    await storage.save_message(msg2)

    results = await storage.search_messages("Python", limit=10)

    assert len(results) >= 1
    assert any("Python" in r.content for r in results)
