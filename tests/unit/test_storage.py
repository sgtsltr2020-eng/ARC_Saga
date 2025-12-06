"""
arc_saga/tests/unit/test_storage.py
Unit tests for SQLite storage backend.

Covers: CRUD operations, FTS search, session grouping, error handling
"""

import pytest
from pathlib import Path
import tempfile
import pytest_asyncio
import shutil
import sqlite3
from unittest.mock import patch, MagicMock

from arc_saga.storage.sqlite import SQLiteStorage
from arc_saga.models import Message, Provider, MessageRole, File, FileType
from arc_saga.exceptions import StorageError


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
async def test_save_and_retrieve_message(storage: SQLiteStorage) -> None:
    """Test saving and retrieving a message."""
    msg = Message(
        provider=Provider.PERPLEXITY,
        role=MessageRole.USER,
        content="Test message",
        tags=["test", "unit"],
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
        tags=["python", "testing"],
    )
    msg2 = Message(
        provider=Provider.ANTHROPIC,
        role=MessageRole.USER,
        content="Another message about Java",
        tags=["java"],
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
        session_id=session_id,
    )
    msg2 = Message(
        provider=Provider.PERPLEXITY,
        role=MessageRole.ASSISTANT,
        content="Second message",
        session_id=session_id,
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
        tags=["python"],
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


@pytest.mark.asyncio
async def test_health_check_failure() -> None:
    """Test health check failure path."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))
        await stor.initialize()

        # Mock connection to raise exception
        with patch.object(
            stor, "_get_connection", side_effect=sqlite3.Error("DB error")
        ):
            is_healthy = await stor.health_check()
            assert is_healthy is False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_get_message_by_id_not_found(storage: SQLiteStorage) -> None:
    """Test get_message_by_id returns None when message not found."""
    result = await storage.get_message_by_id("non-existent-id")
    assert result is None


@pytest.mark.asyncio
async def test_get_file_by_id_not_found(storage: SQLiteStorage) -> None:
    """Test get_file_by_id returns None when file not found."""
    result = await storage.get_file_by_id("non-existent-id")
    assert result is None


@pytest.mark.asyncio
async def test_save_message_database_error() -> None:
    """Test save_message raises StorageError on database error."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))
        await stor.initialize()

        msg = Message(
            provider=Provider.PERPLEXITY,
            role=MessageRole.USER,
            content="Test message",
        )

        # Mock _get_connection to return a connection with a cursor that raises errors
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.Error("DB error")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(stor, "_get_connection", return_value=mock_conn):
            with pytest.raises(StorageError) as exc_info:
                await stor.save_message(msg)
            assert exc_info.value.operation == "save_message"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_save_file_database_error() -> None:
    """Test save_file raises StorageError on database error."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))
        await stor.initialize()

        file = File(
            filename="test.py",
            filepath="files/test.py",
            file_type=FileType.CODE,
            extracted_text="print('hello')",
        )

        # Mock _get_connection to return a connection with a cursor that raises errors
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.Error("DB error")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(stor, "_get_connection", return_value=mock_conn):
            with pytest.raises(StorageError) as exc_info:
                await stor.save_file(file)
            assert exc_info.value.operation == "save_file"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_search_messages_database_error(storage: SQLiteStorage) -> None:
    """Test search_messages raises StorageError on database error."""
    # Mock _get_connection to return a connection with a cursor that raises errors
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = sqlite3.Error("DB error")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(storage, "_get_connection", return_value=mock_conn):
        with pytest.raises(StorageError) as exc_info:
            await storage.search_messages("test")
        assert exc_info.value.operation == "search_messages"


@pytest.mark.asyncio
async def test_get_message_by_id_database_error(storage: SQLiteStorage) -> None:
    """Test get_message_by_id raises StorageError on database error."""
    # Mock _get_connection to return a connection with a cursor that raises errors
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = sqlite3.Error("DB error")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(storage, "_get_connection", return_value=mock_conn):
        with pytest.raises(StorageError) as exc_info:
            await storage.get_message_by_id("test-id")
        assert exc_info.value.operation == "get_message_by_id"


@pytest.mark.asyncio
async def test_get_file_by_id_database_error(storage: SQLiteStorage) -> None:
    """Test get_file_by_id raises StorageError on database error."""
    # Mock _get_connection to return a connection with a cursor that raises errors
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = sqlite3.Error("DB error")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(storage, "_get_connection", return_value=mock_conn):
        with pytest.raises(StorageError) as exc_info:
            await storage.get_file_by_id("test-id")
        assert exc_info.value.operation == "get_file_by_id"


@pytest.mark.asyncio
async def test_get_by_session_database_error(storage: SQLiteStorage) -> None:
    """Test get_by_session raises StorageError on database error."""
    # Mock _get_connection to return a connection with a cursor that raises errors
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = sqlite3.Error("DB error")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(storage, "_get_connection", return_value=mock_conn):
        with pytest.raises(StorageError) as exc_info:
            await storage.get_by_session("session-id")
        assert exc_info.value.operation == "get_by_session"


@pytest.mark.asyncio
async def test_initialize_database_error() -> None:
    """Test initialize raises StorageError on database error."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))

        # Mock sqlite3.connect to raise error
        with patch(
            "arc_saga.storage.sqlite.sqlite3.connect",
            side_effect=sqlite3.Error("DB error"),
        ):
            with pytest.raises(StorageError) as exc_info:
                await stor.initialize()
            assert exc_info.value.operation == "initialize"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_storage_init_with_default_path() -> None:
    """Test SQLiteStorage initialization with db_path=None uses default."""
    with patch("shared.config.SharedConfig.DB_PATH", "~/.arc-saga/test-default.db"):
        stor = SQLiteStorage(db_path=None)
        # Verify it uses the default path
        assert stor.db_path is not None
        assert ".arc-saga" in str(stor.db_path) or "arc-saga" in str(stor.db_path)
