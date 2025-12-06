"""
Unit tests for Perplexity client.

Tests verify:
1. Bug fix: save_message() is called (not store_message())
2. Bug fix: session_id is used (not thread_id)
3. Error handling with log_with_context()
4. Type safety and proper initialization

Coverage target: 95%+
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arc_saga.integrations.perplexity_client import (
    PerplexityAPIError,
    PerplexityClient,
    PerplexityClientError,
    PerplexityStorageError,
)
from arc_saga.models import Message, MessageRole, Provider


class MockStorageBackend:
    """Mock storage backend for testing."""

    def __init__(self) -> None:
        self.saved_messages: list[Message] = []
        self.save_message_calls: int = 0
        self.should_fail_save: bool = False
        self.should_fail_get_session: bool = False

    async def initialize(self) -> None:
        """Initialize mock storage."""
        pass

    async def save_message(self, message: Message) -> str:
        """Save message and return ID."""
        self.save_message_calls += 1
        if self.should_fail_save:
            raise Exception("Mock storage error")
        self.saved_messages.append(message)
        return message.id

    async def save_file(self, file: Any) -> str:
        """Save file placeholder."""
        return str(uuid.uuid4())

    async def search_messages(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 50
    ) -> list[Any]:
        """Search messages placeholder."""
        return []

    async def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        for msg in self.saved_messages:
            if msg.id == message_id:
                return msg
        return None

    async def get_file_by_id(self, file_id: str) -> Optional[Any]:
        """Get file by ID placeholder."""
        return None

    async def get_by_session(self, session_id: str) -> list[Message]:
        """Get messages by session."""
        if self.should_fail_get_session:
            raise Exception("Mock session retrieval error")
        return [msg for msg in self.saved_messages if msg.session_id == session_id]

    async def health_check(self) -> bool:
        """Health check placeholder."""
        return True


class MockStreamChunk:
    """Mock chunk from OpenAI streaming response."""

    def __init__(self, content: Optional[str]) -> None:
        self.choices = [MagicMock()]
        self.choices[0].delta = MagicMock()
        self.choices[0].delta.content = content


async def mock_stream_generator(chunks: list[str]) -> AsyncIterator[MockStreamChunk]:
    """Generate mock stream chunks."""
    for chunk in chunks:
        yield MockStreamChunk(chunk)


class TestPerplexityClientInitialization:
    """Tests for PerplexityClient initialization."""

    def test_init_with_valid_api_key(self) -> None:
        """Test initialization with valid API key succeeds."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-api-key", storage=storage)

        assert client.storage is storage
        assert client.client is not None

    def test_init_with_empty_api_key_raises_error(self) -> None:
        """Test initialization with empty API key raises ValueError."""
        storage = MockStorageBackend()

        with pytest.raises(ValueError, match="API key cannot be empty"):
            PerplexityClient(api_key="", storage=storage)

    def test_init_with_whitespace_api_key_raises_error(self) -> None:
        """Test initialization with whitespace API key raises ValueError."""
        storage = MockStorageBackend()

        with pytest.raises(ValueError, match="API key cannot be empty"):
            PerplexityClient(api_key="   ", storage=storage)


class TestAskStreaming:
    """Tests for ask_streaming method."""

    @pytest.mark.asyncio
    async def test_ask_streaming_uses_save_message_not_store_message(self) -> None:
        """
        BUG FIX TEST: Verify save_message() is called, not store_message().

        This test ensures the bug fix is working - the old code incorrectly
        called store_message() which doesn't exist on StorageBackend.
        """
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        # Mock the OpenAI client
        mock_stream = mock_stream_generator(["Hello", " world", "!"])

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            # Consume the generator
            chunks = []
            async for chunk in client.ask_streaming("Test query"):
                chunks.append(chunk)

            # Verify save_message was called (not store_message)
            assert storage.save_message_calls == 2  # user + assistant
            assert len(storage.saved_messages) == 2

    @pytest.mark.asyncio
    async def test_ask_streaming_uses_session_id_not_thread_id(self) -> None:
        """
        BUG FIX TEST: Verify session_id is used, not thread_id.

        This test ensures the bug fix is working - the old code incorrectly
        used thread_id which doesn't exist on Message model.
        """
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])
        test_session_id = "test-session-123"

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming(
                "Test query",
                session_id=test_session_id
            ):
                pass

            # Verify both messages have session_id set correctly
            for msg in storage.saved_messages:
                assert msg.session_id == test_session_id
                # Verify thread_id attribute doesn't exist
                assert not hasattr(msg, "thread_id")

    @pytest.mark.asyncio
    async def test_ask_streaming_generates_session_id_if_not_provided(self) -> None:
        """Test that session_id is auto-generated when not provided."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming("Test query"):
                pass

            # Verify session_id was generated and is a valid UUID
            assert len(storage.saved_messages) == 2
            session_id = storage.saved_messages[0].session_id
            assert session_id is not None
            # Verify it's a valid UUID format
            uuid.UUID(session_id)
            # Both messages should have same session_id
            assert storage.saved_messages[1].session_id == session_id

    @pytest.mark.asyncio
    async def test_ask_streaming_stores_user_message_with_correct_fields(self) -> None:
        """Test user message is stored with correct provider and role."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])
        query = "What is Python?"

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming(query):
                pass

            user_msg = storage.saved_messages[0]
            assert user_msg.role == MessageRole.USER
            assert user_msg.provider == Provider.PERPLEXITY
            assert user_msg.content == query

    @pytest.mark.asyncio
    async def test_ask_streaming_stores_assistant_message_with_correct_fields(
        self
    ) -> None:
        """Test assistant message is stored with correct provider and role."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Hello", " ", "world!"])

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming("Test"):
                pass

            assistant_msg = storage.saved_messages[1]
            assert assistant_msg.role == MessageRole.ASSISTANT
            assert assistant_msg.provider == Provider.PERPLEXITY
            assert assistant_msg.content == "Hello world!"

    @pytest.mark.asyncio
    async def test_ask_streaming_yields_chunks_correctly(self) -> None:
        """Test streaming yields chunks in correct format."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        expected_chunks = ["Hello", " ", "world"]
        mock_stream = mock_stream_generator(expected_chunks)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            received_chunks = []
            async for chunk in client.ask_streaming("Test"):
                data = json.loads(chunk)
                if data["type"] == "chunk":
                    received_chunks.append(data["content"])

            assert received_chunks == expected_chunks

    @pytest.mark.asyncio
    async def test_ask_streaming_yields_completion_signal(self) -> None:
        """Test streaming yields completion signal at end."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])
        session_id = "test-session"

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            chunks = []
            async for chunk in client.ask_streaming("Test", session_id=session_id):
                chunks.append(json.loads(chunk))

            # Last chunk should be completion signal
            last_chunk = chunks[-1]
            assert last_chunk["type"] == "complete"
            assert last_chunk["session_id"] == session_id
            assert "correlation_id" in last_chunk

    @pytest.mark.asyncio
    async def test_ask_streaming_includes_context_in_api_call(self) -> None:
        """Test context messages are included in API call."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])
        context = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            async for _ in client.ask_streaming("New question", context=context):
                pass

            # Verify API was called with context + new message
            call_args = mock_create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 3  # 2 context + 1 new
            assert messages[0] == context[0]
            assert messages[1] == context[1]
            assert messages[2]["content"] == "New question"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_user_message_storage_failure_raises_error(self) -> None:
        """Test that user message storage failure raises PerplexityStorageError."""
        storage = MockStorageBackend()
        storage.should_fail_save = True
        client = PerplexityClient(api_key="test-key", storage=storage)

        with pytest.raises(PerplexityStorageError, match="Failed to store user message"):
            async for _ in client.ask_streaming("Test"):
                pass

    @pytest.mark.asyncio
    async def test_api_failure_yields_error_and_raises(self) -> None:
        """Test API failure yields error chunk and raises exception."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            chunks = []
            with pytest.raises(PerplexityAPIError, match="Perplexity API call failed"):
                async for chunk in client.ask_streaming("Test"):
                    chunks.append(json.loads(chunk))

            # Should have yielded error chunk before raising
            # Note: User message is stored first, then API fails
            error_chunks = [c for c in chunks if c.get("type") == "error"]
            assert len(error_chunks) == 1
            assert "API Error" in error_chunks[0]["message"]

    @pytest.mark.asyncio
    async def test_assistant_storage_failure_logs_warning_but_completes(self) -> None:
        """Test assistant storage failure logs warning but doesn't fail request."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        mock_stream = mock_stream_generator(["Response"])
        call_count = 0

        async def failing_save(message: Message) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (user message) succeeds
                storage.saved_messages.append(message)
                return message.id
            else:
                # Second call (assistant message) fails
                raise Exception("Storage error")

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream

            with patch.object(storage, "save_message", side_effect=failing_save):
                chunks = []
                async for chunk in client.ask_streaming("Test"):
                    chunks.append(json.loads(chunk))

                # Should still complete successfully
                assert any(c["type"] == "complete" for c in chunks)


class TestGetSessionHistory:
    """Tests for get_session_history method."""

    @pytest.mark.asyncio
    async def test_get_session_history_returns_messages(self) -> None:
        """Test get_session_history returns messages for session."""
        storage = MockStorageBackend()
        client = PerplexityClient(api_key="test-key", storage=storage)

        # Add some messages to storage
        session_id = "test-session"
        msg1 = Message(
            provider=Provider.PERPLEXITY,
            role=MessageRole.USER,
            content="Question",
            session_id=session_id
        )
        msg2 = Message(
            provider=Provider.PERPLEXITY,
            role=MessageRole.ASSISTANT,
            content="Answer",
            session_id=session_id
        )
        storage.saved_messages = [msg1, msg2]

        result = await client.get_session_history(session_id)

        assert len(result) == 2
        assert result[0].content == "Question"
        assert result[1].content == "Answer"

    @pytest.mark.asyncio
    async def test_get_session_history_storage_failure_raises_error(self) -> None:
        """Test storage failure raises PerplexityStorageError."""
        storage = MockStorageBackend()
        storage.should_fail_get_session = True
        client = PerplexityClient(api_key="test-key", storage=storage)

        with pytest.raises(
            PerplexityStorageError,
            match="Failed to get session history"
        ):
            await client.get_session_history("test-session")


class TestExceptionHierarchy:
    """Tests for exception classes."""

    def test_perplexity_client_error_is_base(self) -> None:
        """Test PerplexityClientError is the base exception."""
        assert issubclass(PerplexityAPIError, PerplexityClientError)
        assert issubclass(PerplexityStorageError, PerplexityClientError)

    def test_perplexity_api_error_stores_original(self) -> None:
        """Test PerplexityAPIError stores original exception."""
        original = ValueError("Original error")
        error = PerplexityAPIError("API failed", original_error=original)

        assert error.original_error is original
        assert "API failed" in str(error)

    def test_perplexity_storage_error_stores_original(self) -> None:
        """Test PerplexityStorageError stores original exception."""
        original = IOError("Storage failed")
        error = PerplexityStorageError("Storage error", original_error=original)

        assert error.original_error is original
        assert "Storage error" in str(error)

