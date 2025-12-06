"""
Unit tests for ResponseMode functionality.

Tests streaming vs complete response modes in CopilotReasoningEngine.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc_saga.integrations.copilot_reasoning_engine import CopilotReasoningEngine
from arc_saga.orchestrator.types import AIProvider, AITask, AITaskInput, ResponseMode


@pytest.fixture
def mock_token_store() -> AsyncMock:
    """Mock encrypted token store."""
    store = AsyncMock()
    return store


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Mock HTTP client for testing."""
    return MagicMock()


@pytest.fixture
def copilot_engine(
    mock_token_store: AsyncMock, mock_http_client: MagicMock
) -> CopilotReasoningEngine:
    """Create CopilotReasoningEngine instance for testing."""
    engine = CopilotReasoningEngine(
        client_id="test_client_id",
        client_secret="test_client_secret",
        tenant_id="test_tenant_id",
        token_store=mock_token_store,
        http_client=mock_http_client,  # Provide mock to avoid creating real ClientSession
    )
    # Mock the auth manager
    engine.auth_manager = AsyncMock()
    engine.auth_manager.get_valid_token = AsyncMock(return_value="test_token")
    return engine


@pytest.fixture
def sample_task() -> AITask:
    """Create a sample AITask for testing."""
    return AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.COMPLETE,
    )


@pytest.mark.asyncio
async def test_response_mode_complete_returns_airesult(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify complete mode returns AIResult."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.COMPLETE,
    )

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "Test response", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    await copilot_engine.reason(task)

    # Check if it's an AIResult instance (AIResult is a type alias, check attributes)
    assert hasattr(result, "task_id")
    assert hasattr(result, "success")
    assert hasattr(result, "output_data")
    assert result.success is True
    assert result.output_data is not None
    assert result.output_data.response == "Test response"


@pytest.mark.asyncio
async def test_response_mode_streaming_yields_tokens(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify streaming mode yields tokens."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.STREAMING,
    )

    # Mock HTTP response for reason_complete (called by reason_streaming)
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "Hello world test", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    # Get streaming generator (reason() is async, await to get generator)
    stream_result = await copilot_engine.reason(task)
    assert hasattr(stream_result, "__aiter__")  # Should be async generator

    # Collect tokens
    tokens = []
    async for token in stream_result:
        tokens.append(token)

    # Verify tokens were yielded
    assert len(tokens) > 0
    # Verify full response can be reconstructed
    full_response = "".join(tokens)
    assert (
        "Hello" in full_response or "world" in full_response or "test" in full_response
    )


@pytest.mark.asyncio
async def test_response_mode_defaults_to_complete(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify default response mode is COMPLETE."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
        ),
        # response_mode not specified, should default to COMPLETE
    )

    assert task.response_mode == ResponseMode.COMPLETE


@pytest.mark.asyncio
async def test_response_mode_streaming_empty_response(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify streaming handles empty response gracefully."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.STREAMING,
    )

    # Mock empty response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    stream_result = await copilot_engine.reason(task)
    tokens = []
    async for token in stream_result:
        tokens.append(token)

    # Should handle empty response without error
    assert isinstance(tokens, list)


@pytest.mark.asyncio
async def test_response_mode_complete_with_system_prompt(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify complete mode includes system prompt in request."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="User question",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            system_prompt="You are a helpful assistant",
        ),
        response_mode=ResponseMode.COMPLETE,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "Response", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    result = await copilot_engine.reason(task)

    # Verify request included system prompt
    call_args = copilot_engine.http_client.post.call_args
    assert call_args is not None
    request_body = call_args[1]["json"]
    messages = request_body["messages"]
    assert len(messages) == 2  # System + user
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"


@pytest.mark.asyncio
async def test_response_mode_streaming_multiple_words(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify streaming yields multiple tokens for multi-word response."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.STREAMING,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": "This is a longer response with multiple words",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    stream_result = await copilot_engine.reason(task)
    tokens = []
    async for token in stream_result:
        tokens.append(token)

    # Should yield multiple tokens
    assert len(tokens) >= 3  # At least 3 words
    full_response = "".join(tokens)
    assert "longer" in full_response or "response" in full_response


@pytest.mark.asyncio
async def test_response_mode_complete_sets_stream_available(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify complete mode sets stream_available flag."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.COMPLETE,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "Test", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    result = await copilot_engine.reason(task)

    # Check result attributes (AIResult is a type alias, can't use isinstance)
    assert hasattr(result, "success")
    assert hasattr(result, "stream_available")
    assert result.stream_available is True


@pytest.mark.asyncio
async def test_response_mode_streaming_error_handling(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify streaming mode handles errors gracefully."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.STREAMING,
    )

    # Mock authentication error
    copilot_engine.auth_manager.get_valid_token = AsyncMock(
        side_effect=Exception("Auth failed")
    )

    with pytest.raises(Exception):
        stream_result = copilot_engine.reason(task)
        stream = await stream_result  # Await to get the async generator
        # Try to consume stream (should raise error)
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_response_mode_complete_error_handling(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Verify complete mode handles errors and raises them."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test prompt",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        response_mode=ResponseMode.COMPLETE,
    )

    # Mock HTTP 401 error
    mock_response = MagicMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={"error": {"message": "Unauthorized"}})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    from arc_saga.exceptions.integration_exceptions import AuthenticationError

    with pytest.raises(AuthenticationError):
        await copilot_engine.reason(task)


@pytest.mark.asyncio
async def test_response_mode_enum_values() -> None:
    """Verify ResponseMode enum has correct values."""
    assert ResponseMode.STREAMING == "streaming"
    assert ResponseMode.COMPLETE == "complete"


@pytest.mark.asyncio
async def test_response_mode_task_creation() -> None:
    """Verify tasks can be created with different response modes."""
    task_streaming = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test", model="copilot-gpt4o", provider=AIProvider.COPILOT_CHAT
        ),
        response_mode=ResponseMode.STREAMING,
    )

    task_complete = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test", model="copilot-gpt4o", provider=AIProvider.COPILOT_CHAT
        ),
        response_mode=ResponseMode.COMPLETE,
    )

    assert task_streaming.response_mode == ResponseMode.STREAMING
    assert task_complete.response_mode == ResponseMode.COMPLETE
