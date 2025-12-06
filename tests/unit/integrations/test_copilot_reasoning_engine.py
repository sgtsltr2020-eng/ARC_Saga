"""
Unit tests for CopilotReasoningEngine.

Tests task execution, error handling, response parsing, and resource management.
Target: 98%+ coverage.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from arc_saga.exceptions.integration_exceptions import (
    AuthenticationError,
    InputValidationError,
    RateLimitError,
    TransientError,
)
from arc_saga.integrations.copilot_reasoning_engine import CopilotReasoningEngine
from arc_saga.orchestrator.types import (
    AIProvider,
    AITask,
    AITaskInput,
    TaskStatus,
)


@pytest_asyncio.fixture
async def mock_token_store() -> AsyncMock:
    """Mock encrypted token store."""
    return AsyncMock()


@pytest.fixture
def mock_auth_manager() -> AsyncMock:
    """Mock EntraIDAuthManager."""
    manager = AsyncMock()
    manager.get_valid_token = AsyncMock(return_value="valid_token_123")
    return manager


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Mock HTTP client for testing."""
    return MagicMock()


@pytest.fixture
def copilot_engine(
    mock_token_store: AsyncMock, mock_http_client: MagicMock
) -> CopilotReasoningEngine:
    """Create CopilotReasoningEngine with mocked dependencies."""
    engine = CopilotReasoningEngine(
        client_id="test_client_id",
        client_secret="test_client_secret",
        tenant_id="test_tenant_id",
        token_store=mock_token_store,
        http_client=mock_http_client,  # Provide mock to avoid creating real ClientSession
    )
    # Mock auth manager to avoid OAuth calls
    engine.auth_manager = AsyncMock()
    engine.auth_manager.get_valid_token = AsyncMock(return_value="valid_token_123")
    return engine


@pytest.fixture
def sample_task() -> AITask:
    """Sample AITask for testing."""
    return AITask(
        operation="chat_completion",
        input_data=AITaskInput(
            prompt="Explain quantum computing",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=1000,
            temperature=0.7,
        ),
        metadata={"user_id": "test_user"},
    )


def create_copilot_response(
    content: str = "AI is a field of computer science",
    prompt_tokens: int = 10,
    completion_tokens: int = 50,
    finish_reason: str = "stop",
) -> dict:
    """Create mock Copilot API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@pytest.mark.asyncio
async def test_reason_successful_execution(
    copilot_engine: CopilotReasoningEngine,
    sample_task: AITask,
    mock_http_client: MagicMock,
) -> None:
    """Test successful task execution."""
    response_body = create_copilot_response()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    result = await copilot_engine.reason(sample_task)

    assert result.success is True
    assert result.output_data is not None
    assert result.output_data.response == "AI is a field of computer science"
    assert result.output_data.tokens_used == 60
    assert result.output_data.prompt_tokens == 10
    assert result.output_data.completion_tokens == 50
    assert result.output_data.provider == AIProvider.COPILOT_CHAT
    assert result.output_data.finish_reason == "stop"
    assert result.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_reason_parses_tokens_correctly(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test token parsing from Copilot response."""
    response_body = create_copilot_response(
        prompt_tokens=15, completion_tokens=25, finish_reason="length"
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    result = await copilot_engine.reason(sample_task)

    assert result.output_data is not None
    assert result.output_data.prompt_tokens == 15
    assert result.output_data.completion_tokens == 25
    assert result.output_data.tokens_used == 40
    assert result.output_data.finish_reason == "length"


@pytest.mark.asyncio
async def test_reason_includes_system_prompt(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test system prompt inclusion."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="User question",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            system_prompt="You are a helpful assistant",
            max_tokens=1000,
        ),
    )

    response_body = create_copilot_response()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    await copilot_engine.reason(task)

    # Verify system prompt was included in request
    call_args = mock_session.post.call_args
    assert call_args is not None
    request_body = call_args[1]["json"]
    assert request_body["messages"][0]["role"] == "system"
    assert request_body["messages"][0]["content"] == "You are a helpful assistant"
    assert request_body["messages"][1]["role"] == "user"
    assert request_body["messages"][1]["content"] == "User question"


@pytest.mark.asyncio
async def test_reason_auth_error_on_401(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 401 authentication error."""
    mock_response = MagicMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(
        return_value={"error": {"code": "Unauthorized", "message": "Invalid token"}}
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(AuthenticationError, match="Copilot"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_rate_limit_on_429(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 429 rate limit error."""
    mock_response = MagicMock()
    mock_response.status = 429
    mock_response.json = AsyncMock(return_value={"error": {"message": "Rate limited"}})
    mock_response.headers = {"Retry-After": "60"}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(RateLimitError, match="Retry-After"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_timeout_on_408(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 408 request timeout."""
    mock_response = MagicMock()
    mock_response.status = 408
    mock_response.json = AsyncMock(return_value={})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(TimeoutError, match="timeout"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_timeout_on_504(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 504 gateway timeout."""
    mock_response = MagicMock()
    mock_response.status = 504
    mock_response.json = AsyncMock(return_value={})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(TimeoutError, match="timeout"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_input_too_large_on_413(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 413 payload too large."""
    mock_response = MagicMock()
    mock_response.status = 413
    mock_response.json = AsyncMock(return_value={})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(InputValidationError, match="too large"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_bad_request_on_400(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 400 bad request."""
    mock_response = MagicMock()
    mock_response.status = 400
    mock_response.json = AsyncMock(
        return_value={"error": {"code": "InvalidRequest", "message": "Bad request"}}
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(ValueError, match="Invalid Copilot request"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_service_error_on_500(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test HTTP 500 service error."""
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.json = AsyncMock(
        return_value={
            "error": {"code": "InternalServerError", "message": "Server error"}
        }
    )
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(TransientError, match="service error"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_respects_timeout(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test timeout is respected."""
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt="Test",
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=100,
        ),
        timeout_ms=5000,  # 5 seconds
    )

    response_body = create_copilot_response()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    await copilot_engine.reason(task)

    # Verify timeout was passed to ClientTimeout
    call_args = mock_session.post.call_args
    assert call_args is not None
    timeout = call_args[1]["timeout"]
    assert timeout.total == 5.0


@pytest.mark.asyncio
async def test_reason_with_large_prompt(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test large prompt handling."""
    large_prompt = "x" * 2000
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt=large_prompt,
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=1000,
        ),
    )

    response_body = create_copilot_response()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    result = await copilot_engine.reason(task)
    assert result.success is True


@pytest.mark.asyncio
async def test_reason_with_oversized_prompt(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test oversized prompt (HTTP 413)."""
    oversized_prompt = "x" * 100000  # Very large
    task = AITask(
        operation="chat",
        input_data=AITaskInput(
            prompt=oversized_prompt,
            model="copilot-gpt4o",
            provider=AIProvider.COPILOT_CHAT,
            max_tokens=1000,
        ),
    )

    mock_response = MagicMock()
    mock_response.status = 413
    mock_response.json = AsyncMock(return_value={})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(InputValidationError):
        await copilot_engine.reason(task)


@pytest.mark.asyncio
async def test_reason_missing_choices_in_response(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test missing 'choices' in response."""
    response_body = {"error": {"message": "Service unavailable"}}

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(ValueError, match="missing 'choices'"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_empty_choices_in_response(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test empty 'choices' array."""
    response_body = {"choices": []}

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(ValueError, match="empty or invalid 'choices'"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_missing_message_in_choice(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test missing 'message' in choice."""
    response_body = {"choices": [{"finish_reason": "stop"}]}

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(ValueError, match="missing 'message'"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_missing_content_in_message(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test missing/empty 'content' in message (non-fatal)."""
    response_body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    # Missing content
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    result = await copilot_engine.reason(sample_task)

    # Should succeed but with empty content
    assert result.success is True
    assert result.output_data is not None
    assert result.output_data.response == ""


@pytest.mark.asyncio
async def test_reason_auth_manager_raises_auth_error(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test auth manager raises AuthenticationError."""
    copilot_engine.auth_manager.get_valid_token = AsyncMock(
        side_effect=AuthenticationError("Token invalid")
    )

    with pytest.raises(AuthenticationError):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_network_error_during_request(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test network error during request."""
    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(TransientError, match="Network error"):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_reason_async_timeout_during_request(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test async timeout during request."""
    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(
        side_effect=asyncio.TimeoutError("Request timeout")
    )
    mock_post.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client.post = MagicMock(return_value=mock_post)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await copilot_engine.reason(sample_task)


@pytest.mark.asyncio
async def test_close_cleans_up_resources(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test close() cleans up resources."""
    mock_client = AsyncMock()
    copilot_engine.http_client = mock_client
    copilot_engine._owns_http_client = True

    await copilot_engine.close()

    mock_client.close.assert_called_once()

    # Safe to call multiple times
    await copilot_engine.close()
    assert mock_client.close.call_count == 2


@pytest.mark.asyncio
async def test_close_owned_http_client(
    copilot_engine: CopilotReasoningEngine,
) -> None:
    """Test close() with owned HTTP client."""
    mock_client = AsyncMock()
    copilot_engine.http_client = mock_client
    copilot_engine._owns_http_client = True

    await copilot_engine.close()

    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_external_http_client() -> None:
    """Test close() with external HTTP client."""
    mock_store = AsyncMock()
    mock_client = AsyncMock()

    engine = CopilotReasoningEngine(
        client_id="test",
        client_secret="test",
        tenant_id="test",
        token_store=mock_store,
        http_client=mock_client,  # External client
    )

    await engine.close()

    # Should NOT close external client
    mock_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_constructor_validates_parameters(mock_token_store: AsyncMock) -> None:
    """Test constructor validates parameters."""
    with pytest.raises(ValueError, match="client_id cannot be empty"):
        CopilotReasoningEngine("", "secret", "tenant", mock_token_store)

    with pytest.raises(ValueError, match="client_secret cannot be empty"):
        CopilotReasoningEngine("client_id", "", "tenant", mock_token_store)

    with pytest.raises(ValueError, match="tenant_id cannot be empty"):
        CopilotReasoningEngine("client_id", "secret", "", mock_token_store)


@pytest.mark.asyncio
async def test_reason_logging_context(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test logging context includes correct fields."""
    response_body = create_copilot_response()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_body)
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with patch(
        "arc_saga.integrations.copilot_reasoning_engine.log_with_context"
    ) as mock_log:
        await copilot_engine.reason(sample_task)

        # Verify logging calls
        log_calls = [
            call[0][1] for call in mock_log.call_args_list
        ]  # Extract event names
        assert "copilot_request_start" in log_calls
        assert "copilot_request_complete" in log_calls


@pytest.mark.asyncio
async def test_truncate_helper(copilot_engine: CopilotReasoningEngine) -> None:
    """Test _truncate helper function."""
    long_string = "x" * 1000
    truncated = copilot_engine._truncate(long_string, max_length=500)

    assert len(truncated) == 503  # 500 + "..."
    assert truncated.endswith("...")
    assert truncated.startswith("x" * 500)

    short_string = "short"
    assert copilot_engine._truncate(short_string, max_length=500) == short_string


@pytest.mark.asyncio
async def test_reason_unexpected_status_code(
    copilot_engine: CopilotReasoningEngine, sample_task: AITask
) -> None:
    """Test unexpected status code."""
    mock_response = MagicMock()
    mock_response.status = 418  # I'm a teapot
    mock_response.json = AsyncMock(return_value={})
    mock_response.headers = {}

    mock_post = AsyncMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    copilot_engine.http_client = mock_session

    with pytest.raises(ValueError, match="Unexpected status"):
        await copilot_engine.reason(sample_task)
