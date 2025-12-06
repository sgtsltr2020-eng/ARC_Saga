"""
Copilot Reasoning Engine Implementation.

Microsoft 365 Copilot Chat API integration with Entra ID authentication.
Implements IReasoningEngine protocol for provider-agnostic orchestration.

This module provides:
- CopilotReasoningEngine: IReasoningEngine implementation
- Automatic token management via EntraIDAuthManager
- Defensive response parsing with comprehensive error handling
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any, AsyncGenerator, Optional, Union

import aiohttp

from ..error_instrumentation import log_with_context
from ..exceptions.integration_exceptions import (
    AuthenticationError,
    InputValidationError,
    RateLimitError,
    TransientError,
)
from ..orchestrator.protocols import IEncryptedStore
from ..orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    ResponseMode,
    TaskStatus,
)
from .entra_id_auth_manager import EntraIDAuthManager

# Constants
COPILOT_CHAT_ENDPOINT = "/copilot/chat"
DEFAULT_TIMEOUT_SECONDS = 30


class CopilotReasoningEngine:
    """
    Microsoft Copilot reasoning engine.

    Executes AI tasks via Microsoft Graph Copilot Chat API with automatic
    authentication and token management. Implements IReasoningEngine protocol.

    Attributes:
        auth_manager: Entra ID authentication manager
        http_client: HTTP client for API calls (owned or external)
        graph_endpoint: Microsoft Graph API endpoint base URL
        _owns_http_client: Whether engine owns HTTP client (for cleanup)

    Example:
        >>> store = SQLiteEncryptedTokenStore("~/.arc_saga/tokens.db")
        >>> engine = CopilotReasoningEngine(
        ...     client_id="...",
        ...     client_secret="...",
        ...     tenant_id="...",
        ...     token_store=store,
        ... )
        >>> result = await engine.reason(task)
        >>> await engine.close()
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        token_store: IEncryptedStore,
        http_client: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """
        Initialize Copilot reasoning engine.

        Args:
            client_id: Azure AD application ID
            client_secret: OAuth2 client secret
            tenant_id: Azure AD tenant ID
            token_store: Encrypted token storage backend
            http_client: Optional HTTP client (created if None)

        Raises:
            ValueError: If any required parameter is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id cannot be empty")
        if not client_secret or not client_secret.strip():
            raise ValueError("client_secret cannot be empty")
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")

        self.auth_manager = EntraIDAuthManager(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            token_store=token_store,
        )

        self.http_client = http_client or aiohttp.ClientSession()
        self._owns_http_client = http_client is None
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"

        log_with_context(
            "info",
            "copilot_engine_initialized",
            tenant_id=tenant_id,
            owns_http_client=self._owns_http_client,
        )

    async def reason(self, task: AITask) -> Union[AIResult, AsyncGenerator[str, None]]:
        """
        Execute AI reasoning task via Copilot Chat API.

        Implements IReasoningEngine.reason() protocol. Supports both streaming
        and complete response modes based on task.response_mode.

        Args:
            task: AI task with prompt, model, provider, response_mode, etc.

        Returns:
            - If ResponseMode.COMPLETE: AIResult with full response
            - If ResponseMode.STREAMING: AsyncGenerator[str, None] yielding tokens

        Raises:
            AuthenticationError: If token authentication fails
            RateLimitError: If Copilot rate limit exceeded
            TimeoutError: If request times out
            InputValidationError: If prompt too large
            ValueError: If request format invalid
            TransientError: If transient error occurs

        Example:
            >>> task = AITask(
            ...     operation="chat",
            ...     input_data=AITaskInput(
            ...         prompt="Explain quantum computing",
            ...         provider=AIProvider.COPILOT_CHAT,
            ...         max_tokens=1000,
            ...     ),
            ...     response_mode=ResponseMode.COMPLETE,
            ... )
            >>> result = await engine.reason(task)
            >>> # For streaming:
            >>> task.response_mode = ResponseMode.STREAMING
            >>> async for token in engine.reason(task):
            ...     print(token, end="", flush=True)
        """
        # Dispatch based on response mode
        if task.response_mode == ResponseMode.STREAMING:
            return self.reason_streaming(task)
        else:
            return await self.reason_complete(task)

    async def reason_complete(self, task: AITask) -> AIResult:
        """
        Execute task and return complete AIResult.

        Collects all tokens from streaming response and returns a single
        AIResult with the full response text.

        Args:
            task: AI task to execute

        Returns:
            AIResult with complete response, token usage, and metadata

        Raises:
            Same as reason() method
        """
        start_time = time.time()

        # Extract user_id from task metadata or input_data
        # AITaskInput doesn't have user_id, so check task.metadata
        user_id = "default_user"
        if hasattr(task, "metadata") and isinstance(task.metadata, dict):
            user_id = task.metadata.get("user_id", user_id)

        # Get valid token (may raise AuthenticationError)
        try:
            token = await self.auth_manager.get_valid_token(user_id)
        except AuthenticationError as e:
            log_with_context(
                "error",
                "copilot_auth_failed",
                task_id=task.id,
                user_id=user_id,
                error=str(e)[:200],  # Truncate
                error_type=type(e).__name__,
            )
            raise

        # Build request body
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": task.input_data.prompt,
            }
        ]

        # Include system prompt if provided
        if task.input_data.system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": task.input_data.system_prompt,
                },
            )

        request_body: dict[str, list[dict[str, str]] | int | float] = {
            "messages": messages,
            "max_tokens": task.input_data.max_tokens,
            "temperature": task.input_data.temperature,
        }

        log_with_context(
            "info",
            "copilot_request_start",
            task_id=task.id,
            user_id=user_id,
            prompt_length=len(task.input_data.prompt),
            max_tokens=task.input_data.max_tokens,
            temperature=task.input_data.temperature,
        )

        # Make HTTP request
        timeout_secs = (
            task.timeout_ms / 1000 if task.timeout_ms else DEFAULT_TIMEOUT_SECONDS
        )

        try:
            async with self.http_client.post(
                f"{self.graph_endpoint}{COPILOT_CHAT_ENDPOINT}",
                headers={"Authorization": f"Bearer {token}"},
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout_secs),
            ) as response:
                status_code = response.status
                response_body = await response.json()

                # Handle errors by status code
                if status_code == 401:
                    error_detail = response_body.get("error", {}).get(
                        "message", "Unauthorized"
                    )
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=401,
                        error=self._truncate(str(error_detail), 200),
                    )
                    raise AuthenticationError(f"Copilot: {error_detail}")

                elif status_code == 429:
                    retry_after = response.headers.get("Retry-After", "unknown")
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=429,
                        retry_after=retry_after,
                    )
                    raise RateLimitError(
                        f"Copilot rate limited. Retry-After: {retry_after}"
                    )

                elif status_code in [408, 504]:
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=status_code,
                    )
                    raise TimeoutError(
                        f"Copilot request timeout (status {status_code})"
                    )

                elif status_code == 413:
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=413,
                    )
                    raise InputValidationError(
                        "Prompt too large for Copilot (HTTP 413)"
                    )

                elif status_code == 400:
                    error_detail = response_body.get("error", {}).get(
                        "message", "Bad request"
                    )
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=400,
                        error=self._truncate(str(error_detail), 200),
                    )
                    raise ValueError(f"Invalid Copilot request: {error_detail}")

                elif status_code >= 500:
                    error_detail = response_body.get("error", {}).get(
                        "message", f"Service error {status_code}"
                    )
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=status_code,
                        error=self._truncate(str(error_detail), 200),
                    )
                    raise TransientError(
                        f"Copilot service error: {status_code}: {error_detail}"
                    )

                elif status_code != 200:
                    log_with_context(
                        "error",
                        "copilot_error",
                        task_id=task.id,
                        status_code=status_code,
                    )
                    raise ValueError(f"Unexpected status from Copilot: {status_code}")

                # Parse response defensively (HTTP 200)
                try:
                    response_text, finish_reason, usage = self._parse_copilot_response(
                        response_body
                    )
                except ValueError as e:
                    log_with_context(
                        "error",
                        "copilot_parse_error",
                        task_id=task.id,
                        error=str(e)[:200],  # Truncate
                        response_body_preview=self._truncate(str(response_body), 300),
                    )
                    raise ValueError(f"Failed to parse Copilot response: {e}") from e

                # Extract token usage
                total_tokens = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Build AIResultOutput
                duration_ms = int((time.time() - start_time) * 1000)

                output = AIResultOutput(
                    response=response_text,
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider=AIProvider.COPILOT_CHAT,
                    model="copilot-gpt4o",
                    cost_usd=Decimal("0.0"),  # Computed offline in Step 3
                    finish_reason=finish_reason,
                    latency_ms=duration_ms,
                )

                log_with_context(
                    "info",
                    "copilot_request_complete",
                    task_id=task.id,
                    user_id=user_id,
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=duration_ms,
                    finish_reason=finish_reason,
                )

                return AIResult(
                    task_id=task.id,
                    success=True,
                    output_data=output,
                    status=TaskStatus.COMPLETED,
                    duration_ms=duration_ms,
                    stream_available=True,  # Copilot supports streaming
                )

        except (
            AuthenticationError,
            RateLimitError,
            TimeoutError,
            InputValidationError,
            ValueError,
            TransientError,
        ):
            # Re-raise known exceptions
            raise
        except asyncio.TimeoutError as e:
            log_with_context(
                "error",
                "copilot_error",
                task_id=task.id,
                error_type="TimeoutError",
                error_message=str(e)[:200],  # Truncate
            )
            raise TimeoutError(
                f"Copilot request timed out after {timeout_secs}s"
            ) from e
        except aiohttp.ClientError as e:
            log_with_context(
                "error",
                "copilot_error",
                task_id=task.id,
                error_type=type(e).__name__,
                error_message=str(e)[:200],  # Truncate
            )
            raise TransientError(
                f"Network error during Copilot request: {type(e).__name__}: {str(e)}"
            ) from e
        except Exception as e:
            log_with_context(
                "error",
                "copilot_error",
                task_id=task.id,
                error_type=type(e).__name__,
                error_message=str(e)[:200],  # Truncate
            )
            raise TransientError(f"Unexpected error during Copilot request: {e}") from e

    async def reason_streaming(self, task: AITask) -> AsyncGenerator[str, None]:
        """
        Execute task and stream tokens as they arrive.

        For Copilot API, which currently returns complete responses,
        this method simulates streaming by chunking the response text
        into word-sized tokens. Future implementations may support
        true server-sent events (SSE) streaming if Copilot API adds support.

        Args:
            task: AI task to execute

        Yields:
            Token strings as they become available

        Raises:
            Same as reason() method
        """
        # For now, get complete response and simulate streaming
        # Note: SSE streaming not yet supported by Copilot API; using polling pattern
        result = await self.reason_complete(task)

        if result.output_data:
            # Chunk response into word-sized tokens for streaming simulation
            response_text = result.output_data.response
            words = response_text.split(" ")
            for i, word in enumerate(words):
                # Add space before word (except first)
                token = (" " if i > 0 else "") + word
                yield token

        log_with_context(
            "info",
            "copilot_streaming_complete",
            task_id=task.id,
            tokens_yielded=len(result.output_data.response.split(" "))
            if result.output_data
            else 0,
        )

    def _parse_copilot_response(
        self, response_body: dict[str, Any]
    ) -> tuple[str, str, dict[str, int]]:
        """
        Parse Copilot API response with defensive checks.

        Validates response structure, extracts content, finish_reason, and usage.
        Handles missing or malformed fields gracefully.

        Args:
            response_body: JSON response from Copilot API

        Returns:
            Tuple of (response_text, finish_reason, usage_dict)

        Raises:
            ValueError: If response structure is invalid

        Example:
            >>> text, reason, usage = engine._parse_copilot_response(response_body)
        """
        # Check for choices array
        if "choices" not in response_body:
            raise ValueError(
                f"Copilot response missing 'choices' key. "
                f"Response keys: {list(response_body.keys())}"
            )

        choices = response_body["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                f"Copilot response has empty or invalid 'choices'. "
                f"Type: {type(choices)}, Length: {len(choices) if isinstance(choices, list) else 'N/A'}"
            )

        first_choice = choices[0]
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError(
                f"Copilot response missing 'message' in first choice. "
                f"First choice keys: {list(first_choice.keys())}"
            )

        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "unknown")
        usage = response_body.get("usage", {})

        # Defensive: log if content is empty (may indicate streaming or error)
        if not content or not isinstance(content, str):
            log_with_context(
                "warning",
                "copilot_empty_response",
                content_type=type(content).__name__,
                content_length=len(content) if isinstance(content, str) else 0,
            )

        return str(content), str(finish_reason), dict(usage)

    def _truncate(self, value: str, max_length: int = 500) -> str:
        """
        Truncate string for safe logging.

        Args:
            value: String to truncate
            max_length: Maximum length (default: 500)

        Returns:
            Truncated string with "..." suffix if longer than max_length
        """
        if len(value) <= max_length:
            return value
        return value[:max_length] + "..."

    async def close(self) -> None:
        """
        Clean up resources.

        Closes HTTP client if engine owns it. Safe to call multiple times.

        Example:
            >>> await engine.close()
        """
        if self._owns_http_client and self.http_client:
            try:
                await self.http_client.close()
            except Exception as e:
                log_with_context(
                    "error",
                    "copilot_close_error",
                    error=str(e)[:200],  # Truncate
                )

        log_with_context(
            "info",
            "copilot_engine_closed",
        )
