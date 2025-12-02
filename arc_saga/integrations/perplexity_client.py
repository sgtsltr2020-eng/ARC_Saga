"""
Perplexity API client with automatic ARC Saga integration.

This module provides a streaming client for Perplexity AI that automatically
stores conversations in ARC Saga's memory system.

Follows: Repository Pattern, Dependency Injection, Error Handling Standards
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..error_instrumentation import (
    ErrorContext,
    create_request_context,
    get_correlation_id,
    log_with_context,
    request_context,
)
from ..models import Message, MessageRole, Provider
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
)

if TYPE_CHECKING:
    from ..storage.base import StorageBackend


class PerplexityClientError(Exception):
    """Base exception for Perplexity client errors."""

    pass


class PerplexityAPIError(PerplexityClientError):
    """Error during Perplexity API call."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class PerplexityStorageError(PerplexityClientError):
    """Error during message storage."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class PerplexityClient:
    """
    Perplexity API client with automatic ARC Saga integration.

    This client wraps the Perplexity API and automatically stores
    all conversations in the ARC Saga memory system.

    Attributes:
        client: AsyncOpenAI client configured for Perplexity API
        storage: StorageBackend for persisting conversations

    Example:
        >>> storage = SQLiteStorage("~/.arc-saga/memory.db")
        >>> await storage.initialize()
        >>> client = PerplexityClient(api_key="pplx-xxx", storage=storage)
        >>> async for chunk in client.ask_streaming("What is Python?"):
        ...     print(chunk)
    """

    def __init__(
        self,
        api_key: str,
        storage: StorageBackend,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
    ) -> None:
        """
        Initialize Perplexity client.

        Args:
            api_key: Perplexity API key
            storage: StorageBackend instance for message persistence
            failure_threshold: Circuit breaker failure threshold (default: 5)
            success_threshold: Circuit breaker success threshold (default: 2)
            timeout_seconds: Circuit breaker timeout in seconds (default: 60)

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.storage: StorageBackend = storage

        # Initialize circuit breaker for Perplexity API
        self.circuit_breaker = CircuitBreaker(
            service_name="perplexity",
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
        )

        log_with_context(
            "info",
            "perplexity_client_initialized",
            base_url="https://api.perplexity.ai",
            circuit_breaker_enabled=True,
        )

    async def ask_streaming(
        self,
        query: str,
        context: Optional[list[dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        model: str = "sonar-pro"
    ) -> AsyncIterator[str]:
        """
        Ask Perplexity with streaming response.

        Automatically stores conversation in ARC Saga memory.

        Args:
            query: User's question or prompt
            context: Optional list of previous messages for context
            session_id: Session identifier for grouping messages
            model: Perplexity model to use (default: sonar-pro)

        Yields:
            JSON strings with chunk data, completion signal, or error

        Raises:
            PerplexityAPIError: If API call fails
            PerplexityStorageError: If message storage fails

        Example:
            >>> async for chunk in client.ask_streaming("Hello"):
            ...     data = json.loads(chunk)
            ...     if data["type"] == "chunk":
            ...         print(data["content"], end="")
        """
        # Create request context for tracing
        ctx = create_request_context(service_name="perplexity_client")
        request_context.set(ctx)

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        correlation_id = get_correlation_id()

        log_with_context(
            "info",
            "perplexity_ask_start",
            session_id=session_id,
            query_length=len(query),
            has_context=context is not None,
            context_count=len(context) if context else 0,
            model=model
        )

        # Build messages for API call
        api_messages: list[ChatCompletionMessageParam] = []

        # Add context if provided
        if context:
            for ctx_msg in context:
                api_messages.append(cast(ChatCompletionMessageParam, ctx_msg))

        # Add current query
        api_messages.append({"role": "user", "content": query})

        # Store user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=MessageRole.USER,
            content=query,
            provider=Provider.PERPLEXITY,
            timestamp=datetime.now(timezone.utc),
            metadata={"correlation_id": correlation_id, "model": model}
        )

        try:
            await self.storage.save_message(user_msg)
            log_with_context(
                "info",
                "user_message_stored",
                message_id=user_msg.id,
                session_id=session_id
            )
        except Exception as e:
            error_ctx = ErrorContext(
                operation="store_user_message",
                error=e,
                context={"session_id": session_id, "message_id": user_msg.id}
            )
            error_ctx.log()
            raise PerplexityStorageError(
                f"Failed to store user message: {e}",
                original_error=e
            ) from e

        # Call Perplexity API with circuit breaker and retry
        full_response = ""
        start_time = datetime.now(timezone.utc)

        async def _call_api() -> list[str]:
            """
            Internal function to call API and collect all chunks.

            This function is wrapped by circuit breaker and retry logic.
            """
            stream = await self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True
            )

            chunks: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunks.append(content)

            return chunks

        try:
            # Wrap API call with circuit breaker, then retry with backoff
            async def _api_with_circuit_breaker() -> list[str]:
                """Call API through circuit breaker."""
                return await self.circuit_breaker.call(_call_api)

            # Retry with exponential backoff for transient errors
            chunks = await retry_with_backoff(
                _api_with_circuit_breaker,
                max_attempts=3,
                base_delay=1.0,
                max_delay=60.0
            )

            # Yield chunks to client
            for content in chunks:
                full_response += content
                yield json.dumps({
                    "type": "chunk",
                    "content": content
                })

            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            log_with_context(
                "info",
                "perplexity_api_complete",
                session_id=session_id,
                response_length=len(full_response),
                duration_ms=duration_ms,
                model=model
            )

            # Store complete assistant response
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                provider=Provider.PERPLEXITY,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "correlation_id": correlation_id,
                    "model": model,
                    "duration_ms": duration_ms
                }
            )

            try:
                await self.storage.save_message(assistant_msg)
                log_with_context(
                    "info",
                    "assistant_message_stored",
                    message_id=assistant_msg.id,
                    session_id=session_id,
                    response_length=len(full_response)
                )
            except Exception as e:
                error_ctx = ErrorContext(
                    operation="store_assistant_message",
                    error=e,
                    context={
                        "session_id": session_id,
                        "message_id": assistant_msg.id
                    }
                )
                error_ctx.log()
                # Don't raise here - response was already streamed
                # Log error but yield completion signal
                log_with_context(
                    "warning",
                    "assistant_message_storage_failed",
                    error=str(e),
                    session_id=session_id
                )

            # Send completion signal
            yield json.dumps({
                "type": "complete",
                "session_id": session_id,
                "correlation_id": correlation_id
            })

        except CircuitBreakerOpenError:
            # Circuit breaker is open - graceful degradation
            log_with_context(
                "warning",
                "perplexity_circuit_open_graceful_degradation",
                session_id=session_id,
                circuit_state=self.circuit_breaker.state.value,
                metrics=self.circuit_breaker.metrics.to_dict(),
            )

            # Return cached/fallback response if available
            # For now, yield error message
            yield json.dumps({
                "type": "error",
                "message": "Perplexity service is temporarily unavailable. Circuit breaker is open.",
                "error_type": "CircuitBreakerOpenError",
                "correlation_id": correlation_id,
                "circuit_state": self.circuit_breaker.state.value,
            })

            # Still store user message (it was already stored)
            # Don't store assistant message since we didn't get a response
            return

        except PerplexityStorageError:
            # Re-raise storage errors
            raise

        except Exception as e:
            error_ctx = ErrorContext(
                operation="perplexity_api_call",
                error=e,
                context={
                    "session_id": session_id,
                    "query_length": len(query),
                    "model": model
                }
            )
            error_ctx.log()

            yield json.dumps({
                "type": "error",
                "message": str(e),
                "error_type": type(e).__name__,
                "correlation_id": correlation_id
            })

            raise PerplexityAPIError(
                f"Perplexity API call failed: {e}",
                original_error=e
            ) from e

    async def get_session_history(
        self,
        session_id: str
    ) -> list[Message]:
        """
        Get all messages in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages in chronological order

        Raises:
            PerplexityStorageError: If retrieval fails
        """
        log_with_context(
            "info",
            "get_session_history_start",
            session_id=session_id
        )

        try:
            messages = await self.storage.get_by_session(session_id)
            log_with_context(
                "info",
                "get_session_history_complete",
                session_id=session_id,
                message_count=len(messages)
            )
            return messages
        except Exception as e:
            error_ctx = ErrorContext(
                operation="get_session_history",
                error=e,
                context={"session_id": session_id}
            )
            error_ctx.log()
            raise PerplexityStorageError(
                f"Failed to get session history: {e}",
                original_error=e
            ) from e
