"""
Protocol Definitions for ARC SAGA Orchestrator.

Defines structural typing protocols for reasoning engines and encrypted storage.
Uses typing.Protocol for runtime-checkable interfaces that support duck typing.

This module provides:
- IReasoningEngine: Protocol for AI reasoning engine implementations
- IEncryptedStore: Protocol for encrypted token storage backends
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from arc_saga.orchestrator.types import AIResult, AITask


@runtime_checkable
class IReasoningEngine(Protocol):
    """
    Protocol for AI reasoning engine implementations.

    Defines the contract for executing AI tasks via different providers
    (Copilot, Claude, GPT-4, etc.). All reasoning engines must implement
    this protocol to be used by the orchestrator.

    Example:
        >>> class CopilotReasoningEngine:
        ...     async def reason(self, task: AITask) -> AIResult:
        ...         # Implementation
        ...
        >>> engine = CopilotReasoningEngine()
        >>> assert isinstance(engine, IReasoningEngine)  # Runtime check
    """

    async def reason(self, task: AITask) -> Union[AIResult, AsyncGenerator[str, None]]:
        """
        Execute an AI task and return the result or stream tokens.

        Args:
            task: The AI task to execute, containing prompt, model, provider, etc.
                If task.response_mode == ResponseMode.STREAMING, yields tokens.
                Otherwise, returns AIResult with complete response.

        Returns:
            - If ResponseMode.COMPLETE: AIResult with full response
            - If ResponseMode.STREAMING: AsyncGenerator[str, None] yielding tokens

        Raises:
            AuthenticationError: If authentication fails (permanent)
            RateLimitError: If rate limit exceeded after retries (transient)
            InputValidationError: If input is invalid (permanent)
            TransientError: If network/service error occurs (transient)
            TimeoutError: If request times out (transient)
        """
        ...

    async def close(self) -> None:
        """
        Clean up resources used by the reasoning engine.

        Safe to call multiple times. Should close HTTP clients,
        connection pools, and other resources.

        Example:
            >>> engine = CopilotReasoningEngine(...)
            >>> try:
            ...     result = await engine.reason(task)
            ... finally:
            ...     await engine.close()
        """
        ...


@runtime_checkable
class IEncryptedStore(Protocol):
    """
    Protocol for encrypted token storage backends.

    Defines the contract for securely storing and retrieving OAuth2 tokens
    with encryption at rest. All token storage implementations must implement
    this protocol.

    Example:
        >>> class SQLiteEncryptedTokenStore:
        ...     async def get_token(self, user_id: str) -> Optional[dict]:
        ...         # Implementation
        ...
        >>> store = SQLiteEncryptedTokenStore(...)
        >>> assert isinstance(store, IEncryptedStore)  # Runtime check
    """

    async def get_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored token dictionary for a user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Token dictionary containing access_token, refresh_token, expires_in, etc.
            Returns None if no token found for the user.

        Raises:
            TokenStorageError: If decryption fails, database error, or corrupted data
        """
        ...

    async def save_token(self, user_id: str, token_dict: Dict[str, Any]) -> None:
        """
        Save encrypted token dictionary for a user.

        Args:
            user_id: Unique identifier for the user
            token_dict: Token dictionary to encrypt and store
                Must contain: access_token, refresh_token, expires_in, token_type

        Raises:
            TokenStorageError: If encryption fails, database write fails, or permissions denied
            ValueError: If user_id is empty or token_dict is empty
        """
        ...
