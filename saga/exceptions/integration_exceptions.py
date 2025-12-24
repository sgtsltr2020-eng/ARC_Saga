"""
Integration Exception Classes.

Custom exceptions for AI provider integrations and authentication.
All exceptions inherit from ArcSagaException for consistent error handling.

This module provides:
- AuthenticationError: OAuth2/auth failures (permanent)
- RateLimitError: HTTP 429 rate limiting (transient)
- InputValidationError: Request format errors (permanent)
- TokenStorageError: Encrypted store failures (permanent)
- TransientError: Network/timeout errors (transient)
"""

from __future__ import annotations

from .storage_exceptions import ArcSagaException


class AuthenticationError(ArcSagaException):
    """
    Raised when authentication or token refresh fails.

    This is a permanent error - do not retry in the auth manager.
    Used when:
    - Token invalid or expired
    - Refresh token revoked
    - Storage cannot persist token (atomicity requirement)

    Example:
        >>> raise AuthenticationError("Token refresh failed: invalid credentials")
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        Initialize AuthenticationError.

        Args:
            message: Error description
            cause: Optional underlying exception
        """
        super().__init__(message, error_code="AUTHENTICATION_ERROR")
        self.cause = cause
        if cause:
            self.__cause__ = cause


class RateLimitError(ArcSagaException):
    """
    Raised when rate limit is exceeded (HTTP 429).

    This is a transient error - retry with exponential backoff.
    Raised after exhausting all retries (max 5 attempts, ~31s total wait).

    Attributes:
        retry_after: Suggested retry delay in seconds (from Retry-After header)

    Example:
        >>> raise RateLimitError("Rate limit exceeded", retry_after=60)
    """

    def __init__(
        self,
        message: str,
        retry_after: str | int | None = None,
        cause: Exception | None = None,
    ) -> None:
        """
        Initialize RateLimitError.

        Args:
            message: Error description
            retry_after: Suggested retry delay (from Retry-After header)
            cause: Optional underlying exception
        """
        super().__init__(message, error_code="RATE_LIMIT_ERROR")
        self.retry_after = retry_after
        self.cause = cause
        if cause:
            self.__cause__ = cause


class InputValidationError(ArcSagaException):
    """
    Raised when request format is invalid (HTTP 413, malformed input).

    This is a permanent error - do not retry.
    Used when:
    - Prompt too large (HTTP 413)
    - Request format invalid
    - Required fields missing

    Example:
        >>> raise InputValidationError("Prompt exceeds maximum size limit")
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        Initialize InputValidationError.

        Args:
            message: Error description
            cause: Optional underlying exception
        """
        super().__init__(message, error_code="INPUT_VALIDATION_ERROR")
        self.cause = cause
        if cause:
            self.__cause__ = cause


class TokenStorageError(ArcSagaException):
    """
    Raised when encrypted token storage operation fails.

    This is a permanent error - do not retry at auth manager level.
    Used when:
    - Database connection fails
    - Encryption/decryption fails
    - File permissions denied
    - Corrupted encrypted data

    Example:
        >>> raise TokenStorageError("Failed to decrypt token: invalid key")
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        Initialize TokenStorageError.

        Args:
            message: Error description
            cause: Optional underlying exception
        """
        super().__init__(message, error_code="TOKEN_STORAGE_ERROR")
        self.cause = cause
        if cause:
            self.__cause__ = cause


class TransientError(ArcSagaException):
    """
    Raised when a transient error occurs (network, timeout, service errors).

    This is a transient error - retry with exponential backoff.
    Used when:
    - Network connection fails
    - Request timeout (HTTP 408, 504)
    - Service errors (HTTP 500+)
    - DNS resolution fails

    Example:
        >>> raise TransientError("Network error: connection timeout")
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        Initialize TransientError.

        Args:
            message: Error description
            cause: Optional underlying exception
        """
        super().__init__(message, error_code="TRANSIENT_ERROR")
        self.cause = cause
        if cause:
            self.__cause__ = cause

