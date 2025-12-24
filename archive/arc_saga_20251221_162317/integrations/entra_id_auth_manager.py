"""
Entra ID Authentication Manager.

OAuth2 token lifecycle management with automatic refresh and encrypted persistence.
Handles token expiry detection, refresh with exponential backoff, and atomic storage.

This module provides:
- EntraIDAuthManager: OAuth2 token lifecycle manager
- Automatic token refresh with retry logic
- JWT expiry detection with buffer
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from typing import Optional

import aiohttp

from ..error_instrumentation import log_with_context
from ..exceptions.integration_exceptions import (
    AuthenticationError,
    RateLimitError,
    TokenStorageError,
    TransientError,
)
from ..orchestrator.protocols import IEncryptedStore

# Constants
OAUTH2_TOKEN_ENDPOINT_TEMPLATE = (
    "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
)
DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
TOKEN_BUFFER_SECONDS = 300  # Refresh 5 min before expiry
MAX_REFRESH_RETRIES = 5
BACKOFF_BASE = 1  # 1s, 2s, 4s, 8s, 16s
MAX_TOTAL_WAIT_SECONDS = 31  # Sum of all backoff delays


class EntraIDAuthManager:
    """
    Entra ID OAuth2 token lifecycle manager.

    Manages OAuth2 token refresh, expiry detection, and encrypted persistence.
    Provides transparent token management: users stay logged in across reboots.

    Attributes:
        client_id: Azure AD application ID
        client_secret: OAuth2 client secret
        tenant_id: Azure AD tenant ID
        token_store: Encrypted token storage backend
        scope: OAuth2 scope (default: Microsoft Graph)

    Example:
        >>> store = SQLiteEncryptedTokenStore("~/.arc_saga/tokens.db")
        >>> auth_manager = EntraIDAuthManager(
        ...     client_id="...",
        ...     client_secret="...",
        ...     tenant_id="...",
        ...     token_store=store,
        ... )
        >>> token = await auth_manager.get_valid_token("user123")
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        token_store: IEncryptedStore,
        scope: str = DEFAULT_SCOPE,
    ) -> None:
        """
        Initialize Entra ID auth manager.

        Args:
            client_id: Azure AD application ID
            client_secret: OAuth2 client secret
            tenant_id: Azure AD tenant ID
            token_store: Encrypted token storage backend
            scope: OAuth2 scope (default: Microsoft Graph)

        Raises:
            ValueError: If any required parameter is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id cannot be empty")
        if not client_secret or not client_secret.strip():
            raise ValueError("client_secret cannot be empty")
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")

        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token_store = token_store
        self.scope = scope

        self._token_endpoint = OAUTH2_TOKEN_ENDPOINT_TEMPLATE.format(
            tenant_id=tenant_id
        )

        log_with_context(
            "info",
            "entra_id_auth_manager_initialized",
            tenant_id=tenant_id,
            scope=scope,
        )

    async def get_valid_token(self, user_id: str) -> str:
        """
        Get a valid access token, refreshing if expired.

        Fetches token from store, checks expiry (with buffer), and refreshes
        if needed. Ensures atomicity: token only returned if refresh + storage succeed.

        Args:
            user_id: Unique user identifier

        Returns:
            Valid access token string

        Raises:
            AuthenticationError: If token refresh fails or persistence fails
            TokenStorageError: If token store operation fails

        Example:
            >>> token = await auth_manager.get_valid_token("user123")
            >>> # Use token in Authorization header: f"Bearer {token}"
        """
        log_with_context(
            "info",
            "entra_id_token_check",
            user_id=user_id,
        )

        try:
            # Fetch token from store
            token_dict = await self.token_store.get_token(user_id)

            if token_dict is None:
                log_with_context(
                    "info",
                    "entra_id_token_missing",
                    user_id=user_id,
                )
                raise AuthenticationError(
                    f"No token found for user {user_id}. User must authenticate."
                )

            # Check if token is expired or within buffer
            if self._is_token_expired(token_dict, buffer_seconds=TOKEN_BUFFER_SECONDS):
                log_with_context(
                    "info",
                    "entra_id_token_expired",
                    user_id=user_id,
                    buffer_seconds=TOKEN_BUFFER_SECONDS,
                )

                # Refresh token
                refresh_token = token_dict.get("refresh_token")
                if not refresh_token:
                    raise AuthenticationError(
                        f"No refresh token found for user {user_id}. User must re-authenticate."
                    )

                new_tokens = await self._refresh_token(user_id, str(refresh_token))

                # CRITICAL: Save new tokens atomically
                # If save fails, don't return token (would fail on next call)
                try:
                    await self.token_store.save_token(user_id, new_tokens)
                except TokenStorageError as e:
                    log_with_context(
                        "error",
                        "entra_id_token_persistence_failed",
                        user_id=user_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise AuthenticationError(
                        "Token refresh succeeded but persistence failed. User must re-authenticate."
                    ) from e

                log_with_context(
                    "info",
                    "entra_id_token_stored",
                    user_id=user_id,
                )

                log_with_context(
                    "info",
                    "entra_id_token_refreshed",
                    user_id=user_id,
                )

                return str(new_tokens["access_token"])

            # Token is still valid
            access_token = token_dict.get("access_token")
            if not access_token:
                raise AuthenticationError(
                    f"Token dict missing access_token for user {user_id}"
                )

            # Calculate seconds until expiry for logging
            try:
                exp_timestamp = self._extract_exp_timestamp(str(access_token))
                if exp_timestamp:
                    seconds_until_expiry = int(exp_timestamp - time.time())
                    log_with_context(
                        "info",
                        "entra_id_token_valid",
                        user_id=user_id,
                        seconds_until_expiry=seconds_until_expiry,
                    )
            except Exception:
                # Logging failure shouldn't break token retrieval
                log_with_context(
                    "info",
                    "entra_id_token_valid",
                    user_id=user_id,
                )

            return str(access_token)

        except AuthenticationError:
            raise
        except TokenStorageError as e:
            raise AuthenticationError(
                f"Token storage error for user {user_id}: {e}"
            ) from e
        except Exception as e:
            raise AuthenticationError(
                f"Unexpected error getting token for user {user_id}: {e}"
            ) from e

    async def _refresh_token(
        self, user_id: str, refresh_token: str
    ) -> dict[str, str | int]:
        """
        Execute OAuth2 token refresh flow with exponential backoff.

        POSTs to Entra ID token endpoint to refresh access token.
        Implements exponential backoff on HTTP 429 (max 5 attempts, ~31s total).

        Args:
            user_id: Unique user identifier
            refresh_token: OAuth2 refresh token

        Returns:
            Token dictionary with access_token, refresh_token, expires_in, token_type

        Raises:
            AuthenticationError: On HTTP 401/400 (permanent, don't retry)
            RateLimitError: On HTTP 429 after exhausting retries
            TransientError: On HTTP 500+ or network errors (caller handles retry)
        """
        log_with_context(
            "info",
            "entra_id_refresh_start",
            user_id=user_id,
        )

        request_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "scope": self.scope,
        }

        last_error: Exception | None = None
        retry_after: str | int | None = None

        for attempt in range(MAX_REFRESH_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._token_endpoint,
                        data=request_data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        status_code = response.status
                        response_body = await response.json()

                        if status_code == 200:
                            # Success - parse response
                            access_token = response_body.get("access_token")
                            new_refresh_token = response_body.get(
                                "refresh_token", refresh_token
                            )
                            expires_in = response_body.get("expires_in", 3600)
                            token_type = response_body.get("token_type", "Bearer")

                            if not access_token:
                                raise AuthenticationError(
                                    "Token refresh response missing access_token"
                                )

                            new_tokens = {
                                "access_token": access_token,
                                "refresh_token": new_refresh_token,
                                "expires_in": int(expires_in),
                                "token_type": token_type,
                            }

                            log_with_context(
                                "info",
                                "entra_id_refresh_success",
                                user_id=user_id,
                                attempt=attempt + 1,
                            )

                            return new_tokens

                        elif status_code == 401:
                            # Invalid credentials - permanent error
                            error_detail = response_body.get(
                                "error_description", "Unauthorized"
                            )
                            log_with_context(
                                "error",
                                "entra_id_refresh_failed_permanent",
                                user_id=user_id,
                                status_code=401,
                                error_detail=error_detail[:200],  # Truncate
                            )
                            raise AuthenticationError(
                                f"Invalid refresh token or credentials: {error_detail}"
                            )

                        elif status_code == 400:
                            # Bad request - permanent error
                            error_detail = response_body.get(
                                "error_description", "Bad request"
                            )
                            log_with_context(
                                "error",
                                "entra_id_refresh_failed_permanent",
                                user_id=user_id,
                                status_code=400,
                                error_detail=error_detail[:200],  # Truncate
                            )
                            raise ValueError(
                                f"Invalid refresh token format: {error_detail}"
                            )

                        elif status_code == 429:
                            # Rate limited - retry with exponential backoff
                            retry_after = response.headers.get("Retry-After", "unknown")
                            backoff_seconds = min(
                                BACKOFF_BASE * (2**attempt), 16
                            )  # Cap at 16s
                            jitter = random.uniform(
                                0, 0.1 * backoff_seconds
                            )  # nosec B311
                            total_delay = backoff_seconds + jitter

                            log_with_context(
                                "warning",
                                "entra_id_refresh_retry",
                                user_id=user_id,
                                attempt=attempt + 1,
                                backoff_seconds=total_delay,
                                retry_after=retry_after,
                            )

                            # Don't retry on last attempt
                            if attempt < MAX_REFRESH_RETRIES - 1:
                                await asyncio.sleep(total_delay)
                                continue
                            else:
                                # All retries exhausted
                                raise RateLimitError(
                                    f"Token refresh failed after {MAX_REFRESH_RETRIES} retries. "
                                    f"Total wait: ~{MAX_TOTAL_WAIT_SECONDS}s. Retry-After: {retry_after}",
                                    retry_after=retry_after,
                                )

                        elif status_code >= 500:
                            # Service error - transient
                            error_detail = response_body.get(
                                "error_description", f"Service error {status_code}"
                            )
                            log_with_context(
                                "error",
                                "entra_id_refresh_failed_transient",
                                user_id=user_id,
                                status_code=status_code,
                                error_detail=error_detail[:200],  # Truncate
                            )
                            raise TransientError(
                                f"Entra ID service error: {status_code}: {error_detail}"
                            )

                        else:
                            # Unexpected status code
                            error_detail = response_body.get(
                                "error_description", f"Unexpected status {status_code}"
                            )
                            log_with_context(
                                "error",
                                "entra_id_refresh_failed_transient",
                                user_id=user_id,
                                status_code=status_code,
                                error_detail=error_detail[:200],  # Truncate
                            )
                            raise TransientError(
                                f"Unexpected status from Entra ID: {status_code}: {error_detail}"
                            )

            except (AuthenticationError, ValueError, RateLimitError, TransientError):
                # Re-raise known exceptions
                raise
            except aiohttp.ClientError as e:
                # Network errors - transient
                log_with_context(
                    "error",
                    "entra_id_refresh_network_error",
                    user_id=user_id,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],  # Truncate
                )
                raise TransientError(
                    f"Network error during token refresh: {type(e).__name__}: {str(e)}"
                ) from e
            except asyncio.TimeoutError as e:
                # Timeout - transient
                log_with_context(
                    "error",
                    "entra_id_refresh_network_error",
                    user_id=user_id,
                    error_type="TimeoutError",
                    error_message=str(e)[:200],  # Truncate
                )
                raise TransientError("Network timeout during token refresh") from e
            except Exception as e:
                # Unexpected error - transient
                last_error = e
                log_with_context(
                    "error",
                    "entra_id_refresh_failed_transient",
                    user_id=user_id,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],  # Truncate
                )
                # Retry on last attempt will raise
                if attempt < MAX_REFRESH_RETRIES - 1:
                    backoff_seconds = min(BACKOFF_BASE * (2**attempt), 16)
                    await asyncio.sleep(backoff_seconds)
                    continue

        # All retries exhausted
        if last_error:
            raise TransientError(
                f"Token refresh failed after {MAX_REFRESH_RETRIES} attempts: {last_error}"
            ) from last_error
        raise TransientError(
            f"Token refresh failed after {MAX_REFRESH_RETRIES} attempts"
        )

    def _is_token_expired(
        self,
        token_dict: dict[str, str | int],
        buffer_seconds: int = TOKEN_BUFFER_SECONDS,
    ) -> bool:
        """
        Check if token is expired or about to expire.

        Decodes JWT access_token to extract 'exp' claim and compares with
        current time + buffer. Handles malformed tokens defensively.

        Args:
            token_dict: Token dictionary with 'access_token' key
            buffer_seconds: Seconds before expiry to trigger refresh (default: 300)

        Returns:
            True if expired or within buffer, False otherwise

        Example:
            >>> expired = auth_manager._is_token_expired(token_dict, buffer_seconds=300)
        """
        access_token = token_dict.get("access_token")
        if not access_token:
            # Missing token - trigger refresh
            return True

        if not isinstance(access_token, str):
            # Invalid type - trigger refresh
            return True

        # Extract exp timestamp
        exp_timestamp = self._extract_exp_timestamp(access_token)
        if exp_timestamp is None:
            # Can't parse - trigger refresh (safe default)
            return True

        # Check if expired or within buffer
        now = time.time()
        expiry_with_buffer = exp_timestamp - buffer_seconds

        return now >= expiry_with_buffer

    def _extract_exp_timestamp(self, access_token: str) -> Optional[float]:
        """
        Extract 'exp' claim from JWT access token.

        Decodes JWT payload (no validation) to extract expiration timestamp.
        Handles malformed tokens gracefully.

        Args:
            access_token: JWT access token string

        Returns:
            Expiration timestamp (Unix epoch seconds) or None if cannot parse
        """
        try:
            # Split JWT into parts: header.payload.signature
            parts = access_token.split(".")
            if len(parts) != 3:
                # Malformed JWT - return None (will trigger refresh)
                return None

            payload = parts[1]

            # Add padding if needed (base64url doesn't pad)
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            # Decode base64url
            decoded_bytes = base64.urlsafe_b64decode(payload)
            payload_dict = json.loads(decoded_bytes.decode("utf-8"))

            # Extract 'exp' claim
            exp = payload_dict.get("exp")
            if exp is None:
                # Missing exp claim - return None (will trigger refresh)
                return None

            # Convert to float (timestamp)
            if isinstance(exp, (int, float)):
                return float(exp)

            # Invalid exp type - return None
            return None

        except Exception:
            # Any error parsing JWT - return None (will trigger refresh)
            return None

