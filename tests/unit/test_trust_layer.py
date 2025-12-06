"""
Targeted trust-layer edge coverage to lift coverage toward 98%.

Focus:
- JWT expiry parsing fallbacks (_extract_exp_timestamp / _is_token_expired)
- Authentication guardrails when refresh material is missing
- Perplexity client storage failure propagation
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock

import pytest

from arc_saga.exceptions.integration_exceptions import AuthenticationError
from arc_saga.integrations.entra_id_auth_manager import EntraIDAuthManager
from arc_saga.integrations.perplexity_client import (
    PerplexityClient,
    PerplexityStorageError,
)


@pytest.fixture
def auth_manager() -> EntraIDAuthManager:
    """Provide an EntraIDAuthManager with a mocked token store."""
    mock_store = AsyncMock()
    return EntraIDAuthManager(
        client_id="client",
        client_secret="secret",
        tenant_id="tenant",
        token_store=mock_store,
    )


def test_extract_exp_timestamp_invalid_json_returns_none(
    auth_manager: EntraIDAuthManager,
) -> None:
    """_extract_exp_timestamp should fail closed when payload JSON is invalid."""
    # Build a token with payload that base64-decodes but is not valid JSON
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode())
        .decode()
        .rstrip("=")
    )
    bad_payload = base64.urlsafe_b64encode(b"{not-json").decode().rstrip("=")
    token = f"{header}.{bad_payload}.sig"

    assert auth_manager._extract_exp_timestamp(token) is None


def test_is_token_expired_with_non_string_token_triggers_refresh(
    auth_manager: EntraIDAuthManager,
) -> None:
    """Non-string access_token values should be treated as expired."""
    token_dict = {"access_token": {"not": "string"}, "refresh_token": "rt"}

    assert auth_manager._is_token_expired(token_dict) is True


@pytest.mark.asyncio
async def test_get_valid_token_missing_refresh_token_with_unparseable_access_token(
    auth_manager: EntraIDAuthManager,
) -> None:
    """
    When access_token is unparseable and no refresh_token exists,
    get_valid_token must raise AuthenticationError (no silent pass-through).
    """
    # access_token is non-string; refresh_token missing
    auth_manager.token_store.get_token = AsyncMock(
        return_value={"access_token": {"bad": "token"}}
    )

    with pytest.raises(AuthenticationError, match="No refresh token"):
        await auth_manager.get_valid_token("user123")


@pytest.mark.asyncio
async def test_perplexity_client_raises_storage_error_on_user_save_failure() -> None:
    """ask_streaming should raise PerplexityStorageError if user message persistence fails."""
    failing_store = AsyncMock()
    failing_store.save_message.side_effect = Exception("disk error")
    failing_store.get_by_session = AsyncMock(return_value=[])

    client = PerplexityClient(api_key="test-key", storage=failing_store)  # type: ignore[arg-type]

    with pytest.raises(PerplexityStorageError, match="store user message"):
        async for _ in client.ask_streaming("hello"):
            pass
