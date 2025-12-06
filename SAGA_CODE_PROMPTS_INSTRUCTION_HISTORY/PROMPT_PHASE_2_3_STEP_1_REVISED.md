# Phase 2.3 Step 1: Copilot Reasoning Engine & EntraID Auth — FINAL CURSOR PROMPT (REVISED)

## Executive Summary

Implement Microsoft Copilot integration for ARC SAGA as the default reasoning engine. This involves 9 focused tasks across protocols, exceptions, encryption, authentication, API integration, and comprehensive testing. The deliverable is production-ready code with 98%+ coverage, full type safety, transparent token persistence across reboots, and robust handling of edge cases.

---

## Architecture Overview

### Multi-Engine Cockpit Design

```
User Controls (Main Tab)
  ├─ Provider Dropdown: Copilot, Claude, GPT-4, Perplexity, LiteLLM
  └─ Mode Dropdown: Quick, Deep, Smart, Custom

Provider Router
  ├─ If Provider="Copilot" → CopilotReasoningEngine
  ├─ If Provider="Claude" → (Phase 3+)
  └─ If Provider="LiteLLM" → (Phase 3+)

CopilotReasoningEngine
  ├─ EntraIDAuthManager (OAuth2 token lifecycle)
  ├─ EncryptedTokenStore (persistent, encrypted storage)
  └─ HTTP client (aiohttp async)
```

### Phase 2.3 Scope

- **Primary**: Copilot as default brain (internal shared tenant, no per-user licensing)
- **Secondary**: Foundation for multi-provider switching (steps 2-5)
- **Out of scope**: Fallback logic (deferred to Step 4), document generation (Phase 3+)

---

## Context & Dependencies

### Existing ARC SAGA Infrastructure

1. **Types** (`arc_saga/orchestrator/types.py`):
   - `AITask`, `AITaskInput`, `AIResultOutput`, `AIResult`
   - `TaskStatus`, `WorkflowPattern`
   - `AIProvider` enum (need to add `COPILOT_CHAT`)

2. **Logging** (`arc_saga/error_instrumentation.py`):
   - `log_with_context(event_name, **context)` — Use for all logging

3. **Exceptions** (`arc_saga/exceptions/`):
   - Base class: `ArcSagaException`
   - Extend with integration-specific exceptions

4. **Token Manager** (`arc_saga/orchestrator/token_manager.py`):
   - `TokenBudgetManager`, `TokenUsage` (will integrate in Step 3)

### External Dependencies (Requirements)

- `aiohttp` — Async HTTP client
- `cryptography` — AES-256 encryption (Fernet)
- `aiosqlite` — Async SQLite
- `python-dotenv` — Environment variables

---

## Tasks 1-9: Implementation Breakdown

### Task 1: Protocol Definitions (`arc_saga/orchestrator/protocols.py`)

**Deliverable**: New file defining two protocols

```python
# 1. IReasoningEngine
- async reason(task: AITask) -> AIResult
- Must match existing pattern if protocol already exists

# 2. IEncryptedStore
- async get_token(user_id: str) -> Optional[dict]
- async save_token(user_id: str, token_dict: dict) -> None
```

**Key points**:
- Use `typing.Protocol` with `@runtime_checkable`
- Docstrings explain contract and error handling
- No implementation, only signatures

---

### Task 2: Exception Classes (`arc_saga/exceptions/integration_exceptions.py`)

**Deliverable**: New file with 5 custom exceptions

1. **`AuthenticationError(ArcSagaException)`**
   - OAuth2 token refresh failures, invalid credentials, persistence failures
   - Permanent error (don't retry in auth manager)
   - Used when: token invalid, refresh token revoked, storage can't persist token

2. **`RateLimitError(ArcSagaException)`**
   - HTTP 429 with `retry_after` support
   - Transient (retry with exponential backoff, but max 5 attempts)
   - Raised after exhausting all retries (~31s max wait: 1+2+4+8+16)

3. **`InputValidationError(ArcSagaException)`**
   - Request format errors (HTTP 413, malformed input, oversized prompt)
   - Permanent (don't retry)

4. **`TokenStorageError(ArcSagaException)`**
   - Encrypted store operation failures (DB error, encryption failure, permissions)
   - Permanent (don't retry at auth manager level; orchestrator handles)

5. **`TransientError(ArcSagaException)`**
   - Network, timeout, service errors (HTTP 500+, connection failures)
   - Transient (retry with backoff, but cap at 5 attempts)

**Key points**:
- All inherit from `ArcSagaException`
- Include `error_code` field for categorization
- Docstring explains when/why each is raised
- All support message parameter and optional cause chain (raise X from Y)

---

### Task 3: Encrypted Token Store (`arc_saga/integrations/encrypted_token_store.py`)

**Deliverable**: New file with `SQLiteEncryptedTokenStore` implementing `IEncryptedStore`

**Class**: `SQLiteEncryptedTokenStore`

**Constructor**:
```python
def __init__(
    self,
    db_path: str,  # e.g., ~/.arc_saga/tokens.db
    encryption_key: Optional[str] = None,  # From env or generate
) -> None
```

**Database Schema**:
```sql
CREATE TABLE IF NOT EXISTS encrypted_tokens (
    user_id TEXT PRIMARY KEY,
    encrypted_data BLOB NOT NULL,  -- Fernet-encrypted JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Methods**:

1. `async get_token(user_id: str) -> Optional[dict]`
   - Query DB, decrypt data, return dict
   - Return `None` if not found
   - Raise `TokenStorageError` on decrypt failure (corrupted data, wrong key, DB error)

2. `async save_token(user_id: str, token_dict: dict) -> None`
   - Serialize token_dict to JSON
   - Encrypt with Fernet
   - Insert/update in DB
   - Raise `TokenStorageError` on failure
   - **Critical**: If this fails, caller (`get_valid_token`) must raise `AuthenticationError` (don't return token if persistence failed)

3. `_get_or_create_encryption_key() -> str`
   - Check env var `ARC_SAGA_TOKEN_ENCRYPTION_KEY`
   - If not set, generate new key and store in `~/.arc_saga/.token_key`
   - Load from file if it exists
   - Set file permissions to 0600 (user-readable only)

**Encryption**:
- Use `cryptography.fernet.Fernet` (symmetric, easy to rotate)
- Key derivation: Fernet base64-encoded 32-byte key
- Store key securely (env var > ~/.arc_saga/.token_key with 0600 perms)

**Logging**:
- `"token_store_get"` — Token retrieved
- `"token_store_save"` — Token saved
- `"token_store_error"` — Error on get/save (with error_type, user_id)

**Error Handling**:
- Invalid encryption key → `TokenStorageError("Invalid encryption key")`
- DB connection failure → `TokenStorageError("Database connection failed")`
- Concurrent access → Use SQLite locks (aiosqlite handles automatically)
- Corrupted encrypted data → `TokenStorageError("Failed to decrypt token data")`

---

### Task 4: EntraID Auth Manager (`arc_saga/integrations/entra_id_auth_manager.py`)

**Deliverable**: New file (~250 lines) with `EntraIDAuthManager`

**Class**: `EntraIDAuthManager`

**Constructor**:
```python
def __init__(
    self,
    client_id: str,                 # Azure AD app ID
    client_secret: str,             # OAuth client secret
    tenant_id: str,                 # Azure AD tenant
    token_store: IEncryptedStore,   # Encrypted store instance
    scope: str = "https://graph.microsoft.com/.default",
) -> None
```

**Validation**:
- Raise `ValueError` if any parameter is empty string
- Log initialization with context

**Constants**:
```python
OAUTH2_TOKEN_ENDPOINT = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
TOKEN_BUFFER_SECONDS = 300  # Refresh 5 min before expiry
MAX_REFRESH_RETRIES = 5
BACKOFF_BASE = 1  # 1s, 2s, 4s, 8s, 16s
MAX_TOTAL_WAIT_SECONDS = 31  # Sum of all backoff delays
```

**Method 1**: `async get_valid_token(user_id: str) -> str`

**Logic**:
1. Fetch token dict from `token_store.get_token(user_id)`
2. If token exists:
   - Check expiry: `_is_token_expired(token_dict, buffer_seconds=300)`
   - If not expired, log and return `token_dict["access_token"]`
3. If token missing or expired:
   - Get `refresh_token` from stored dict
   - Call `_refresh_token(user_id, refresh_token)`
   - Store new tokens: `await token_store.save_token(user_id, new_tokens)`
     - **If save_token() fails**: Raise `AuthenticationError` with message "Token refresh succeeded but persistence failed. User must re-authenticate."
     - Log: `"entra_id_token_persistence_failed"` with error details
     - This prevents returning a token that can't be stored (would fail on next call)
   - Return new `access_token`
4. If store get_token error or no refresh token, raise `AuthenticationError`

**Logging**:
- `"entra_id_token_valid"` — Token checked, still valid (include: user_id, seconds_until_expiry)
- `"entra_id_token_expired"` — Token expired, triggering refresh (include: user_id, buffer_seconds)
- `"entra_id_token_refreshed"` — Refresh successful (include: user_id, new_exp_timestamp)
- `"entra_id_token_stored"` — Token persisted (include: user_id)
- `"entra_id_token_persistence_failed"` — Store failed (include: user_id, error, error_type)
- Include: `user_id`, `buffer_seconds`, `exp_timestamp`, error info as applicable

**Return**: Valid access token string, never None

**Method 2**: `async _refresh_token(user_id: str, refresh_token: str) -> dict`

**HTTP Flow**:
- POST to: `https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`
- Body:
  ```json
  {
    "client_id": self.client_id,
    "client_secret": self.client_secret,
    "refresh_token": refresh_token,
    "grant_type": "refresh_token",
    "scope": self.scope
  }
  ```

**Response Parsing** (HTTP 200):
- Extract: `access_token`, `refresh_token`, `expires_in` (seconds)
- Build dict: `{"access_token": str, "refresh_token": str, "expires_in": int, "token_type": "Bearer"}`
- Return dict

**Error Handling with Retry Logic**:
- HTTP 401 (invalid credentials) → `AuthenticationError("Invalid refresh token or credentials")`
  - Permanent, don't retry
  - Log: `"entra_id_refresh_failed_permanent"` with status_code=401
- HTTP 400 (bad request) → `ValueError("Invalid refresh token format")`
  - Permanent, don't retry
  - Log: `"entra_id_refresh_failed_permanent"` with status_code=400
- HTTP 429 (rate limited) → Implement **exponential backoff**:
  - Max 5 retries with delays: 1s, 2s, 4s, 8s, 16s
  - Add random jitter: ±10%
  - Extract `Retry-After` header if present
  - Log each retry: `"entra_id_refresh_retry"` with attempt_number, backoff_seconds
  - On final failure (after 5 attempts totaling ~31s): 
    ```python
    raise RateLimitError(
        f"Token refresh failed after {MAX_REFRESH_RETRIES} retries. "
        f"Total wait: ~{MAX_TOTAL_WAIT_SECONDS}s. Retry-After: {retry_after}"
    )
    ```
- HTTP 500+ (service error) → `TransientError("Entra ID service error: {status_code}")`
  - Transient (caller handles retry)
  - Log: `"entra_id_refresh_failed_transient"` with status_code
- Connection error (timeout, DNS, etc.) → `TransientError("Network error during token refresh")`
  - Log: `"entra_id_refresh_network_error"` with error_type

**Logging**:
- `"entra_id_refresh_start"` — Before POST (include: user_id)
- `"entra_id_refresh_success"` — After successful 200 (include: user_id, new_exp_timestamp)
- `"entra_id_refresh_retry"` — On 429, before backoff (include: user_id, attempt, backoff_seconds, retry_after)
- `"entra_id_refresh_failed_permanent"` — On 401/400 (include: user_id, status_code, error_detail)
- `"entra_id_refresh_failed_transient"` — On 500+ or network error (include: user_id, status_code or error_type)
- All logs include: `user_id`, `status_code` (if HTTP), `attempt`, `retry_after` (if in response), `error` (truncated)

**Docstring**: Google-style with Args, Returns, Raises, Examples

**Method 3**: `_is_token_expired(token_dict: dict, buffer_seconds: int = 300) -> bool`

**JWT Decoding** (no validation):
1. Extract `access_token` from `token_dict["access_token"]`
2. Split by `.` → `[header, payload, signature]`
3. **Defensive check**: If split doesn't produce 3 parts, return `True` (trigger refresh)
4. Decode `payload` (base64url decode without padding):
   ```python
   import json, base64
   payload = token_dict["access_token"].split(".")[1]
   # Add padding if needed (base64url doesn't pad)
   padding = 4 - len(payload) % 4
   if padding != 4:
       payload += "=" * padding
   try:
       decoded = json.loads(base64.urlsafe_b64decode(payload))
   except Exception:
       # Malformed payload, return True to trigger refresh
       return True
   ```
5. Compare: `exp_timestamp < now() + buffer_seconds`?
6. Return `True` if expired/within buffer, `False` otherwise

**Edge Cases** (all handled defensively):
- Malformed token (can't split by `.`) → Return `True` (trigger refresh)
- Missing `exp` claim → Return `True` (trigger refresh)
- Corrupted base64 payload → Return `True` (trigger refresh), no exception
- Token expiry in past → Return `True` (already expired)
- Token expiry 0 or invalid → Return `True` (safety default)

**Docstring**: Explain JWT structure, buffer purpose, edge case handling, return value

---

### Task 5: Copilot Reasoning Engine (`arc_saga/integrations/copilot_reasoning_engine.py`)

**Deliverable**: New file (~350 lines) with `CopilotReasoningEngine` implementing `IReasoningEngine`

**Class**: `CopilotReasoningEngine`

**Constructor**:
```python
def __init__(
    self,
    client_id: str,                              # Azure AD app ID
    client_secret: str,                          # OAuth secret
    tenant_id: str,                              # Azure tenant
    token_store: IEncryptedStore,                # Encrypted token store
    http_client: Optional[aiohttp.ClientSession] = None,
) -> None
```

**Initialization**:
- Validate all parameters (raise `ValueError` if empty)
- Create `EntraIDAuthManager` with provided credentials
- Create HTTP client if not provided:
  ```python
  self.http_client = http_client or aiohttp.ClientSession()
  self._owns_http_client = http_client is None  # Track ownership for cleanup
  ```
- Store `graph_endpoint = "https://graph.microsoft.com/v1.0"`
- Log initialization

**Constants**:
```python
COPILOT_CHAT_ENDPOINT = "/copilot/chat"
DEFAULT_TIMEOUT_SECONDS = 30
```

**Method 1**: `async reason(task: AITask) -> AIResult`

**Contract**: Implements `IReasoningEngine.reason()` protocol

**Logic**:

1. **Start timing**:
   ```python
   start_time = time.time()
   ```

2. **Get user_id**:
   ```python
   user_id = task.input_data.get("user_id", "default_user") if hasattr(task.input_data, 'get') else getattr(task.input_data, 'user_id', 'default_user')
   ```

3. **Get valid token** (may raise `AuthenticationError`):
   ```python
   try:
       token = await self.auth_manager.get_valid_token(user_id)
   except AuthenticationError as e:
       log_with_context(
           "copilot_auth_failed",
           task_id=task.id,
           user_id=user_id,
           error=str(e),
           error_type=type(e).__name__,
       )
       raise
   ```

4. **Build request body**:
   ```python
   request_body = {
       "messages": [
           {
               "role": "user",
               "content": task.input_data.prompt,
           }
       ],
       "max_tokens": task.input_data.max_tokens,
       "temperature": task.input_data.temperature,
   }
   
   if task.input_data.system_prompt:
       request_body["messages"].insert(0, {
           "role": "system",
           "content": task.input_data.system_prompt,
       })
   ```

5. **Log request**:
   ```python
   log_with_context(
       "copilot_request_start",
       task_id=task.id,
       user_id=user_id,
       prompt_length=len(task.input_data.prompt),
       max_tokens=task.input_data.max_tokens,
       temperature=task.input_data.temperature,
   )
   ```

6. **Make HTTP request**:
   ```python
   timeout_secs = task.timeout_ms / 1000 if task.timeout_ms else DEFAULT_TIMEOUT_SECONDS
   try:
       async with self.http_client.post(
           f"{self.graph_endpoint}{COPILOT_CHAT_ENDPOINT}",
           headers={"Authorization": f"Bearer {token}"},
           json=request_body,
           timeout=aiohttp.ClientTimeout(total=timeout_secs),
       ) as response:
           response_body = await response.json()
           status_code = response.status
   except asyncio.TimeoutError:
       raise TimeoutError(f"Copilot request timed out after {timeout_secs}s")
   except aiohttp.ClientError as e:
       raise TransientError(f"Network error during Copilot request: {type(e).__name__}: {str(e)}")
   ```

7. **Handle errors** (HTTP status codes):
   ```python
   if status_code == 401:
       error_detail = response_body.get('error', {}).get('message', 'Unauthorized')
       log_with_context("copilot_error", task_id=task.id, status_code=401, error=_truncate(error_detail, 200))
       raise AuthenticationError(f"Copilot: {error_detail}")
   
   elif status_code == 429:
       retry_after = response.headers.get("Retry-After", "unknown")
       log_with_context("copilot_error", task_id=task.id, status_code=429, retry_after=retry_after)
       raise RateLimitError(f"Copilot rate limited. Retry-After: {retry_after}")
   
   elif status_code in [408, 504]:
       log_with_context("copilot_error", task_id=task.id, status_code=status_code)
       raise TimeoutError(f"Copilot request timeout (status {status_code})")
   
   elif status_code == 413:
       log_with_context("copilot_error", task_id=task.id, status_code=413)
       raise InputValidationError("Prompt too large for Copilot (HTTP 413)")
   
   elif status_code == 400:
       error_detail = response_body.get('error', {}).get('message', 'Bad request')
       log_with_context("copilot_error", task_id=task.id, status_code=400, error=_truncate(error_detail, 200))
       raise ValueError(f"Invalid Copilot request: {error_detail}")
   
   elif status_code >= 500:
       error_detail = response_body.get('error', {}).get('message', f'Service error {status_code}')
       log_with_context("copilot_error", task_id=task.id, status_code=status_code, error=_truncate(error_detail, 200))
       raise TransientError(f"Copilot service error: {status_code}: {error_detail}")
   
   elif status_code != 200:
       log_with_context("copilot_error", task_id=task.id, status_code=status_code)
       raise ValueError(f"Unexpected status from Copilot: {status_code}")
   ```

8. **Parse response defensively** (HTTP 200):
   ```python
   def _parse_copilot_response(response_body: dict) -> tuple:
       """
       Parse Copilot API response with defensive checks.
       Returns: (response_text, finish_reason, usage_dict)
       Raises: ValueError if response structure invalid
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
               "copilot_empty_response",
               content_type=type(content).__name__,
               content_length=len(content) if isinstance(content, str) else 0,
           )
       
       return content, finish_reason, usage
   
   try:
       response_text, finish_reason, usage = _parse_copilot_response(response_body)
   except ValueError as e:
       log_with_context("copilot_parse_error", error=str(e), response_body_preview=_truncate(str(response_body), 300))
       raise ValueError(f"Failed to parse Copilot response: {e}") from e
   
   total_tokens = usage.get("total_tokens", 0)
   prompt_tokens = usage.get("prompt_tokens", 0)
   completion_tokens = usage.get("completion_tokens", 0)
   ```

9. **Build AIResultOutput**:
   ```python
   duration_ms = int((time.time() - start_time) * 1000)
   
   output = AIResultOutput(
       response=response_text,
       tokens_used=total_tokens,
       prompt_tokens=prompt_tokens,
       completion_tokens=completion_tokens,
       provider=AIProvider.COPILOT_CHAT,
       model="copilot-gpt4o",  # Or extract from response if available
       cost_usd=Decimal("0.0"),  # Computed offline in Step 3
       finish_reason=finish_reason,
       latency_ms=duration_ms,
   )
   ```

10. **Log success**:
    ```python
    log_with_context(
        "copilot_request_complete",
        task_id=task.id,
        user_id=user_id,
        tokens_used=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=duration_ms,
        finish_reason=finish_reason,
    )
    ```

11. **Return AIResult**:
    ```python
    return AIResult(
        task_id=task.id,
        success=True,
        output_data=output,
        status=TaskStatus.COMPLETED,
        duration_ms=duration_ms,
    )
    ```

**Error Handling** (catch-all):
- All exceptions caught, logged with full context
- Reraise original exception (don't suppress)
- Log format: `"copilot_error"` or `"copilot_auth_failed"` with `error_type`, `status_code` (if HTTP), `task_id`, `user_id`
- Never log full token or sensitive values; truncate to first 200 chars

**Helper Function**: `def _truncate(value: str, max_length: int = 500) -> str`
- Return first `max_length` chars + "..." if longer
- Safe for logging sensitive data

**Docstring**: Google-style with Args, Returns, Raises, Examples

**Method 2**: `async close() -> None`

**Logic**:
```python
if self._owns_http_client and self.http_client:
    try:
        await self.http_client.close()
    except Exception as e:
        log_with_context("copilot_close_error", error=str(e))
log_with_context("copilot_engine_closed")
```

**Purpose**: Resource cleanup; safe to call multiple times

---

### Task 6: Update Provider Enum (`arc_saga/orchestrator/types.py`)

**Modification**: Add to `AIProvider` enum

```python
class AIProvider(str, Enum):
    # ... existing values ...
    COPILOT_CHAT = "copilot_chat"  # Microsoft Copilot (new)
```

**Verification**: Ensure no conflicts with existing values

---

### Task 7: Unit Tests — EntraIDAuthManager (`tests/unit/integrations/test_entra_id_auth_manager.py`)

**Deliverable**: ~400 lines, 98%+ coverage

**Mocking Strategy**:
- Mock `IEncryptedStore` with `unittest.mock.AsyncMock`
- Mock `aiohttp.ClientSession` for HTTP calls
- Use `pytest.mark.asyncio` for async tests

**Test Cases**:

1. **Successful token refresh flow** (`test_get_valid_token_refreshes_expired_token`)
   - Store returns expired token dict
   - Manager calls `_refresh_token()`
   - HTTP POST to Entra ID returns 200 with new tokens
   - New tokens stored in store
   - New `access_token` returned
   - Verify logging includes all events

2. **Token already valid** (`test_get_valid_token_returns_valid_token`)
   - Store returns non-expired token
   - Manager returns immediately (no HTTP call)
   - Verify `_refresh_token()` not called

3. **Token expiry detection with buffer** (`test_is_token_expired_with_buffer`)
   - Test expiry detection with 300s buffer
   - Test token within buffer (should return True)
   - Test token far from expiry (should return False)
   - Test token already expired (should return True)

4. **Malformed JWT handling** (`test_is_token_expired_malformed_jwt`)
   - Token with < 3 parts (invalid structure) → Return `True`
   - Token with valid structure but corrupted payload → Return `True`
   - Token missing `exp` claim → Return `True`
   - No exceptions raised; defensive graceful handling

5. **Partial JWT (2 parts)** (`test_is_token_expired_partial_jwt`)
   - Token like "header.payload" (no signature part)
   - Should return `True` without crashing

6. **Refresh failures - 401** (`test_refresh_token_auth_error_on_401`)
   - HTTP 401 → `AuthenticationError` raised
   - Verify error message includes response body (truncated)
   - Verify logging includes status_code=401

7. **Refresh failures - 400** (`test_refresh_token_bad_request_on_400`)
   - HTTP 400 → `ValueError` raised

8. **Rate limit with exponential backoff** (`test_refresh_token_rate_limit_with_backoff`)
   - HTTP 429 on first two attempts
   - HTTP 200 on third attempt
   - Verify exponential backoff delays occur: 1s, 2s (then succeeds)
   - Verify logging of each retry attempt
   - Verify `Retry-After` header extraction

9. **Rate limit exhausts retries** (`test_refresh_token_rate_limit_max_retries_exhausted`)
   - HTTP 429 on all 5 attempts
   - Verify `RateLimitError` raised after ~31s total wait
   - Verify error message includes retry count and total wait time
   - Verify no 6th attempt made

10. **Refresh failures - 500** (`test_refresh_token_service_error_on_500`)
    - HTTP 500+ → `TransientError` raised
    - Verify logging includes status_code

11. **Network error during refresh** (`test_refresh_token_network_error`)
    - Connection timeout or DNS failure
    - → `TransientError` raised
    - Verify error message includes network error type

12. **Token storage error** (`test_get_valid_token_storage_error`)
    - `token_store.get_token()` raises `TokenStorageError`
    - Manager propagates as `AuthenticationError`
    - Verify logging

13. **Token persistence failure** (`test_get_valid_token_persistence_failure`)
    - Refresh succeeds (HTTP 200)
    - `token_store.save_token()` raises `TokenStorageError`
    - Manager raises `AuthenticationError("Token refresh succeeded but persistence failed...")`
    - Verify token NOT returned
    - Verify caller must re-authenticate

14. **Constructor validation** (`test_constructor_validates_parameters`)
    - Empty `client_id` → `ValueError`
    - Empty `client_secret` → `ValueError`
    - Empty `tenant_id` → `ValueError`
    - Non-empty parameters pass

**Coverage**: Assert 98%+ line coverage using `pytest-cov`

---

### Task 8: Unit Tests — CopilotReasoningEngine (`tests/unit/integrations/test_copilot_reasoning_engine.py`)

**Deliverable**: ~500 lines, 98%+ coverage

**Mocking Strategy**:
- Mock `aiohttp.ClientSession` for HTTP
- Mock `EntraIDAuthManager` for token management
- Use `pytest.mark.asyncio` for async tests
- Use `pytest` fixtures for common setup

**Test Cases**:

1. **Successful task execution** (`test_reason_successful_execution`)
   - Create valid AITask
   - Mock auth manager returns valid token
   - Mock HTTP POST returns 200 with valid response structure
   - Call `reason(task)`
   - Verify: `success=True`, `tokens_used > 0`, `finish_reason='stop'`
   - Verify logging: `copilot_request_start`, `copilot_request_complete`

2. **Token parsing from response** (`test_reason_parses_tokens_correctly`)
   - Response includes `usage: {prompt_tokens: 10, completion_tokens: 20, total_tokens: 30}`
   - Verify AIResultOutput includes all three values correctly

3. **Response with system prompt** (`test_reason_includes_system_prompt`)
   - AITask with `system_prompt="You are a helpful assistant"`
   - Verify HTTP request includes system message at position 0 in messages array
   - Verify user message follows

4. **HTTP error 401 (authentication)** (`test_reason_auth_error_on_401`)
   - HTTP 401 response
   - Verify: `AuthenticationError` raised
   - Verify logging: `copilot_error` with status_code=401

5. **HTTP error 429 (rate limit)** (`test_reason_rate_limit_on_429`)
   - HTTP 429 with `Retry-After: 60` header
   - Verify: `RateLimitError` raised
   - Verify: error message includes Retry-After value
   - Verify logging

6. **HTTP error 408 (request timeout)** (`test_reason_timeout_on_408`)
   - HTTP 408
   - Verify: `TimeoutError` raised

7. **HTTP error 504 (gateway timeout)** (`test_reason_timeout_on_504`)
   - HTTP 504
   - Verify: `TimeoutError` raised

8. **HTTP error 413 (payload too large)** (`test_reason_input_too_large_on_413`)
   - HTTP 413
   - Verify: `InputValidationError` raised

9. **HTTP error 400 (bad request)** (`test_reason_bad_request_on_400`)
   - HTTP 400 with error details in body
   - Verify: `ValueError` raised
   - Verify error message includes error detail

10. **HTTP error 500 (service error)** (`test_reason_service_error_on_500`)
    - HTTP 500+
    - Verify: `TransientError` raised

11. **Timeout during request** (`test_reason_respects_timeout`)
    - Set `task.timeout_ms = 5000` (5 seconds)
    - Mock HTTP client to simulate timeout
    - Verify `TimeoutError` raised
    - Verify timeout passed correctly to `aiohttp.ClientTimeout(total=5.0)`

12. **Large prompt handling** (`test_reason_with_large_prompt`)
    - Prompt of 2000+ characters (still valid)
    - Verify request sent successfully
    - HTTP 200 returned

13. **Oversized prompt (HTTP 413)** (`test_reason_with_oversized_prompt`)
    - Prompt exceeds Copilot limit
    - HTTP 413 returned
    - Verify `InputValidationError` raised

14. **Defensive response parsing - missing choices** (`test_reason_missing_choices_in_response`)
    - HTTP 200 but response missing 'choices' key
    - Verify `ValueError("missing choices")` raised
    - Verify error logged with response_body_preview

15. **Defensive response parsing - empty choices** (`test_reason_empty_choices_in_response`)
    - HTTP 200 but 'choices' is empty list
    - Verify `ValueError` raised

16. **Defensive response parsing - missing message** (`test_reason_missing_message_in_choice`)
    - HTTP 200 but first choice missing 'message' key
    - Verify `ValueError` raised

17. **Defensive response parsing - missing content** (`test_reason_missing_content_in_message`)
    - HTTP 200, has message, but content is empty or missing
    - Verify content is empty string (not error)
    - Verify logging: `"copilot_empty_response"` with warning

18. **Authentication failure** (`test_reason_auth_manager_raises_auth_error`)
    - Auth manager raises `AuthenticationError`
    - Verify error propagated
    - Verify logging: `copilot_auth_failed`

19. **Network error during request** (`test_reason_network_error_during_request`)
    - HTTP client raises `aiohttp.ClientError`
    - Verify `TransientError` raised

20. **Async timeout during request** (`test_reason_async_timeout_during_request`)
    - HTTP client raises `asyncio.TimeoutError`
    - Verify `TimeoutError` raised

21. **close() method** (`test_close_cleans_up_resources`)
    - Create engine, call `close()`
    - Verify HTTP client closed
    - Call again: safe (no error)
    - Verify logging: `copilot_engine_closed`

22. **close() with owned HTTP client** (`test_close_owned_http_client`)
    - Engine created without provided http_client (owns it)
    - Verify close() calls client.close()

23. **close() with external HTTP client** (`test_close_external_http_client`)
    - Engine created with provided http_client (doesn't own it)
    - Verify close() doesn't call client.close()

24. **Constructor validation** (`test_constructor_validates_parameters`)
    - Empty `client_id` → `ValueError`
    - Empty `client_secret` → `ValueError`
    - Empty `tenant_id` → `ValueError`

25. **Logging verification** (`test_reason_logging_context`)
    - Use `caplog` fixture to capture logs
    - Verify `copilot_request_start` includes `task_id`, `user_id`, `prompt_length`, `max_tokens`
    - Verify `copilot_request_complete` includes `tokens_used`, `latency_ms`, `finish_reason`
    - Verify no full tokens or secrets logged

**Coverage**: Assert 98%+ line coverage

---

### Task 9: Unit Tests — EncryptedTokenStore (`tests/unit/integrations/test_encrypted_token_store.py`)

**Deliverable**: ~300 lines, 98%+ coverage

**Fixtures**:
- Temporary SQLite DB path (use `tmp_path` pytest fixture)
- Sample token dict: `{"access_token": "eyJ...", "refresh_token": "0.A...", "expires_in": 3600, "token_type": "Bearer"}`
- Encryption key (fixed for testing)

**Test Cases**:

1. **Encryption/decryption roundtrip** (`test_token_encryption_roundtrip`)
   - Save token dict with encryption
   - Retrieve and verify matches original exactly

2. **Token storage and retrieval** (`test_save_and_get_token`)
   - Save token for user_id="test_user"
   - Retrieve and verify all fields present and correct

3. **Key derivation from environment** (`test_key_derivation_from_env`)
   - Set `ARC_SAGA_TOKEN_ENCRYPTION_KEY` env var to base64 key
   - Verify key used correctly
   - Verify token can be decrypted

4. **Key derivation from file** (`test_key_derivation_from_file`)
   - Unset env var
   - Create `~/.arc_saga/.token_key` with key
   - Verify key loaded from file

5. **Key generation if not exists** (`test_key_generation_if_not_exists`)
   - Unset env var
   - No key file exists
   - Initialize store
   - Verify key generated and stored in file
   - Verify file permissions are 0600 (user-readable only)

6. **Non-existent token returns None** (`test_get_token_returns_none_if_not_found`)
   - Query for non-existent user_id
   - Verify returns `None` (not error)

7. **Update existing token** (`test_update_existing_token`)
   - Save token for user_id
   - Save different token for same user_id
   - Retrieve and verify new token (not old one)
   - Verify timestamp updated

8. **Concurrent access** (`test_concurrent_token_operations`)
   - Multiple concurrent writes/reads (asyncio.gather)
   - Verify no corruption or race conditions
   - Verify SQLite locks handle concurrency

9. **Error on corrupted encrypted data** (`test_error_on_corrupted_encryption`)
   - Manually corrupt encrypted_data in DB
   - Attempt to retrieve
   - Verify `TokenStorageError` raised with decryption error

10. **Error on invalid encryption key** (`test_error_on_invalid_key`)
    - Save token with one key
    - Initialize new store with different key
    - Attempt to retrieve
    - Verify `TokenStorageError` raised

11. **DB error handling** (`test_error_on_db_failure`)
    - Close DB connection
    - Attempt save/get
    - Verify `TokenStorageError` raised

12. **Large token handling** (`test_large_token_storage`)
    - Token with many fields and large values
    - Verify stored and retrieved correctly

13. **Special characters in token** (`test_special_characters_in_token`)
    - Token containing special characters, emojis, etc.
    - Verify JSON encoding/decoding handles correctly

**Coverage**: Assert 98%+ line coverage

---

## Implementation Verification Checklist

### Code Quality

- [ ] **Type Safety**: `mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py` passes
  - No `Any` without comment justification
  - All function signatures fully annotated
  - Protocols properly implemented with correct signatures

- [ ] **Linting**: `pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/` scores 8.0+
  - No unresolved imports
  - No undefined names
  - No unused variables

- [ ] **Security**: `bandit -r arc_saga/integrations/ arc_saga/exceptions/` reports 0 issues
  - Tokens never logged in full (truncate first 200 chars or use `_truncate()`)
  - HTTP calls use HTTPS only
  - No hardcoded credentials
  - No insecure deserialization
  - Database file and key file have proper permissions

- [ ] **Formatting**: 
  - `black --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py` passes
  - `isort --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/` passes

### Testing

- [ ] **Unit Tests Pass**: `pytest tests/unit/integrations/ -v --tb=short` all pass
  - No skipped tests
  - No warnings (except expected deprecations)

- [ ] **Coverage**: `pytest tests/unit/integrations/ --cov=arc_saga.integrations --cov=arc_saga.exceptions --cov-report=term-missing`
  - 98%+ coverage for both modules
  - Any excluded lines documented

- [ ] **Async Safety**: All async tests use `pytest.mark.asyncio`
  - No blocking calls in async functions
  - Proper await statements
  - Fixtures are async-aware where needed

### Documentation

- [ ] **Docstrings**: All public methods include Google-style docstrings
  - Args with types
  - Returns with type and description
  - Raises with exception types and when raised
  - Examples for complex methods (especially `_is_token_expired`, `_parse_copilot_response`)

- [ ] **Inline Comments**: Complex logic explained
  - JWT decoding steps and edge cases
  - Encryption/decryption process
  - Exponential backoff calculation
  - Error handling rationale
  - Retry logic boundaries (max 5 attempts, ~31s total)

- [ ] **README**: Document created explaining:
  - Architecture diagram (auth flow, token persistence, Copilot API)
  - Token persistence mechanism (encryption, storage, key management)
  - Error categories and handling (permanent vs. transient)
  - Integration with ARC SAGA
  - Edge cases and defensive parsing

### Integration Points

- [ ] **AIProvider Enum**: `COPILOT_CHAT` added to `arc_saga/orchestrator/types.py`
  - No conflicts with existing values
  - Used consistently in code

- [ ] **Exception Imports**: All exceptions importable from `arc_saga.exceptions`
  - Base class `ArcSagaException` available
  - New exceptions inherit correctly
  - Can be caught by caller appropriately

- [ ] **Logging Integration**: All events logged via `log_with_context()`
  - Correct event names (snake_case)
  - Sufficient context included (task_id, user_id, tokens, latency, error details)
  - No sensitive data logged (tokens, secrets, credentials)
  - Sensitive values truncated to 200 chars max

---

## Production Considerations & Edge Cases

### Retry Logic Boundaries

✅ **Auth token refresh**: Max 5 retries on 429, ~31s total wait  
✅ **Copilot API calls**: Errors propagated to caller (FallbackStrategy handles retry in Step 4)  
✅ **Token storage**: Failures propagated as `AuthenticationError` (don't retry)  

### Token Persistence Guarantees

✅ **Transparency**: User stays logged in across reboots  
✅ **Atomicity**: Token saved only if refresh successful AND storage succeeds  
✅ **Rollback**: If storage fails, token not returned (caller must re-authenticate)  
✅ **Encryption**: Fernet symmetric, easy to rotate, no plaintext keys in code  

### Response Parsing Robustness

✅ **Defensive checks**: All response fields validated before access  
✅ **Graceful degradation**: Empty/missing fields logged, not fatal  
✅ **Future-proofing**: Ready for streaming, chunked responses (logged as empty_response, caller decides)  

### Security

✅ **Token logging**: Never logged in full; always truncated or redacted  
✅ **HTTPS-only**: All HTTP calls use https:// URIs  
✅ **Key storage**: Environment variable or user-only file (0600 permissions)  
✅ **Encryption**: Fernet (industry-standard, cryptography library)  
✅ **No credentials in code**: All secrets passed via constructor/env  

---

## Success Criteria (Final Checklist)

✅ **All code type-safe, fully annotated, no `Any` without justification**

✅ **mypy, pylint, bandit, black, isort all pass**

✅ **98%+ test coverage for all modules**

✅ **Comprehensive docstrings (Google-style)**

✅ **Error handling for all HTTP codes (401, 429, 408, 504, 400, 413, 500+)**

✅ **Detailed logging with context (task_id, user_id, tokens, latency, no secrets)**

✅ **Token persistence transparent: user stays logged in across reboots**

✅ **Encryption uses Fernet (symmetric, production-ready)**

✅ **SQLite backend for encrypted tokens with proper schema and permissions**

✅ **Exponential backoff on rate limits (1s, 2s, 4s, 8s, 16s max, 5 retries)**

✅ **Defensive response parsing: all fields validated, edge cases handled gracefully**

✅ **Retry logic bounded: max 5 attempts, ~31s total wait, then fail-fast**

✅ **Token persistence guaranteed: saved only if refresh + storage succeed**

✅ **Integrates cleanly with existing ARC SAGA types, logging, exceptions**

✅ **Ready for Step 2 (ResponseMode + ProviderRouter)**

---

## Deliverables Summary

| File | Lines | Purpose |
|------|-------|---------|
| `arc_saga/orchestrator/protocols.py` | 20–30 | Protocol definitions |
| `arc_saga/exceptions/integration_exceptions.py` | 60–100 | Exception classes (5 types) |
| `arc_saga/integrations/encrypted_token_store.py` | 150–200 | Encrypted SQLite token storage |
| `arc_saga/integrations/entra_id_auth_manager.py` | 250–300 | OAuth2 token lifecycle, exponential backoff |
| `arc_saga/integrations/copilot_reasoning_engine.py` | 300–400 | Copilot Chat API, defensive parsing |
| `arc_saga/orchestrator/types.py` | 1 line (enum) | Add COPILOT_CHAT |
| `tests/unit/integrations/test_entra_id_auth_manager.py` | 350–450 | Auth manager tests (14+ cases) |
| `tests/unit/integrations/test_copilot_reasoning_engine.py` | 500–600 | Engine tests (25+ cases) |
| `tests/unit/integrations/test_encrypted_token_store.py` | 250–350 | Store tests (13+ cases) |

**Total Production Code**: ~900–1,200 lines  
**Total Test Code**: ~1,100–1,400 lines  
**Total Coverage**: 98%+

---

## References

- **Microsoft Graph Copilot Chat API**: https://learn.microsoft.com/en-us/graph/api/resources/chat
- **Entra ID OAuth2**: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
- **Token Refresh**: https://learn.microsoft.com/en-us/entra/identity-platform/refresh-tokens
- **Fernet Encryption**: https://cryptography.io/en/latest/fernet/
- **ARC SAGA Types**: `arc_saga/orchestrator/types.py`
- **Logging**: `arc_saga/error_instrumentation.py`
- **Attached**: MICROSOFT_COPILOT_INTEGRATION_GUIDE.txt, COPILOT_BRAINSTORM_12_4_RATE_LIMITS_WORKflow.txt

---

## Next Steps (After Step 1 Verification)

1. **Step 2**: ResponseMode + ReasoningEngineRegistry + ProviderRouter
2. **Step 3**: Extend Orchestrator.execute_workflow() with provider/mode support + TokenBudgetManager integration
3. **Step 4**: FallbackStrategy + circuit breaker error handling
4. **Step 5**: Integration tests + end-to-end workflow validation

---

## Messaging: "The Trust Layer"

**For stakeholders & developers:**

> **Phase 2.3 Step 1: The Trust Layer**
>
> Every reasoning call in ARC SAGA goes through the **trust layer** — secure authentication, encrypted token persistence, and comprehensive logging. This step guarantees that Copilot integration is not just functional, but reliable and auditable.
>
> **What you get:**
> - Seamless authentication: users stay logged in across reboots
> - Transparent token management: refresh happens automatically, invisibly
> - Full observability: every call logged with context for debugging and compliance
> - Production-ready error handling: distinguishes transient failures (retry) from permanent ones (escalate)
>
> This foundation enables the multi-engine cockpit (Steps 2–5) without fighting auth or token issues.

---

## Usage Example (After Implementation)

```python
# Create auth manager + reasoning engine
token_store = SQLiteEncryptedTokenStore(db_path="~/.arc_saga/tokens.db")
engine = CopilotReasoningEngine(
    client_id="your_client_id",
    client_secret="your_client_secret",
    tenant_id="your_tenant_id",
    token_store=token_store,
)

# Execute a task
task = AITask(
    id="task_001",
    input_data=AITaskInput(
        prompt="Explain quantum computing",
        provider=AIProvider.COPILOT_CHAT,
        max_tokens=1000,
        temperature=0.7,
        user_id="joshua",
    ),
    timeout_ms=30000,
)

try:
    result = await engine.reason(task)
    print(f"Tokens used: {result.output_data.tokens_used}")
    print(f"Response: {result.output_data.response}")
except AuthenticationError:
    print("User must re-authenticate")
except RateLimitError:
    print("Rate limited; queue task for later")
except TimeoutError:
    print("Request timed out; retry or escalate")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Clean up
    await engine.close()
```

---

**READY FOR CURSOR**: Copy this entire revised prompt into Cursor and execute. Cursor will generate all 9 tasks with production-ready code, comprehensive tests, defensive parsing, robust retry logic, and full documentation.
