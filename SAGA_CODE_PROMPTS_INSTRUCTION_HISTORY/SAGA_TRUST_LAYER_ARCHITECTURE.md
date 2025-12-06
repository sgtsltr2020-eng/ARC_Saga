# Phase 2.3 Step 1: Orchestration Spine — Sequence Diagrams & Architecture

## Architecture Overview: The Trust Layer

ARC SAGA's reasoning foundation flows through **five integrated layers**, each with explicit responsibilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER / ORCHESTRATOR                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ AITask(prompt, user_id, timeout_ms)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             Layer 1: CopilotReasoningEngine (Reasoning Brain)               │
│  • Implements IReasoningEngine protocol                                     │
│  • async reason(task) → AIResult                                           │
│  • Manages HTTP client, token coordination                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Extract user_id, build request body
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         Layer 2: EntraIDAuthManager (OAuth2 Token Lifecycle)                │
│  • async get_valid_token(user_id) → str (access_token)                     │
│  • Checks expiry (buffer 300s), refreshes if needed                        │
│  • Retry logic: max 5 attempts, ~31s total wait                            │
│  • Raises: AuthenticationError (permanent), RateLimitError (transient)     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Token missing/expired?
                                 │ → Call _refresh_token()
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│      Layer 3: SQLiteEncryptedTokenStore (Persistent, Encrypted Storage)     │
│  • async get_token(user_id) → Optional[dict]                               │
│  • async save_token(user_id, token_dict) → None                            │
│  • Fernet (AES-256) encryption, 0600 file permissions                      │
│  • Raises: TokenStorageError on decrypt/DB failure                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Key: from env or ~/.arc_saga/.token_key
                                 │ DB: ~/.arc_saga/tokens.db (encrypted blobs)
                                 │
                                 ├─ GET: Decrypt, return token dict
                                 │         (or None if not found)
                                 │
                                 └─ SAVE: Encrypt, upsert to DB
                                          (fail → raise TokenStorageError)
```

---

## Detailed Task Flow: Happy Path

### Scenario: Reasoning on a New User (First Login)

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    USER INITIATES REASONING TASK                            ║
║                    AITask(id, prompt, user_id, timeout_ms)                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                    [CopilotReasoningEngine.reason(task)]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          Log: copilot_request_start         Get user_id from task
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
        [EntraIDAuthManager.get_valid_token("joshua")]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: entra_id_token_check         Fetch from store
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
    [SQLiteEncryptedTokenStore.get_token("joshua")]
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
              Query DB for user_id            Not found (first login)
                    │                                │
              [No row found]              Return None (OK for first login)
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                    Log: entra_id_token_missing
                                    │
                                    ▼
                [Must obtain new tokens via refresh flow]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          Problem: No refresh_token yet!      (First login scenario)
                    │                               │
        Expected: App has initialized user with    │
        initial refresh token via OAuth login      │
        (outside scope of Step 1 — assumes token   │
        already obtained in earlier OAuth flow)    │
                    │                               │
                    ▼
    [EntraIDAuthManager._refresh_token("joshua", refresh_token)]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: entra_id_refresh_start         POST to Entra ID:
                    │                    https://login.microsoftonline.com/
                    │                    {tenant}/oauth2/v2.0/token
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                        [HTTP 200 OK] ← Entra ID returns:
                                    │  {
                                    │    access_token: "eyJ...",
                                    │    refresh_token: "0.A...",
                                    │    expires_in: 3600,
                                    │    token_type: "Bearer"
                                    │  }
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: entra_id_refresh_success       Parse response
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
            [SQLiteEncryptedTokenStore.save_token("joshua", new_tokens)]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              Encrypt token dict        Serialize to JSON:
              with Fernet key           {access_token, refresh_token, ...}
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                          [Upsert into DB]
                    encrypted_tokens(user_id, encrypted_data)
                                    │
                        [INSERT/UPDATE succeeds]
                                    │
                    Log: entra_id_token_stored
                                    │
                                    ▼
                    Return access_token to engine
                                    │
                                    ▼
        [CopilotReasoningEngine — Build Copilot request]
                                    │
            request_body = {
              messages: [{role: user, content: "Explain AI"}],
              max_tokens: 1000,
              temperature: 0.7
            }
                                    │
            Log: copilot_request_start (prompt_length=12, max_tokens=1000)
                                    │
                                    ▼
        [POST to Microsoft Graph: /v1.0/copilot/chat]
            Authorization: Bearer eyJ...
                                    │
                        [HTTP 200 OK] ← Copilot returns:
                                    │  {
                                    │    choices: [{
                                    │      message: {content: "AI is..."},
                                    │      finish_reason: "stop"
                                    │    }],
                                    │    usage: {
                                    │      prompt_tokens: 10,
                                    │      completion_tokens: 50,
                                    │      total_tokens: 60
                                    │    }
                                    │  }
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: copilot_parse_start           Defensive parsing:
                    │                    Check 'choices' array
                    │                    Check 'message' object
                    │                    Check 'content' string
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                    [Parse succeeds — all fields present]
                                    │
                                    ▼
        Build AIResultOutput:
        {
          response: "AI is...",
          tokens_used: 60,
          prompt_tokens: 10,
          completion_tokens: 50,
          provider: COPILOT_CHAT,
          model: "copilot-gpt4o",
          cost_usd: Decimal("0.0"),
          finish_reason: "stop",
          latency_ms: 1250
        }
                                    │
                    Log: copilot_request_complete
                    (tokens_used=60, latency_ms=1250)
                                    │
                                    ▼
                Return AIResult {
                  task_id: "task_001",
                  success: True,
                  output_data: AIResultOutput(...),
                  status: COMPLETED,
                  duration_ms: 1250
                }
                                    │
╔═════════════════════════════════════════════════════════════════════════════╗
║                    RETURN TO ORCHESTRATOR / USER                            ║
║                         ✓ Task Complete                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## Error Path: Token Refresh Rate Limited

### Scenario: Entra ID Returns HTTP 429

```
                    [EntraIDAuthManager._refresh_token("joshua", rt)]
                                    │
                                    ▼
                    POST to Entra ID (1st attempt)
                                    │
                    [HTTP 429 — Too Many Requests]
                    Retry-After: 2 (seconds)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: entra_id_refresh_retry         Exponential backoff:
        (attempt=1, backoff_seconds=1)      wait 1s + jitter
                    │                               │
                    │                    (Retry-After: 2, use max)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                            await asyncio.sleep(2.1)
                                    │
                                    ▼
                    POST to Entra ID (2nd attempt)
                                    │
                    [HTTP 429 — Too Many Requests]
                    Retry-After: 3 (seconds)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: entra_id_refresh_retry         Exponential backoff:
        (attempt=2, backoff_seconds=2)      wait 2s + jitter
                    │                               │
                    │                    (Retry-After: 3, use max)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                            await asyncio.sleep(3.1)
                                    │
                                    ▼
                    POST to Entra ID (3rd attempt)
                                    │
                    [HTTP 200 OK] ← Success!
                                    │
                    Proceed with token storage...
                    (save to DB, return to engine)
```

If all 5 attempts exhausted (~31s total):

```
                    POST to Entra ID (5th attempt)
                                    │
                    [HTTP 429 — Too Many Requests]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Attempts exhausted: 5      Total wait: ~31s
                    │
                    ▼
        raise RateLimitError(
            "Token refresh failed after 5 retries. "
            "Total wait: ~31s. Retry-After: 60"
        )
                    │
                    ▼
        [Propagate to CopilotReasoningEngine.reason()]
                    │
                    ▼
        [Log: copilot_auth_failed]
                    │
                    ▼
        [Re-raise RateLimitError to orchestrator]
                    │
        Orchestrator (Step 4 FallbackStrategy) decides:
        - Retry with Copilot (maybe later)
        - Queue task for later
        - Fallback to LiteLLM
        - Escalate to user
```

---

## Error Path: Malformed Copilot Response

### Scenario: HTTP 200 but Missing 'choices'

```
        [CopilotReasoningEngine — HTTP 200 received]
                                    │
                        response_body = await response.json()
                                    │
                        {
                          "error": {
                            "code": "ServiceUnavailable",
                            "message": "Server error"
                          }
                        }
                                    │
                        (No 'choices' key!)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: copilot_parse_error           [_parse_copilot_response()]
                    │                               │
                    │              if "choices" not in response_body:
                    │                  raise ValueError(
                    │                      "Copilot response missing 'choices' key. "
                    │                      f"Response keys: {list(response_body.keys())}"
                    │                  )
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                        raise ValueError(...)
                                    │
                    [Caught in reason() catch-all]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Log: copilot_error                 Re-raise ValueError
        (error_type=ValueError,
         response_body_preview="...error...")
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
        [Propagate to orchestrator]
                    │
        Orchestrator (Step 4) decides: retry, fallback, or escalate
```

---

## Error Path: Token Persistence Failure

### Scenario: Save Token Fails (Disk Full, Permissions)

```
                    [EntraIDAuthManager.get_valid_token()]
                    Token refresh HTTP 200 successful
                                    │
                    Store new tokens:
                    await token_store.save_token("joshua", new_tokens)
                                    │
                    [SQLiteEncryptedTokenStore.save_token()]
                                    │
                    Encrypt token dict with Fernet key
                    Attempt to INSERT/UPDATE DB
                                    │
                    [DB Error: disk full / permission denied]
                                    │
                    raise TokenStorageError(
                        "Database write failed: [OSError details]"
                    )
                                    │
                    [Caught in get_valid_token()]
                                    │
        ┌───────────────────────────────────────┐
        │ CRITICAL DECISION POINT               │
        │                                       │
        │ Token refresh succeeded but storage   │
        │ failed. DO NOT return token.          │
        │ Caller would use token once, then fail│
        │ on next call when token not persisted.│
        └───────────────────────────────────────┘
                                    │
                                    ▼
        Log: entra_id_token_persistence_failed
        (user_id="joshua", error_type=TokenStorageError)
                                    │
                                    ▼
        raise AuthenticationError(
            "Token refresh succeeded but persistence failed. "
            "User must re-authenticate."
        )
                                    │
                    [Propagate to CopilotReasoningEngine.reason()]
                                    │
                    ▼
        Log: copilot_auth_failed
        (error_type=AuthenticationError)
                                    │
                    ▼
        [Re-raise AuthenticationError to orchestrator]
                                    │
        User/orchestrator must re-authenticate
        (start OAuth login flow again)
```

---

## Token Expiry Detection: Edge Cases

### JWT Parsing Defensive Checks

```
    _is_token_expired(token_dict, buffer_seconds=300)
                                    │
                                    ├─ Extract access_token
                                    │
                                    ▼
                    Split by "." → [header, payload, sig]
                                    │
            ┌───────────┬───────────┼───────────┬────────────┐
            │           │           │           │            │
        (3 parts) (2 parts)  (1 part)  (4+ parts) (malformed)
            │           │           │           │            │
            ▼           ▼           ▼           ▼            ▼
        Decode   Return  Return  Return     Return
        payload   TRUE    TRUE    TRUE      TRUE
            │     (refresh)(refresh)(refresh)(refresh)
            │
            ▼
    Decode base64url(payload):
            │
        ┌───┴────────────────────────────┐
        │                                │
    (Valid)                      (Corrupted base64)
        │                                │
        ▼                                ▼
    Extract "exp"              return TRUE
    claim                       (refresh)
        │
        ├─ Missing "exp"
        │  └─ return TRUE (refresh)
        │
        ├─ exp <= now()
        │  └─ return TRUE (expired)
        │
        ├─ exp < now() + 300s
        │  └─ return TRUE (within buffer)
        │
        └─ exp >= now() + 300s
           └─ return FALSE (still valid)
```

**Key insight**: All edge cases default to "expired" (return TRUE), triggering refresh. This is safe: if token is actually valid, refresh will return same token again.

---

## Logging Taxonomy

### Event Markers (Every Major Milestone Logged)

```
ENTRA ID AUTHENTICATION
├─ entra_id_token_check ..................... Fetching token from store
├─ entra_id_token_valid ..................... Token exists and not expired
├─ entra_id_token_expired ................... Token expired, triggering refresh
├─ entra_id_refresh_start ................... Beginning OAuth token refresh
├─ entra_id_refresh_retry ................... Retry on 429 with backoff
├─ entra_id_refresh_success ................. HTTP 200, new tokens obtained
├─ entra_id_refresh_failed_permanent ........ 401/400 — permanent error
├─ entra_id_refresh_failed_transient ........ 500+/network — transient error
├─ entra_id_token_stored .................... New tokens persisted to store
├─ entra_id_token_persistence_failed ........ Storage write failed (critical)

COPILOT API
├─ copilot_request_start .................... Building and logging request
├─ copilot_parse_start ...................... Parsing response
├─ copilot_parse_error ...................... Defensive parsing failed
├─ copilot_empty_response ................... Content empty (streaming?)
├─ copilot_request_complete ................. HTTP 200, response parsed
├─ copilot_error ............................ HTTP error (status_code logged)
├─ copilot_auth_failed ...................... Token auth failed
├─ copilot_engine_closed .................... Resources cleaned up

TOKEN STORE
├─ token_store_get .......................... Retrieving token from DB
├─ token_store_save ......................... Persisting encrypted token
├─ token_store_error ........................ DB/encryption error

EXAMPLE LOG OUTPUT
─────────────────────────────────────────────────────

[2025-12-04 18:45:30.123] copilot_request_start
  task_id=task_001
  user_id=joshua
  prompt_length=47
  max_tokens=1000
  temperature=0.7

[2025-12-04 18:45:30.145] entra_id_token_check
  user_id=joshua

[2025-12-04 18:45:30.156] token_store_get
  user_id=joshua

[2025-12-04 18:45:30.167] entra_id_token_valid
  user_id=joshua
  seconds_until_expiry=2847

[2025-12-04 18:45:30.789] copilot_request_complete
  task_id=task_001
  user_id=joshua
  tokens_used=60
  prompt_tokens=10
  completion_tokens=50
  latency_ms=644
  finish_reason=stop
```

---

## State Diagram: Token Lifecycle

```
                    ┌──────────────────────────┐
                    │   Token Not Found / Null  │
                    │   (First Login or Revoked)│
                    └────────────┬─────────────┘
                                 │
                    Must initiate OAuth login
                    (outside Step 1 scope)
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Stored (not encrypted)  │
                    │  Refresh token obtained  │
                    │  from initial OAuth      │
                    └────────────┬─────────────┘
                                 │
        [User calls get_valid_token() on app startup]
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Loading from Encrypted   │
                    │ Store (disk read)        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┬────────────┐
                    │            │            │
            (Found & valid)  (Expired)   (Not found)
                    │            │            │
                    ▼            ▼            ▼
            ┌─────────────┐  ┌─────────┐  ┌─────────┐
            │ Valid Token │  │Refreshing│  │Error:   │
            │ (In Buffer) │  │ via OAuth│  │Must Re- │
            └─────────────┘  │  (Retry  │  │Authenticate
                    │        │ logic)   │  └─────────┘
                    │        └────┬────┘
                    │             │
                    │    ┌────────┴────────┐
                    │    │                 │
                    │  (Success)       (Failure)
                    │    │                 │
                    │    ▼                 ▼
                    │ ┌───────────┐  ┌──────────────┐
                    │ │New Token  │  │Error (Perm or│
                    │ │Obtained   │  │Transient)    │
                    │ └─────┬─────┘  └──────────────┘
                    │       │
                    │    Encrypt & Save
                    │       │
                    │    ┌──┴──────────────┐
                    │    │                 │
                    │(Success)        (Failure)
                    │    │                 │
                    │    ▼                 ▼
                    │ ┌─────────┐  ┌──────────────┐
                    │ │Persisted│  │Error: Disk   │
                    │ │in Store │  │full/Perms    │
                    │ └────┬────┘  │(Raise Auth   │
                    │      │       │Error, user   │
                    │      │       │must re-auth) │
                    │      │       └──────────────┘
                    │      │
                    └──────┼──────────────────────┐
                           │                      │
                           ▼                      │
                    ┌────────────────┐             │
                    │ Return Token   │             │
                    │ to Engine      │             │
                    └────────────────┘             │
                           │                      │
                    Engine makes request to Copilot
                    with token as Authorization
                           │
                    ┌───────┴───────┐
                    │               │
              (HTTP 200)         (HTTP 401)
              ✓ Success        ✗ Token invalid
                    │               │
                    │       [Next call: Token expired,
                    │        re-authenticate]
                    │
                    ▼
            ┌────────────────────┐
            │ Return AIResult    │
            │ to Orchestrator    │
            └────────────────────┘
```

---

## Resource Management: HTTP Client Ownership

```
SCENARIO 1: Engine Creates HTTP Client (Owns It)

    engine = CopilotReasoningEngine(
        client_id="...",
        client_secret="...",
        tenant_id="...",
        token_store=store,
        http_client=None  ← Not provided
    )
    
    In __init__:
    ├─ self.http_client = aiohttp.ClientSession()  ← Created
    └─ self._owns_http_client = True                ← Mark as owned
    
    ...later...
    
    await engine.close()
    
    In close():
    ├─ if self._owns_http_client and self.http_client:
    │  └─ await self.http_client.close()  ← We close it
    └─ log: copilot_engine_closed


SCENARIO 2: Caller Provides HTTP Client (We Don't Own It)

    http_client = aiohttp.ClientSession()  ← Created externally
    
    engine = CopilotReasoningEngine(
        client_id="...",
        client_secret="...",
        tenant_id="...",
        token_store=store,
        http_client=http_client  ← Provided
    )
    
    In __init__:
    ├─ self.http_client = http_client  ← Use provided
    └─ self._owns_http_client = False   ← Mark as NOT owned
    
    ...later...
    
    await engine.close()
    
    In close():
    ├─ if self._owns_http_client and self.http_client:
    │  └─ # Skipped: _owns_http_client is False
    └─ log: copilot_engine_closed (client NOT closed by engine)
    
    # Caller must close:
    await http_client.close()
```

---

## Integration with Orchestrator (Step 2+)

### Provider Switching: Copilot → Claude → LiteLLM

```
                    User selects provider from dropdown
                                    │
                    Dropdown: "Provider" = "Claude"
                                    │
                                    ▼
        [Orchestrator.execute_workflow(..., provider=CLAUDE, ...)]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        Get ReasoningEngineRegistry         Lookup engine
                    │                    for provider=CLAUDE
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    engine = registry.get_engine(CLAUDE)
                    → AnthropicReasoningEngine instance
                                    │
                                    ▼
                    Invoke: result = await engine.reason(task)
                                    │
                    Same AITask/AIResult contract
                    Different underlying implementation
                                    │
                                    ▼
                    Return AIResult to user
```

---

## Summary: Trust Layer Benefits

| Aspect | Guarantee |
|--------|-----------|
| **Authentication** | Transparent: users stay logged in across reboots |
| **Token Persistence** | Atomic: saved only if refresh succeeds + storage succeeds |
| **Error Semantics** | Clear: permanent vs transient, retry logic bounded |
| **Observability** | Complete: every event logged with context (task_id, user_id, tokens) |
| **Security** | Hardened: Fernet encryption, 0600 perms, no credential leaks |
| **Resilience** | Production-ready: 98%+ coverage, defensive parsing, edge cases handled |

This foundation enables Steps 2–5 to focus on orchestration, fallback, and cost tracking without fighting auth headaches.

---

## Deployment Checklist: Trust Layer Ready

- [ ] All 9 tasks implemented + 98%+ coverage verified
- [ ] Env var `ARC_SAGA_TOKEN_ENCRYPTION_KEY` documented (or auto-generate ~/.arc_saga/.token_key)
- [ ] DB path `~/.arc_saga/tokens.db` created with proper perms (0600)
- [ ] All logging events configured in observability stack
- [ ] Error codes mapped in orchestrator's error handler
- [ ] Rate limit handling (5 retries, ~31s max) communicated to users
- [ ] OAuth login flow (outside Step 1) ensures initial refresh token obtained
- [ ] Step 2 (ResponseMode + ProviderRouter) ready to integrate

---

## References

- **Sequence diagrams**: All happy path, rate limit, and persistence failure scenarios
- **State machine**: Token lifecycle from null → stored → valid → expired → refresh
- **Resource ownership**: HTTP client lifecycle management
- **Provider switching**: Integration point with Steps 2–5
- **Logging taxonomy**: 15+ event markers for complete observability
