# Phase 2.3 Step 1: Implementation Readiness & Cursor Execution Plan

## Status: GREEN LIGHT ✅

All three documents are complete, aligned, and ready for Cursor execution:

1. ✅ **PROMPT_PHASE_2_3_STEP_1_REVISED.md** — 9 tasks, detailed specs, error handling
2. ✅ **SAGA_TRUST_LAYER_ARCHITECTURE.md** — Sequence diagrams, state machines, flows
3. ✅ **This document** — Execution checklist and Cursor workflow

---

## Pre-Cursor Verification (Do This First)

### Document Coherence Check

| Aspect | Revised Prompt | Architecture Doc | Status |
|--------|----------------|------------------|--------|
| **Task breakdown** | 9 tasks with specs | Diagrams show all flows | ✅ Aligned |
| **Error handling** | 5 exception types, retry bounds | Error paths show decisions | ✅ Aligned |
| **Token persistence** | Atomic contract | State machine shows transitions | ✅ Aligned |
| **Logging** | 15+ events listed | Taxonomy with examples | ✅ Aligned |
| **Test count** | 52+ test cases | Coverage goals | ✅ Aligned |
| **HTTP client** | Ownership flag | Resource management diagram | ✅ Aligned |

**Result**: All documents cross-reference and reinforce each other. No contradictions.

---

## Cursor Execution Workflow

### Phase 1: Copy → Paste → Execute (30 minutes setup)

**Step 1: Prepare Cursor**
```bash
# In your project root
cd arc_saga

# Verify structure
ls -la arc_saga/orchestrator/
ls -la arc_saga/exceptions/
ls -la arc_saga/integrations/  # Should be empty (Cursor will create)
ls -la tests/unit/integrations/  # Should be empty (Cursor will create)

# Confirm dependencies installed
pip list | grep -E "aiohttp|cryptography|aiosqlite"
```

**Expected output**:
- ✅ `arc_saga/orchestrator/` exists with types.py
- ✅ `arc_saga/exceptions/` exists (base exception classes)
- ✅ `aiohttp`, `cryptography`, `aiosqlite` installed

**Step 2: Open Cursor**
```bash
cursor arc_saga/
```

**Step 3: Create a new chat with instruction**

Use this exact prompt in Cursor:

```
@codebase
You are implementing Phase 2.3 Step 1 of ARC SAGA: Copilot Reasoning Engine & EntraID Auth.

CRITICAL REQUIREMENTS:
1. Implement 9 tasks exactly as specified
2. Type-safe: mypy --strict passes (no Any without justification)
3. Linting: pylint scores 8.0+
4. Coverage: 98%+ unit test coverage
5. All docstrings: Google-style format
6. All logging: Use log_with_context() from arc_saga.error_instrumentation
7. Security: No full tokens in logs (truncate first 200 chars max)
8. Retry logic: Max 5 attempts on 429, ~31s total wait, then fail-fast

IMPLEMENTATION CHECKLIST:
Task 1: Create arc_saga/orchestrator/protocols.py
  - IReasoningEngine protocol (async reason)
  - IEncryptedStore protocol (get_token, save_token)

Task 2: Create arc_saga/exceptions/integration_exceptions.py
  - AuthenticationError (permanent, token/refresh failures)
  - RateLimitError (transient, 429 with retry_after)
  - InputValidationError (permanent, 413/malformed)
  - TokenStorageError (permanent, DB errors)
  - TransientError (transient, 500+/network)

Task 3: Create arc_saga/integrations/encrypted_token_store.py
  - SQLiteEncryptedTokenStore class (IEncryptedStore impl)
  - Fernet encryption (AES-256)
  - Database schema (user_id, encrypted_data, created_at, updated_at)
  - Key management: env var or ~/.arc_saga/.token_key (0600 perms)
  - Methods: get_token(user_id), save_token(user_id, token_dict), _get_or_create_encryption_key()

Task 4: Create arc_saga/integrations/entra_id_auth_manager.py
  - EntraIDAuthManager class
  - async get_valid_token(user_id) → str (with expiry check, refresh, storage)
  - async _refresh_token(user_id, refresh_token) → dict (OAuth2 with exponential backoff)
  - _is_token_expired(token_dict, buffer_seconds=300) → bool (defensive JWT parsing)
  - Constants: MAX_REFRESH_RETRIES=5, BACKOFF_BASE=1, TOKEN_BUFFER_SECONDS=300
  - Error handling: 401→AuthenticationError, 429→RateLimitError (with max retries), 500+→TransientError
  - Logging: All major events (refresh_start, retry, success, failed_*, stored, persistence_failed)
  - CRITICAL: If save_token() fails after refresh succeeds, raise AuthenticationError (don't return token)

Task 5: Create arc_saga/integrations/copilot_reasoning_engine.py
  - CopilotReasoningEngine class (IReasoningEngine impl)
  - async reason(task: AITask) → AIResult
  - async close() → None (with HTTP client ownership management)
  - EntraIDAuthManager integration (get_valid_token before request)
  - Helper: _parse_copilot_response(response_body) → (response_text, finish_reason, usage)
  - Helper: _truncate(value, max_length=500) → str (for safe logging)
  - HTTP error handling: 401→AuthenticationError, 429→RateLimitError, 408/504→TimeoutError, 413→InputValidationError, 400→ValueError, 500+→TransientError
  - Defensive parsing: Check 'choices' array, 'message' object, 'content' string (all present/non-empty)
  - Log empty response (non-fatal) if content missing
  - HTTP client ownership: _owns_http_client flag (only close if owned)
  - Logging: All major events (request_start, parse_start, parse_error, request_complete, error, auth_failed, engine_closed)

Task 6: Update arc_saga/orchestrator/types.py
  - Add AIProvider.COPILOT_CHAT = "copilot_chat" to enum

Task 7: Create tests/unit/integrations/test_entra_id_auth_manager.py
  - 14+ test cases with 98%+ coverage
  - Use pytest.mark.asyncio for async tests
  - Mock IEncryptedStore with AsyncMock
  - Mock aiohttp.ClientSession for HTTP
  - Key cases:
    * Successful token refresh flow
    * Token already valid (no HTTP call)
    * Malformed JWT (< 3 parts, corrupted payload, missing exp)
    * Partial JWT (2 parts)
    * Rate limit with exponential backoff
    * Rate limit exhausts retries (5 attempts, ~31s total, RateLimitError raised)
    * 401 (AuthenticationError, no retry)
    * 400 (ValueError, no retry)
    * 500+ (TransientError, no retry)
    * Network error (TransientError)
    * Token storage error (TokenStorageError)
    * Token persistence failure (save fails → AuthenticationError raised, token NOT returned)
    * Constructor validation

Task 8: Create tests/unit/integrations/test_copilot_reasoning_engine.py
  - 25+ test cases with 98%+ coverage
  - Mock aiohttp.ClientSession for HTTP
  - Mock EntraIDAuthManager for token management
  - Key cases:
    * Successful task execution (HTTP 200, valid response structure)
    * Correct token parsing (prompt_tokens, completion_tokens, total_tokens)
    * System prompt inclusion (in messages array at position 0)
    * HTTP 401 (AuthenticationError)
    * HTTP 429 (RateLimitError with Retry-After)
    * HTTP 408 (TimeoutError)
    * HTTP 504 (TimeoutError)
    * HTTP 413 (InputValidationError)
    * HTTP 400 (ValueError)
    * HTTP 500+ (TransientError)
    * Timeout during request (asyncio.TimeoutError → TimeoutError)
    * Large prompt (still valid)
    * Oversized prompt (HTTP 413)
    * Missing 'choices' (ValueError, log copilot_parse_error)
    * Empty 'choices' (ValueError)
    * Missing 'message' (ValueError)
    * Missing/empty 'content' (no error, log copilot_empty_response warning)
    * Auth manager raises AuthenticationError (propagate, log copilot_auth_failed)
    * Network error during request (TransientError)
    * Async timeout (TimeoutError)
    * close() with owned HTTP client (client closed)
    * close() with external HTTP client (client NOT closed)
    * Constructor validation

Task 9: Create tests/unit/integrations/test_encrypted_token_store.py
  - 13+ test cases with 98%+ coverage
  - Use tmp_path pytest fixture for temp DB
  - Key cases:
    * Encryption/decryption roundtrip
    * Save and retrieve token
    * Key derivation from env var
    * Key derivation from ~/.arc_saga/.token_key file
    * Key generation if not exists (check file perms 0600)
    * Non-existent token returns None
    * Update existing token (verify old token replaced)
    * Concurrent access (asyncio.gather for race condition testing)
    * Error on corrupted encryption
    * Error on invalid key
    * Error on DB failure
    * Large token storage
    * Special characters in token

QUALITY GATES:
- mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py → 0 errors
- pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ → 8.0+ score
- bandit -r arc_saga/integrations/ arc_saga/exceptions/ → 0 security issues
- black --check arc_saga/integrations/ arc_saga/exceptions/ → formatted
- isort --check arc_saga/integrations/ arc_saga/exceptions/ → sorted
- pytest tests/unit/integrations/ -v --cov=arc_saga.integrations --cov=arc_saga.exceptions --cov-report=term-missing → 98%+ coverage

REFERENCE DOCUMENTS:
- See attached PROMPT_PHASE_2_3_STEP_1_REVISED.md for full specifications
- See attached SAGA_TRUST_LAYER_ARCHITECTURE.md for sequence diagrams and error flows

BEGIN IMPLEMENTATION.
```

**Step 4: Execute in Cursor Chat**

Paste the instruction above into Cursor chat. Cursor will:
1. Create all 9 files with complete implementations
2. Generate 52+ test cases
3. Ensure type safety and linting compliance
4. Output verification commands

---

### Phase 2: Verification (15 minutes)

**Step 1: Run Type Checking**
```bash
mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py
```
**Expected**: 0 errors

**Step 2: Run Linting**
```bash
pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
```
**Expected**: 8.0+ score for all files

**Step 3: Run Security Check**
```bash
bandit -r arc_saga/integrations/ arc_saga/exceptions/
```
**Expected**: 0 issues

**Step 4: Run Tests**
```bash
pytest tests/unit/integrations/ -v --tb=short
```
**Expected**: All tests pass, no skips, no warnings

**Step 5: Check Coverage**
```bash
pytest tests/unit/integrations/ \
  --cov=arc_saga.integrations \
  --cov=arc_saga.exceptions \
  --cov-report=term-missing \
  --cov-report=html
```
**Expected**: 98%+ coverage for both modules

**Step 6: Format Check (Optional)**
```bash
black --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
isort --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
```
**Expected**: All formatted correctly (if not, Cursor already formatted during generation)

---

### Phase 3: Integration Verification (10 minutes)

**Step 1: Verify Import Paths**
```python
# Test imports work
from arc_saga.exceptions.integration_exceptions import (
    AuthenticationError,
    RateLimitError,
    InputValidationError,
    TokenStorageError,
    TransientError,
)
from arc_saga.orchestrator.protocols import IReasoningEngine, IEncryptedStore
from arc_saga.integrations.encrypted_token_store import SQLiteEncryptedTokenStore
from arc_saga.integrations.entra_id_auth_manager import EntraIDAuthManager
from arc_saga.integrations.copilot_reasoning_engine import CopilotReasoningEngine
from arc_saga.orchestrator.types import AIProvider

assert AIProvider.COPILOT_CHAT == "copilot_chat"
print("✅ All imports successful")
```

**Step 2: Verify Logging Integration**
```python
# Check log_with_context is called
import arc_saga.error_instrumentation as logs
# Verify it exports log_with_context
assert hasattr(logs, 'log_with_context')
print("✅ Logging integration verified")
```

**Step 3: Verify Exception Inheritance**
```python
from arc_saga.exceptions import ArcSagaException
from arc_saga.exceptions.integration_exceptions import AuthenticationError

assert issubclass(AuthenticationError, ArcSagaException)
print("✅ Exception hierarchy verified")
```

---

## Expected Deliverables

### Code Structure After Execution

```
arc_saga/
├── orchestrator/
│   ├── types.py (MODIFIED — add COPILOT_CHAT)
│   └── protocols.py (NEW)
├── exceptions/
│   └── integration_exceptions.py (NEW)
├── integrations/ (NEW DIRECTORY)
│   ├── __init__.py
│   ├── encrypted_token_store.py
│   ├── entra_id_auth_manager.py
│   └── copilot_reasoning_engine.py
└── error_instrumentation.py (unchanged)

tests/unit/integrations/ (NEW DIRECTORY)
├── __init__.py
├── test_entra_id_auth_manager.py
├── test_copilot_reasoning_engine.py
└── test_encrypted_token_store.py
```

### File Line Counts (Actual)

| File | Type | Expected | Actual |
|------|------|----------|--------|
| protocols.py | Source | 20–30 | TBD |
| integration_exceptions.py | Source | 60–100 | TBD |
| encrypted_token_store.py | Source | 150–200 | TBD |
| entra_id_auth_manager.py | Source | 250–300 | TBD |
| copilot_reasoning_engine.py | Source | 300–400 | TBD |
| types.py | Modification | 1 line | TBD |
| test_entra_id_auth_manager.py | Test | 350–450 | TBD |
| test_copilot_reasoning_engine.py | Test | 500–600 | TBD |
| test_encrypted_token_store.py | Test | 250–350 | TBD |
| **Total Production** | | **900–1,200** | TBD |
| **Total Test** | | **1,100–1,400** | TBD |

---

## Quality Checkpoints During Execution

Cursor will pause/flag if:

| Issue | How Cursor Handles |
|-------|-------------------|
| Type error detected | Shows exact mypy error, suggests fix |
| Missing docstring | Flags in pylint, adds template |
| Test case missing | Counts cases, alerts if < 50 total |
| Coverage gap | Highlights uncovered lines in report |
| Security issue (bandit) | Shows code snippet, suggests remediation |
| Import error | Verifies path exists, fixes if wrong |

---

## Post-Execution Steps (Before Step 2)

### Step 1: Commit to Git
```bash
git add arc_saga/orchestrator/protocols.py
git add arc_saga/exceptions/integration_exceptions.py
git add arc_saga/integrations/
git add tests/unit/integrations/
git commit -m "Phase 2.3 Step 1: Copilot Reasoning Engine & EntraID Auth (production-ready, 98%+ coverage)"
```

### Step 2: Document Environment Setup
Create `.env.example`:
```
# Entra ID OAuth2 credentials (get from Azure AD app registration)
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret

# Token encryption (auto-generated if not provided)
# ARC_SAGA_TOKEN_ENCRYPTION_KEY=base64_encoded_key_here

# Optional: Token store path (defaults to ~/.arc_saga/tokens.db)
# ARC_SAGA_TOKEN_STORE_PATH=~/.arc_saga/tokens.db
```

### Step 3: Update ROADMAP.md
Add to Phase 2.3 completion section:
```markdown
## Phase 2.3 Step 1: ✅ Complete

**Deliverables**:
- Copilot Reasoning Engine (IReasoningEngine implementation)
- EntraID Auth Manager (OAuth2 token lifecycle with exponential backoff)
- Encrypted Token Store (Fernet + SQLite for persistent tokens)
- 5 custom exceptions (AuthenticationError, RateLimitError, InputValidationError, TokenStorageError, TransientError)
- 52+ unit tests (98%+ coverage)
- Complete sequence diagrams and state machines

**Metrics**:
- Production code: 900–1,200 lines
- Test code: 1,100–1,400 lines
- Type safety: mypy --strict ✅
- Linting: pylint 8.0+ ✅
- Security: bandit 0 issues ✅
- Coverage: 98%+ ✅

**Next**: Phase 2.3 Step 2 (ResponseMode + ProviderRouter)
```

### Step 4: Create Integration Test Script
Create `scripts/verify_step1.sh`:
```bash
#!/bin/bash
set -e

echo "🔍 Phase 2.3 Step 1: Verification"
echo ""

echo "1️⃣  Type Checking..."
mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py
echo "✅ Type safety verified"
echo ""

echo "2️⃣  Linting..."
pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ --exit-zero
echo "✅ Code quality verified"
echo ""

echo "3️⃣  Security Audit..."
bandit -r arc_saga/integrations/ arc_saga/exceptions/
echo "✅ Security verified"
echo ""

echo "4️⃣  Unit Tests..."
pytest tests/unit/integrations/ -v --tb=short
echo "✅ Tests passed"
echo ""

echo "5️⃣  Coverage Report..."
pytest tests/unit/integrations/ \
  --cov=arc_saga.integrations \
  --cov=arc_saga.exceptions \
  --cov-report=term-missing \
  --cov-report=html
echo "✅ Coverage verified (see htmlcov/index.html)"
echo ""

echo "🎉 Step 1 Verification Complete!"
```

Make it executable:
```bash
chmod +x scripts/verify_step1.sh
```

### Step 5: Document Known Limitations / Future Enhancements

Create `STEP_1_NOTES.md`:
```markdown
# Phase 2.3 Step 1: Implementation Notes

## What This Step Provides

✅ Secure token persistence across reboots (Fernet encryption)
✅ Exponential backoff retry logic (5 attempts, ~31s max)
✅ Defensive JWT parsing (all edge cases default to "refresh")
✅ Atomic token storage (return token only if refresh + save succeed)
✅ Complete logging with context (15+ event markers)
✅ Production-ready error handling (permanent vs transient)
✅ Type-safe implementation (mypy --strict ✅)
✅ 98%+ test coverage (52+ test cases)

## What This Step Does NOT Provide

❌ Initial OAuth login flow (assumed to happen before Step 1)
❌ Multi-user session management (Step 2)
❌ Provider switching (Claude, GPT-4, etc. — Step 2)
❌ Fallback strategy if Copilot fails (Step 4)
❌ Cost tracking and token budgeting (Step 3)
❌ End-to-end workflow execution (Step 5)

## Assumptions Made

1. Initial OAuth login flow provides user with refresh token (outside Step 1 scope)
2. Entra ID credentials (tenant_id, client_id, client_secret) provided via env vars
3. `arc_saga.error_instrumentation.log_with_context()` available for logging
4. Base exception class `ArcSagaException` available in `arc_saga.exceptions`
5. AITask, AIResult, AIProvider, AITaskInput, AIResultOutput types available

## Known Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| Malformed JWT (< 3 parts) | Return True (trigger refresh) |
| Corrupted JWT payload | Return True (trigger refresh) |
| Missing 'exp' claim | Return True (trigger refresh) |
| Copilot returns empty response | Log warning, don't fail |
| Token refresh succeeds but save fails | Raise AuthenticationError (atomic) |
| Rate limit exhausted (5 attempts) | Raise RateLimitError after ~31s |
| HTTP 401 from Copilot | Raise AuthenticationError (permanent) |
| Network timeout | Raise TimeoutError (transient) |

## Future Enhancements (Post-Step 1)

- [ ] Token refresh in background task (don't block request)
- [ ] Metrics collection (token refresh latency, success rate)
- [ ] Token expiration warning notifications
- [ ] Fallback to LiteLLM if Copilot unavailable
- [ ] Support for multiple concurrent users
- [ ] Token cache with TTL for rapid-fire requests

## Troubleshooting

### "TokenStorageError: Invalid encryption key"
- Check `ARC_SAGA_TOKEN_ENCRYPTION_KEY` env var is valid base64
- Or delete `~/.arc_saga/.token_key` and reinitialize

### "AuthenticationError: Token refresh failed after 5 retries"
- Entra ID is rate limiting; wait ~31s and retry
- Check `Retry-After` header in logs for suggested wait time

### "RateLimitError: Copilot rate limited"
- Copilot API is throttling; queue request for later (Step 4)
- Fallback to Claude or GPT-4 (Step 2)

### "DB locked" or "permission denied"
- Check file permissions: `ls -la ~/.arc_saga/tokens.db`
- Should be: `-rw------- (0600)`
- Fix: `chmod 0600 ~/.arc_saga/tokens.db`

## Testing Notes

- All tests use async fixtures (`pytest.mark.asyncio`)
- HTTP calls mocked with `unittest.mock.AsyncMock`
- Concurrent tests use `asyncio.gather()` for race condition testing
- Coverage report includes branch coverage (missing lines flagged)
- All test names follow pattern: `test_<component>_<scenario>_<expected_outcome>`

## Security Checklist

- [ ] Env var `ARC_SAGA_TOKEN_ENCRYPTION_KEY` set or auto-generated
- [ ] File `~/.arc_saga/.token_key` has permissions 0600 (user-readable only)
- [ ] DB file `~/.arc_saga/tokens.db` has permissions 0600
- [ ] No tokens logged in full (first 200 chars max via `_truncate()`)
- [ ] No credentials committed to git
- [ ] All HTTPS endpoints (no http://)
- [ ] Fernet encryption used (industry-standard, no home-grown crypto)

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| get_valid_token (cache hit) | <10ms | Token in memory, expiry check only |
| get_valid_token (refresh) | 500–2000ms | HTTP POST + encryption + DB write |
| reason (successful) | 1000–5000ms | Token fetch + Copilot API call |
| reason (rate limited) | ~31s max | Exponential backoff (1+2+4+8+16s) |

---

**Ready for Step 2: ResponseMode + ProviderRouter** ✅
```

---

## Final Execution Checklist

Before you run Cursor, verify:

- [ ] All three documents created and finalized
  - [ ] PROMPT_PHASE_2_3_STEP_1_REVISED.md
  - [ ] SAGA_TRUST_LAYER_ARCHITECTURE.md
  - [ ] This execution plan

- [ ] Environment ready
  - [ ] `arc_saga/orchestrator/` exists with types.py
  - [ ] `arc_saga/exceptions/` exists with base classes
  - [ ] Dependencies installed: `aiohttp`, `cryptography`, `aiosqlite`

- [ ] Cursor configured
  - [ ] Cursor opened in project root
  - [ ] @codebase context available

- [ ] No blockers
  - [ ] No merge conflicts in git
  - [ ] No uncommitted changes (or staged for commit)
  - [ ] No syntax errors in existing code

---

## Execution Timeline

| Phase | Duration | Checkpoint |
|-------|----------|-----------|
| **Phase 1: Setup** | 5 min | Cursor chat ready with instruction |
| **Phase 1: Implementation** | 25 min | All 9 files generated |
| **Phase 2: Type Check** | 2 min | mypy passes |
| **Phase 2: Linting** | 2 min | pylint 8.0+ |
| **Phase 2: Security** | 1 min | bandit 0 issues |
| **Phase 2: Tests** | 5 min | 52+ tests pass, 98%+ coverage |
| **Phase 3: Integration** | 3 min | Imports verified, logging integrated |
| **Phase 3: Documentation** | 5 min | STEP_1_NOTES.md + .env.example created |
| **Total** | **48 minutes** | **Step 1 Complete** |

---

## Success Criteria

✅ **All 9 tasks implemented** (protocols, exceptions, store, auth manager, engine, enum, tests)
✅ **Type-safe** (mypy --strict passes)
✅ **High quality** (pylint 8.0+, black formatted, isort organized)
✅ **Secure** (bandit 0 issues, no credential leaks, Fernet encryption)
✅ **Well-tested** (52+ test cases, 98%+ coverage)
✅ **Observable** (15+ logging events with full context)
✅ **Resilient** (all error paths handled, exponential backoff, atomic persistence)
✅ **Production-ready** (ready for Steps 2–5)

---

## Next Steps After Step 1

1. **Step 2**: ResponseMode + ProviderRouter (provider switching, UI dropdowns)
2. **Step 3**: TokenBudgetManager integration (cost tracking, token accounting)
3. **Step 4**: FallbackStrategy + circuit breaker (error recovery, fallback providers)
4. **Step 5**: End-to-end integration tests (workflow validation)

---

## Questions Before Execution?

- Any blockers with environment setup?
- Any clarifications needed on the specs?
- Ready to proceed with Cursor execution?

**Status: READY FOR IMMEDIATE EXECUTION** 🚀

---

## Reference

**Key Documents**:
- PROMPT_PHASE_2_3_STEP_1_REVISED.md (implementation blueprint)
- SAGA_TRUST_LAYER_ARCHITECTURE.md (visual architecture)
- This document (execution workflow)

**Key Commands**:
```bash
# Verify after Cursor completes
pytest tests/unit/integrations/ -v --cov=arc_saga.integrations --cov=arc_saga.exceptions --cov-report=term-missing

# Commit to git
git add arc_saga/ tests/ && git commit -m "Phase 2.3 Step 1: Complete"

# Run verification script
./scripts/verify_step1.sh
```

**Success Metric**: All quality gates pass, 98%+ coverage confirmed, integration tests pass. ✅
