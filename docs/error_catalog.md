# Error Catalog

# Comprehensive Database of All Errors Encountered

# Use this for debugging and error prevention

---

## 🎯 HOW TO USE THIS DOCUMENT

1. **When you encounter an error**, search this catalog
2. **If it exists**, use the proven fix + prevention strategy
3. **If it doesn't exist**, work through the debugging framework, then document it here
4. **Reference in prompts**: "@codebase We've seen this before - see ERROR-001 in error_catalog.md"

**Format:**

```
## ERROR-XXX: [Error Name]

**Error Message:**
The exact error text

**Frequency:**
Low / Medium / High

**Root Cause:**
What actually causes this

**Context Clues:**
What to look for in logs

**Proven Fixes:**
1. Fix A (success rate: X%)
2. Fix B (success rate: Y%)

**Prevention:**
How to prevent going forward

**Similar Errors:**
- ERROR-YYY: [Related error]

**Debug Checklist:**
- [ ] Step 1
- [ ] Step 2
```

---

## Template Errors (Fill These In As You Encounter Issues)

## ERROR-001: Database Connection Timeout

**Error Message:**

```
asyncpg.exceptions.PoolAcquireTimeoutError: cannot acquire connection to database within 5.000 seconds
```

**Frequency:** Medium (2-3 times per day during testing)

**Root Cause:**

1. Database overloaded (too many queries)
2. Network latency spike
3. Connection pool too small
4. Long-running queries blocking pool

**Context Clues:**

- High database CPU usage
- Slow query logs showing long queries
- Connection pool size < 5
- Network latency spikes in metrics

**Proven Fixes:**

1. **Increase timeout** (success rate: 40%)

   - Change: DB_TIMEOUT = 10000 (from 5000)
   - Reason: Gives slow queries more time
   - When to use: If queries generally slow but work

   ```python
   async def get_pool():
       return await asyncpg.create_pool(
           DATABASE_URL,
           timeout=10,  # 10 seconds instead of 5
           min_size=10,  # Increased pool
           max_size=20
       )
   ```

2. **Increase connection pool** (success rate: 50%)

   - Change: min_size=10, max_size=20 (from 5, 10)
   - Reason: More connections available
   - When to use: Under high load

   ```python
   pool = await asyncpg.create_pool(
       DATABASE_URL,
       min_size=10,
       max_size=20,
       timeout=5
   )
   ```

3. **Optimize slow queries** (success rate: 80%)

   - Check: EXPLAIN ANALYZE on slow queries
   - Fix: Add missing indexes
   - When to use: Query performance the issue

   ```sql
   CREATE INDEX idx_conversations_provider_date
   ON conversations(provider, created_at DESC);
   ```

4. **Add connection pooling** (success rate: 60%)

   - Implement: SQLAlchemy async session pooling
   - When to use: If not using pooling already

5. **Implement circuit breaker** (success rate: 90%)

   - When database overloaded, fail fast instead of queuing
   - Allows database to recover

   ```python
   db_circuit_breaker = CircuitBreaker(
       failure_threshold=5,
       recovery_timeout=60
   )

   async def get_connection():
       return await db_circuit_breaker.call(
           pool.acquire
       )
   ```

**Prevention:**

- [ ] Monitor pool usage continuously
- [ ] Alert if pool usage > 80%
- [ ] Create indexes on all WHERE columns
- [ ] Implement query timeout monitoring
- [ ] Use connection pooling (min_size >= 5)
- [ ] Keep max_size sized for peak load

**Seen in ARC SAGA:**

- 2024-12-01 14:23: Fixed by increasing pool size to 20
- 2024-11-28 09:45: Fixed by optimizing conversation_capture query
- 2024-11-25 16:30: Fixed by adding index on (provider, created_at)

**Related Errors:**

- ERROR-002: Connection Refused
- ERROR-003: Deadlock

---

## ERROR-002: Connection Refused

**Error Message:**

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Frequency:** Low (happens when database stopped)

**Root Cause:**

1. Database server not running
2. Database crashed
3. Wrong host/port in connection string
4. Firewall blocking connection

**Context Clues:**

- Database process not running
- Connection refused immediately (no timeout)
- Network connectivity issues

**Proven Fixes:**

1. **Restart database** (success rate: 95%)

   ```bash
   systemctl restart postgresql
   ```

2. **Verify connection string** (success rate: 50%)

   ```python
   print(DATABASE_URL)  # Check if correct
   ```

3. **Check firewall** (success rate: 40%)
   ```bash
   telnet localhost 5432  # Can you connect?
   ```

**Prevention:**

- [ ] Health checks monitoring database availability
- [ ] Alerts on connection refused
- [ ] Automated restart of database on failure

---

## ERROR-003: Rate Limit Exceeded

**Error Message:**

```
RateLimitError: You have exceeded your API quota. Rate limit: 100 requests/minute
```

**Frequency:** Medium (when hitting external API limits)

**Root Cause:**

1. Sending requests too fast
2. No backoff strategy
3. Concurrent requests overwhelming API
4. No rate limit awareness

**Context Clues:**

- Multiple requests sent rapidly
- HTTP 429 responses
- API response headers: `X-RateLimit-Remaining: 0`

**Proven Fixes:**

1. **Implement exponential backoff** (success rate: 80%)

   ```python
   async def call_with_backoff(func, max_attempts=5):
       for attempt in range(max_attempts):
           try:
               return await func()
           except RateLimitError:
               if attempt == max_attempts - 1:
                   raise
               delay = min(2 ** attempt, 60)
               await asyncio.sleep(delay)
   ```

2. **Implement rate limiter** (success rate: 90%)

   ```python
   rate_limiter = RateLimiter(
       rate=100,  # 100 requests
       per=60,    # per 60 seconds
   )

   async def call_api():
       await rate_limiter.wait()
       return await api_call()
   ```

3. **Add queue + throttle** (success rate: 95%)

   ```python
   # Queue requests, process at controlled rate
   queue = asyncio.Queue()

   async def process_queue():
       while True:
           request = await queue.get()
           try:
               await api_call(request)
           except RateLimitError:
               await asyncio.sleep(60)  # Back off
   ```

**Prevention:**

- [ ] Check rate limit headers before each call
- [ ] Track API usage
- [ ] Alert if approaching limit
- [ ] Never send burst of requests
- [ ] Implement exponential backoff

---

## ERROR-004: Null Pointer / Attribute Error

**Error Message:**

```
AttributeError: 'NoneType' object has no attribute 'id'
```

**Frequency:** High (usually code bug, not runtime issue)

**Root Cause:**

1. Variable is None when shouldn't be
2. Function returned None unexpectedly
3. Query returned no results
4. Missing error checking

**Context Clues:**

- Line accesses attribute on None
- Function didn't return expected value
- Database query returned no rows

**Proven Fixes:**

1. **Add None check** (success rate: 100%)

   ```python
   # ❌ Bad
   user = db.get_user(user_id)
   print(user.name)  # Fails if None

   # ✅ Good
   user = db.get_user(user_id)
   if not user:
       raise UserNotFound(f"User {user_id} not found")
   print(user.name)
   ```

2. **Use Optional type hints** (success rate: 95%)

   ```python
   from typing import Optional

   def get_user(user_id: str) -> Optional[User]:
       """Returns User or None if not found."""
       return db.query(User).filter(User.id == user_id).first()

   # Caller knows it might be None
   user = get_user(user_id)
   if user:
       print(user.name)
   ```

3. **Add mypy strict checking** (success rate: 100%)
   - Forces you to handle Optional types
   - Catches these bugs at type-check time

**Prevention:**

- [ ] Use type hints (Optional for nullable values)
- [ ] Use mypy --strict (enforces None handling)
- [ ] Add None checks before accessing attributes
- [ ] Return meaningful errors, not None

---

## ERROR-005: Stack Overflow / Recursion Limit

**Error Message:**

```
RecursionError: maximum recursion depth exceeded
```

**Frequency:** Low (usually code design issue)

**Root Cause:**

1. Infinite recursion
2. Recursive call without base case
3. Circular dependency
4. Recursive data structure

**Context Clues:**

- Stack trace shows same function calling itself
- Base case missing or unreachable
- Circular reference in data

**Proven Fixes:**

1. **Add base case** (success rate: 100%)

   ```python
   # ❌ Bad
   def traverse(node):
       process(node)
       traverse(node.next)  # No base case!

   # ✅ Good
   def traverse(node):
       if not node:
           return  # Base case
       process(node)
       traverse(node.next)
   ```

2. **Use iteration instead of recursion** (success rate: 100%)
   ```python
   # ✅ Better
   def traverse(node):
       while node:
           process(node)
           node = node.next
   ```

**Prevention:**

- [ ] Always include base case in recursive functions
- [ ] Test recursive functions with depth
- [ ] Prefer iteration over recursion
- [ ] Use linting to detect complex recursion

---

## ERROR-010: Budget Exceeded

**Error Message:**

```
BudgetExceededRoutingError: Estimated cost <x> exceeds max $<max_usd>
```

**Frequency:** Low

**Root Cause:**

1. Cost estimation for a task exceeds configured hard limit.
2. Token estimate overflow (>1,000,000 tokens) triggers safety cap.

**Context Clues:**

- `cost_optimizer_ranked` log absent; failure raised before routing attempts.
- Prometheus counter `arc_saga_tier_escalations_total` increments with reason `budget_limit` or `token_overflow` when enabled.

**Proven Fixes:**

1. Increase `SAGA_COST_MAX_USD` to match workload ceiling.
2. Reduce prompt size or token targets before routing.
3. Disable hard limits only in trusted environments (`SAGA_COST_ENFORCE_HARD_LIMITS=false`).

**Prevention:**

- Keep provider cost profiles updated.
- Set environment-specific budgets with alerts.
- Validate prompt size and expected tokens upstream.

**Similar Errors:**

- Provider rate-limit/quota errors (see ERROR-003 patterns)

---

## Structure for New Errors (Copy and Fill In)

```markdown
## ERROR-XXX: [Error Name]

**Error Message:**
```

[Paste exact error text]

````

**Frequency:** Low / Medium / High

**Root Cause:**
1. [Primary cause]
2. [Secondary cause]
3. [Tertiary cause]

**Context Clues:**
- [What to look for]
- [What indicates this]

**Proven Fixes:**
1. **Fix A** (success rate: X%)
   - What: [What to do]
   - Why: [Why it works]
   - When: [When to use]
   ```python
   [Example code]
````

2. **Fix B** (success rate: Y%)
   - What: [What to do]
   - Why: [Why it works]
   - When: [When to use]

**Prevention:**

- [ ] [Preventive measure 1]
- [ ] [Preventive measure 2]

**Seen in ARC SAGA:**

- [Date]: [Summary of incident]

**Related Errors:**

- ERROR-XXX: [Related error]

```

---

## Debugging Decision Tree

When you encounter an error:

```

1. CATEGORIZE THE ERROR
   ├─ Compilation Error (Python won't even start)
   ├─ Type Error (mypy caught it)
   ├─ Runtime Error (crashes during execution)
   ├─ Logic Error (runs but gives wrong result)
   └─ Performance Error (too slow)

2. READ THE ERROR MESSAGE
   ├─ What line of code?
   ├─ What function?
   ├─ What's the exact error?
   ├─ What's the stack trace?
   └─ What's the context?

3. CHECK ERROR CATALOG
   ├─ Does this error exist?
   ├─ What are proven fixes?
   ├─ What's the success rate?
   ├─ When to use each fix?
   └─ What's the prevention?

4. IF IN CATALOG: Apply fix
5. IF NOT IN CATALOG: Debug systematically

6. DEBUGGING SYSTEMATIC APPROACH
   ├─ Add logging at the failure point
   ├─ Log all relevant state
   ├─ Run with narrower inputs
   ├─ Check assumptions
   ├─ Verify external dependencies
   └─ Think backwards from error to root cause

7. FIND ROOT CAUSE
   ├─ This is the underlying issue (not the symptom)
   ├─ Might be different from error location
   ├─ Ask "why" 5 times to get to root

8. FIX ROOT CAUSE (not just symptom)

9. DOCUMENT IT
   ├─ Add to error_catalog.md
   ├─ Update decision_catalog.md if needed
   ├─ Add prevention steps
   └─ Create test to prevent regression

10. CREATE TEST
    ├─ Test reproduces the error
    ├─ Test verifies the fix
    ├─ Test prevents regression
    └─ Add to test suite

```

---

## Version History
- v1.0 (2024-12-01): Initial error catalog with common database and runtime errors
```
