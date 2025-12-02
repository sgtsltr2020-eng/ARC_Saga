# Decision Catalog

# Comprehensive Reference for All Architectural Decisions

# Use this to make smarter decisions faster

---

## 🎯 HOW TO USE THIS DOCUMENT

1. **When you encounter a design decision**, search this catalog
2. **If it exists**, reference the proven options + tradeoffs
3. **If it doesn't exist**, think through the framework, then document it
4. **Reference in prompts**: "@codebase This is similar to [Decision Name] in decision_catalog.md"

---

## DATABASE DECISIONS

### Decision: How to Handle Database Connection Failures

**Problem:** Database connections fail transiently (network hiccup) or persistently (server down). Must not crash system.

**Options Evaluated:**

**Option 1: Immediate Fail**

- How: Raise exception immediately on connection error
- Pros: Simple, forces client to handle
- Cons: No resilience, cascades failures
- Success rate: 10%
- When to use: Never in production

**Option 2: Retry Forever**

- How: Loop until connection succeeds
- Pros: Eventually succeeds if transient
- Cons: Hangs indefinitely if persistent, resource leak
- Success rate: 20%
- When to use: Never (better options exist)

**Option 3: Retry with Exponential Backoff (CHOSEN for ARC SAGA)**

- How: Retry up to N times with delay = min(base \* 2^attempt, max_delay) + jitter
- Pros: Handles transient + persistent, prevents thundering herd, resource efficient
- Cons: User experiences delay on persistent failures
- Success rate: 85-90%
- When to use: Always for external services

**Option 4: Retry with Circuit Breaker (BEST for Production)**

- How: Use exponential backoff + circuit breaker + fallback + health checks
- Pros: Handles transient, detects persistent, prevents cascade, includes recovery
- Cons: More complex implementation
- Success rate: 95%+
- When to use: Always for critical services

**Chosen Solution:** Option 4 (Retry + Circuit Breaker)

**Implementation:**

```python
async def connect_to_database():
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=asyncpg.PoolAcquireTimeoutError
    )

    try:
        return await circuit_breaker.call(
            asyncpg.create_pool,
            DATABASE_URL,
            timeout=5
        )
    except CircuitBreakerOpen:
        # Graceful degradation
        return await connect_to_read_replica()
    except Exception as e:
        log_error("database_connection_failed", error=e)
        raise
```

**Tradeoffs:**

- Accept: User sees delay on persistent failures (better than cascade)
- Accept: Must implement fallback strategy (worth it)
- Get: Resilient system that handles common failures

**Related Decisions:**

- [How to Handle Circuit Breaker Open] - What to do when all retries exhausted
- [How to Implement Fallback Strategy] - What to do when primary fails

**Failure Modes:**

1. Timeout on connection attempt

   - Mitigation: Set explicit timeout (5000ms)
   - Detection: Log timeout errors
   - Recovery: Circuit breaker opens, fallback used

2. Persistent database down

   - Mitigation: Health checks
   - Detection: Circuit breaker opens
   - Recovery: Switch to read replica or cached data

3. Network partition
   - Mitigation: Exponential backoff prevents thundering herd
   - Detection: Circuit breaker detects
   - Recovery: Wait for network recovery + health checks

---

### Decision: How to Query for Performance

**Problem:** Queries can be slow (high latency), especially with large datasets or complex joins.

**Options Evaluated:**

**Option 1: Retrieve All, Filter in Application**

- How: Query all rows, filter in Python
- Pros: Flexible, easy to implement
- Cons: High memory, slow network, defeats database indexing
- Success rate: 30% (fails at scale)
- When to use: Never (for large datasets)

**Option 2: Parameterized Query with WHERE Clause (CHOSEN)**

- How: Build query with WHERE clause based on criteria, use parameterized queries
- Pros: Database handles filtering, indexes used, efficient
- Cons: Need to know query structure ahead of time
- Success rate: 90%
- When to use: Always for structured queries

**Option 3: Full-Text Search Index**

- How: Create full-text search index, use search API
- Pros: Fast text search, fuzzy matching
- Cons: Limited to text search, requires index maintenance
- Success rate: 95% for text search
- When to use: When searching text fields

**Option 4: Vector Search (for semantic search)**

- How: Use vector embeddings, search in vector space
- Pros: Semantic understanding, powerful
- Cons: Requires embedding model, external service
- Success rate: 85-90% (depends on embeddings)
- When to use: Semantic search (finding "similar" conversations)

**Chosen Solutions:**

- Use Option 2 for structure queries (ID lookups, date ranges, status filters)
- Use Option 3 for full-text search
- Use Option 4 for semantic search

**Implementation:**

```python
# Option 2: Parameterized query
query = """
SELECT * FROM conversations
WHERE provider = :provider
  AND created_at > :cutoff_date
ORDER BY created_at DESC
LIMIT :limit
"""
results = await session.execute(query, {
    "provider": "perplexity",
    "cutoff_date": cutoff_date,
    "limit": 100
})

# Option 3: Full-text search
query = """
SELECT * FROM conversations
WHERE to_tsvector('english', content) @@ plainto_tsquery(:search_term)
"""
results = await session.execute(query, {"search_term": "bug fix"})

# Option 4: Vector search
embeddings = await embeddings_service.embed("bug fix")
results = await qdrant_client.search(
    collection_name="conversations",
    query_vector=embeddings,
    limit=10
)
```

**Testing Strategy:**

- Unit: Query generates correct SQL
- Integration: Query returns correct results
- Performance: Query completes < 100ms (p95) with 100k rows

**Performance Benchmarks:**

- Parameterized query: < 50ms (p95) with index
- Full-text search: < 100ms (p95)
- Vector search: < 200ms (p95) with Qdrant

---

## ERROR HANDLING DECISIONS

### Decision: When to Retry vs When to Fail

**Problem:** Operation failed. Do we retry or fail immediately?

**Retry if (Transient Errors):**

- Network timeout (temporary network issue)
- Connection refused (service temporarily down)
- Rate limit (temporary resource exhaustion)
- Deadlock (concurrent access issue)
- Retriable HTTP 5xx (temporary server error)

**Fail Immediately if (Permanent Errors):**

- Invalid input (will keep failing)
- Authentication failure (credentials wrong)
- Authorization failure (permission denied)
- Resource not found (won't reappear)
- Permanent HTTP 4xx (client error)
- Constraint violation (data structure violated)

**Implementation:**

```python
RETRIABLE_ERRORS = (
    asyncio.TimeoutError,
    ConnectionError,
    asyncpg.PoolAcquireTimeoutError,
    RateLimitError,
)

PERMANENT_ERRORS = (
    ValueError,
    InvalidInput,
    NotFoundError,
    AuthenticationError,
    PermissionDenied,
)

async def call_with_intelligent_retry(func, *args, **kwargs):
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except RETRIABLE_ERRORS as e:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(2 ** attempt, 60))
        except PERMANENT_ERRORS:
            raise  # Fail immediately, no retry
```

---

## LOGGING DECISIONS

### Decision: How Much to Log

**Problem:** Too little logging = can't debug. Too much logging = noise + performance cost + storage cost.

**Levels:**

**DEBUG (Verbose, only in development):**

- Variable values at each step
- Function entry/exit
- Loop iterations
- Conditional branches taken

**INFO (Normal operations):**

- Operation start/end
- Significant decisions
- State changes
- Performance metrics

**WARNING (Unusual but handled):**

- Retry attempts
- Fallback activation
- Threshold exceeded
- Deprecated API used

**ERROR (Failures):**

- Operation failed
- Exception raised
- Recovery needed
- Actionable problem

**CRITICAL (System broken):**

- Service unavailable
- Data loss
- Security breach
- Must-fix-now issues

**Chosen for ARC SAGA:**

- Log at INFO level for all major operations
- Log at ERROR level for all failures
- Include context: request_id, user_id, operation, duration
- Never log: passwords, API keys, tokens, secrets

---

## TESTING DECISIONS

### Decision: How Much Test Coverage is Enough

**Problem:** 0% coverage = untested code. 100% coverage = slow development. Need balance.

**Options:**

**Option 1: No Testing (Not acceptable)**

- Coverage: 0%
- When to use: Never

**Option 2: Manual Testing Only (Not acceptable)**

- Coverage: Unknown, usually 20-30%
- When to use: Never for critical code

**Option 3: 70% Coverage**

- Coverage: 70%
- Pros: Faster development
- Cons: Many edge cases untested
- Success rate: 60% (bugs in uncovered code)
- When to use: Non-critical features

**Option 4: 90% Coverage (Good)**

- Coverage: 90%
- Pros: Good balance of speed and confidence
- Cons: Some edge cases missed
- Success rate: 85%
- When to use: Most production code

**Option 5: 95%+ Coverage (CHOSEN for ARC SAGA)**

- Coverage: 95%+
- Pros: High confidence, catches most bugs
- Cons: Takes time
- Success rate: 95%+
- When to use: All critical code

**Chosen: 95%+ for ARC SAGA**

**What to test:**

- Happy path (normal operation)
- Error paths (each exception)
- Edge cases (empty, null, max, min)
- Boundaries (off-by-one, just-over)
- Concurrency (multiple threads/tasks)
- Performance (latency targets)

---

## ARCHITECTURE DECISIONS

### Decision: Monolith vs Microservices

**Problem:** ARC SAGA is starting small but will grow. How to structure?

**Option 1: Monolith (CHOSEN for Phase 1)**

- How: All code in one application
- Pros: Simple to build, easy to test, fast communication
- Cons: Harder to scale later, coupled components
- Success rate: 90% (at small scale)
- When to use: < 50k lines of code

**Option 2: Microservices**

- How: Separate services for each bounded context
- Pros: Independent scaling, loose coupling
- Cons: Complex deployment, network latency, debugging harder
- Success rate: 70% (complexity costs more than it saves at small scale)
- When to use: > 200k lines of code with clear boundaries

**Chosen for ARC SAGA (Phase 1): Monolith with Module Boundaries**

- Start as monolith (easier to build)
- Organize into bounded contexts (can evolve to microservices)
- Use event bus internally (simulate services)
- Can extract to microservices when needed

**Future Consideration:** Migrate to microservices when any bounded context needs independent scaling.

---

## CACHING DECISIONS

### Decision: What to Cache and How

**Problem:** Without caching, system is slow. With too much caching, data gets stale.

**Options:**

**Option 1: No Caching (Not acceptable for production)**

- Latency: High (every request hits database)
- Success rate: 30%
- When to use: Never

**Option 2: Cache Everything**

- Latency: Low
- Staleness: High (data wrong)
- Consistency: Bad (cache invalidation nightmare)
- Success rate: 20% (too many consistency bugs)
- When to use: Never

**Option 3: Cache Reads Only (CHOSEN)**

- What: Cache frequently read data (conversations, user profiles)
- How: TTL-based invalidation (30 min default)
- Why: Reads dominate, stale data acceptable for reads
- Success rate: 85%
- When to use: Read-heavy operations

**Option 4: Cache Reads + Event-Based Invalidation**

- What: Cache reads, invalidate when writes occur
- How: When write happens, invalidate related cache
- Why: Immediate consistency + performance
- Success rate: 90%
- When to use: Critical data where stale is not acceptable

**Chosen for ARC SAGA:**

- Cache conversation metadata (TTL 30min)
- Cache search results (TTL 5min, invalidate on write)
- Don't cache: User credentials, settings (too risky)
- Invalidation strategy: Event-based for writes

```python
class ConversationService:
    async def get_conversation(self, conversation_id: str):
        # Try cache first
        cached = await cache.get(f"conversation:{conversation_id}")
        if cached:
            return cached

        # Load from DB
        conversation = await db.get_conversation(conversation_id)

        # Cache for 30 minutes
        await cache.set(f"conversation:{conversation_id}", conversation, ttl=1800)

        return conversation

    async def update_conversation(self, conversation_id: str, updates: dict):
        # Update DB
        updated = await db.update_conversation(conversation_id, updates)

        # Invalidate cache immediately
        await cache.delete(f"conversation:{conversation_id}")

        # Publish event for other services
        await event_bus.publish(ConversationUpdatedEvent(conversation_id))

        return updated
```

---

## SECURITY DECISIONS

### Decision: How to Authenticate Users

**Problem:** Need to verify user identity without storing passwords.

**Options:**

**Option 1: Store Password Hash**

- How: Hash password with bcrypt, store hash, verify on login
- Pros: Standard, well-understood
- Cons: If DB breached, hashes are target
- Success rate: 90%
- When to use: Traditional web apps

**Option 2: OAuth / External Provider (CHOSEN for ARC SAGA Phase 2)**

- How: Delegate to Google/GitHub/Microsoft
- Pros: Don't store passwords, leverages their security, familiar to users
- Cons: Dependency on external service
- Success rate: 95%
- When to use: Web applications

**Option 3: JWT Tokens (Phase 1)**

- How: Generate signed token on login, verify token on requests
- Pros: Stateless, scalable
- Cons: Token revocation complex, expiration management needed
- Success rate: 85%
- When to use: APIs, microservices

**Current for ARC SAGA (Phase 1): JWT Tokens from OAuth**

- Use OAuth to get initial token (delegate auth)
- Generate JWT for session
- Verify JWT on requests
- Refresh token before expiration

---

## PERFORMANCE DECISIONS

### Decision: How to Optimize Slow Queries

**Problem:** Query takes 5 seconds, should be < 100ms.

**Diagnosis:**

1. Check if index exists on WHERE columns
2. Check if query does N+1 (joins missing)
3. Check if query returns too many rows
4. Check if complex JOIN logic

**Solutions in Order:**

1. **Add index** (if index missing) - Usually fixes (80%)
2. **Add JOIN** (if fetching related data in loop) - Usually fixes (15%)
3. **Add pagination** (if returning too many rows) - Sometimes fixes (5%)
4. **Optimize logic** (if query is complex) - Rarely needed (< 1%)

**Implementation:**

```sql
-- Create index on frequently searched columns
CREATE INDEX idx_conversations_provider_date
ON conversations(provider, created_at DESC);

-- Verify query uses index
EXPLAIN ANALYZE
SELECT * FROM conversations
WHERE provider = 'perplexity' AND created_at > now() - interval '7 days'
LIMIT 100;
```

---

## DEPLOYMENT DECISIONS

### Decision: When to Deploy

**Never deploy if:**

- [ ] Any test failing
- [ ] Coverage below 95%
- [ ] Linting score below 8.0
- [ ] Security scan shows issues
- [ ] Type checking shows errors
- [ ] Code review found blockers
- [ ] Performance benchmarks not met

**Always deploy if:**

- [ ] All tests passing
- [ ] Coverage >= 95%
- [ ] Linting score >= 8.0
- [ ] Security scan 0 issues
- [ ] Type checking clean
- [ ] Code review approved
- [ ] Performance benchmarks met

**Process:**

1. Push to develop branch
2. GitHub Actions runs all checks
3. If all pass: auto-commit + auto-push
4. Create PR to main
5. Code review
6. Merge to main (auto-deploy)

---

## Version History

- v1.0 (2024-12-01): Initial decision catalog for ARC SAGA
