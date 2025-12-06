"""
ARC Saga Memory API Server
Provides unified memory layer for Perplexity + Copilot
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..observability import ObservabilityMiddleware, init_observability
from ..validators import (
    validate_capture_request,
    validate_perplexity_request,
    validate_search_request,
)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore
except Exception:  # pragma: no cover - optional dep guard
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    generate_latest = None  # type: ignore[assignment]
try:
    from slowapi import Limiter  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
    from slowapi.middleware import SlowAPIMiddleware  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
except Exception:  # pragma: no cover - optional dep guard
    Limiter = None  # type: ignore
    RateLimitExceeded = None  # type: ignore
    SlowAPIMiddleware = None  # type: ignore
    get_remote_address = None  # type: ignore

# Import from Phase 1a (verified working)
from ..models import Message, MessageRole, Provider

# Import new services
from ..services.auto_tagger import AutoTagger
from ..services.file_processor import FileProcessor
from ..storage.sqlite import SQLiteStorage

# Import health monitoring
from .health_monitor import (
    LatencyTrackingMiddleware,
    check_database_health,
    check_storage_space,
    determine_health_status,
    health_monitor,
)

# Initialize services
storage_path = Path.home() / ".arc_saga" / "memory.db"
storage_path.parent.mkdir(parents=True, exist_ok=True)

storage = SQLiteStorage(str(storage_path))
auto_tagger = AutoTagger()
file_processor = FileProcessor(storage_path.parent / "files")

# Perplexity client (optional - only if API key provided)
perplexity_client = None
if os.getenv("PPLX_API_KEY"):
    try:
        from ..integrations.perplexity_client import PerplexityClient

        perplexity_client = PerplexityClient(
            api_key=os.getenv("PPLX_API_KEY"), storage=storage
        )
        # Register circuit breaker with health monitor
        health_monitor.register_circuit_breaker(
            "perplexity", perplexity_client.circuit_breaker
        )
    except Exception as e:
        print(f"Perplexity client disabled: {e}")

# Pydantic models for requests


class CaptureRequest(BaseModel):
    source: str  # "perplexity" or "copilot"
    role: str  # "user" or "assistant"
    content: str
    thread_id: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    search_type: str = "keyword"
    sources: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 20


class PerplexityRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    inject_context: bool = True


# ═══════════════════════════════════════════════════════════
# LIFESPAN CONTEXT MANAGER - INITIALIZE DATABASE
# ═══════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    try:
        await storage.initialize()
        print(f"✅ Database initialized at {storage_path}")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")

    yield

    # Shutdown (if needed in future)


# Initialize FastAPI with lifespan context manager
app = FastAPI(
    title="ARC Saga Memory API",
    description="Unified conversation memory for AI assistants",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter (gracefully disabled if slowapi not installed)
limiter = (
    Limiter(key_func=get_remote_address) if Limiter and get_remote_address else None
)
if limiter and SlowAPIMiddleware:
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
# CORS for VSCode extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["vscode-webview://*", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add latency tracking middleware
app.add_middleware(LatencyTrackingMiddleware)
# Add observability middleware (metrics + OTEL spans, no-op if deps missing)
app.add_middleware(ObservabilityMiddleware)
init_observability(app)

if RateLimitExceeded:

    @app.exception_handler(RateLimitExceeded)  # type: ignore[arg-type]
    async def rate_limit_handler(request, exc):  # type: ignore[no-untyped-def]
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry shortly."},
        )


# ═══════════════════════════════════════════════════════════
# CONVERSATION CAPTURE
# ═══════════════════════════════════════════════════════════


@app.post("/capture")
async def capture_message(request: Request, capture_req: CaptureRequest):
    """Store a conversation message from any source"""

    validate_capture_request(
        source=capture_req.source,
        role=capture_req.role,
        content=capture_req.content,
        metadata_keys=(capture_req.metadata or {}).keys(),
    )

    # Map source string to Provider enum (verified from diagnostic)
    provider_map = {
        "perplexity": Provider.PERPLEXITY,
        "copilot": Provider.OPENAI,
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "google": Provider.GOOGLE,
        "antigravity": Provider.ANTIGRAVITY,
        "groq": Provider.GROQ,
        "test": Provider.OPENAI,
    }

    provider = provider_map.get(capture_req.source.lower(), Provider.OPENAI)

    # Map role string to MessageRole enum
    role = (
        MessageRole.USER
        if capture_req.role.lower() == "user"
        else MessageRole.ASSISTANT
    )

    # Create Message object (matches verified Phase 1a model)
    message = Message(
        provider=provider,
        role=role,
        content=capture_req.content,
        tags=[],  # Will be filled by auto_tagger
        session_id=capture_req.thread_id,  # thread_id maps to session_id
        metadata=capture_req.metadata or {},
    )

    # Store in database (MUST use await - verified async)
    try:
        message_id = await storage.save_message(message)

        # Auto-tag
        tags = auto_tagger.extract_tags(capture_req.content)

        return {
            "message_id": message_id,
            "thread_id": capture_req.thread_id or message.session_id,
            "tags": tags,
            "status": "stored",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")


# ═══════════════════════════════════════════════════════════
# CONTEXT RETRIEVAL
# ═══════════════════════════════════════════════════════════


@app.get("/context/recent")
async def get_recent_context(
    request: Request, sources: Optional[str] = None, limit: int = 10
):
    """Get recent conversation context across all sources"""

    try:
        # Use non-empty query to avoid FTS5 error
        results = await storage.search_messages(query="a", limit=limit * 2)

        # Get full messages
        messages = []
        for r in results:
            msg = await storage.get_message_by_id(r.entity_id)
            if msg:
                messages.append(msg)

        # Filter by source if requested
        if sources:
            source_list = sources.split(",")
            provider_map = {
                "perplexity": "perplexity",
                "copilot": "openai",
                "openai": "openai",
                "anthropic": "anthropic",
            }
            allowed_providers = [provider_map.get(s.lower()) for s in source_list]
            messages = [m for m in messages if m.provider.value in allowed_providers]

        # Take only requested limit
        messages = messages[:limit]

        return {
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "provider": msg.provider.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "session_id": msg.session_id,
                }
                for msg in messages
            ],
            "count": len(messages),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/thread/{thread_id}")
async def get_thread(request: Request, thread_id: str):
    """Get complete thread history"""

    try:
        # Use get_by_session (verified method name)
        messages = await storage.get_by_session(thread_id)

        if not messages:
            raise HTTPException(status_code=404, detail="Thread not found")

        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "provider": msg.provider.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "tags": msg.tags,
                    "metadata": msg.metadata,
                }
                for msg in messages
            ],
            "count": len(messages),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


# ═══════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════


@app.post("/search")
async def search_memory(request: Request, search_req: SearchRequest):
    """Search across all conversations"""

    validate_search_request(query=search_req.query or "", limit=search_req.limit)

    try:
        # Handle empty query for "recent" search
        query = search_req.query.strip() if search_req.query else "test"

        results = await storage.search_messages(
            query=query, tags=None, limit=search_req.limit
        )

        return {
            "query": search_req.query,
            "results": [
                {
                    "id": r.entity_id,
                    "content": r.content,
                    "tags": r.tags,
                    "timestamp": r.timestamp.isoformat(),
                    "score": r.relevance_score,
                    "snippet": r.content[:200] + "..."
                    if len(r.content) > 200
                    else r.content,
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ═══════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════


@app.post("/attach/file")
async def attach_file(thread_id: str, file: UploadFile = File(...)):
    """Attach file to conversation thread"""

    try:
        # Save file and extract text
        file_id = str(uuid.uuid4())
        filepath, extracted_text = await file_processor.process_file(
            file_id=file_id, file=file
        )

        # Note: Phase 1a doesn't have file storage yet
        # This is a placeholder for Phase 1b

        return {
            "file_id": file_id,
            "filename": file.filename,
            "thread_id": thread_id,
            "extracted_text_length": len(extracted_text) if extracted_text else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


# ═══════════════════════════════════════════════════════════
# PERPLEXITY INTEGRATION
# ═══════════════════════════════════════════════════════════


@app.post("/perplexity/ask")
async def ask_perplexity(request: Request, perplexity_req: PerplexityRequest):
    """Ask Perplexity with automatic context injection"""

    validate_perplexity_request(query=perplexity_req.query)

    if not perplexity_client:
        raise HTTPException(
            status_code=503,
            detail="Perplexity integration not available (missing API key)",
        )

    # Get context if requested
    context_messages = []
    if perplexity_req.inject_context and perplexity_req.thread_id:
        messages = await storage.get_by_session(perplexity_req.thread_id)
        context_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

    # Stream response
    async def stream_response():
        async for chunk in perplexity_client.ask_streaming(
            query=perplexity_req.query,
            context=context_messages,
            thread_id=perplexity_req.thread_id,
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# HEALTH CHECK ENDPOINTS
# ═══════════════════════════════════════════════════════════


# Pydantic models for health responses
class HealthStatusResponse(BaseModel):
    """Basic health status response."""

    status: str  # "healthy", "degraded", or "unhealthy"
    timestamp: str


class DatabaseHealth(BaseModel):
    """Database health information."""

    healthy: bool
    latency_ms: float
    reachable: bool
    error: Optional[str] = None


class StorageSpace(BaseModel):
    """Storage space information."""

    available_mb: float
    total_mb: float
    used_mb: float
    usage_percent: float
    error: Optional[str] = None


class CircuitBreakerInfo(BaseModel):
    """Circuit breaker information."""

    state: str
    metrics: dict[str, Any]


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""

    status: str
    timestamp: str
    database: DatabaseHealth
    storage_space: StorageSpace
    circuit_breakers: dict[str, CircuitBreakerInfo]
    endpoint_latencies: dict[str, dict[str, float]]
    last_errors: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    """Metrics response."""

    circuit_breakers: dict[str, dict[str, Any]]
    endpoint_latencies: dict[str, dict[str, float]]
    recent_errors: list[dict[str, Any]]
    timestamp: str


@app.get("/health", response_model=HealthStatusResponse)
async def health_check() -> HealthStatusResponse:
    """
    Basic health check endpoint.

    Returns:
        HealthStatusResponse with status and timestamp
    """
    try:
        # Check database health
        db_health = await check_database_health(storage)

        # Check storage space
        storage_space_info = check_storage_space(storage_path)

        # Get circuit breaker states
        circuit_breakers = health_monitor.get_circuit_breaker_states()

        # Get endpoint latencies
        endpoint_latencies = health_monitor.get_endpoint_latencies()

        # Determine overall status
        status = determine_health_status(
            db_health=db_health,
            storage_space=storage_space_info,
            circuit_breakers=circuit_breakers,
            endpoint_latencies=endpoint_latencies,
        )

        return HealthStatusResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        # If health check itself fails, system is unhealthy
        health_monitor.record_error(
            operation="health_check",
            error=e,
            context={},
        )

        return HealthStatusResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@app.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """
    Detailed health check with full system diagnostics.

    Returns:
        DetailedHealthResponse with complete system health information
    """
    try:
        # Check database health
        db_health = await check_database_health(storage)

        # Check storage space
        storage_space_info = check_storage_space(storage_path)

        # Get circuit breaker states
        circuit_breakers = health_monitor.get_circuit_breaker_states()

        # Get endpoint latencies
        endpoint_latencies = health_monitor.get_endpoint_latencies()

        # Get recent errors
        recent_errors = health_monitor.get_recent_errors()

        # Determine overall status
        status = determine_health_status(
            db_health=db_health,
            storage_space=storage_space_info,
            circuit_breakers=circuit_breakers,
            endpoint_latencies=endpoint_latencies,
        )

        # Convert circuit breakers to response format
        circuit_breaker_info: dict[str, CircuitBreakerInfo] = {}
        for service_name, breaker_data in circuit_breakers.items():
            circuit_breaker_info[service_name] = CircuitBreakerInfo(
                state=breaker_data["state"],
                metrics=breaker_data["metrics"],
            )

        return DetailedHealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            database=DatabaseHealth(**db_health),
            storage_space=StorageSpace(**storage_space_info),
            circuit_breakers=circuit_breaker_info,
            endpoint_latencies=endpoint_latencies,
            last_errors=recent_errors,
        )
    except Exception as e:
        # If detailed health check fails, record error and return minimal info
        health_monitor.record_error(
            operation="detailed_health_check",
            error=e,
            context={},
        )

        # Return unhealthy status with minimal information
        return DetailedHealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            database=DatabaseHealth(
                healthy=False,
                latency_ms=0.0,
                reachable=False,
                error=str(e),
            ),
            storage_space=StorageSpace(
                available_mb=0.0,
                total_mb=0.0,
                used_mb=0.0,
                usage_percent=0.0,
                error=str(e),
            ),
            circuit_breakers={},
            endpoint_latencies={},
            last_errors=health_monitor.get_recent_errors(),
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Get performance and error metrics.

    Returns cached metrics if available (10s TTL) to avoid performance impact.

    Returns:
        MetricsResponse with circuit breaker metrics, endpoint latencies, and errors
    """
    try:
        # Check cache first
        cached_metrics = health_monitor.get_cached_metrics()

        if cached_metrics is not None:
            return MetricsResponse(**cached_metrics)

        # Build metrics
        circuit_breakers: dict[str, dict[str, Any]] = {}
        for (
            service_name,
            breaker_data,
        ) in health_monitor.get_circuit_breaker_states().items():
            circuit_breakers[service_name] = breaker_data["metrics"]

        endpoint_latencies = health_monitor.get_endpoint_latencies()
        recent_errors = health_monitor.get_recent_errors()

        metrics = MetricsResponse(
            circuit_breakers=circuit_breakers,
            endpoint_latencies=endpoint_latencies,
            recent_errors=recent_errors,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Cache metrics
        health_monitor.set_cached_metrics(metrics.model_dump())

        return metrics
    except Exception as e:
        health_monitor.record_error(
            operation="get_metrics",
            error=e,
            context={},
        )

        # Return empty metrics on error
        return MetricsResponse(
            circuit_breakers={},
            endpoint_latencies={},
            recent_errors=health_monitor.get_recent_errors(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@app.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    """
    Expose Prometheus-compatible metrics stream.

    Falls back gracefully if prometheus_client is not installed.
    """
    if generate_latest is None:
        return Response(
            content="prometheus_client not installed",
            media_type="text/plain",
            status_code=503,
        )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/threads")
async def list_threads(limit: int = 50):
    """List all conversation threads"""

    try:
        # Use a non-empty query to avoid FTS5 syntax error
        results = await storage.search_messages(query="a", limit=500)

        threads = {}
        for r in results:
            # Get message to extract session_id
            msg = await storage.get_message_by_id(r.entity_id)
            if not msg:
                continue

            sid = msg.session_id or "default"
            if sid not in threads:
                threads[sid] = {
                    "thread_id": sid,
                    "first_message": msg.content[:100],
                    "last_updated": msg.timestamp,
                    "message_count": 1,
                    "providers": [msg.provider.value],
                }
            else:
                threads[sid]["message_count"] += 1
                if msg.timestamp > threads[sid]["last_updated"]:
                    threads[sid]["last_updated"] = msg.timestamp
                if msg.provider.value not in threads[sid]["providers"]:
                    threads[sid]["providers"].append(msg.provider.value)

        # Convert to list and sort by last_updated
        thread_list = list(threads.values())
        thread_list.sort(key=lambda x: x["last_updated"], reverse=True)

        # Convert timestamps to ISO format
        for thread in thread_list:
            thread["last_updated"] = thread["last_updated"].isoformat()

        return {"threads": thread_list[:limit], "total": len(threads)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread listing error: {str(e)}")


if limiter:
    capture_message = limiter.limit("30/minute")(capture_message)  # type: ignore[assignment]
    search_memory = limiter.limit("60/minute")(search_memory)  # type: ignore[assignment]
    get_recent_context = limiter.limit("60/minute")(get_recent_context)  # type: ignore[assignment]
    get_thread = limiter.limit("60/minute")(get_thread)  # type: ignore[assignment]
    ask_perplexity = limiter.limit("20/minute")(ask_perplexity)  # type: ignore[assignment]

# Server startup
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8421)
