"""
ARC Saga Memory API Server
Provides unified memory layer for Perplexity + Copilot
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import from Phase 1a (verified working)
from ..storage.sqlite import SQLiteStorage
from ..models import Message, MessageRole, Provider

# Import new services
from ..services.auto_tagger import AutoTagger
from ..services.file_processor import FileProcessor

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
            api_key=os.getenv("PPLX_API_KEY"),
            storage=storage
        )
    except Exception as e:
        print(f"Perplexity client disabled: {e}")

# Pydantic models for requests


class CaptureRequest(BaseModel):
    source: str  # "perplexity" or "copilot"
    role: str    # "user" or "assistant"
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
    lifespan=lifespan
)

# CORS for VSCode extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["vscode-webview://*", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# CONVERSATION CAPTURE
# ═══════════════════════════════════════════════════════════


@app.post("/capture")
async def capture_message(request: CaptureRequest):
    """Store a conversation message from any source"""

    # Map source string to Provider enum (verified from diagnostic)
    provider_map = {
        "perplexity": Provider.PERPLEXITY,
        "copilot": Provider.OPENAI,
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "google": Provider.GOOGLE,
        "antigravity": Provider.ANTIGRAVITY,
        "groq": Provider.GROQ,
        "test": Provider.OPENAI
    }

    provider = provider_map.get(request.source.lower(), Provider.OPENAI)

    # Map role string to MessageRole enum
    role = MessageRole.USER if request.role.lower() == "user" else MessageRole.ASSISTANT

    # Create Message object (matches verified Phase 1a model)
    message = Message(
        provider=provider,
        role=role,
        content=request.content,
        tags=[],  # Will be filled by auto_tagger
        session_id=request.thread_id,  # thread_id maps to session_id
        metadata=request.metadata or {}
    )

    # Store in database (MUST use await - verified async)
    try:
        message_id = await storage.save_message(message)

        # Auto-tag
        tags = auto_tagger.extract_tags(request.content)

        return {
            "message_id": message_id,
            "thread_id": request.thread_id or message.session_id,
            "tags": tags,
            "status": "stored"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

# ═══════════════════════════════════════════════════════════
# CONTEXT RETRIEVAL
# ═══════════════════════════════════════════════════════════


@app.get("/context/recent")
async def get_recent_context(
    sources: Optional[str] = None,
    limit: int = 10
):
    """Get recent conversation context across all sources"""

    try:
        # Use non-empty query to avoid FTS5 error
        results = await storage.search_messages(
            query="a",
            limit=limit * 2
        )

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
                "anthropic": "anthropic"
            }
            allowed_providers = [
                provider_map.get(
                    s.lower()) for s in source_list]
            messages = [
                m for m in messages if m.provider.value in allowed_providers]

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
                    "session_id": msg.session_id
                }
                for msg in messages
            ],
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str):
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
                    "metadata": msg.metadata
                }
                for msg in messages
            ],
            "count": len(messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {
                str(e)}")

# ═══════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════


@app.post("/search")
async def search_memory(request: SearchRequest):
    """Search across all conversations"""

    try:
        # Handle empty query for "recent" search
        query = request.query.strip() if request.query else "test"

        results = await storage.search_messages(
            query=query,
            tags=None,
            limit=request.limit
        )

        return {
            "query": request.query,
            "results": [
                {
                    "id": r.entity_id,
                    "content": r.content,
                    "tags": r.tags,
                    "timestamp": r.timestamp.isoformat(),
                    "score": r.relevance_score,
                    "snippet": r.content[:200] + "..." if len(r.content) > 200 else r.content
                }
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# ═══════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════


@app.post("/attach/file")
async def attach_file(
    thread_id: str,
    file: UploadFile = File(...)
):
    """Attach file to conversation thread"""

    try:
        # Save file and extract text
        file_id = str(uuid.uuid4())
        filepath, extracted_text = await file_processor.process_file(
            file_id=file_id,
            file=file
        )

        # Note: Phase 1a doesn't have file storage yet
        # This is a placeholder for Phase 1b

        return {
            "file_id": file_id,
            "filename": file.filename,
            "thread_id": thread_id,
            "extracted_text_length": len(extracted_text) if extracted_text else 0}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing error: {
                str(e)}")

# ═══════════════════════════════════════════════════════════
# PERPLEXITY INTEGRATION
# ═══════════════════════════════════════════════════════════


@app.post("/perplexity/ask")
async def ask_perplexity(request: PerplexityRequest):
    """Ask Perplexity with automatic context injection"""

    if not perplexity_client:
        raise HTTPException(
            status_code=503,
            detail="Perplexity integration not available (missing API key)"
        )

    # Get context if requested
    context_messages = []
    if request.inject_context and request.thread_id:
        messages = await storage.get_by_session(request.thread_id)
        context_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    # Stream response
    async def stream_response():
        async for chunk in perplexity_client.ask_streaming(
            query=request.query,
            context=context_messages,
            thread_id=request.thread_id
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

# ═══════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════


@app.get("/health")
async def health_check():
    """Server health check"""
    db_healthy = await storage.health_check()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": str(storage_path),
        "timestamp": datetime.now().isoformat()
    }


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
                    "providers": [msg.provider.value]
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

        return {
            "threads": thread_list[:limit],
            "total": len(threads)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Thread listing error: {
                str(e)}")

# Server startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8421)
