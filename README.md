# ARC Saga Memory Layer

Enterprise-grade persistent memory for AI conversations.

## Phase 1a: Foundation Complete ✅

- ✅ Shared configuration module
- ✅ Core data models with Pydantic validation
- ✅ SQLite storage backend with full-text search
- ✅ Structured logging (JSON)
- ✅ Custom exceptions
- ✅ Unit tests (80%+ coverage)

## Next Steps

- Phase 1b: Monitoring services
- Phase 1c: Validator integration

## Usage

```python
from arc_saga.arc_saga.models import Message, Provider, MessageRole
from arc_saga.arc_saga.storage import SQLiteStorage

# Initialize storage
storage = SQLiteStorage()
await storage.initialize()

# Save a message
message = Message(
    provider=Provider.PERPLEXITY,
    role=MessageRole.USER,
    content="How do I implement rate limiting?"
)
await storage.save_message(message)

# Search messages
results = await storage.search_messages("rate limiting")
```
