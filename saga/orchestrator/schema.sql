-- Orchestrator Events Schema
-- Event sourcing table for ARC SAGA orchestrator
-- Supports append-only event storage with efficient querying

CREATE TABLE IF NOT EXISTS orchestrator_events (
    id TEXT PRIMARY KEY,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,  -- JSON serialized event payload
    created_at TEXT NOT NULL,  -- ISO8601 timestamp
    correlation_id TEXT,       -- Links related events across operations
    sequence_number INTEGER,   -- Per-aggregate ordering
    
    -- Metadata for debugging and auditing
    source TEXT,               -- Component that emitted the event
    version INTEGER DEFAULT 1  -- Event schema version for migrations
);

-- Index for querying events by aggregate (e.g., workflow_id)
CREATE INDEX IF NOT EXISTS idx_events_aggregate_created 
    ON orchestrator_events(aggregate_id, created_at);

-- Index for querying events since a timestamp (projections/replay)
CREATE INDEX IF NOT EXISTS idx_events_created_at 
    ON orchestrator_events(created_at);

-- Index for tracing related events across operations
CREATE INDEX IF NOT EXISTS idx_events_correlation_id 
    ON orchestrator_events(correlation_id);

-- Index for filtering by event type (e.g., all WorkflowCompleted events)
CREATE INDEX IF NOT EXISTS idx_events_type 
    ON orchestrator_events(event_type);








