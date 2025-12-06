# time-out-prevention

## Purpose

Prevent timeouts during long-running operations by outputting periodic status updates.

## Behavior

If an operation takes longer than 60 seconds, output a status message every 30-60 seconds to confirm activity and prevent connection timeouts.

## Status Message Format

```
Still working on [task name], everything is fine. Progress: [current step/description]
```

## Examples

- "Still working on implementing CopilotReasoningEngine, everything is fine. Progress: Writing error handling for HTTP 429 responses"
- "Still working on test suite generation, everything is fine. Progress: Creating test_entra_id_auth_manager.py (14/25 test cases complete)"
- "Still working on quality gate verification, everything is fine. Progress: Running mypy type checking (2/5 checks complete)"

## Implementation Notes

- Output status messages proactively, not reactively
- Include specific progress indicators when possible
- Use friendly, reassuring tone
- Don't spam - wait at least 30 seconds between messages
- If operation completes quickly (<60s), no status message needed
