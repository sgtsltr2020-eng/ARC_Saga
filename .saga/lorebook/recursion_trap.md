# LoreBook Entry: The Recursion Trap

## Incident

During the implementation of SAGA Tier 2.5 (Memory Compaction), integration tests utilizing a forced "Memory Pressure" loop (15+ iterations) systematically failed with `GraphRecursionError`.

## Diagnostics

- **Symptom 1**: `Warden.solve_request` returned an empty history/state, appearing as a silent failure.
- **Symptom 2**: Debug logs revealed "Recursion limit of 50 reached" deep in the stack trace.
- **Root Cause**: The default LangGraph recursion limit (often 25 or 50) is insufficient for long-running agent loops designed to stress-test memory compaction. A 15-iteration loop with a 4-node cycle consumes 60 steps, exceeding the default.

## Solution Pattern

1. **Headroom Calculation**: Set `recursion_limit` based on the formula:
   `Limit >= (Max_Iterations * Nodes_Per_Cycle) + Safety_Buffer`
   For SAGA Tier 2.5: `150` was chosen to accommodate deep reasoning chains.

2. **Graceful Recovery**: Implement a `try/except` block around `graph.invoke` to catch recursion errors.
   **CRITICAL**: In the `except` block, attempt to retrieve the _last valid state snapshot_ using `graph.get_state(config)`. This preserves the partial history for debugging ("Red Line" analysis) rather than losing all context.

## Code Example

```python
# warden.py
try:
    final_state = await graph.invoke(state, config={"recursion_limit": 150})
except Exception as e:
    logger.error(f"Graph crashed: {e}")
    # FALLBACK: Retrieve what we have so far
    snapshot = await graph.get_state(config)
    history = snapshot.values.get("history", [])
```

## Tags

Topography: Memory, LangGraph, Debugging, Stability
date: 2025-12-24
