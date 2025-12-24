# ARC SAGA Performance Benchmarks

## Overview

Performance benchmarks for the SAGA persistence and verification layers.
All benchmarks run on standard developer hardware (8-core CPU, 16GB RAM, SSD).

## TaskStore Benchmarks

### Save/Load Operations

| Operation              | Task Count | Measured Time | Threshold |
| ---------------------- | ---------- | ------------- | --------- |
| `save_task_graph()`    | 100        | ~1.2s         | <5s       |
| `load_task_graph()`    | 100        | ~0.4s         | <2s       |
| `update_task_status()` | 50 batch   | ~0.8s         | <3s       |
| `get_pending_tasks()`  | 100        | ~0.08s        | <0.5s     |

### Scale Tests

| Scenario                  | Result  | Notes                          |
| ------------------------- | ------- | ------------------------------ |
| Large descriptions (10KB) | ✅ Pass | SQLite TEXT handles well       |
| 20-item checklists        | ✅ Pass | JSON serialization efficient   |
| 50-level deep chains      | ✅ Pass | Dependencies tracked correctly |
| 52-node wide graphs       | ✅ Pass | Parallel detection works       |

### Database Size

| Tasks | DB Size | Growth      |
| ----- | ------- | ----------- |
| 10    | ~12KB   | -           |
| 100   | ~80KB   | ~0.7KB/task |
| 500   | ~400KB  | ~0.8KB/task |

## TaskVerifier Benchmarks

### Verification Levels

| Level    | Description       | Time (small file) |
| -------- | ----------------- | ----------------- |
| `exists` | File existence    | <10ms             |
| `syntax` | Python AST parse  | <50ms             |
| `import` | Import resolution | <200ms            |
| `tests`  | pytest discovery  | <500ms            |
| `mimiry` | Oracle validation | varies            |

### Batch Audit

| Tasks | Level  | Time  |
| ----- | ------ | ----- |
| 10    | syntax | ~0.3s |
| 50    | syntax | ~1.2s |
| 100   | syntax | ~2.5s |

## LoreBook / VectorSearch

### Decision Retrieval

| Decisions | Query Time | Notes            |
| --------- | ---------- | ---------------- |
| 100       | <20ms      | In-memory TF-IDF |
| 500       | <50ms      | Scales linearly  |
| 1000      | <100ms     | Still fast       |

### Pattern Matching

| Patterns | Match Time |
| -------- | ---------- |
| 50       | <10ms      |
| 200      | <30ms      |

## Crash Recovery

| Scenario                         | Recovery Time |
| -------------------------------- | ------------- |
| Resume after clean shutdown      | <500ms        |
| Resume after crash (WAL mode)    | <500ms        |
| Concurrent updates (10 parallel) | No data loss  |

## Running Benchmarks

```bash
# Run TaskStore benchmarks
python -m pytest tests/test_taskstore_enhanced.py -v -k "performance"

# Run all enhanced tests
python -m pytest tests/test_taskstore_enhanced.py tests/test_verifier_enhanced.py -v
```

## Thresholds

Benchmarks are designed to pass with these thresholds:

```python
# Performance thresholds (seconds)
SAVE_100_TASKS = 5.0
LOAD_100_TASKS = 2.0
UPDATE_50_TASKS = 3.0
QUERY_PENDING = 0.5

# Scale thresholds
MAX_DESCRIPTION_SIZE = 100_000  # 100KB
MAX_CHECKLIST_ITEMS = 100
MAX_DEPENDENCY_DEPTH = 100
MAX_PARALLEL_TASKS = 200
```

## Notes

1. **SQLite WAL mode**: Improves concurrent write performance
2. **In-memory TF-IDF**: No external vector DB required
3. **Async I/O**: All database operations are async (aiosqlite)
4. **Lazy loading**: Large fields loaded on demand
