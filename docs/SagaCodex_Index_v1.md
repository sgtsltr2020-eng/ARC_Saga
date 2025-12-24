# SagaCodex Index v1 Schema

This file defines the JSON structure of `sagacodex_index.json`.

The index is the single source of truth for machine-readable Cortex rules used by Warden, Mimiry, SagaConstitution, and DebateManager/LoreBook logic. It allows programmatic access to the standards defined in `sagacodex_python_fastapi.md`.

## Top-Level JSON Structure

The index file uses the following top-level structure:

```json
{
  "version": "1.0.0",
  "language": "Python",
  "framework": "FastAPI",
  "generated_at": "2025-12-21T00:00:00Z",
  "rules": [
    /* Rule objects */
  ]
}
```

### Fields

| Field          | Type   | Description                                                                              |
| -------------- | ------ | ---------------------------------------------------------------------------------------- |
| `version`      | string | Schema version (e.g., "1.0.0").                                                          |
| `language`     | string | The primary programming language (e.g., "Python"). Matches the active SagaCodex profile. |
| `framework`    | string | The primary framework (e.g., "FastAPI").                                                 |
| `generated_at` | string | ISO 8601 timestamp of when the index was generated.                                      |
| `rules`        | array  | List of Rule Objects defining the codex.                                                 |

## Rule Object Schema

Each rule represents a specific standard or requirement.

### Fields

| Field                | Type     | Required | Description                                                                    |
| -------------------- | -------- | -------- | ------------------------------------------------------------------------------ |
| `id`                 | string   | Yes      | Unique rule ID (e.g., "1", "SEC-001"). Must align with SagaCodex rule numbers. |
| `title`              | string   | Yes      | Short human-readable name (e.g., "Type Safety Required").                      |
| `severity`           | string   | Yes      | "CRITICAL", "WARNING", or "INFO". Drives escalation/debate behavior.           |
| `category`           | string   | Yes      | High-level grouping (e.g., "Security", "Reliability", "Architecture").         |
| `tags`               | string[] | Yes      | Free-form keywords (e.g., "fastapi", "db", "async", "auth").                   |
| `affected_artifacts` | string[] | No       | Artifact types this rule applies to (e.g., "endpoint", "model").               |
| `enforcement_phase`  | string   | Yes      | "design", "pre-merge", or "runtime". When this is checked.                     |
| `description`        | string   | Yes      | Concise description of the rule intent.                                        |
| `checklist_item`     | string   | Yes      | Single actionable checklist item for Warden/agents.                            |
| `detection_hint`     | string   | No       | Guidance for static analysis/AST detection.                                    |
| `examples`           | object[] | No       | List of correct usage examples.                                                |
| `antipatterns`       | object[] | No       | List of incorrect patterns with fixes.                                         |
| `references`         | string[] | No       | Pointers to documentation sections.                                            |
| `related_rules`      | string[] | No       | IDs of related rules.                                                          |

### Sub-objects

#### Example Object

```json
{
  "code": "def foo() -> int: ...",
  "description": "Correct usage with return type annotation."
}
```

#### Antipattern Object

```json
{
  "code": "def foo(): ...",
  "description": "Missing return type.",
  "fix": "Add -> type annotation."
}
```

## Example: Rule 1 – Type Safety Required

```json
{
  "id": "1",
  "title": "Type Safety Required",
  "severity": "CRITICAL",
  "category": "Reliability",
  "tags": ["types", "mypy", "python"],
  "affected_artifacts": ["all"],
  "enforcement_phase": "pre-merge",
  "description": "All code must be fully typed and pass static analysis checks.",
  "checklist_item": "All functions have complete type hints and pass mypy --strict.",
  "detection_hint": "Check for missing type hints in signatures or 'Any' usage.",
  "examples": [
    {
      "code": "def calculate_total(items: list[Item]) -> float:\n    return sum(item.price for item in items)",
      "description": "Function with complete argument and return type annotations."
    }
  ],
  "antipatterns": [
    {
      "code": "def calculate_total(items):\n    return sum(item.price for item in items)",
      "description": "Missing type annotations makes code unsafe and hard to verify.",
      "fix": "Add type hints: (items: list[Item]) -> float"
    },
    {
      "code": "def process_data(data: Any) -> Any:\n    ...",
      "description": "Overuse of 'Any' defeats the purpose of type checking.",
      "fix": "Use specific types like dict[str, str] or custom Pydantic models."
    }
  ]
}
```

## Example: Rule 2 – Async for I/O Operations

```json
{
  "id": "2",
  "title": "Async for IO Operations",
  "severity": "CRITICAL",
  "category": "Performance",
  "tags": ["async", "fastapi", "db", "io", "sqlalchemy"],
  "affected_artifacts": ["endpoint", "service"],
  "enforcement_phase": "runtime",
  "description": "Blocking I/O operations must be avoided in async endpoints to prevent event loop starvation.",
  "checklist_item": "Use async/await for all DB and network operations in FastAPI routes.",
  "detection_hint": "Flag sync calls (e.g., requests.get, db.query) inside 'async def' functions.",
  "examples": [
    {
      "code": "@app.get('/items')\nasync def read_items(db: AsyncSession = Depends(get_db)):\n    result = await db.execute(select(Item))\n    return result.scalars().all()",
      "description": "Correct use of AsyncSession and await for database query."
    }
  ],
  "antipatterns": [
    {
      "code": "@app.get('/items')\nasync def read_items(db: Session = Depends(get_db)):\n    return db.query(Item).all()",
      "description": "Sync database call inside async endpoint will block the event loop.",
      "fix": "Switch to AsyncSession and use await db.execute(...)."
    }
  ]
}
```

## Usage by Components

### Warden

Warden uses the index to generate dynamic, task-specific checklists.

- **Input**: Task type (e.g., "create FastAPI endpoint")
- **Action**: Query index for rules tagged `"fastapi"` and `"endpoint"`.
- **Output**: Extracts `checklist_item` strings to build the `verification_checklist`.

### Mimiry

Mimiry uses the rich metadata to explain violations and guide corrections.

- **Input**: Detected violation of a specific rule.
- **Action**: Retrieve `description`, `examples`, and `antipatterns` for the rule.
- **Output**: Generates a detailed explanation for the user or agent, referencing the exact standard.

### SagaConstitution / DebateManager

These components use the index to determine escalation behavior.

- **Input**: Runtime violation context.
- **Action**: Map violation to rule `id` and `severity`.
- **Decision**: Trigger `AdminApprovalRequest` if severity is `CRITICAL` or if conflict arises. Use rule `title` and `id` in the request sent to the user.

### LoreBook

LoreBook uses the index for pattern learning and tagging.

- **Input**: Completed debate/decision.
- **Action**: Store `violated_rules` (IDs) and `tags` in the `DebateRecord`.
- **Learning**: Aggregates decisions with similar tags to refine future prompts or detect persistent misunderstandings.

---

_Note: `sagacodex_index.json` is generated from the authoritative `sagacodex_python_fastapi.md` and `sagacodex_profiles.py`. This schema is versioned to support future language/framework profiles while maintaining a consistent structure for SAGA's core reasoning engines._
