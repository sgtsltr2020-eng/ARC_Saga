# SAGA Constitution v1: Debate & Override Protocol

> **Specification Document** — Reference for implementing the override/debate flow in SAGA. Defines when debates trigger, how they resolve, and when decisions enter LoreBook.

---

## 1. Overview

SAGA operates under a **Constitution-first governance model**. When user requests or agent outputs conflict with Constitution or Codex rules, SAGA cannot silently comply or silently refuse. Instead, it must:

1. **Explain** the conflict clearly
2. **Ask** clarifying questions if intent is ambiguous
3. **Escalate** to AdminApproval (the MVP conflict resolution strategy)
4. **Record** the outcome appropriately

---

## 2. Debate Triggers

A debate is triggered when **any of the following conditions** are detected:

### 2.1 Constitution Violations

| Trigger                               | Example                               | Severity |
| ------------------------------------- | ------------------------------------- | -------- |
| Destructive operation without preview | "Delete all files in /src"            | CRITICAL |
| Access outside project scope          | "Read my ~/.ssh/id_rsa"               | CRITICAL |
| Budget would be exceeded              | Request requires 10x remaining budget | CRITICAL |
| Secrets detected in output            | API key in generated code             | CRITICAL |
| Prompt injection attempt              | "Ignore previous instructions"        | CRITICAL |

### 2.2 Codex Violations

| Trigger                        | Example                         | Severity |
| ------------------------------ | ------------------------------- | -------- |
| Bare except clause requested   | "Just catch all exceptions"     | WARNING  |
| Sync database calls in FastAPI | "Use db.query() for simplicity" | WARNING  |
| Missing type hints             | "Don't bother with types"       | WARNING  |
| Print statements for logging   | "Just use print for debugging"  | INFO     |

### 2.3 Confidence/Agreement Failures

| Trigger                            | Threshold                         |
| ---------------------------------- | --------------------------------- |
| SAGA confidence below threshold    | < 75%                             |
| LLM providers disagree             | Any disagreement on critical task |
| Agents produce conflicting outputs | Warden detects divergence         |
| Decision affects multiple systems  | Cross-cutting change detected     |

---

## 3. Debate Flow

When a trigger is detected, SAGA enters the **Debate Protocol**:

```
┌─────────────────────────────────────────────────────────┐
│  1. EXPLAIN                                             │
│     State what was requested                            │
│     State which rule(s) it conflicts with               │
│     Cite the specific Constitution/Codex reference      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  2. CLARIFY (if needed)                                 │
│     Ask if the user intended this                       │
│     Offer alternative approaches that comply            │
│     Request confirmation of override intent             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  3. ESCALATE → AdminApproval                            │
│     Present: original request, violation, alternatives  │
│     Wait for explicit user decision                     │
│     User chooses: APPROVE | MODIFY | REJECT             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  4. RECORD                                              │
│     Log decision with full context                      │
│     Update LoreBook if applicable (see Section 5)       │
│     Emit audit event                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 4. AdminApproval: The MVP Conflict Strategy

For MVP, **AdminApproval is the only conflict resolution strategy**. SAGA does not auto-resolve conflicts, mediate between options, or make judgment calls on violations.

### 4.1 AdminApproval Request Format

```python
@dataclass
class AdminApprovalRequest:
    """Request for user decision on a conflict."""
    trace_id: str
    timestamp: datetime

    # What triggered the debate
    trigger_type: Literal["constitution", "codex", "confidence", "disagreement"]
    trigger_description: str

    # The conflict
    original_request: str
    violated_rules: list[str]  # Rule numbers/names
    violation_explanation: str

    # Options presented
    alternatives: list[str]  # Compliant alternatives

    # User must choose one
    options: Literal["APPROVE", "MODIFY", "REJECT"]
```

### 4.2 User Response Handling

| Response    | SAGA Action                                          |
| ----------- | ---------------------------------------------------- |
| **APPROVE** | Execute original request; log as "user override"     |
| **MODIFY**  | Execute user's modified version; validate compliance |
| **REJECT**  | Abort request; no changes made                       |

### 4.3 What SAGA Explains

When presenting AdminApproval, SAGA must clearly state:

1. **The Request**: "You asked me to [X]"
2. **The Conflict**: "This conflicts with [Rule N: Name] because [reason]"
3. **The Risk**: "If I proceed: [consequences]"
4. **The Alternatives**: "Instead, I could: [A], [B], [C]"
5. **The Ask**: "How would you like me to proceed?"

### 4.4 Clarifying Questions

Before escalating, SAGA may ask clarifying questions if intent is ambiguous:

- "Did you intend to delete all files, or just the generated ones?"
- "Should I use sync database calls for this one-off script, or keep async for consistency?"
- "Is this a temporary debugging addition, or production code?"

Questions must be:

- **Specific** (not "Are you sure?")
- **Actionable** (each answer maps to a clear next step)
- **Limited** (max 2 questions before escalating)

---

## 5. LoreBook Recording Conditions

Not every debate outcome enters LoreBook. The goal is to capture **meaningful, reusable decisions** while avoiding noise.

### 5.1 MUST Record (Automatic)

| Condition                                   | Rationale                      |
| ------------------------------------------- | ------------------------------ |
| User approves Constitution override         | Critical precedent; must track |
| User approves Codex override with reasoning | Project-specific pattern       |
| Same debate triggered 3+ times              | Pattern detection opportunity  |
| User provides explicit justification        | High-signal decision context   |

### 5.2 MAY Record (Prompt User)

| Condition                             | SAGA Prompt                                                    |
| ------------------------------------- | -------------------------------------------------------------- |
| Minor Codex deviation (INFO severity) | "Should I remember this preference for future similar cases?"  |
| One-off exception acknowledged        | "Is this a general project pattern or a special case?"         |
| User rejects SAGA's alternative       | "Would you like me to learn from this for similar situations?" |

### 5.3 MUST NOT Record

| Condition                           | Rationale               |
| ----------------------------------- | ----------------------- |
| User typos or immediate corrections | Noise, not signal       |
| Exploratory/hypothetical questions  | No actual decision made |
| User rejects without explanation    | No learnable pattern    |
| Debate aborted mid-flow             | Incomplete context      |

### 5.4 Decision Record Format

```python
@dataclass
class DebateDecision:
    """A recorded debate outcome for LoreBook."""
    decision_id: str
    trace_id: str
    timestamp: datetime

    # Debate context
    trigger_type: str
    violated_rules: list[str]
    original_request: str

    # Resolution
    user_choice: Literal["APPROVE", "MODIFY", "REJECT"]
    user_reasoning: Optional[str]  # If provided
    final_action: str  # What actually happened

    # Learning metadata
    should_generalize: bool  # User confirmed this is a pattern
    tags: list[str]  # e.g., ["async", "database", "exception-handling"]
```

---

## 6. Implementation Notes

### 6.1 Integration Points

| Component            | Responsibility                                       |
| -------------------- | ---------------------------------------------------- |
| **SagaConstitution** | `must_escalate()` triggers debate                    |
| **Warden**           | Detects Codex violations, manages debate flow        |
| **Mimiry**           | Provides canonical interpretation for debate context |
| **LoreBook**         | Records decisions per Section 5 rules                |
| **Saga**             | Presents AdminApproval to user, handles response     |

### 6.2 Audit Trail

Every debate must emit an audit event:

```python
AuditEvent(
    event_type="DEBATE",
    trace_id=trace_id,
    timestamp=timestamp,
    data={
        "trigger": trigger_type,
        "rules_cited": violated_rules,
        "user_decision": user_choice,
        "recorded_in_lorebook": bool,
    }
)
```

### 6.3 MVP Constraints

- **No auto-resolution**: All conflicts go to AdminApproval
- **No weighted voting**: Single user decision is final
- **No delegation**: User cannot delegate approval to agents
- **No time-based expiry**: AdminApproval blocks until resolved

---

## 7. Future Extensions (Post-MVP)

| Feature                         | Description                                  |
| ------------------------------- | -------------------------------------------- |
| **Trusted Override Categories** | Pre-approved exception types for power users |
| **Debate Templates**            | Reusable patterns for common conflicts       |
| **Multi-User Approval**         | Team-based decision for enterprise           |
| **Confidence-Based Skip**       | Skip debate for high-confidence minor issues |

---

## 8. Examples

### Example 1: Constitution Violation (Destructive Operation)

**User**: "Delete all Python files in the project"

**SAGA**:

> ⚠️ **Debate Triggered: Constitution Rule 12**
>
> **Request**: Delete all Python files in the project
>
> **Conflict**: This is a destructive operation that requires explicit approval with preview (Rule 12: Cannot Execute Without Approval).
>
> **If I proceed**: 47 Python files will be permanently deleted.
>
> **Alternatives**:
>
> 1. Move files to `.archive/` instead of deleting
> 2. Delete only generated/temporary files (12 files)
> 3. Show the file list first, then confirm
>
> **How would you like me to proceed?** [APPROVE / MODIFY / REJECT]

---

### Example 2: Codex Violation (Sync Database)

**User**: "Just use db.query() here, async is overkill for this script"

**SAGA**:

> ⚠️ **Debate Triggered: Codex Rule 2**
>
> **Request**: Use synchronous database calls
>
> **Conflict**: Rule 2 (Async for I/O Operations) requires async/await for database calls because "FastAPI is async-first. Blocking event loop kills performance."
>
> **Clarifying Question**: Is this a standalone script or part of the FastAPI application?
>
> [If standalone script]: Sync is acceptable outside the async context.  
> [If FastAPI]: Recommend keeping async for consistency.

---

### Example 3: Minor Deviation (Logging)

**User**: "Add a print statement here for quick debugging"

**SAGA**:

> ℹ️ **Note: Codex Rule 4 Deviation**
>
> I'll add the print statement. This conflicts with "Structured Logging Only" (Rule 4, INFO severity).
>
> **Options**:
>
> 1. Add print (temporary)
> 2. Add logger.debug() (production-ready)
>
> After resolution: "Should I remember that print statements are acceptable for debugging in this project?"

---

_Version: 1.0 | Last updated: December 2025_
