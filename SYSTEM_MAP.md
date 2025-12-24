# SAGA System Map

> **Canonical Architecture Reference** — This document is the single authoritative map of SAGA's internal components ("organs") and their interactions. For implementation details, see the referenced source files.

---

## Overview

SAGA is a **local-first, multi-agent AI orchestrator** designed for autonomous software development with strict governance and empirical learning. It operates under a 3-tier authority model:

1. **SagaConstitution** — Immutable meta-rules governing SAGA's own behavior
2. **SagaCodex** — Language/framework-specific coding standards
3. **LoreBook** — Project-specific learned patterns and decisions

This file defines the purpose, inputs, outputs, and invocation context for every major component.

---

## Governance Layer

The governance layer defines SAGA's operating principles and quality standards. Authority flows downward: **Constitution > Codex > LoreBook > SagaRules**.

### SagaConstitution

**Purpose:** 15 immutable meta-rules that govern SAGA's decision-making, escalation, and safety. These rules protect against hallucination, overconfidence, runaway automation, cost overruns, and security violations. They cannot be overridden by users or agents.

- **Inputs:** `EscalationContext` (confidence level, agent agreement, conflict status, budget state)
- **Outputs:** `can_saga_act_autonomously()`, `must_escalate()`, `get_verification_strategy()`
- **Invocation:** Every decision point in the orchestration loop; pre-decision, post-generation, and runtime phases

> Source: [saga/config/sagarules_embedded.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/config/sagarules_embedded.py)

---

### SagaCodex

**Purpose:** Language and framework-specific coding standards. Defines required patterns (type safety, async I/O, structured logging), anti-patterns to avoid, elite patterns to emulate, and testing/performance requirements. MVP: Python/FastAPI profile.

- **Inputs:** Language/framework profile selection, code to validate
- **Outputs:** `CodeStandard` rules, anti-pattern violations, profile metadata
- **Invocation:** During prompt construction (`PromptBuilder`), code verification (`Warden`), and oracle consultation (`Mimiry`)

> Source: [saga/config/sagacodex_profiles.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/config/sagacodex_profiles.py)  
> Reference Doc: [saga/config/sagacodex_python_fastapi.md](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/config/sagacodex_python_fastapi.md)

---

### SagaRules (User-Defined)

**Purpose:** Project-specific rules defined by the user. These supplement SagaCodex but have lowest authority. Users can override Codex with protest (logged dissent), but cannot override Constitution.

- **Inputs:** Project configuration files, user preferences
- **Outputs:** Custom rule set that modifies or extends Codex
- **Invocation:** During project initialization and prompt construction

> Status: Planned (Phase 3+)

---

### LoreBook

**Purpose:** Empirical memory system that stores decisions, tracks outcomes, and extracts patterns. Balances Mimiry's idealism with project-specific learned behaviors. Consulted FIRST (fast, local) before Mimiry (slower, authoritative).

- **Inputs:** Question, context dict, trace_id
- **Outputs:** `Decision` (past decision with confidence), `Pattern` (generalized learning), `Outcome` (success/failure metrics)
- **Invocation:** Before every non-trivial decision by Warden; after decision outcomes are known for learning

> Source: [saga/core/lorebook.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/core/lorebook.py)  
> Storage: `.saga/lorebook.db` (SQLite + vector embeddings)

---

## Orchestration Layer

The orchestration layer manages task execution, agent coordination, and quality enforcement.

### Saga (Orchestrator)

**Purpose:** Top-level coordinator that receives user requests, interprets intent, delegates to Warden, and synthesizes final results. Responsible for the main execution loop and user interaction.

- **Inputs:** User request, project context, trace_id
- **Outputs:** Synthesized response, execution status, AdminApproval requests
- **Invocation:** Entry point for all user interactions; runs the main loop

> Status: Core orchestration implemented; full API in development

---

### Warden (Delegation & Enforcement)

**Purpose:** SAGA's delegation and arbitration agent. Receives proposals from Saga, decomposes into tasks, spawns coding agents, verifies outputs, and enforces quality gates. Always consults LoreBook FIRST, then Mimiry for conflicts.

- **Inputs:** `saga_request`, `context`, `trace_id`, agent outputs
- **Outputs:** `WardenProposal` (approved/modified/rejected), `TaskGraph`, verification results
- **Invocation:** After Saga interprets user request; before and after agent execution; during conflict resolution

> Source: [saga/core/warden.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/core/warden.py)

---

### Mimiry (Oracle)

**Purpose:** The immutable oracle of software truth. Embodies SagaCodex as "what must be true." Does not advise or suggest—states canonical interpretations. Consulted for discrepancies, rule interpretation, code measurement against ideal, and agent conflict resolution.

- **Inputs:** Question/code, context, agent outputs (for conflict resolution)
- **Outputs:** `OracleResponse`, `CanonicalInterpretation`, `ConflictResolution`
- **Invocation:** When Warden detects agent disagreement; when user asks "What does SagaCodex demand?"; when measuring code quality

> Source: [saga/core/mimiry.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/core/mimiry.py)

---

### Sub-Agents

Specialized agents that execute specific task types. All operate under SagaCodex-enforcing prompts.

| Agent           | Purpose                                         | Invocation               |
| --------------- | ----------------------------------------------- | ------------------------ |
| **CodingAgent** | Generates production code + tests per SagaCodex | Task execution by Warden |
| **Reviewer**    | Reviews code against Codex standards            | Pre-merge verification   |
| **Docs**        | Generates/updates documentation                 | After code changes       |
| **Testing**     | Expands test coverage, runs test suites         | Verification phase       |
| **GitHub**      | Manages PRs, branches, CI integration           | Deployment workflows     |
| **Onboarding**  | Guides new project setup                        | Project initialization   |

> Source: [saga/agents/coder.py](file:///c:/Users/sgtsl/PROJECTS/ARC_Saga/saga/agents/coder.py), agent framework

---

### Escalation Engine

**Purpose:** Enforces Constitution rules for escalation. Determines when SAGA must defer to user (AdminApproval) rather than proceeding autonomously.

**Escalation Triggers:**

- `must_escalate()` returns true (confidence < 75%, conflict detected, secrets found, budget exceeded)
- LLM providers disagree
- Decision affects multiple systems
- User explicitly requests escalation

**Escalation Flow:**

1. Warden detects escalation condition
2. If resolvable via Mimiry → debate protocol
3. If unresolvable → present all options to user with trade-offs
4. User decision recorded; if override → LoreBook updated

- **Inputs:** `EscalationContext`, agent disagreements, verification failures
- **Outputs:** AdminApproval request, debate log, override protocol
- **Invocation:** Any point where `SagaConstitution.must_escalate()` returns true

---

## Memory and Learning

### Decision Recording Flow

```
Warden receives proposal
    └─> Consult LoreBook FIRST (local, fast)
        └─> Match found? → Use historical decision
        └─> No match? → Consult Mimiry (authoritative)
            └─> Mimiry returns canonical ruling
            └─> Conflict with LoreBook? → Trigger debate
                └─> User resolves → Record Decision in LoreBook
```

**Recording Conditions:**

- Debate or override leads to a stored `Decision`
- New pattern emerges from repeated successful decisions
- User explicitly confirms a decision should be remembered

**Minor/Unresolved Cases:**

- SAGA should prompt: "Should I remember this decision for future reference?"
- User can decline, accept, or mark for review

---

## Onboarding and IDE Integration

### Onboarding Flow (Planned)

1. Detect missing project scaffolding (README, requirements/pyproject.toml, .cursorrules)
2. Prompt user for project metadata
3. Generate initial configuration files
4. Initialize LoreBook database
5. Validate IDE integration

### IDE Integration

**Principle:** "If the IDE can do it, SAGA should be able to command it."

- **Cursor/AG Integration:** SAGA Local Server API exposes endpoints for Cursor commands
- **Configuration:** `.cursorrules` suite for editor-level enforcement
- **Sync:** Changes in IDE reflected in SAGA state; SAGA can trigger IDE actions

> Reference: Cursor FAANG setup docs, `.cursorrules` files

---

## End-to-End Flow Summary

**Happy-Path Pipeline (10 steps):**

1. **User Request** — User issues a request to SAGA
2. **Saga Interprets** — Saga parses intent, identifies request type
3. **Mimiry Consultation (if needed)** — For ambiguous requests, consult Mimiry for canonical interpretation
4. **Warden Plans** — Warden decomposes request into TaskGraph, consults LoreBook for patterns
5. **Task Assignment** — Warden picks appropriate agents (Coder, Reviewer, etc.)
6. **Agent Execution** — Agents execute with SagaCodex-enforcing prompts + LoreBook context
7. **Warden Verification** — Warden verifies outputs against quality gates; escalates if needed
8. **Conflict Resolution** — If agents disagree, Warden consults Mimiry for canonical judgment
9. **Result Synthesis** — Saga synthesizes verified outputs, presents for AdminApproval
10. **LoreBook Update** — If meaningful decision or override occurred, record in LoreBook

---

## Quick Reference

| Component        | Authority Level | Mutability              | Primary Role          |
| ---------------- | --------------- | ----------------------- | --------------------- |
| SagaConstitution | 1 (highest)     | Immutable               | SAGA's behavior       |
| SagaCodex        | 2               | Immutable (per version) | Code quality          |
| LoreBook         | 3               | Mutable (learning)      | Project patterns      |
| SagaRules        | 4 (lowest)      | Mutable (user config)   | Project customization |

---

_Last updated: December 2025_
