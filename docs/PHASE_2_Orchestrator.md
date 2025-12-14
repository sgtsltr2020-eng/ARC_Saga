# PHASE 2: ORCHESTRATOR (In Progress)

> [!NOTE]
> This document outlines the active development phase. For the master roadmap, see `docs/ROADMAP.md`.

## 1. Objective

Make ARC SAGA the central nervous system for Python/FastAPI development by orchestrating multiple specialized agents (Copilot, Perplexity, etc.) through a unified interface.

**Status:** 🔨 **IN PROGRESS**
**Timeline:** Weeks 2-4

## 2. Key Components

### A. Orchestrator Agent (`arc_saga/orchestrator/core.py`)

- **Role:** Central coordinator.
- **Responsibilities:**
  - Intent classification (Plan vs. Code vs. Debug).
  - Policy enforcement (Quality Gates).
  - Context packaging.
  - Routing actions to the best provider.

### B. Agent Registry (`arc_saga/orchestrator/registry.py`)

- **Role:** Dynamic provider management.
- **Capabilities:**
  - Pluggable interface for OpenAI, Anthropic, etc.
  - Capability descriptors (Streaming, Tool-use).
  - Hot-swappable providers.

### C. Provider Router

- **Role:** Intelligence routing.
- **Logic:**
  - Transient errors -> Retry/Circuit Break.
  - Permanent errors -> Fallback to next provider.
  - Cost/Quality optimization logic.

## 3. Workflows

1.  **Sequential:** Generate -> Test -> Review -> Deploy
2.  **Parallel:** Comparison of multiple model outputs.
3.  **Dynamic:** Adaptive execution based on task complexity.

## 4. Quality Gates

- **Pre-Execution:** Audit input/context.
- **Post-Execution:** Run `mypy`, `pytest`, `pylint`, `bandit`.
- **Blocker:** Code is rejected if gates fail.

## 5. Reference Links

- [Detailed Roadmap](file:///docs/ROADMAP.md)
- [Agent Onboarding](file:///docs/AGENT_ONBOARDING.md)
