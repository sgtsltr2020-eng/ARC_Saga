# PHASE 3: MEMORY SYSTEMS (Planned)

> [!NOTE]
> Detailed specification for the "System 2" memory architecture.

## 1. Goal

Make ARC SAGA learn and improve over time by capturing reasoning traces and building a knowledge graph.

## 2. Key Features

### A. Knowledge Graph

- **Graph Relationships:** Map dependencies between modules.
- **Entities:** Code structure, Architecture limits, Decisions.
- **Implementation:** Neo4j or NetworkX.

### B. Reasoning Trace Capture

- **System 2 Learning:** Capture step-by-step problem solving.
- **Replay:** Ability to "reply" reasoning to train small models.
- **Data Model:** `ReasoningTrace` (problem, steps, decision_points).

### C. Pattern Recognition

- Identify recurring anti-patterns.
- Suggest reusable components from `decision_catalog.md`.
