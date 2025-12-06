# ARC Saga — Feature Ideas Backlog

Purpose: A lightweight, living backlog of planned ideas and guardrails to ensure ARC Saga stays replaceable, controllable, and enterprise-grade. Each item is intentionally concise.

Status: Ideas only. Not commitments. Reference AGENT_ONBOARDING.md for current implementation status.

## Orchestrator & Control Plane

- Orchestrator Agent (central coordinator)
  - A controller inside Saga that classifies intent, enforces policies, packages context, and routes actions to IDEs/providers. Prevents IDEs from acting on brainstorms.

- Policy Engine (user-customizable)
  - Loads project-wide rules ("cursorrules"), quality gates, and guardrails. Blocks or requires confirmation for actions based on intent and policy.

- CommandEnvelope (standard action payload)
  - Provider-agnostic command package including intent, constraints, attachments (rules/docs), acceptance criteria, and rollback policy. Ensures consistency and quality.

- Context Packager (automatic attachments)
  - Automatically attaches .cursorrules, onboarding docs, verification checklist, decision/error catalogs, and relevant code excerpts to every command.

- Agent Registry & Capabilities
  - Register and track connected agents (IDE, LLM, controller) with capability descriptors (streaming, tool-use, max tokens) used to route tasks.

- Intent Classification & Action Gating
  - Detect brainstorming, planning, decision, task, code_change. Block auto-execution for early stages; require confirmation for decisions; draft-only for code changes.

## Replaceability Primitives

- ProviderAdapter ABC + Registry
  - Formal adapter interface for all providers (Perplexity, OpenAI, Anthropic, Groq, etc.) with a central registry for hot-swapping and capability lookup.

- IdeAdapter ABC
  - Standard interface for IDE integrations (Cursor, VSCode, Copilot) so orchestrator sends uniform commands regardless of IDE specifics.

- API Versioning (additive)
  - Add api_version field to requests/responses. Evolve contracts additively to avoid breaking clients.

- Idempotency Envelope
  - Require message_id and correlation_id in write endpoints to deduplicate and trace operations reliably.

- RetrievalStrategy Abstraction
  - Pluggable retrieval layer (KeywordFTS, SemanticVector, Hybrid) so context injection can evolve without changing orchestrator logic.

## Stability & Resilience

- Circuit Breaker (external calls)
  - Guard Perplexity and other provider calls with open/half-open/closed states to prevent cascading failures.

- Retry with Exponential Backoff + Jitter
  - Uniform transient error handling for all network interactions.

- Rate Limiting (per endpoint/client)
  - Control usage for fairness and protection under load.

- Configurable Server Port & Settings
  - Move hardcoded 8421 to config; expose env vars (ARC_SAGA_PORT, etc.).

## Audit & UX

- Audit Dashboard Tab (real-time)
  - UI showing health, metrics, errors, and rule compliance with live updates (WebSocket). Pulls from error_instrumentation and storage health.

- Agent Activity Tab
  - Shows each agent's current status, intent, active session, and last action. Ties into agent registry and orchestrator events.

- Command Console Tab
  - Text box to send follow-ups; routes through Orchestrator with automatic attachments. Includes intent override and confirmation toggles.

- Actionable Audit Notes
  - Users create "AuditNotes" with severity/category/suggested action and attachments. Orchestrator routes them to the right agent as CommandEnvelopes.

- WebSocket Event Bus
  - /ws/events streams core events (MessageCaptured, IntentClassified, ActionRequested, ActionApproved, NoteCreated/Routed/Completed, ErrorLogged).

## OSS Default Model & BYOK

- Default Open-Source Orchestrator LLM
  - Local, no token/rate limits by default (e.g., Llama 3.1 70B via vLLM; fallback Mixtral/Qwen). Used for governance, rules enforcement, and planning.

- BYOK Provider Support
  - Users can configure OpenAI/Anthropic/Groq/etc. via adapters; capability registry handles routing decisions.

## Perplexity & Multi-Provider Flows

- ARC Saga–Proxied Perplexity Endpoint
  - /perplexity/ask performs context injection, calls provider, streams response, and persists messages. Centralizes retries/logging/policies.

- Direct Capture/Search Flow
  - Agents use /capture and /context/recent against Saga for write/read, keeping memory consistent even when providers are called directly.

## Advanced Architecture

- Event Sourcing
  - Immutable event log for audit, replay, and incident forensics (Phase 2 feature).

- CQRS
  - Separate read/write models; optimize read projections and maintain eventual consistency.

- Vector/Semantic Search
  - Embedding-based retrieval using Qdrant/Pinecone to complement FTS5; switchable via RetrievalStrategy.

- OAuth Authentication
  - Google/GitHub/Microsoft SSO with JWT; per-agent permissions and access scopes.

## Enterprise & Ops

- Multi-Tenancy & RBAC
  - Team/org isolation, roles, and granular permissions for agents and endpoints.

- Analytics Dashboard
  - Usage metrics, search analytics, performance monitoring; policy violation tracking.

- Export/Import & Backups
  - Data portability, backup scheduling, and restore tooling.

## Quality Gates & Standards

- Enforcement of Type Hints & Docstrings
  - Fail actions that produce code without full type hints and Google-style docstrings.

- Logging & Error Handling Requirements
  - Require contextual logging and structured error handling in all generated code.

- Test Coverage & Security Gates
  - Accept only changes that include unit/integration tests, meet coverage targets, and pass bandit/mypy/pylint.

## Developer Experience

- Prompts Library Expansion
  - Curated prompts tied to policies and acceptance criteria for repeatable, high-quality generation across agents.

- Decision & Error Catalog Attachments
  - Automatically attach relevant decisions/errors from catalogs to guide agents toward known patterns and pitfalls.

- Health Check & Diagnostics Endpoints
  - /audit/summary, /metrics, and diagnostic utilities exposed for quick triage.

---

Maintenance:

- Keep entries brief; link to detailed specs when available.
- Update statuses in AGENT_ONBOARDING.md when ideas transition to planned/implemented.