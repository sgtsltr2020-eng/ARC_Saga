# Agent Onboarding Guide

This guide provides the essential information for agents and contributors working on ARC Saga.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Project Structure](#3-project-structure)
4. [Development Workflow](#4-development-workflow)
5. [Coding Standards](#5-coding-standards)
6. [Testing](#6-testing)
7. [Documentation](#7-documentation)
8. [Planned Features & Roadmap](#8-planned-features--roadmap)
   - 8.1 [Feature Ideas Backlog (living)](#feature-ideas-backlog-living)

---

## 1. Introduction

ARC Saga is an enterprise-grade persistent memory layer for AI conversations with full-text search, auto-tagging, and multi-provider support.

## 2. Getting Started

Refer to the [README.md](../README.md) for installation and setup instructions.

## 3. Project Structure

- `arc_saga/` - Core package
- `docs/` - Documentation
- `tests/` - Test suite
- `shared/` - Shared utilities

## 4. Development Workflow

Follow the standard GitHub flow: branch, develop, test, PR, review, merge.

## 5. Coding Standards

Adhere to the project's `.cursorrules` and the patterns documented in `docs/decision_catalog.md`.

## 6. Testing

Aim for 95%+ test coverage. See `docs/verification_checklist.md` for quality gates.

## 7. Documentation

Keep documentation up to date. Reference `docs/arc_saga_master_index.md` for the document index.

## 8. Planned Features & Roadmap

### Phase 1c: Monitoring & Resilience (Next)

- Enhanced monitoring services
- Validator integration
- Improved error handling and resilience patterns

### Feature Ideas Backlog (living)

A concise, living backlog of feature and control-plane ideas lives in docs/feature_ideas_backlog.md. This file is intended to capture brainstormed features, orchestration patterns, UX ideas (Audit tab, Agent Activity, Command Console), provider adaptors, and stability primitives so they are not lost. The backlog items are grouped by their rough phase alignment below for discoverability; consult the standalone file for full details.

- Phase 1c (near-term): Audit & UX (Audit Dashboard, WebSocket events, Actionable Audit Notes), Capture API idempotency/validation, Perplexity integration fixes, ProviderAdapter refactor (foundations for replaceability).
- Phase 2: Orchestrator & Control Plane (Policy Engine, Intent Classification, CommandEnvelope), Provider registry, Vector/Semantic RetrievalStrategy.
- Phase 3: Enterprise features (Multi-tenancy, Analytics Dashboard, Event Sourcing / CQRS, OAuth, backups).

Refer to docs/feature_ideas_backlog.md for the full backlog and concise one-line descriptions.

### Phase 2: Orchestrator & Control Plane

- Policy Engine and Intent Classification
- Provider registry
- Vector/Semantic RetrievalStrategy

### Phase 3: Enterprise Features

- Multi-tenancy support
- Analytics Dashboard
- Event Sourcing / CQRS
- OAuth integration
- Backup solutions
