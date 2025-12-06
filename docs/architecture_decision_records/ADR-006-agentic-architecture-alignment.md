# ADR-006: Align with BARC 2025 Agentic Architecture Standard

**Status**: Accepted  
**Date**: 2025-12-06

## Context

BARC research note “From Data to Agents” (Sep 2025) defines the required data/model/agentic workflows for enterprise agentic AI success.

## Decision

ARC SAGA already implements the complete stack:

- Data layer → message.py + SQLite
- Model layer → AIProvider enum + registry
- Agentic layer → ProviderRouter + CostOptimizer with Decimal scoring

We formally adopt this document as the architectural north star.

## Consequences

- All future components must map to one of the three layers
- CostOptimizer is officially the governance control plane
- We are 12–24 months ahead of enterprise timelines
