"""
MAE - Modular Agentic Ecosystem
===============================

The "Nervous System" and "Muscles" of Saga.

This package implements the orchestration layer that ensures Saga
uses the USMA wisdom foundation by making Codex checks a hard-coded
requirement for every action.

Components:
- FQL Gateway (fql_schema.py): Contractual JSON Schema for Mimiry queries
- Governor (governor.py): Turn-Control and AgentDropout logic
- Swarm (swarm.py): Agent coordination and pruning

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

from saga.core.mae.fql_schema import (
    ComplianceResult,
    FQLGovernance,
    FQLHeader,
    FQLPacket,
    FQLPayload,
    ValidationCache,
    create_fql_packet,
)
from saga.core.mae.governor import (
    AgentMode,
    AgentUtilityScore,
    Governor,
    MASSTriggeredReason,
    TurnMetrics,
)
from saga.core.mae.swarm import (
    AgentDropout,
    AgentType,
    DropoutDecision,
    DropoutReason,
    SwarmAgent,
    SwarmCoordinator,
)

__all__ = [
    # FQL Schema
    "FQLHeader",
    "FQLPayload",
    "FQLGovernance",
    "FQLPacket",
    "ComplianceResult",
    "ValidationCache",
    "create_fql_packet",
    # Governor
    "Governor",
    "TurnMetrics",
    "AgentUtilityScore",
    "MASSTriggeredReason",
    "SimulationFailureType",
    "AgentMode",
    # Swarm
    "AgentDropout",
    "SwarmCoordinator",
    "SwarmAgent",
    "AgentType",
    "DropoutReason",
    "DropoutDecision",
]

