"""
Role Definitions for ARC SAGA Orchestrator.

Defines the roles and profiles for agents within an arbitration or
multi-agent workflow. This allows separating "who is speaking"
(Logic/Role) from "which model is speaking" (Provider/Engine).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from arc_saga.orchestrator.types import AIProvider


class AgentRole(str, Enum):
    """
    Role of an agent in a workflow.

    Attributes:
        WORKER: Performs the initial task (Drafting, coding, planning).
        REVIEWER: Critiques the output of a Worker.
        JUDGE: Arbitrates between conflicting reviews or synthesizes final decision.
        MANAGER: Coordinates other agents (meta-role).
    """

    WORKER = "worker"
    REVIEWER = "reviewer"
    JUDGE = "judge"
    MANAGER = "manager"


@dataclass(frozen=True)
class SubagentProfile:
    """
    Profile definition for a subagent.

    A Subagent is a specific instantiation of a Role, potentially
    bound to a specific AI Provider or containing specific instructions.

    Attributes:
        name: Human-readable name (e.g., "Senior Python Architect").
        role: The functional role this agent plays.
        provider_preference: Optional specific provider to use (overrides router).
        system_prompt_override: Optional specific system prompt.
        description: Description of the agent's expertise.
    """

    name: str
    role: AgentRole
    provider_preference: Optional[AIProvider] = None
    system_prompt_override: Optional[str] = None
    description: str = ""
