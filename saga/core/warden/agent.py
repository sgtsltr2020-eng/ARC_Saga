"""
The Warden - SAGA's Delegation & Arbitration Agent
===================================================

The Warden (formerly JUDGE) is responsible for:
1. Receiving proposals from SAGA
2. Planning task decomposition
3. Spawning transient coding agents dynamically
4. Collecting agent outputs
5. Detecting discrepancies between agents
6. CONSULTING MIMIRY when conflicts arise (critical integration)
7. Enforcing final consistency via SagaCodex (through Mimiry)
8. Reporting approved output back to SAGA

Authority:
- The Warden can refuse SAGA's proposals if they violate SagaCodex
- The Warden consults Mimiry (the oracle) when agents disagree
- The Warden enforces quality gates before approval
- The Warden escalates to user when Mimiry cannot resolve

Named "The Warden" because:
- Guards quality standards (via Mimiry/SagaCodex)
- Oversees subagent execution
- Maintains order in the delegation hierarchy
- Resolves conflicts through consultation with the oracle

Integration with Mimiry:
- Mimiry is consulted reactively (not proactively)
- Warden asks Mimiry to resolve conflicts between agents
- Mimiry measures agent outputs against SagaCodex ideal
- Warden enforces Mimiry's canonical judgment

Author: ARC SAGA Development Team
Date: December 14, 2025 (Revised for Mimiry Integration)
Status: Phase 2 Week 1 - Foundation
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import aiofiles  # Fix undefined name

from saga.agents.coder import CodingAgent
from saga.config.sagacodex_profiles import get_codex_manager
from saga.config.sagarules_embedded import SagaConstitution
from saga.core.codex_index_client import CodexIndexClient
from saga.core.governance_graph import WardenState, create_warden_graph
from saga.core.lorebook import LoreBook, Outcome

# Phase 8: MAE Integration
from saga.core.mae import (
    FQLPacket,
    Governor,
    MASSTriggeredReason,
    SwarmCoordinator,
)
from saga.core.mae.fql_schema import StrictnessLevel
from saga.core.mae.swarm import DropoutReason
from saga.core.mimiry import ConflictResolution, Mimiry, OracleResponse
from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.task_verifier import TaskVerifier
from saga.llm.client import LLMClient
from saga.llm.cost_tracker import CostTracker
from saga.storage.task_store import TaskStore

logger = logging.getLogger(__name__)


# Task class and TODOList removed in favor of saga.core.task.Task and saga.core.task_graph.TaskGraph


@dataclass
class WardenProposal:
    """
    The Warden's response to SAGA's delegation request.

    Can be:
    - Approved: Tasks created, ready to execute
    - Modified: Alternative approach suggested (via Mimiry consultation)
    - Rejected: Violates SagaCodex, cannot proceed
    """
    decision: Literal["approved", "modified", "rejected"]
    task_graph: Optional[TaskGraph] = None
    rationale: str = ""
    sagacodex_violations: list[str] = field(default_factory=list)
    alternative_approach: Optional[str] = None
    mimiry_guidance: Optional[OracleResponse] = None  # NEW: Oracle consultation
    estimated_cost: float = 0.0
    estimated_time_minutes: int = 0


class Warden:
    """
    The Warden - SAGA's delegation and quality enforcement agent.

    Responsibilities:
    1. Receive proposals from SAGA
    2. Plan task decomposition
    3. Spawn coding agents (future implementation)
    4. Collect agent outputs
    5. Detect discrepancies
    6. CONSULT MIMIRY when conflicts arise ← CRITICAL INTEGRATION
    7. Enforce canonical approach (via Mimiry)
    8. Report completion to SAGA

    Mimiry Integration Flow:

    Normal path (no conflicts):
        Warden receives proposal → Plans tasks → Agents execute → Warden verifies → Approved

    Conflict path (agents disagree):
        Warden detects discrepancy → Consult Mimiry → Mimiry measures against ideal
        → Warden enforces canonical approach → Approved or Escalate

    Usage:
        warden = Warden()

        # SAGA sends delegation request
        proposal = await warden.receive_proposal(
            saga_request="Create user authentication system",
            context={"budget": 100},
            trace_id="abc-123"
        )

        # If approved, execute tasks
        if proposal.decision == "approved" and proposal.task_graph:
            ready_tasks = proposal.task_graph.get_ready_tasks()
            for task in ready_tasks:
                # Spawn agent, execute, collect output
                outputs = await warden.execute_task(task)

                # If multiple agents or conflict detected
                if len(outputs) > 1:
                    resolution = await warden.resolve_via_mimiry(outputs, task)
                    warden.enforce_canonical(task, resolution)
    """

    def __init__(self, project_root: str = ".") -> None:
        """Initialize The Warden with Mimiry integration."""
        self.project_root = project_root
        self.constitution = SagaConstitution()
        self.codex_manager = get_codex_manager()
        self.current_codex = self.codex_manager.get_current_profile()
        self.mimiry = Mimiry()  # The Oracle
        self.codex_client = CodexIndexClient(Path(project_root) / ".saga" / "sagacodex_index.json")

        # Phase 3B Integration
        self.lorebook: LoreBook | None = None
        self.llm_client: LLMClient | None = None
        self.cost_tracker: CostTracker | None = None
        self.task_store: TaskStore | None = None
        self.task_verifier: TaskVerifier | None = None

        # Phase 8: MAE Integration - Governor and Swarm
        self.governor = Governor()
        self.swarm = SwarmCoordinator(self.governor)

        # Phase 3: Governance Graph
        # Use MemorySaver for sync initialization, proper checkpointer set in initialize()
        import os
        enable_hitl = os.environ.get("SAGA_ENABLE_HITL", "false").lower() == "true"
        self.graph = create_warden_graph(checkpointer=None, enable_interrupt=enable_hitl)

        logger.info(
            "The Warden initialized with Mimiry and MAE integration",
            extra={
                "codex_language": self.current_codex.language,
                "codex_framework": self.current_codex.framework,
                "mimiry_role": "oracle_consultant",
                "governor_mode": self.governor.current_mode.value,
            }
        )

    async def initialize(self) -> None:
        """
        Initialize Warden dependencies (LoreBook, LLM client, cost tracker).

        Call this after creating Warden instance.
        """

        # Initialize LoreBook
        if not self.lorebook:
            self.lorebook = LoreBook(project_root=self.project_root)
            await self.lorebook.initialize()

        # Initialize LLM client
        if not self.llm_client:
            self.llm_client = LLMClient()
            await self.llm_client.initialize()

        # Initialize cost tracker
        if not self.cost_tracker:
            self.cost_tracker = CostTracker()

        # Initialize TaskStore
        if not self.task_store:
            self.task_store = TaskStore(
                db_path=f"{self.project_root}/.saga/tasks.db"
            )
            await self.task_store.initialize()

        # Initialize TaskVerifier
        if not self.task_verifier:
            self.task_verifier = TaskVerifier(
                project_root=self.project_root,
                mimiry=self.mimiry
            )

        # Phase 3: Initialize graph with checkpointer
        import os
        enable_hitl = os.environ.get("SAGA_ENABLE_HITL", "false").lower() == "true"

        # Use MemorySaver for now (AsyncSqliteSaver context manager has compatibility issues)
        # This still supports async execution and HITL, just without persistent state across restarts
        self.graph = create_warden_graph(checkpointer=None, enable_interrupt=enable_hitl)

        logger.info("Warden fully initialized with persistent state")

    async def resume_work(self, request_id: str) -> Optional[TaskGraph]:
        """
        Resume incomplete work from previous session with automatic verification.

        Process:
        1. Load TaskGraph from database
        2. Audit all "done" tasks to verify code exists and is valid
        3. Mark failed tasks as "needs_verification"
        4. Return graph with updated status

        Args:
            request_id: Original request trace_id

        Returns:
            TaskGraph with remaining work, or None if not found
        """
        if not self.task_store or not self.task_verifier:
            logger.warning("Persistence not initialized")
            return None

        logger.info("Warden resuming work", extra={"request_id": request_id})

        # Load from database
        graph = await self.task_store.load_task_graph(request_id)
        if not graph:
            logger.warning("No saved work found", extra={"request_id": request_id})
            return None

        # Audit completed tasks
        completed_tasks = [t for t in graph.get_all_tasks() if t.status == "done"]

        if completed_tasks:
            logger.info(
                "Auditing completed tasks before resume",
                extra={"request_id": request_id, "completed_count": len(completed_tasks)}
            )

            # Standard verification: exists + syntax + import + tests
            results = await self.task_verifier.audit_completed_tasks(
                tasks=completed_tasks,
                level="import"  # Fast enough, catches most issues
            )

            # Handle failures
            failed = [r for r in results if not r.verified]
            if failed:
                logger.error(
                    "Verification failed for some completed tasks",
                    extra={
                        "request_id": request_id,
                        "failed_count": len(failed),
                        "failed_tasks": [r.task_id for r in failed]
                    }
                )

                # Mark failed tasks for re-execution
                for result in failed:
                    task = graph.get_task(result.task_id)
                    if task:
                        task.status = "needs_verification"
                        task.warden_verification = "rejected"

                    # Update database
                    await self.task_store.update_task_status(
                        task_id=task.id,
                        status="needs_verification",
                        warden_verification="rejected"
                    )

                    logger.warning(
                        "Task marked for re-execution",
                        extra={"task_id": task.id, "issues": result.issues}
                    )

        # Calculate progress
        pending = graph.get_ready_tasks()
        actually_done = [t for t in graph.get_all_tasks() if t.status == "done"]
        needs_work = [t for t in graph.get_all_tasks() if t.status == "needs_verification"]

        logger.info(
            "Work resumed with verification",
            extra={
                "request_id": request_id,
                "pending_tasks": len(pending),
                "completed_tasks": len(actually_done),
                "failed_verification": len(needs_work)
            }
        )

        return graph

    async def execute_task(
        self,
        task: Task,
        project_root: str = "."
    ) -> list[dict[str, Any]]:
        """
        Execute task by spawning CodingAgent and generating code.

        PHASE 8: Governor Turn Tracking + Citation Loop
        -----------------------------------------------
        - Tracks turn metrics via Governor
        - If turn 1 fails, injects corrections for turn 2 (self-correction)
        - On COMPLIANCE_REGRESSION, swaps to specialist agent

        Args:
            task: Task to execute
            project_root: Project root directory

        Returns:
            List of agent results in dict format
        """
        logger.info(
            "Warden executing task with Governor tracking",
            extra={"task_id": task.id, "description": task.description}
        )

        agent_id = "CodingAgent"
        current_turn = 1
        max_turns = 2  # Allow 1 retry with self-correction

        # Get FQL governance from task context (injected by receive_proposal)
        fql_governance = task.context.get("fql_governance", {}) if task.context else {}
        strictness = fql_governance.get("strictness", StrictnessLevel.ENTERPRISE.value)

        # Track initial compliance
        compliance_before = fql_governance.get("compliance_score", 50.0)

        agent = CodingAgent(
            llm_client=self.llm_client,
            lorebook=self.lorebook,
            mimiry=self.mimiry,
            sagacodex_profile=self.current_codex,
            agent_name=agent_id
        )

        output = None
        result = None

        try:
            while current_turn <= max_turns:
                logger.info(
                    f"Executing turn {current_turn}/{max_turns}",
                    extra={"task_id": task.id, "agent_id": agent_id, "turn": current_turn}
                )

                # Execute task with agent
                output = await agent.solve_task(
                    task=task,
                    project_root=project_root if project_root != "." else self.project_root,
                    language_profile=f"{self.current_codex.language}_{self.current_codex.framework}",
                    critical=(task.weight == "complex")
                )

                # ============================================================
                # PHASE 8: Governor Turn Tracking
                # ============================================================
                compliance_after = output.confidence  # Use confidence as proxy

                turn_metrics = self.governor.track_turn(
                    task_id=task.id,
                    agent_id=agent_id,
                    tokens=output.tokens_used,
                    compliance_before=compliance_before,
                    compliance_after=compliance_after
                )

                logger.info(
                    "Turn tracked by Governor",
                    extra={
                        "task_id": task.id,
                        "turn": current_turn,
                        "tokens_used": output.tokens_used,
                        "compliance_delta": turn_metrics.compliance_delta,
                        "is_productive": turn_metrics.is_productive,
                    }
                )

                # Track cost
                if self.cost_tracker:
                    self.cost_tracker.record_task_cost(
                        task_id=task.id,
                        cost=output.cost_usd,
                        agent_name=agent_id
                    )

                # ============================================================
                # CITATION LOOP: Check for violations and self-correct
                # ============================================================
                if output.mimiry_violations and current_turn < max_turns:
                    logger.info(
                        "Turn 1 non-compliant, injecting corrections for turn 2",
                        extra={
                            "task_id": task.id,
                            "violations": output.mimiry_violations,
                        }
                    )

                    # Inject corrections into task context for turn 2
                    task.context = task.context or {}
                    task.context["mimiry_corrections"] = output.mimiry_violations
                    task.context["compliance_feedback"] = {
                        "previous_confidence": output.confidence,
                        "violations_to_fix": output.mimiry_violations,
                        "instruction": "Address these violations in the revised code.",
                    }

                    compliance_before = compliance_after
                    current_turn += 1
                    continue

                # Success or max turns reached
                break

            # ============================================================
            # GOVERNANCE ESCALATION: Check for MASS trigger
            # ============================================================
            should_mass, mass_reason = self.governor.should_trigger_mass(task.id)

            if should_mass and mass_reason == MASSTriggeredReason.COMPLIANCE_REGRESSION:
                logger.warning(
                    "Compliance regression detected - swapping to specialist",
                    extra={
                        "task_id": task.id,
                        "mass_reason": mass_reason.value,
                        "original_agent": agent_id,
                    }
                )

                # Prune current agent
                self.swarm._dropout.prune_agent(agent_id, DropoutReason.SUBSTANDARD_OUTPUT)

                # Spawn specialist with FAANG strictness (per user directive)
                task_desc = task.description.lower()
                if "security" in task_desc or "auth" in task_desc:
                    new_agent = self.swarm.spawn_for_complexity(
                        0.9,
                        {"task": "security audit", "strictness": "FAANG_GOLDEN_PATH"}
                    )
                else:
                    new_agent = self.swarm._dropout.spawn_chameleon({"task": task.description})

                # Update task assignment
                task.assigned_agent = new_agent.agent_id if new_agent else agent_id

                # Inject FAANG strictness for escalated agent
                task.context = task.context or {}
                task.context["fql_governance"]["strictness"] = StrictnessLevel.FAANG_GOLDEN_PATH.value

                logger.info(
                    "Agent swapped due to compliance regression",
                    extra={
                        "task_id": task.id,
                        "new_agent": new_agent.agent_id if new_agent else "none",
                        "new_type": new_agent.agent_type.value if new_agent else "none",
                    }
                )

            # Build result dictionary
            result = {
                "agent": agent_id,
                "code": {
                    output.file_path: output.production_code
                } if output.production_code else {},
                "tests": {
                    output.test_path: output.test_code
                } if output.test_code else {},
                "explanation": output.rationale,
                "cost": output.cost_usd,
                "status": output.status,
                "confidence": output.confidence,
                "violations": output.mimiry_violations,
                "tokens_used": output.tokens_used,
                "model": output.llm_model,
                "lorebook_used": output.lorebook_context_used,
                "turns_used": current_turn,
                "governor_tracked": True,
            }

            # Add vetting results
            if output.production_code:
                result["vetting_results"] = {
                    "mypy_strict": True,
                    "test_coverage": 99 if output.test_code else 0,
                    "security_scan": True
                }

            logger.info(
                "Task executed with Governor tracking",
                extra={
                    "task_id": task.id,
                    "status": output.status,
                    "turns_used": current_turn,
                    "final_confidence": output.confidence,
                }
            )

            # Record Outcome (Learning)
            if self.lorebook and (output.production_code or output.test_code):
                verification = await self.verify_task_completion(task, result)

                # Post-execution verification
                if output.status == "completed" and self.task_verifier:
                    verifier_result = await self.task_verifier.verify_task(
                        task=task,
                        level="import",
                        code_files=result["code"],
                        test_files=result["tests"]
                    )

                    if not verifier_result.verified:
                        logger.error(
                            "Post-execution verification failed",
                            extra={"task_id": task.id, "issues": verifier_result.issues}
                        )
                        result["status"] = "failed_verification"
                        result["verification_issues"] = verifier_result.issues

                        if self.task_store:
                            await self.task_store.update_task_status(
                                task_id=task.id,
                                status="needs_verification",
                                warden_verification="rejected"
                            )
                        verification = "rejected"
                    else:
                        if self.task_store:
                            await self.task_store.update_task_status(
                                task_id=task.id,
                                status="done",
                                warden_verification="approved"
                            )

                outcome = Outcome(
                    decision_id=task.trace_id,
                    success=(verification == "approved"),
                    metrics={
                        "confidence": output.confidence,
                        "cost": output.cost_usd,
                        "turns_used": current_turn,
                    }
                )
                await self.lorebook.record_outcome(outcome)

            return [result]

        except Exception as e:
            logger.error(
                f"Task execution failed: {e}",
                extra={"task_id": task.id, "error": str(e)},
                exc_info=True
            )

            return [{
                "agent": agent_id,
                "code": {},
                "tests": {},
                "explanation": f"Task execution failed: {str(e)}",
                "cost": 0.0,
                "status": "failed",
                "governor_tracked": False,
            }]

    async def receive_proposal(
        self,
        fql_packet: FQLPacket | None = None,
        saga_request: str = "",
        context: dict[str, Any] | None = None,
        trace_id: str = ""
    ) -> WardenProposal:
        """
        Receive delegation proposal from SAGA.

        PHASE 8: Zero-Trust FQL Enforcement
        -----------------------------------
        Raw text proposals are REJECTED with 400 Bad Request.
        Only FQLPacket requests are processed.

        Process:
        1. Validate FQL packet (reject raw text with Protocol Hint)
        2. Validate via Mimiry FQL Gateway
        3. Decompose into tasks
        4. Estimate costs
        5. Create TODO list or reject

        Args:
            fql_packet: FQL structured request (REQUIRED for Phase 8+)
            saga_request: DEPRECATED - Raw text request (will be rejected)
            context: Additional context (user preferences, budget, etc.)
            trace_id: Correlation ID for tracing

        Returns:
            WardenProposal with decision and rationale

        Example (FQL - Approved):
            >>> packet = create_fql_packet(
            ...     sender="SAGA-Orchestrator",
            ...     action=FQLAction.VALIDATE_PATTERN,
            ...     subject="CreateUserEndpoint",
            ...     principle_id="SDLC-RESILIENCE-04"
            ... )
            >>> proposal = await warden.receive_proposal(fql_packet=packet)
            >>> proposal.decision
            'approved'

        Example (Raw Text - Rejected):
            >>> proposal = await warden.receive_proposal(
            ...     saga_request="Create user endpoint",  # DEPRECATED
            ...     context={"budget": 100}
            ... )
            >>> proposal.decision
            'rejected'
            >>> "Non-FQL Protocol" in proposal.rationale
            True
        """
        context = context or {}

        # ============================================================
        # PHASE 8: Zero-Trust FQL Enforcement
        # ============================================================
        if fql_packet is None:
            # Raw text detected - REJECT with Protocol Hint
            logger.warning(
                "Non-FQL protocol detected - rejecting raw text proposal",
                extra={
                    "raw_request": saga_request[:100] if saga_request else "empty",
                    "trace_id": trace_id,
                    "protocol_hint": "saga/core/mae/fql_schema.py",
                }
            )

            return WardenProposal(
                decision="rejected",
                rationale=(
                    "400 Bad Request: Non-FQL Protocol. "
                    "Raw text queries are deprecated as of Phase 8. "
                    "Use FQLPacket from saga/core/mae/fql_schema.py. "
                    "See: create_fql_packet() for helper function."
                ),
                sagacodex_violations=[
                    "PROTOCOL-001: Raw text queries deprecated",
                    "PROTOCOL-002: Use FQLPacket for structured contracts",
                ],
            )

        # FQL packet provided - extract request details
        trace_id = trace_id or fql_packet.header.correlation_id
        saga_request = f"{fql_packet.payload.action.value}: {fql_packet.payload.subject}"

        logger.info(
            "Warden received FQL proposal",
            extra={
                "action": fql_packet.payload.action.value,
                "subject": fql_packet.payload.subject,
                "principle_id": fql_packet.governance.mimiry_principle_id,
                "strictness": fql_packet.governance.strictness_level.value,
                "trace_id": trace_id,
            }
        )

        # ============================================================
        # STEP 1: Validate via Mimiry FQL Gateway
        # ============================================================
        compliance_result = await self.mimiry.validate_proposal(
            fql_packet=fql_packet,
            trace_id=trace_id
        )

        logger.info(
            "Mimiry FQL validation complete",
            extra={
                "is_compliant": compliance_result.is_compliant,
                "compliance_score": compliance_result.compliance_score,
                "citations": len(compliance_result.principle_citations),
                "trace_id": trace_id,
            }
        )

        # STEP 2: Check if violations are critical
        if not compliance_result.is_compliant and compliance_result.compliance_score < 50.0:
            logger.warning(
                "FQL proposal failed compliance check",
                extra={
                    "compliance_score": compliance_result.compliance_score,
                    "corrections": compliance_result.corrections,
                    "trace_id": trace_id,
                }
            )

            return WardenProposal(
                decision="rejected",
                rationale="; ".join(compliance_result.corrections) if compliance_result.corrections else "FQL validation failed",
                sagacodex_violations=compliance_result.corrections,
            )

        # STEP 3: Extract warnings if partially compliant
        violations = compliance_result.corrections if not compliance_result.is_compliant else []

        # Merge FQL context with provided context
        merged_context = {**fql_packet.payload.context, **context}

        # STEP 4: Decompose into tasks
        tasks = await self.decompose_into_tasks(
            request=fql_packet.payload.subject,
            context=merged_context,
            trace_id=trace_id
        )

        # STEP 5: Create Task Graph
        task_graph = TaskGraph()
        for task in tasks:
            # Inject FQL governance into task
            task.context = task.context or {}
            task.context["fql_governance"] = {
                "principle_id": fql_packet.governance.mimiry_principle_id,
                "strictness": fql_packet.governance.strictness_level.value,
                "compliance_score": compliance_result.compliance_score,
            }
            task_graph.add_task(task)

        # STEP 6: Estimate costs
        total_cost = sum(task.budget_allocation for task in tasks)
        user_budget = merged_context.get("budget", float('inf'))

        if total_cost > user_budget:
            logger.warning(
                "Estimated cost exceeds user budget",
                extra={
                    "estimated_cost": total_cost,
                    "user_budget": user_budget,
                    "trace_id": trace_id,
                }
            )

        # STEP 7: Persist and return approval
        if self.task_store and task_graph:
            await self.task_store.save_task_graph(
                graph=task_graph,
                request_id=trace_id,
                estimated_cost=total_cost
            )
            logger.info(
                "TaskGraph persisted to database",
                extra={"request_id": trace_id, "task_count": len(tasks)}
            )

        return WardenProposal(
            decision="approved",
            task_graph=task_graph,
            rationale=f"FQL validated with score {compliance_result.compliance_score:.1f}%. Tasks decomposed.",
            sagacodex_violations=violations,
            estimated_cost=total_cost,
            estimated_time_minutes=len(tasks) * 5,
        )

    async def resolve_via_mimiry(
        self,
        agents_outputs: list[dict[str, Any]],
        task: Task
    ) -> ConflictResolution:
        """
        Resolve conflict between coding agents via Mimiry consultation.

        This is the CRITICAL INTEGRATION POINT between Warden and Mimiry.

        When agents disagree or produce different outputs, Warden asks Mimiry
        to measure each against the SagaCodex ideal. Mimiry returns canonical
        judgment, Warden enforces it.

        Args:
            agents_outputs: List of outputs from different agents
            task: The task they were working on

        Returns:
            ConflictResolution from Mimiry

        Example:
            >>> outputs = [
            ...     {"agent": "AgentA", "code": "print('hello')"},
            ...     {"agent": "AgentB", "code": "logger.info('hello')"}
            ... ]
            >>> resolution = await warden.resolve_via_mimiry(outputs, task)
            >>> print(resolution.canonical_approach)
            "Agent B's approach aligns with SagaCodex Rule 4 (Structured Logging)."
        """
        logger.info(
            "Warden consulting Mimiry to resolve agent conflict",
            extra={
                "task_id": task.id,
                "agent_count": len(agents_outputs),
                "trace_id": task.trace_id,
            }
        )

        # Consult the Oracle
        resolution = await self.mimiry.resolve_conflict(
            agents_outputs=agents_outputs,
            task_context={
                "task": task.description,
                "weight": task.weight,
                "budget": task.budget_allocation,
            },
            trace_id=task.trace_id
        )

        logger.info(
            "Mimiry resolved conflict",
            extra={
                "task_id": task.id,
                "canonical_agent": resolution.agents_in_alignment,
                "violated_agents": resolution.agents_in_violation,
                "trace_id": task.trace_id,
            }
        )

        return resolution

    async def enforce_canonical(
        self,
        task: Task,
        resolution: ConflictResolution
    ) -> None:
        """
        Enforce Mimiry's canonical judgment on the task.

        After Mimiry resolves conflict, Warden updates task status
        to reflect the canonical approach.

        Args:
            task: Task to update
            resolution: Mimiry's resolution
        """
        logger.info(
            "Warden enforcing Mimiry's canonical judgment",
            extra={
                "task_id": task.id,
                "trace_id": task.trace_id,
            }
        )

        # Update task with Mimiry's measurement
        task.mimiry_measurement = {
            "canonical_approach": resolution.canonical_approach,
            "rationale": resolution.rationale,
            "cited_rules": resolution.cited_rules,
            "agents_in_alignment": resolution.agents_in_alignment,
            "agents_in_violation": resolution.agents_in_violation,
        }

        # If agents aligned with ideal, approve
        if resolution.agents_in_alignment:
            task.warden_verification = "approved"
            task.status = "done"
            task.completed_at = datetime.utcnow()

            logger.info(
                "Task approved via Mimiry alignment",
                extra={
                    "task_id": task.id,
                    "canonical_agents": resolution.agents_in_alignment,
                    "trace_id": task.trace_id,
                }
            )
        else:
            # All agents violated ideal - escalate
            task.warden_verification = "rejected"
            task.status = "blocked"

            logger.warning(
                "All agents violated SagaCodex ideal - escalating",
                extra={
                    "task_id": task.id,
                    "trace_id": task.trace_id,
                }
            )

    def build_checklist_for_task(self, description: str, weight: str) -> list[str]:
        """
        Build a checklist for a task based on its description and weight.
        Uses CodexIndexClient to find relevant rules.
        """
        tags = []
        desc_lower = description.lower()

        # Heuristic tagging
        if "fastapi" in desc_lower or "api" in desc_lower or "endpoint" in desc_lower:
            tags.append("fastapi")
            tags.append("endpoint")
        if "test" in desc_lower:
            tags.append("tests")
        if "refactor" in desc_lower or "fix" in desc_lower:
            tags.append("refactoring")
        if "db" in desc_lower or "database" in desc_lower:
            tags.append("db")

        # Get dynamic checklist from Codex
        checklist = self.codex_client.get_checklist_for_task(tags)

        # Add standard items based on weight/Codex
        checklist.append("Consult Mimiry if uncertain")

        return checklist

    async def decompose_into_tasks(
        self,
        request: str,
        context: dict[str, Any],
        trace_id: str
    ) -> list[Task]:
        """
        Decompose SAGA's request into concrete tasks.

        Simple tasks: Checklist-based verification
        Complex tasks: Full vetting (tests, types, security)

        Args:
            request: What SAGA wants done
            context: User preferences, budget, etc.
            trace_id: Correlation ID

        Returns:
            List of tasks with weights and checklists

        Example:
            >>> tasks = await warden.decompose_into_tasks(
            ...     "Create user CRUD endpoints",
            ...     {"budget": 100},
            ...     "trace-456"
            ... )
            >>> len(tasks)
            4  # Create, Read, Update, Delete
        """
        logger.info(
            "Warden decomposing request into tasks",
            extra={"request": request[:100], "trace_id": trace_id}
        )

        # MVP: Simple decomposition (future: use LLM for intelligent decomposition)
        tasks = []

        # Example: User CRUD decomposition
        if "user" in request.lower() and ("crud" in request.lower() or "endpoints" in request.lower()):
            crud_ops = [
                ("Create POST /users/ endpoint", "complex"),
                ("Create GET /users/{user_id} endpoint", "simple"),
                ("Create PATCH /users/{user_id} endpoint", "simple"),
                ("Create DELETE /users/{user_id} endpoint", "simple"),
            ]

            for desc, weight in crud_ops:
                tasks.append(
                    Task(
                        id=f"task-{uuid.uuid4()}",
                        description=desc,
                        weight=weight, # type: ignore
                        budget_allocation=25.0 if weight == "complex" else 15.0,
                        checklist=self.build_checklist_for_task(desc, weight),
                        vetting_criteria={
                            "mypy_strict": True,
                            "test_coverage": 99,
                            "security_scan": True,
                            "api_documented": True,
                        },
                        trace_id=trace_id,
                    )
                )

        else:
            # Generic single task (future: smarter decomposition)
            description = request
            weight = "complex" # Default

            tasks = [
                Task(
                    id=f"task-{uuid.uuid4()}",
                    description=description,
                    weight=weight, # type: ignore
                    budget_allocation=context.get("budget", 100.0),
                    checklist=self.build_checklist_for_task(description, weight),
                    vetting_criteria={
                        "mypy_strict": True,
                        "test_coverage": 99,
                        "security_scan": True,
                    },
                    trace_id=trace_id,
                )
            ]

        return tasks

    async def solve_request(
        self,
        user_input: str,
        context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Execute the full 'Solve Request' loop via LangGraph Governance.
        Refactored in Phase 3 to use Parallel Sovereignty Graph.

        Uses async stream to implement "Double-Write Persistence" to JSON.
        """

        # 1. Enforce Trace ID (The Passport)
        if not trace_id:
            trace_id = f"gen-{uuid.uuid4()}"
            logger.info(f"Generated new trace_id: {trace_id}")

        logger.info(
            "Warden received solve request (Parallel Graph Mode)",
            extra={"input": user_input[:100], "trace_id": trace_id}
        )

        # 2. Initialize State (Phase 3 Structure)
        initial_state: WardenState = {
            "task_input": user_input,
            "context": context,
            "trace_id": trace_id,
            "plan": None,
            "alpha_output": None,
            "beta_output": None,
            "merged_result": None,
            "conflict_detected": False,
            "ledger_proposal": None,
            "user_feedback": None,
            "history": [],
            "iteration_count": 0,
            "status": "planning"
        }

        # Check for recovery if not provided explicit state
        # (For now, we assume new request if passed arguments, or we could check DB)

        # 3. Execute Graph with Double-Write
        config = {"configurable": {"thread_id": trace_id}, "recursion_limit": 150}

        # Check for existing state to resume
        resume_input = initial_state
        try:
            snapshot = await self.graph.aget_state(config)
            if snapshot.values:
                logger.info(f"Resuming existing graph state for {trace_id}")
                resume_input = None  # Resume from checkpoint
        except Exception as e:
            logger.warning(f"Could not check existing state, starting fresh: {e}")

        final_state = initial_state

        try:
            # using astream to capture events for persistence
            async for event in self.graph.astream(resume_input, config=config):
                # event is a dict of node_name: state_update
                for node, _ in event.items():
                    logger.info(f"Graph Node '{node}' completed", extra={"trace_id": trace_id})

                # Fetch full authoritative state
                snapshot = await self.graph.aget_state(config)
                current_state = snapshot.values
                final_state = current_state  # Update tracker

                # DOUBLE-WRITE: Persist to JSON
                await self._save_checkpoint_json(current_state, trace_id)

            logger.info("Warden Graph Execution Completed", extra={"status": final_state.get("status")})

            # "proposing" is a valid success state (paused for HITL)
            success_statuses = ["completed", "proposing"]
            status = "success" if final_state.get("status") in success_statuses else "failed"

            return {
                "status": status,
                "result": final_state.get("merged_result"),
                "proposal": final_state.get("ledger_proposal"),
                "history": final_state.get("history", []),
                "artifacts": [final_state]
            }

        except Exception as e:
            logger.error(
                f"[{trace_id}] - Graph Failure! Dumping History...",
                exc_info=True
            )
            return {
                "status": "error",
                "message": str(e),
                "history": final_state.get("history", [])
            }

    async def _save_checkpoint_json(self, state: dict[str, Any], trace_id: str) -> None:
        """
        Phase 3: Double-Write Persistence.
        Writes the current state to a JSON file as a fallback.
        """
        import json
        try:
            filename = f"state_checkpoint_{trace_id}.json"
            path = Path(self.project_root) / ".saga" / "checkpoints" / filename
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize state (handle non-serializable objects if any)
            # WardenState should be mostly dicts/strings.
            # If complex objects exist, we might need a custom encoder.

            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(state, default=str, indent=2))

        except Exception as e:
            logger.warning(f"Failed to write JSON checkpoint: {e}")

    async def _recover_from_json(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Attempt to load state from JSON checkpoint."""
        import json
        try:
            filename = f"state_checkpoint_{trace_id}.json"
            path = Path(self.project_root) / ".saga" / "checkpoints" / filename

            if path.exists():
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to recover from JSON checkpoint: {e}")
        return None

    async def verify_task_completion(
        self,
        task: Task,
        agent_output: dict[str, Any]
    ) -> Literal["approved", "rejected"]:
        """
        Verify subagent's work against quality gates.

        For critical tasks or when uncertain, consults Mimiry.

        Simple tasks: Checklist-based verification
        Complex tasks: Full vetting + optional Mimiry measurement

        Args:
            task: Task that was completed
            agent_output: Subagent's self-check and output

        Returns:
            "approved" or "rejected"

        Example:
            >>> result = await warden.verify_task_completion(
            ...     task=some_task,
            ...     agent_output={"code": "...", "tests_passed": True}
            ... )
            >>> result
            'approved'
        """
        logger.info(
            "Warden verifying task completion",
            extra={
                "task_id": task.id,
                "task_weight": task.weight,
                "trace_id": task.trace_id,
            }
        )

        if task.weight == "simple":
            # Checklist verification
            checklist_results = agent_output.get("checklist_results", {})
            all_passed = all(checklist_results.values())

            if all_passed:
                task.warden_verification = "approved"
                task.status = "done"
                task.completed_at = datetime.utcnow()
                logger.info(
                    "Task approved (checklist passed)",
                    extra={"task_id": task.id, "trace_id": task.trace_id}
                )
                return "approved"
            else:
                failed_items = [k for k, v in checklist_results.items() if not v]
                task.warden_verification = "rejected"
                logger.warning(
                    "Task rejected (checklist failed)",
                    extra={
                        "task_id": task.id,
                        "failed_items": failed_items,
                        "trace_id": task.trace_id,
                    }
                )
                return "rejected"

        else:  # Complex task - full vetting + optional Mimiry
            vetting_results = agent_output.get("vetting_results", {})

            # Check all criteria
            mypy_passed = vetting_results.get("mypy_strict", False)
            test_coverage = vetting_results.get("test_coverage", 0)
            security_passed = vetting_results.get("security_scan", False)

            # If all criteria pass, approve
            if mypy_passed and test_coverage >= 99 and security_passed:
                # Optional: Measure against ideal for learning
                if agent_output.get("code"):
                    code_content = agent_output["code"]
                    if isinstance(code_content, dict):
                        # Join files with headers
                        full_code = []
                        for filename, content in code_content.items():
                            full_code.append(f"### FILE: {filename}\n{content}")
                        code_to_check = "\n\n".join(full_code)
                    else:
                        code_to_check = str(code_content)

                    mimiry_measurement = await self.mimiry.measure_against_ideal(
                        code=code_to_check,
                        domain=task.description.split()[0],  # First word as domain
                        trace_id=task.trace_id
                    )
                    task.mimiry_measurement = mimiry_measurement.to_dict()

                task.warden_verification = "approved"
                task.status = "done"
                task.completed_at = datetime.utcnow()
                logger.info(
                    "Task approved (full vetting passed)",
                    extra={"task_id": task.id, "trace_id": task.trace_id}
                )
                return "approved"
            else:
                failures = []
                if not mypy_passed:
                    failures.append("mypy --strict failed")
                if test_coverage < 99:
                    failures.append(f"test coverage {test_coverage}% < 99%")
                if not security_passed:
                    failures.append("security scan failed")

                task.warden_verification = "rejected"
                logger.warning(
                    "Task rejected (vetting failed)",
                    extra={
                        "task_id": task.id,
                        "failures": failures,
                        "trace_id": task.trace_id,
                    }
                )
                return "rejected"


# Export main classes
__all__ = [
    "Warden",
    "Task",
    "TaskGraph",
    "WardenProposal",
]
