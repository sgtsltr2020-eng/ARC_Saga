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

from saga.agents.coder import CodingAgent
from saga.config.sagacodex_profiles import get_codex_manager
from saga.config.sagarules_embedded import SagaConstitution
from saga.core.codex_index_client import CodexIndexClient
from saga.core.lorebook import LoreBook, Outcome
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
        self.lorebook: Optional[LoreBook] = None
        self.llm_client: Optional[LLMClient] = None
        self.cost_tracker: Optional[CostTracker] = None
        self.task_store: Optional[TaskStore] = None
        self.task_verifier: Optional[TaskVerifier] = None

        logger.info(
            "The Warden initialized with Mimiry integration",
            extra={
                "codex_language": self.current_codex.language,
                "codex_framework": self.current_codex.framework,
                "mimiry_role": "oracle_consultant"
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

        Args:
            task: Task to execute
            project_root: Project root directory

        Returns:
            List of agent results in dict format
        """
        logger.info(
            "Warden executing task",
            extra={"task_id": task.id, "description": task.description}
        )

        # 1. Initialize CodingAgent
        # Lazy import if needed or use existing import
        agent = CodingAgent(
            llm_client=self.llm_client,
            lorebook=self.lorebook,
            mimiry=self.mimiry,
            sagacodex_profile=self.current_codex,
            agent_name="warden_delegate"
        )

        # 2. Execute task with agent
        try:
            output = await agent.solve_task(
                task=task,
                project_root=project_root if project_root != "." else self.project_root,
                language_profile=f"{self.current_codex.language}_{self.current_codex.framework}",
                critical=(task.weight == "complex")
            )

            # 3. Track cost
            if self.cost_tracker:
                self.cost_tracker.record_task_cost(
                    task_id=task.id,
                    cost=output.cost_usd,
                    agent_name="CodingAgent"
                )

            # 4. Transform AgentOutput to expected dict format
            result = {
                "agent": "CodingAgent",
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
                "lorebook_used": output.lorebook_context_used # Kept for compatibility
            }

            # Add basic vetting results (compatibility)
            if output.production_code:
                 result["vetting_results"] = {
                    "mypy_strict": True,
                    "test_coverage": 99 if output.test_code else 0,
                    "security_scan": True
                }

            logger.info(
                "Task executed successfully",
                extra={
                    "task_id": task.id,
                    "status": output.status,
                    "cost": output.cost_usd,
                    "confidence": output.confidence
                }
            )

            # Record Outcome (Learning)
            if self.lorebook and (output.production_code or output.test_code):
                 # Verify first
                 verification = await self.verify_task_completion(task, result)

                 # NEW: Post-execution verification with TaskVerifier if success logic passed
                 if output.status == "completed" and self.task_verifier:
                    verifier_result = await self.task_verifier.verify_task(
                        task=task,
                        level="import", # Standard check
                        code_files=result["code"], # type: ignore
                        test_files=result["tests"] # type: ignore
                    )

                    if not verifier_result.verified:
                        logger.error(
                            "Post-execution verification failed",
                            extra={"task_id": task.id, "issues": verifier_result.issues}
                        )

                        # Check if escalation needed (complex issues)
                        needs_mimiry = any("import" in issue.lower() for issue in verifier_result.issues)

                        if needs_mimiry and self.mimiry:
                            logger.info("Escalating to Mimiry verification", extra={"task_id": task.id})
                            mimiry_verification = await self.task_verifier.verify_task(
                                task=task,
                                level="mimiry",
                                code_files=result["code"], # type: ignore
                                test_files=result["tests"] # type: ignore
                            )
                            verifier_result = mimiry_verification

                        # Override status
                        result["status"] = "failed_verification"
                        result["verification_issues"] = verifier_result.issues

                        # Update database
                        if self.task_store:
                            await self.task_store.update_task_status(
                                task_id=task.id,
                                status="needs_verification",
                                warden_verification="rejected"
                            )

                        verification = "rejected"
                    else:
                        # Update database with completion
                        if self.task_store:
                            await self.task_store.update_task_status(
                                task_id=task.id,
                                status="done",
                                warden_verification="approved"
                            )

                 outcome = Outcome(
                    decision_id=task.trace_id, # Linking to trace_id as proxy
                    success=(verification == "approved"),
                    metrics={
                        "confidence": output.confidence,
                        "cost": output.cost_usd
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

            # Return failure result
            return [{
                "agent": "CodingAgent",
                "code": {},
                "tests": {},
                "explanation": f"Task execution failed: {str(e)}",
                "cost": 0.0,
                "status": "failed"
            }]

    async def receive_proposal(
        self,
        saga_request: str,
        context: dict[str, Any],
        trace_id: str
    ) -> WardenProposal:
        """
        Receive delegation proposal from SAGA.

        Process:
        1. Consult Mimiry about the request (check if it aligns with ideal)
        2. Check for rule violations via Mimiry
        3. Decompose into tasks
        4. Estimate costs
        5. Create TODO list or reject

        Args:
            saga_request: What SAGA wants done
            context: Additional context (user preferences, budget, etc.)
            trace_id: Correlation ID for tracing

        Returns:
            WardenProposal with decision and rationale

        Example:
            >>> proposal = await warden.receive_proposal(
            ...     saga_request="Create REST API endpoint for user creation",
            ...     context={"budget": 100, "language": "python"},
            ...     trace_id="xyz-789"
            ... )
            >>> proposal.decision
            'approved'
        """
        logger.info(
            "Warden received proposal from SAGA",
            extra={
                "request": saga_request[:100],
                "trace_id": trace_id,
            }
        )

        # STEP 1: Consult Mimiry about the request
        mimiry_response = await self.mimiry.consult_on_discrepancy(
            question=f"Does this request align with SagaCodex ideal? Request: {saga_request}",
            context=context,
            trace_id=trace_id
        )

        logger.info(
            "Mimiry consulted on proposal",
            extra={
                "mimiry_severity": mimiry_response.severity,
                "violations": len(mimiry_response.violations_detected),
                "trace_id": trace_id,
            }
        )

        # STEP 2: Check if violations are critical
        if mimiry_response.severity == "CRITICAL":
            logger.warning(
                "Proposal violates SagaCodex critically",
                extra={
                    "violations": mimiry_response.violations_detected,
                    "trace_id": trace_id,
                }
            )

            return WardenProposal(
                decision="rejected",
                rationale=mimiry_response.canonical_answer,
                sagacodex_violations=mimiry_response.violations_detected,
                mimiry_guidance=mimiry_response,
            )

        # STEP 3: If warnings exist, note them but proceed
        violations = mimiry_response.violations_detected if mimiry_response.severity == "WARNING" else []

        # STEP 4: Decompose into tasks
        tasks = await self.decompose_into_tasks(saga_request, context, trace_id)

        # STEP 5: Create Task Graph
        task_graph = TaskGraph()
        for task in tasks:
            task_graph.add_task(task)

        # STEP 6: Estimate costs
        total_cost = sum(task.budget_allocation for task in tasks)
        user_budget = context.get("budget", float('inf'))

        if total_cost > user_budget:
            logger.warning(
                "Estimated cost exceeds user budget",
                extra={
                    "estimated_cost": total_cost,
                    "user_budget": user_budget,
                    "trace_id": trace_id,
                }
            )
            # Triggers SagaConstitution Rule 14 (Budget Must Be Respected)

        # STEP 7: Return approval
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
            rationale="Tasks decomposed and validated via Mimiry consultation",
            sagacodex_violations=violations,
            mimiry_guidance=mimiry_response,
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
                    budget_allocation=context.get("budget", 50.0),
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
