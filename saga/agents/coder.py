"""
Coding Agent - Production Code Generator
=========================================

Generates SagaCodex-compliant code via LLM, validates with Mimiry.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3B - Agent Execution Framework
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from saga.config.sagacodex_profiles import LanguageProfile, SagaCodex
from saga.core.lorebook import LoreBook, Pattern
from saga.core.mimiry import Mimiry
from saga.core.task import Task
from saga.llm.client import LLMClient
from saga.llm.prompts import ExtractedCode, PromptBuilder
from saga.resilience.async_utils import with_retry, with_timeout

logger = logging.getLogger(__name__)

# Use LLMClient from original import but alias if needed.
# The detailed code specified "UniversalLLMClient" but we have "LLMClient" in saga/llm/client.py
# We will use LLMClient.

@dataclass
class AgentOutput:
    """Output from coding agent."""

    task_id: str
    status: str

    # Generated code
    production_code: str = ""
    test_code: str = ""
    file_path: str = ""
    test_path: str = ""

    # Metadata
    rationale: str = ""
    confidence: float = 0.0
    mimiry_violations: list[str] = field(default_factory=list)

    # Cost tracking
    cost_usd: float = 0.0
    tokens_used: int = 0

    # LLM details
    # We store basic details, not the full object to keep it serializable if needed
    llm_model: str = ""


class CodingAgent:
    """
    Generates production code via LLM.

    Workflow:
        1. Build SagaCodex-enforcing prompt
        2. Call LLM (via LLMClient)
        3. Extract code blocks
        4. Self-validate with Mimiry
        5. Return AgentOutput

    Usage:
        agent = CodingAgent(
            llm_client=client,
            lorebook=lorebook,
            sagacodex_profile=profile
        )

        output = await agent.solve_task(task, project_context)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        lorebook: Optional[LoreBook] = None,
        mimiry: Optional[Mimiry] = None,
        sagacodex_profile: Optional[SagaCodex] = None,
        agent_name: str = "coding_agent",
        prompt_engine: Any = None # Compatibility arg
    ):
        """
        Initialize coding agent.

        Args:
            llm_client: Universal LLM client
            lorebook: LoreBook for learned patterns
            mimiry: Mimiry oracle for validation
            sagacodex_profile: SagaCodex standards
            agent_name: Agent name for LLM config
        """
        self.llm_client = llm_client
        self.lorebook = lorebook
        self.mimiry = mimiry or Mimiry()
        self.agent_name = agent_name

        # Load SagaCodex profile
        if sagacodex_profile:
            self.sagacodex = sagacodex_profile
        else:
            from saga.config.sagacodex_profiles import get_codex_manager
            # Create a manager to get the profile
            manager = get_codex_manager()
            self.sagacodex = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(self.sagacodex)

    @with_timeout(120)
    @with_retry(max_attempts=2, backoff=2.0)
    async def solve_task(
        self,
        task: Task,
        project_root: str, # Compatibility arg
        language_profile: str = "python_fastapi", # Compatibility arg
        critical: bool = False, # Compatibility arg
    ) -> AgentOutput:
        """
        Execute task - generate code via LLM.

        Args:
            task: Task to execute
            project_root: Project root path
            language_profile: Profile name
            critical: Critical tasks use verification

        Returns:
            AgentOutput with generated code and validation results
        """
        logger.info(
            "CodingAgent executing task",
            extra={
                "task_id": task.id,
                "description": task.description,
                "agent": self.agent_name
            }
        )

        project_context = {
            "root": project_root,
            "profile": language_profile
        }

        # 1. Get LoreBook patterns
        lorebook_patterns = await self._get_lorebook_patterns()

        # 2. Build prompt
        messages = self.prompt_builder.build_messages(
            task=task,
            lorebook_patterns=lorebook_patterns,
            project_context=project_context
        )

        # 3. Call LLM
        try:
            if critical:
                # Use verification flow if critical
                verified_response = await self.llm_client.chat_with_verification(
                    messages=messages,
                    task_type="coding_critical"
                )
                llm_response = verified_response.canonical_response
            else:
                llm_response = await self.llm_client.chat(
                    messages=messages
                )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return AgentOutput(
                task_id=task.id,
                status="failed",
                rationale=f"LLM call failed: {str(e)}"
            )

        # 4. Extract code
        try:
            extracted = self.prompt_builder.extract_code(llm_response.text)
        except ValueError as e:
            logger.error(f"Code extraction failed: {e}")
            # Try to return raw response for debugging
            return AgentOutput(
                task_id=task.id,
                status="failed",
                rationale=f"Code extraction failed: {str(e)}\n\nRaw: {llm_response.text[:200]}...",
                cost_usd=getattr(llm_response, "cost", 0.0),
                tokens_used=getattr(llm_response, "usage", {}).get("total_tokens", 0)
            )

        # 5. Self-validate with Mimiry
        violations = await self._validate_with_mimiry(extracted)

        # 6. Calculate confidence
        confidence = self._calculate_confidence(extracted, violations)

        # 7. Build output
        status = "completed" if confidence > 0.7 else "needs_review"

        output = AgentOutput(
            task_id=task.id,
            status=status,
            production_code=extracted.production_code,
            test_code=extracted.test_code,
            file_path=extracted.file_path,
            test_path=extracted.test_path,
            rationale=extracted.rationale,
            confidence=confidence,
            mimiry_violations=violations,
            cost_usd=getattr(llm_response, "cost", 0.0),
            tokens_used=getattr(llm_response, "usage", {}).get("total_tokens", 0),
            llm_model=llm_response.model
        )

        logger.info(
            "CodingAgent completed task",
            extra={
                "task_id": task.id,
                "status": status,
                "confidence": confidence,
                "violations": len(violations),
                "cost_usd": output.cost_usd
            }
        )

        return output

    # Compatibility properties for Warden outputs
    # Warden expects .code_files dict, .tests dict
    # We can add a property/adapter if needed.
    # But easier to update Warden to accept AgentOutput directly or update AgentOutput
    # to include those fields as @property.

    async def _get_lorebook_patterns(self) -> list[Pattern]:
        """Get learned patterns from LoreBook."""
        if not self.lorebook:
            return []

        try:
            patterns = await self.lorebook.get_project_patterns()
            return patterns[:5]  # Top 5 patterns
        except Exception as e:
            logger.warning(f"Failed to get LoreBook patterns: {e}")
            return []

    async def _validate_with_mimiry(self, extracted: ExtractedCode) -> list[str]:
        """
        Validate code against SagaCodex via Mimiry.

        Args:
            extracted: Extracted code

        Returns:
            List of violations (empty if valid)
        """
        if not self.mimiry:
            return []

        try:
            # Simple validation: check for SagaCodex violations
            # In Phase 3B we do simple heuristic checks here as a pre-commit hook
            violations = []

            code = extracted.production_code

            # Check 1: Type hints
            if "def " in code and "->" not in code:
                violations.append("Missing return type hints on functions")

            # Check 2: Async I/O
            if any(word in code.lower() for word in ["database", "redis", "http"]) and "async def" not in code:
                violations.append("I/O operations should use async/await")

            # Check 3: Pydantic
            if "FastAPI" in code and "BaseModel" not in code:
                violations.append("FastAPI endpoints should use Pydantic models")

            # Check 4: Logging
            if "print(" in code:
                violations.append("Use logger.info() instead of print()")

            # Check 5: Tests
            if not extracted.test_code:
                violations.append("No tests generated")

            return violations

        except Exception as e:
            logger.warning(f"Mimiry validation failed: {e}")
            return ["Mimiry validation error"]

    def _calculate_confidence(
        self,
        extracted: ExtractedCode,
        violations: list[str]
    ) -> float:
        """
        Calculate confidence score for generated code.

        Factors:
            - SagaCodex violations (0.2 penalty each)
            - Code length (too short = low confidence)
            - Test coverage estimate
        """
        confidence = 1.0

        # Penalty for violations
        confidence -= len(violations) * 0.15

        # Penalty for short code (likely incomplete)
        if len(extracted.production_code) < 100:
            confidence -= 0.3

        # Penalty for no tests
        if len(extracted.test_code) < 50:
            confidence -= 0.2

        # Bonus for rationale
        if len(extracted.rationale) > 20:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))
