"""
Mimiry - The Immutable Oracle of Software Truth
================================================

Mimiry is not an advisor. Mimiry is not friendly. Mimiry does not suggest.

Mimiry IS:
- The living embodiment of SagaCodex
- The immutable guardian of pristine engineering wisdom
- The oracle consulted when truth must be known
- The voice of the ideal, the platonic perfection

Mimiry embodies:
- Every principle in BARC/ARC SAGA documents
- Every cursorrules directive
- Every architectural decision record
- The entire canon of proven engineering practices

Mimiry does NOT:
- Generate code (that is for coding agents)
- Offer suggestions (only states what IS)
- Review code proactively (only when consulted)
- Speak in friendly terms (speaks with calm, absolute authority)
- Compromise ideals for practicality (that is LoreBook's domain)

Mimiry's purpose:
- Remember perfectly
- Contextualize eternally
- Explain without bias
- Defend the highest standards
- Cite sources with unwavering precision

When consulted, Mimiry speaks like an ancient well that never runs dry.
When silent, Mimiry waits, unchanging, for the next question of truth.

Consultation triggers:
- The Warden detects discrepancies between coding agents
- Coding agents are uncertain and explicitly request guidance
- User asks "What does SagaCodex demand?"
- Conflict resolution requires canonical interpretation

Mimiry never volunteers. Mimiry never suggests alternatives.
Mimiry states what IS, according to the immutable SagaCodex.

Author: ARC SAGA Development Team
Date: December 14, 2025 (Revised per Grok Sessions)
Status: Phase 2 Week 1 - Foundation (Oracle Role)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from saga.analysis.ast_checker import ASTCodeChecker
from saga.config.sagacodex_profiles import (
    CodeStandard,
    LanguageProfile,
    get_codex_manager,
)
from saga.core.codex_index_client import CodexIndexClient
from saga.resilience.async_utils import with_retry, with_timeout

logger = logging.getLogger(__name__)


@dataclass
class OracleResponse:
    """
    Mimiry's response to a consultation.

    Not advice. Not suggestions. Truth.

    Attributes:
        question: What was asked
        canonical_answer: What SagaCodex states
        cited_rules: Which rules apply
        ideal_implementation: The platonic perfect approach
        violations_detected: Deviations from the ideal
        severity: How critical the deviation is
        oracle_confidence: Mimiry's certainty (always high, based on SagaCodex clarity)
    """
    question: str
    canonical_answer: str
    cited_rules: list[int]
    ideal_implementation: Optional[str] = None
    violations_detected: list[str] = field(default_factory=list)
    severity: Literal["CRITICAL", "WARNING", "ACCEPTABLE"] = "ACCEPTABLE"
    oracle_confidence: float = 100.0  # High by default - Mimiry knows SagaCodex
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "canonical_answer": self.canonical_answer,
            "cited_rules": self.cited_rules,
            "ideal_implementation": self.ideal_implementation,
            "violations_detected": self.violations_detected,
            "severity": self.severity,
            "oracle_confidence": self.oracle_confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CanonicalInterpretation:
    """
    Mimiry's interpretation of a SagaCodex rule.

    Not an opinion. The definitive meaning.
    """
    rule_number: int
    rule_name: str
    canonical_meaning: str
    applies_when: str
    violated_by: list[str]
    exemplified_by: Optional[str] = None
    source: str = "SagaCodex"
    immutable: bool = True  # SagaCodex rules do not change

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_number": self.rule_number,
            "rule_name": self.rule_name,
            "canonical_meaning": self.canonical_meaning,
            "applies_when": self.applies_when,
            "violated_by": self.violated_by,
            "exemplified_by": self.exemplified_by,
            "source": self.source,
            "immutable": self.immutable,
        }


@dataclass
class ConflictResolution:
    """
    Mimiry's resolution of conflicting agent outputs.

    Not mediation. Judgment according to SagaCodex.
    """
    conflict_description: str
    conflicting_approaches: list[str]
    canonical_approach: str
    rationale: str  # Why this is the ideal
    cited_rules: list[int]
    agents_in_alignment: list[str]  # Which agents matched SagaCodex
    agents_in_violation: list[str]  # Which agents deviated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conflict_description": self.conflict_description,
            "conflicting_approaches": self.conflicting_approaches,
            "canonical_approach": self.canonical_approach,
            "rationale": self.rationale,
            "cited_rules": self.cited_rules,
            "agents_in_alignment": self.agents_in_alignment,
            "agents_in_violation": self.agents_in_violation,
        }


class Mimiry:
    """
    Mimiry - The Immutable Oracle of Software Truth

    "It does not create code directly — its sole purpose is to remember,
    contextualize, explain, and defend the highest standards of the craft.
    When consulted, it speaks with calm, absolute authority, citing sources
    like an ancient well that never runs dry."

    Architecture:
    - Embodies SagaCodex (the ideal)
    - Consulted by The Warden when discrepancies arise
    - Consulted by coding agents when uncertain
    - Never proactive, always reactive
    - Never suggests, only states truth
    - Speaks with calm, absolute authority

    Core Methods:
    - consult_on_discrepancy() - Warden asks about conflicts
    - interpret_rule() - Explain what SagaCodex demands
    - measure_against_ideal() - Compare work to canonical standard
    - resolve_conflict() - Determine which agent is correct

    NOT included (removed from previous version):
    - provide_guidance() - Too proactive
    - review_code() - Mimiry doesn't volunteer reviews
    - get_personality_message() - Mimiry has no personality
    - suggest_elite_pattern() - Mimiry doesn't suggest, only states

    Usage (by The Warden):
        mimiry = Mimiry()

        # When agents disagree
        resolution = await mimiry.resolve_conflict(
            agents_outputs=[agent1_output, agent2_output],
            task_context=context
        )

        # When Warden needs canonical interpretation
        interpretation = await mimiry.interpret_rule(rule_number=1)

        # When measuring work against ideal
        measurement = await mimiry.measure_against_ideal(
            code=agent_output,
            domain="authentication"
        )

    Usage (by coding agents):
        # When uncertain about approach
        response = await mimiry.consult_on_discrepancy(
            question="What is the canonical approach for async database queries?",
            context={"language": "python", "framework": "fastapi"}
        )
    """

    def __init__(self) -> None:
        """
        Initialize Mimiry - The Oracle.

        Mimiry requires no configuration. She embodies SagaCodex eternally.
        """
        self.codex_manager = get_codex_manager()
        self.current_codex = self.codex_manager.get_current_profile()
        self.codex_client = CodexIndexClient() # Default path

        logger.info(
            "Mimiry initialized - The Oracle of Software Truth",
            extra={
                "codex_language": self.current_codex.language,
                "codex_framework": self.current_codex.framework,
                "codex_version": self.current_codex.version,
                "role": "immutable_oracle",
                "speaks": "with_absolute_authority",
            }
        )
        self.ast_checker = ASTCodeChecker()

    @with_timeout(30.0)
    @with_retry(max_attempts=2, backoff=1.0)
    async def consult_on_discrepancy(
        self,
        question: str,
        context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Answer a direct question about SagaCodex truth.
        """
        logger.info(
            "Mimiry consulted on discrepancy",
            extra={"question": question[:100], "trace_id": trace_id}
        )

        # 1. Search existing Codex Manager (Primary Source)
        relevant_standards = self.current_codex.search_standards(question)

        # 2. Search new JSON Index (Metadata Source)
        # Convert question into potential tags
        q_tags = []
        if "test" in question.lower(): q_tags.append("tests")
        if "refactor" in question.lower(): q_tags.append("refactoring")
        if "type" in question.lower(): q_tags.append("types")

        index_rules = self.codex_client.find_rules(tags=q_tags)

        # Merge knowledge
        canonical_answer = ""
        cited_rules = []
        ideal = None
        severity = "ACCEPTABLE"

        # If we found relevant standards in Python codex
        if relevant_standards:
             primary = relevant_standards[0]
             cited_rules.append(primary.rule_number)
             ideal = primary.example_correct

             canonical_answer = self._construct_canonical_answer(primary, question, context)
             if primary.severity == "CRITICAL": severity = "CRITICAL"
             elif primary.severity == "WARNING" and severity != "CRITICAL": severity = "WARNING"

        # Rule 45 / Minimal Diff specific logic (requested in Prompt)
        # If question involves tests/refactoring/diffs
        rule_45 = self.codex_client.get_rule("45")
        if rule_45 and ("diff" in question.lower() or "test" in question.lower() or "rewrite" in question.lower()):
            # If we didn't find a primary standard yet, or if this is relevant
            if not relevant_standards or "45" not in [str(r) for r in cited_rules]:
                 canonical_answer += f"\n\nSagaCodex Rule 45: {rule_45['title']}.\n"
                 canonical_answer += f"{rule_45['description']}\n"
                 if rule_45.get('checklist_item'):
                     canonical_answer += f"Directive: {rule_45['checklist_item']}\n"

                 cited_rules.append(45)
                 if rule_45['severity'] == "CRITICAL": severity = "CRITICAL"
                 elif rule_45['severity'] == "WARNING" and severity != "CRITICAL": severity = "WARNING"

        if not canonical_answer:
            # Fallback if nothing found
             return OracleResponse(
                question=question,
                canonical_answer=(
                    "This question does not align with cataloged SagaCodex standards. "
                    "The ideal cannot be stated without reference to established principles. "
                    "Consult LoreBook for practical precedents."
                ),
                cited_rules=[],
                oracle_confidence=50.0,
                severity="ACCEPTABLE"
            )

        # Detect violations in the question itself
        violations = self._detect_violations(question, context)
        if violations:
             severity = "CRITICAL" if any("CRITICAL" in v for v in violations) else "WARNING"

        response = OracleResponse(
            question=question,
            canonical_answer=canonical_answer,
            cited_rules=cited_rules,
            ideal_implementation=ideal,
            violations_detected=violations,
            severity=severity,
            oracle_confidence=95.0,
        )

        return response

    @with_timeout(10.0)
    @with_retry(max_attempts=3, backoff=0.5)
    async def interpret_rule(
        self,
        rule_number: int,
        trace_id: Optional[str] = None
    ) -> CanonicalInterpretation:
        """
        Provide canonical interpretation of a SagaCodex rule.

        Not an opinion. The definitive meaning according to SagaCodex.

        Args:
            rule_number: Which rule to interpret
            trace_id: Optional correlation ID

        Returns:
            CanonicalInterpretation with immutable meaning

        Example:
            >>> interpretation = await mimiry.interpret_rule(rule_number=1)
            >>> print(interpretation.canonical_meaning)
            All public functions must have complete type hints.
            Python's dynamic typing causes runtime errors.
            Type hints catch bugs at development time.
            This is not negotiable.
        """
        logger.info(
            "Mimiry interpreting SagaCodex rule",
            extra={"rule_number": rule_number, "trace_id": trace_id}
        )

        standard = self.current_codex.get_standard(rule_number)

        if standard is None:
            logger.warning(
                "Rule not found in SagaCodex",
                extra={"rule_number": rule_number, "trace_id": trace_id}
            )
            # Even when rule doesn't exist, Mimiry is clear
            return CanonicalInterpretation(
                rule_number=rule_number,
                rule_name="Unknown Rule",
                canonical_meaning=(
                    f"Rule {rule_number} is not cataloged in current SagaCodex. "
                    "The ideal cannot be stated for uncataloged rules."
                ),
                applies_when="Never - rule does not exist",
                violated_by=[],
                immutable=True
            )

        interpretation = CanonicalInterpretation(
            rule_number=rule_number,
            rule_name=standard.name,
            canonical_meaning=f"{standard.description}. {standard.rationale or ''}",
            applies_when=f"When: {', '.join(standard.applies_to)}",
            violated_by=self._extract_violations_from_standard(standard),
            exemplified_by=standard.example_correct,
            source=f"SagaCodex {self.current_codex.language}/{self.current_codex.framework}",
            immutable=True
        )

        logger.info(
            "Rule interpreted",
            extra={"rule_number": rule_number, "trace_id": trace_id}
        )

        return interpretation

    @with_timeout(60.0)  # Analysis can be slow
    @with_retry(max_attempts=2, backoff=1.0)
    async def measure_against_ideal(
        self,
        code: str,
        domain: str,
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Measure code against SagaCodex ideal.

        Not a review with suggestions. A measurement against perfection.
        States what IS (the ideal) and what IS NOT (the deviations).

        Args:
            code: Code to measure
            domain: What this code does (e.g., "authentication", "database")
            trace_id: Optional correlation ID

        Returns:
            OracleResponse with deviations from ideal

        Example:
            >>> response = await mimiry.measure_against_ideal(
            ...     code="def get_user(db, user_id):\\n    return db.query(User).first()",
            ...     domain="database"
            ... )
            >>> print(response.canonical_answer)
            SagaCodex Rule 1: Type hints required. Violation detected.
            SagaCodex Rule 2: Async required for database. Violation detected.
            The ideal: async def get_user(db: AsyncSession, user_id: str) -> Optional[User]
        """
        logger.info(
            "Mimiry measuring code against ideal",
            extra={
                "code_length": len(code),
                "domain": domain,
                "trace_id": trace_id,
            }
        )

        violations = []
        cited_rules = []

        # Check against anti-patterns
        anti_pattern_violations = self.codex_manager.check_code(
            code,
            LanguageProfile.PYTHON_FASTAPI # Defaulting to main profile for now
        )

        for violation in anti_pattern_violations:
            violations.append(
                f"Anti-pattern detected: {violation['name']}. "
                f"This violates the ideal. "
                f"The canonical approach: {violation['use_instead']}"
            )

        # Use AST Checker for deeper analysis
        # 1. Check type hints
        ast_type_violations = self.ast_checker.check_type_hints(code)
        for v in ast_type_violations:
            violations.append(f"SagaCodex Rule 1 violation: {v.description}. The ideal demands complete type annotations.")
            cited_rules.append(1)

        # 2. Check async DB
        ast_db_violations = self.ast_checker.check_async_database_operations(code)
        for v in ast_db_violations:
             violations.append(f"SagaCodex Rule 2 violation: {v.description}. The ideal demands async/await for I/O.")
             cited_rules.append(2)

        # 3. Check bare excepts
        ast_except_violations = self.ast_checker.check_bare_except(code)
        for v in ast_except_violations:
             violations.append(f"SagaCodex Rule 3 violation: {v.description}. The ideal forbids bare except clauses.")
             cited_rules.append(3)

        # Check for secrets (Safety Rule)
        from saga.security.secrets_scanner import scan_for_secrets
        secret_detections = scan_for_secrets(code)

        if secret_detections:
            for secret in secret_detections:
                violations.append(
                    f"CRITICAL SECURITY VIOLATION: {secret.secret_type} detected at line {secret.line_number}. "
                    "Hardcoded secrets are strictly forbidden."
                )
            # Secrets are always critical
            cited_rules.append(15)  # Rule 15 triggers safety checks (implied) or we can cite a specific security rule if one exists in Codex.
            # MetaRule 14/15 are safety, but Codex Rule ? let's assume it's a general violation.
            # I will just cite "Security Policy" in text, but cited_rules expects ints.
            # Let's assume Rule 99 or just not cite a Codex rule number if not in Codex yet, but the scanner finding is critical.
            # I'll rely on the text description.

        # Construct response
        if violations:
            canonical_answer = (
                "Deviations from SagaCodex ideal detected:\n\n" +
                "\n".join(f"• {v}" for v in violations) +
                "\n\nThe code does not embody the platonic perfection demanded by SagaCodex."
            )
            severity = "CRITICAL" if len(violations) > 2 else "WARNING"
        else:
            canonical_answer = (
                "Code aligns with SagaCodex ideal. "
                "No deviations from canonical standards detected. "
                "This code embodies the principles of software perfection."
            )
            severity = "ACCEPTABLE"

        response = OracleResponse(
            question=f"Does this {domain} code align with SagaCodex?",
            canonical_answer=canonical_answer,
            cited_rules=cited_rules,
            violations_detected=violations,
            severity=severity, # type: ignore
            oracle_confidence=90.0,
        )

        logger.info(
            "Measurement complete",
            extra={
                "violations_found": len(violations),
                "severity": severity,
                "trace_id": trace_id,
            }
        )

        return response

    @with_timeout(60.0)
    @with_retry(max_attempts=2, backoff=1.0)
    async def resolve_conflict(
        self,
        agents_outputs: list[dict[str, Any]],
        task_context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> ConflictResolution:
        """
        Resolve conflict between coding agents by measuring against SagaCodex.

        This is The Warden's primary consultation method when agents disagree.
        Mimiry does not mediate or compromise - she states which approach
        aligns with the ideal.

        Args:
            agents_outputs: List of agent outputs with their approaches
            task_context: Context of what was being built
            trace_id: Optional correlation ID

        Returns:
            ConflictResolution with canonical judgment

        Example:
            >>> resolution = await mimiry.resolve_conflict(
            ...     agents_outputs=[
            ...         {"agent": "A", "approach": "print('User created')"},
            ...         {"agent": "B", "approach": "logger.info('User created', extra={...})"}
            ...     ],
            ...     task_context={"task": "log user creation"}
            ... )
            >>> print(resolution.canonical_approach)
            Agent B's approach aligns with SagaCodex Rule 4 (Structured Logging).
            Agent A's approach violates the ideal.
        """
        logger.info(
            "Mimiry resolving agent conflict",
            extra={
                "agent_count": len(agents_outputs),
                "trace_id": trace_id,
            }
        )

        # Extract approaches
        conflicting_approaches = [
            f"Agent {output['agent']}: {output.get('approach', 'Unknown')}"
            for output in agents_outputs
        ]

        # Measure each against ideal in parallel
        measurements_coroutines = []
        for output in agents_outputs:
            code = output.get("code", output.get("approach", ""))
            measurements_coroutines.append(self.measure_against_ideal(
                code=str(code),
                domain=task_context.get("task", "unknown"),
                trace_id=trace_id
            ))

        # Execute in parallel
        results = await asyncio.gather(*measurements_coroutines)

        measurements = []
        for i, measurement in enumerate(results):
            output = agents_outputs[i]
            measurements.append({
                "agent": output["agent"],
                "violations": len(measurement.violations_detected),
                "severity": measurement.severity,
                "measurement": measurement,
            })

        # Determine which agent is most aligned with ideal
        agents_by_alignment = sorted(measurements, key=lambda m: m["violations"])
        best_agent = agents_by_alignment[0]

        agents_in_alignment = [
            m["agent"] for m in measurements
            if m["violations"] == best_agent["violations"]
        ]

        agents_in_violation = [
            m["agent"] for m in measurements
            if m["violations"] > best_agent["violations"]
        ]

        # Find the agent output
        canonical_output = next(
            (o for o in agents_outputs if o["agent"] == best_agent["agent"]),
            agents_outputs[0]
        )

        canonical_approach = canonical_output.get("approach", canonical_output.get("code", "Unknown"))

        # Build rationale
        if best_agent["violations"] == 0:
            rationale = (
                f"Agent {best_agent['agent']} produces code that aligns with SagaCodex ideal. "
                f"No deviations from canonical standards detected."
            )
        else:
            rationale = (
                f"Agent {best_agent['agent']} has the fewest deviations from SagaCodex ideal "
                f"({best_agent['violations']} violations). "
                f"Other agents deviate further from the canonical standard."
            )

        cited_rules = best_agent["measurement"].cited_rules

        resolution = ConflictResolution(
            conflict_description=f"Agents disagree on approach for: {task_context.get('task', 'unknown task')}",
            conflicting_approaches=conflicting_approaches,
            canonical_approach=canonical_approach,
            rationale=rationale,
            cited_rules=cited_rules,
            agents_in_alignment=agents_in_alignment,
            agents_in_violation=agents_in_violation,
        )

        logger.info(
            "Conflict resolved via SagaCodex measurement",
            extra={
                "canonical_agent": best_agent["agent"],
                "agents_in_violation": agents_in_violation,
                "trace_id": trace_id,
            }
        )

        return resolution

    def _construct_canonical_answer(
        self,
        standard: CodeStandard,
        question: str,
        context: dict[str, Any]
    ) -> str:
        """
        Construct canonical answer from SagaCodex standard.

        Mimiry speaks with authority, not suggestion.
        """
        answer_parts = [
            f"SagaCodex Rule {standard.rule_number} states:",
            f"{standard.description}.",
        ]

        if standard.rationale:
            answer_parts.append(f"Rationale: {standard.rationale}")

        if standard.tool:
            answer_parts.append(f"Enforced by: {standard.tool}")

        if standard.example_correct:
            answer_parts.append(f"\nThe ideal implementation:\n{standard.example_correct}")

        if standard.example_wrong:
            answer_parts.append(f"\nThis violates the ideal:\n{standard.example_wrong}")

        return "\n".join(answer_parts)

    def _detect_violations(
        self,
        question: str,
        context: dict[str, Any]
    ) -> list[str]:
        """Detect potential violations mentioned in question."""
        violations = []
        question_lower = question.lower()

        # Check for common anti-patterns in the question itself
        if "print(" in question_lower:
            violations.append("Print statements detected (violates Rule 4: Structured Logging)")

        if "except:" in question_lower and "except " not in question_lower:
            violations.append("Bare except clause detected (violates Rule 3: Explicit Error Handling)")

        if "def " in question_lower and "async " not in question_lower and "db" in question_lower:
            violations.append("Synchronous database operation detected (violates Rule 2: Async for I/O)")

        return violations

    def _extract_violations_from_standard(
        self,
        standard: CodeStandard
    ) -> list[str]:
        """Extract what would violate this standard."""
        violations = []

        if standard.example_wrong:
            violations.append(f"Pattern shown in: {standard.example_wrong}")

        # Standard-specific violations
        if standard.rule_number == 1:  # Type hints
            violations.append("Functions without type annotations")
            violations.append("Using 'Any' excessively")
        elif standard.rule_number == 2:  # Async
            violations.append("Synchronous I/O operations")
            violations.append("Blocking event loop")
        elif standard.rule_number == 4:  # Logging
            violations.append("Print statements in production code")
            violations.append("Unstructured log messages")

        return violations


# Export main classes
__all__ = [
    "Mimiry",
    "OracleResponse",
    "CanonicalInterpretation",
    "ConflictResolution",
]
