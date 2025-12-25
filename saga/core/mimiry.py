from __future__ import annotations

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
import inspect
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from saga.analysis.ast_checker import ASTCodeChecker
from saga.config.sagacodex_profiles import (
    CodeStandard,
    LanguageProfile,
    get_codex_manager,
)
from saga.core.codex_index_client import CodexIndexClient
from saga.knowledge.retriever import HybridRetriever
from saga.knowledge.vector_store import ChromaVectorStore, DocumentChunk
from saga.resilience.async_utils import with_retry, with_timeout

# Conditional import to avoid circular dependency
if TYPE_CHECKING:
    from saga.core.mae.fql_schema import (
        ComplianceResult,
        FQLPacket,
        PrincipleCitation,
        RejectedAlternative,
        ValidationCache,
    )

logger = logging.getLogger(__name__)


@dataclass
class Directive:
    """
    A compressed, actionable instruction derived from SagaCodex.
    "Contextual Compression" - reduces token usage for agents.
    """
    title: str
    instruction: str
    priority: Literal["CRITICAL", "HIGH", "MEDIUM"]
    source_rule: int


@dataclass
class OracleResponse:
    """
    Mimiry's response to a consultation.

    Not advice. Not suggestions. Truth.

    Attributes:
        question: What was asked
        canonical_answer: What SagaCodex states
        directives: Actionable compressed instructions (Contextual Compression)
        cited_rules: Which rules apply
        ideal_implementation: The platonic perfect approach
        violations_detected: Deviations from the ideal
        severity: How critical the deviation is
        oracle_confidence: Mimiry's certainty
    """
    question: str
    canonical_answer: str
    directives: list[Directive] = field(default_factory=list)
    cited_rules: list[int] = field(default_factory=list)
    ideal_implementation: Optional[str] = None
    violations_detected: list[str] = field(default_factory=list)
    severity: Literal["CRITICAL", "WARNING", "ACCEPTABLE"] = "ACCEPTABLE"
    oracle_confidence: float = 100.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "canonical_answer": self.canonical_answer,
            "directives": [
                {"title": d.title, "instruction": d.instruction, "priority": d.priority, "source_rule": d.source_rule}
                for d in self.directives
            ],
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
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        retriever: Optional[HybridRetriever] = None
    ) -> None:
        """
        Initialize Mimiry - The Oracle of Software Truth.
        Now powered by Advanced RAG (ChromaDB + BM25).
        """
        self.codex_manager = get_codex_manager()
        self.current_codex = self.codex_manager.get_current_profile()
        self.codex_client = CodexIndexClient()

        # Initialize Advanced RAG Stack
        # Allow dependency injection for testing
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = retriever or HybridRetriever(self.vector_store)

        # Ingest Codex immediately on startup (for now)
        # In prod this would be async or pre-built
        self._ingest_codex()

        # FQL Gateway: Validation cache for efficient repeat queries
        # Lazy-loaded to avoid circular imports
        self._validation_cache: Optional[ValidationCache] = None
        self._fql_enabled = True  # Zero-trust mode by default

        logger.info(
            "Mimiry initialized - The Semantic Oracle with FQL Gateway",
            extra={
                "role": "immutable_oracle",
                "fql_gateway": "enabled",
                "memory_tier": "semantic",
                "rag_engine": "hybrid_fusion"
            }
        )
        self.ast_checker = ASTCodeChecker()

    def _ingest_codex(self):
        """
        Ingest SagaCodex into the Tri-Tier Memory.
        Implements 'Parent-Document' splitting strategy.
        """
        chunks = []
        parent_docs = {}
        bm25_corpus = []
        bm25_ids = []

        for std in self.current_codex.standards:
            # 1. Create Parent Document (Full Context)
            parent_id = f"rule-{std.rule_number}-parent"
            full_text = (
                f"SagaCodex Rule {std.rule_number}: {std.name}\n"
                f"Description: {std.description}\n"
                f"Rationale: {std.rationale or ''}\n"
                f"Tooling: {std.tool or 'None'}\n"
                f"Ideal: {std.example_correct}\n"
                f"Violation: {std.example_wrong}"
            )
            parent_docs[parent_id] = full_text

            # 2. Create Child Chunks (Precise Search Targets)
            # Chunk A: The Rule itself
            chunks.append(DocumentChunk(
                doc_id=f"rule-{std.rule_number}-desc",
                content=f"Rule {std.rule_number} {std.name}: {std.description}",
                metadata={"rule_number": std.rule_number, "type": "rule_def"},
                parent_doc_id=parent_id
            ))

            # Chunk B: The Violation (allows finding rules by describing the error)
            if std.example_wrong:
                chunks.append(DocumentChunk(
                    doc_id=f"rule-{std.rule_number}-wrong",
                    content=f"Violation of Rule {std.rule_number}: {std.example_wrong}",
                    metadata={"rule_number": std.rule_number, "type": "violation_example"},
                    parent_doc_id=parent_id
                ))

            # Chunk C: Specific Tooling/Libraries
            if std.tool:
                chunks.append(DocumentChunk(
                    doc_id=f"rule-{std.rule_number}-tool",
                    content=f"Tool enforced by Rule {std.rule_number}: {std.tool}",
                    metadata={"rule_number": std.rule_number, "type": "tooling"},
                    parent_doc_id=parent_id
                ))

            # For BM25, index the full text to catch exact version numbers/keywords
            bm25_corpus.append(full_text)
            bm25_ids.append(parent_id)

        # 3. Load into Stores
        self.vector_store.add_documents(chunks, parent_docs)
        self.retriever.fit_bm25(bm25_corpus, bm25_ids)
        logger.info("Mimiry has ingested the SagaCodex.")

    @with_timeout(30.0)
    @with_retry(max_attempts=2, backoff=1.0)
    async def consult_on_discrepancy(
        self,
        question: str,
        context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Answer a question using Hybrid RAG (Semantic + Keyword) with Contextual Compression.
        """
        logger.info(
            "Mimiry consulted via Hybrid RAG",
            extra={"question": question[:100], "trace_id": trace_id}
        )

        # 1. Retrieve Canonical Knowledge (Parent Docs)
        # 1. Retrieve Canonical Knowledge (Parent Docs)
        hits = self.retriever.search(question, k=5)



        canonical_answer = ""
        directives: list[Directive] = []
        cited_rules = []
        violations_detected = self._detect_violations(question, context)
        severity = "ACCEPTABLE"

        if hits:
            # Consolidate hits into an authoritative answer AND Directives
            canonical_answer = "SagaCodex decrees the following ideals:\n\n"
            for hit in hits:
                content = hit["content"]
                meta = hit["metadata"]

                # Extract rule number if present in metadata or content
                rule_num = meta.get("rule_number", 0)
                if rule_num and rule_num not in cited_rules:
                    cited_rules.append(rule_num)

                # Create Compressed Directive
                # "Contextual Compression": Turn the hit into a short instruction
                directives.append(Directive(
                    title=f"Rule {rule_num} Enforcement",
                    instruction=f"Ensure compliance with: {content.splitlines()[0]}",  # First line usually has rule name
                    priority="CRITICAL" if any(r in [15, 99] for r in [rule_num]) else "HIGH",
                    source_rule=rule_num
                ))

                canonical_answer += f"--- {meta.get('type', 'Standard')} ---\n{content}\n\n"

            canonical_answer += "Deviating from these principles is unacceptable."

            if any(d.priority == "CRITICAL" for d in directives):
                severity = "CRITICAL"
        else:
            canonical_answer = (
                "This query does not map to cataloged SagaCodex standards. "
                "Consult the LoreBook for established precedents."
            )
            severity = "ACCEPTABLE"

        return OracleResponse(
            question=question,
            canonical_answer=canonical_answer,
            directives=directives,
            cited_rules=cited_rules,
            ideal_implementation=None,
            violations_detected=violations_detected,
            severity=severity,
            oracle_confidence=95.0 if hits else 50.0,
        )

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

    @with_timeout(60.0)
    @with_retry(max_attempts=2, backoff=1.0)
    async def measure_against_ideal(
        self,
        code: str,
        domain: str,
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Measure code against SagaCodex ideal.
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
            LanguageProfile.PYTHON_FASTAPI  # Defaulting to main profile for now
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
            cited_rules.append(15)

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
            severity=severity,
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

    # ============================================================
    # FQL GATEWAY METHODS (Phase 8 - MAE Foundation)
    # ============================================================

    def _get_validation_cache(self) -> ValidationCache:
        """Lazy-load ValidationCache to avoid circular imports."""
        if self._validation_cache is None:
            from saga.core.mae.fql_schema import ValidationCache
            self._validation_cache = ValidationCache()
        return self._validation_cache

    @with_timeout(30.0)
    @with_retry(max_attempts=2, backoff=1.0)
    async def validate_proposal(
        self,
        fql_packet: FQLPacket,
        trace_id: Optional[str] = None
    ) -> ComplianceResult:
        """
        Stateless FQL validation endpoint.

        CRITICAL: This is the canonical way to query Mimiry in Phase 8+.
        Raw text queries are still supported but deprecated.

        The FQL Gateway acts as a "Software Engineering Firewall":
        1. Schema validation first (Pydantic)
        2. Then Codex check (semantic + AST)
        3. Returns ComplianceResult without storing failures

        Args:
            fql_packet: Validated FQL packet with action, subject, governance
            trace_id: Optional correlation ID

        Returns:
            ComplianceResult with compliance score and principle citations.
            Failures return corrections without polluting Mimiry's memory.
        """
        from saga.core.mae.fql_schema import (
            ComplianceResult,
            FQLAction,
            PrincipleCitation,
            RejectedAlternative,
        )

        logger.info(
            "FQL Gateway: validate_proposal",
            extra={
                "action": fql_packet.payload.action.value,
                "subject": fql_packet.payload.subject,
                "strictness": fql_packet.governance.strictness_level.value,
                "trace_id": trace_id or fql_packet.header.correlation_id,
            }
        )

        # 1. Check cache first (efficient repeat queries)
        cache = self._get_validation_cache()
        cached_result = cache.get(fql_packet)
        if cached_result is not None:
            logger.debug("FQL Gateway: Cache hit")
            return cached_result

        # 2. Route based on action type
        action = fql_packet.payload.action
        subject = fql_packet.payload.subject
        context = fql_packet.payload.context
        principle_id = fql_packet.governance.mimiry_principle_id

        citations: list[PrincipleCitation] = []
        corrections: list[str] = []
        rejected_alternatives: list[RejectedAlternative] = []
        compliance_score = 100.0
        validated_approach = ""
        rationale = ""

        if action == FQLAction.VALIDATE_PATTERN:
            # Check AST meta-tags against Codex patterns
            result = await self._validate_pattern_fql(
                subject, context, fql_packet.payload.ast_tags, trace_id
            )
            compliance_score = result["score"]
            citations = result["citations"]
            corrections = result["corrections"]
            rejected_alternatives = result["rejected"]
            validated_approach = result.get("validated_approach", subject)
            rationale = result.get("rationale", "")

        elif action == FQLAction.INTERPRET_RULE:
            # Extract rule number from principle_id or context
            rule_num = context.get("rule_number", 0)
            if "-" in principle_id:
                try:
                    rule_num = int(principle_id.split("-")[-1])
                except ValueError:
                    pass
            interp = await self.interpret_rule(rule_num, trace_id)
            compliance_score = 100.0 if interp.immutable else 75.0
            citations.append(PrincipleCitation(
                rule_id=interp.rule_number,
                rule_name=interp.rule_name,
                relevance="HIGH",
                excerpt=interp.canonical_meaning[:200]
            ))
            validated_approach = interp.canonical_meaning
            rationale = f"Rule {rule_num} interpretation from SagaCodex"

        elif action == FQLAction.MEASURE_CODE:
            # Measure code against ideal
            code = context.get("code", "")
            domain = context.get("domain", "general")
            oracle_response = await self.measure_against_ideal(code, domain, trace_id)

            # Convert OracleResponse to ComplianceResult
            compliance_score = 100.0 - (len(oracle_response.violations_detected) * 10)
            compliance_score = max(0.0, min(100.0, compliance_score))

            for rule_id in oracle_response.cited_rules:
                std = self.current_codex.get_standard(rule_id)
                if std:
                    citations.append(PrincipleCitation(
                        rule_id=rule_id,
                        rule_name=std.name,
                        relevance="CRITICAL" if rule_id in [15, 99] else "HIGH",
                        excerpt=std.description[:100]
                    ))

            corrections = oracle_response.violations_detected
            validated_approach = oracle_response.canonical_answer[:200]
            rationale = f"Measured against SagaCodex ideal. Severity: {oracle_response.severity}"

        elif action == FQLAction.RESOLVE_CONFLICT:
            # Agent conflict resolution
            agents_outputs = context.get("agents_outputs", [])
            task_context = context.get("task_context", {})
            resolution = await self.resolve_conflict(agents_outputs, task_context, trace_id)

            # Determine compliance based on resolution
            if resolution.agents_in_violation:
                compliance_score = 70.0
                for agent in resolution.agents_in_violation:
                    corrections.append(f"Agent {agent} deviates from SagaCodex ideal")
            else:
                compliance_score = 95.0

            for rule_id in resolution.cited_rules:
                std = self.current_codex.get_standard(rule_id)
                if std:
                    citations.append(PrincipleCitation(
                        rule_id=rule_id,
                        rule_name=std.name,
                        relevance="HIGH"
                    ))

            validated_approach = resolution.canonical_approach
            rationale = resolution.rationale

            # Add rejected approaches as alternatives
            for approach in resolution.conflicting_approaches:
                if approach != resolution.canonical_approach:
                    rejected_alternatives.append(RejectedAlternative(
                        approach=approach,
                        rejection_reason="Deviates further from SagaCodex ideal"
                    ))

        # 3. Determine compliance based on strictness level
        from saga.core.mae.fql_schema import StrictnessLevel
        strictness = fql_packet.governance.strictness_level

        if strictness == StrictnessLevel.FAANG_GOLDEN_PATH:
            is_compliant = compliance_score >= 95.0 and len(corrections) == 0
        elif strictness == StrictnessLevel.SENIOR_DEV:
            is_compliant = compliance_score >= 85.0 and len([c for c in corrections if "CRITICAL" in c]) == 0
        else:  # ENTERPRISE
            is_compliant = compliance_score >= 70.0

        # 4. Build result (stateless - not stored on failure)
        result = ComplianceResult(
            is_compliant=is_compliant,
            compliance_score=compliance_score,
            principle_citations=citations,
            corrections=corrections,
            rejected_alternatives=rejected_alternatives,
            validation_hash=fql_packet.compute_packet_hash(),
            validated_approach=validated_approach,
            rationale=rationale,
        )

        # 5. Cache the result
        cache.put(fql_packet, result)

        logger.info(
            "FQL Gateway: validation complete",
            extra={
                "is_compliant": is_compliant,
                "compliance_score": compliance_score,
                "corrections_count": len(corrections),
                "trace_id": trace_id,
            }
        )

        return result

    async def _validate_pattern_fql(
        self,
        subject: str,
        context: dict[str, Any],
        ast_tags: list,
        trace_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Validate a pattern using FQL AST meta-tags.

        Schema-driven validation: Uses structured AST tags instead
        of natural language to prevent hallucination.
        """
        from saga.core.mae.fql_schema import PrincipleCitation, RejectedAlternative

        citations: list[PrincipleCitation] = []
        corrections: list[str] = []
        rejected: list[RejectedAlternative] = []
        score = 100.0

        # Use hybrid retriever to find relevant Codex rules
        query = f"{subject} {' '.join(str(tag) for tag in ast_tags)}"
        hits = self.retriever.search(query, k=3)

        validated_approach = subject
        rationale = "Pattern validated against SagaCodex"

        for hit in hits:
            rule_num = hit.get("metadata", {}).get("rule_number", 0)
            if rule_num:
                std = self.current_codex.get_standard(rule_num)
                if std:
                    citations.append(PrincipleCitation(
                        rule_id=rule_num,
                        rule_name=std.name,
                        relevance="HIGH",
                        excerpt=hit.get("content", "")[:100]
                    ))

        # Check AST tags against known patterns
        for tag in ast_tags:
            pattern_type = getattr(tag, "pattern_type", None)
            implementation = getattr(tag, "implementation", "")

            # Validate common patterns
            if pattern_type and "SINGLETON" in implementation:
                if "THREAD_SAFE" not in implementation:
                    corrections.append(
                        "Singleton pattern must be thread-safe (SINGLETON_THREAD_SAFE)"
                    )
                    score -= 15
                    rejected.append(RejectedAlternative(
                        approach="Non-thread-safe Singleton",
                        rejection_reason="Race condition vulnerability in async context"
                    ))

        return {
            "score": max(0.0, score),
            "citations": citations,
            "corrections": corrections,
            "rejected": rejected,
            "validated_approach": validated_approach,
            "rationale": rationale,
        }


# ============================================================
# LEGACY ADAPTER (Zero-Trust Transition)
# ============================================================

class LegacyMimiryAdapter:
    """
    Adapter for backward compatibility during FQL migration.

    Logs DeprecationWarning when raw text queries are attempted,
    identifying the violating file for easier refactoring.

    Usage:
        # Instead of calling Mimiry directly with text:
        adapter = LegacyMimiryAdapter(mimiry)
        response = await adapter.legacy_consult("raw text question", {...})
        # This will work but emit a deprecation warning

    The goal is to identify all code paths that need FQL migration.
    """

    def __init__(self, mimiry: Mimiry) -> None:
        """Initialize the legacy adapter."""
        self._mimiry = mimiry
        self._violation_log: list[dict[str, Any]] = []

    def _log_violation(self, method_name: str, caller_info: str) -> None:
        """Log a Zero-Trust policy violation."""
        violation = {
            "method": method_name,
            "caller": caller_info,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._violation_log.append(violation)

        # Emit deprecation warning with caller info
        warnings.warn(
            f"Zero-Trust Policy Violation: Direct Mimiry call to '{method_name}' "
            f"from {caller_info}. Use FQL Gateway (validate_proposal) instead. "
            "This method will be removed in Phase 9.",
            DeprecationWarning,
            stacklevel=3
        )

        logger.warning(
            "Legacy Mimiry call detected - migrate to FQL",
            extra={
                "method": method_name,
                "caller": caller_info,
                "action": "migrate_to_fql",
            }
        )

    def _get_caller_info(self) -> str:
        """Get information about the calling code."""
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            filename = caller_frame.f_code.co_filename
            lineno = caller_frame.f_lineno
            funcname = caller_frame.f_code.co_name
            return f"{filename}:{lineno} in {funcname}()"
        return "unknown"

    async def legacy_consult(
        self,
        question: str,
        context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Deprecated: Use FQL Gateway instead.

        This method wraps consult_on_discrepancy and logs a warning.
        """
        self._log_violation("consult_on_discrepancy", self._get_caller_info())
        return await self._mimiry.consult_on_discrepancy(question, context, trace_id)

    async def legacy_measure(
        self,
        code: str,
        domain: str,
        trace_id: Optional[str] = None
    ) -> OracleResponse:
        """
        Deprecated: Use FQL Gateway with MEASURE_CODE action.

        This method wraps measure_against_ideal and logs a warning.
        """
        self._log_violation("measure_against_ideal", self._get_caller_info())
        return await self._mimiry.measure_against_ideal(code, domain, trace_id)

    async def legacy_resolve(
        self,
        agents_outputs: list[dict[str, Any]],
        task_context: dict[str, Any],
        trace_id: Optional[str] = None
    ) -> ConflictResolution:
        """
        Deprecated: Use FQL Gateway with RESOLVE_CONFLICT action.

        This method wraps resolve_conflict and logs a warning.
        """
        self._log_violation("resolve_conflict", self._get_caller_info())
        return await self._mimiry.resolve_conflict(agents_outputs, task_context, trace_id)

    def get_violation_report(self) -> list[dict[str, Any]]:
        """
        Get all logged Zero-Trust violations.

        Returns:
            List of violation records with method, caller, and timestamp.
        """
        return self._violation_log.copy()

    def clear_violations(self) -> int:
        """Clear violation log and return count of cleared entries."""
        count = len(self._violation_log)
        self._violation_log.clear()
        return count


# Export main classes
__all__ = [
    "Mimiry",
    "OracleResponse",
    "CanonicalInterpretation",
    "ConflictResolution",
    "Directive",
    "LegacyMimiryAdapter",
]
