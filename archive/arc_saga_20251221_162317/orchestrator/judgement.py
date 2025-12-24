"""
Judgement Definitions for ARC SAGA Orchestrator.

Defines the structure for Judge outputs and arbitration logic.
A Verdict is the authoritative output of a Judge agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerdictStatus(str, Enum):
    """
    Status of a Judge's verdict.

    Attributes:
        APPROVE: The changes are approved as-is.
        REJECT: The changes are rejected (blocker found).
        REVISE: The changes need revision (non-blocking but necessary).
    """

    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    ESCALATE = "escalate"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class Verdict:
    """
    Structured output from a Judge agent.

    Attributes:
        status: The final decision.
        rationale: Explanation of the decision.
        discrepancies: Summary of disagreements between reviewers.
        required_changes: List of specific changes needed (if REVISE/REJECT).
    """

    status: VerdictStatus
    rationale: str
    discrepancies: list[str] = field(default_factory=list)
    required_changes: list[str] = field(default_factory=list)

    def is_blocking(self) -> bool:
        """Check if verdict prevents moving forward."""
        return self.status in (VerdictStatus.REJECT, VerdictStatus.ESCALATE)


class VerdictParser:
    """Robust parser for LLM verdict outputs."""

    @staticmethod
    def parse(llm_output: str) -> Verdict:
        """
        Parse LLM output into a Verdict object.
        
        Attempts to find and parse JSON blob. 
        Basic repair logic can be added here.
        """
        import json
        import re

        # strip markdown code blocks
        clean_text = re.sub(r"```json\s*|\s*```", "", llm_output, flags=re.IGNORECASE).strip()
        
        # Determine strictness? For now, standard json load
        try:
            data = json.loads(clean_text)
            
            # Helper to map string to Enum safely
            status_str = data.get("status", "").lower()
            try:
                status = VerdictStatus(status_str)
            except ValueError:
                # Fallback or strict error? 
                # Policy: If status invalid -> REJECT (Fail Safe)
                status = VerdictStatus.REJECT
                data["rationale"] = f"INVALID STATUS '{status_str}' PARSED. DEFAULTING TO REJECT. " + data.get("rationale", "")

            return Verdict(
                status=status,
                rationale=data.get("rationale", "No rationale provided."),
                discrepancies=data.get("discrepancies", []),
                required_changes=data.get("required_changes", [])
            )
        except json.JSONDecodeError:
             # Fail Safe
             return Verdict(
                 status=VerdictStatus.REJECT,
                 rationale=f"Failed to parse Judge JSON output. Raw: {llm_output[:100]}...",
                 discrepancies=["Output format error"]
             )

def validate_artifacts(artifacts: list[Any]) -> None:
    """
    Guard against empty inputs.
    
    Raises:
        ValueError: If any artifact is missing or empty.
    """
    if not artifacts:
        raise ValueError("No input artifacts provided to Judge.")
    
    # Check for empty strings in list
    for i, art in enumerate(artifacts):
        if hasattr(art, '__len__') and len(art) == 0:
             raise ValueError(f"Artifact at index {i} is empty.")
