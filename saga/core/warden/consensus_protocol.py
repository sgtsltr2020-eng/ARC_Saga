"""
Multi-Agent Critique (Consensus Mechanism)
==========================================

Reliable consensus via Multi-Agent Consensus Protocol.
Requires 2/3 majority (Planner, Coder, Critic) before agreeing to bold syntheses.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: Reliable Thinking Layer
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class Vote(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

@dataclass
class Critique:
    critic_id: str
    vote: Vote
    reasoning: str
    challenges: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: 0.0)

# ═══════════════════════════════════════════════════════════════
# CRITIC AGENT
# ═══════════════════════════════════════════════════════════════

class CriticAgent:
    """
    Dedicated Critic agent in the swarm.
    Challenges assumptions and verifies alignment.
    """

    def __init__(self, agent_id: str = "critic_01", strictness: float = 0.7):
        self.agent_id = agent_id
        self.strictness = strictness  # 0.0 to 1.0 (Higher = more skepticism)

    async def evaluate_proposal(self, proposal: dict[str, Any], context: dict[str, Any]) -> Critique:
        """
        Evaluate a proposal and return a critique.

        Args:
            proposal: The proposal to evaluate
            context: Context including novelty score, domain, etc.

        Returns:
            Critique object with vote and reasoning
        """
        novelty = context.get("novelty_score", 0.0)
        is_cross_domain = context.get("cross_domain", False)

        # Heuristic Logic for Prototype (in real system, this calls LLM)
        # If highly novel, apply higher scrutiny

        challenges = []
        vote = Vote.APPROVE
        reasoning = "Proposal aligns with patterns."

        # 1. Check Provenance
        if novelty > 0.8 and not proposal.get("provenance_citations"):
            vote = Vote.REJECT
            reasoning = "High novelty proposal lacks provenance citations."
            challenges.append("Cite source of inspiration")

        # 2. Check Modularity
        if "core_logic" in proposal.get("affected_areas", []) and self.strictness > 0.5:
             # Higher bar for core logic
             # Mock check: assume failure if no formal proof provided
             if not proposal.get("formal_verification"):
                 vote = Vote.REJECT
                 reasoning = "Core logic change requires formal verification plan."
                 challenges.append("Prove modularity preservation")

        logger.info(f"Critic {self.agent_id} voted {vote.value}: {reasoning}")
        return Critique(
            critic_id=self.agent_id,
            vote=vote,
            reasoning=reasoning,
            challenges=challenges
        )

# ═══════════════════════════════════════════════════════════════
# CONSENSUS PROTOCOL
# ═══════════════════════════════════════════════════════════════

class ConsensusProtocol:
    """
    Orchestrates the 2/3 consensus mechanism.

    Participants:
    1. Planner (Implicitly approves its own plan)
    2. Coder (Evaluates feasibility)
    3. Critic (Evaluates safety/alignment)
    """

    def __init__(self, critic: CriticAgent, chronicler: Any = None):
        self.critic = critic
        self.chronicler = chronicler

    async def request_consensus(
        self,
        proposal: dict[str, Any],
        planner_vote: Vote = Vote.APPROVE
    ) -> bool:
        """
        Execute voting cycle.

        Logic:
        - If Routine (novelty <= 0.7, no cross-domain): Bypass consensus (Auto-Approve).
        - If Novel/Risky: Trigger 2/3 majority vote (Planner, Coder, Critic).

        Returns:
            True if approved (either bypassed or voted).
        """
        context = proposal.get("context", {})
        novelty = context.get("novelty_score", 0.0)
        is_cross_domain = context.get("cross_domain", False)

        # ADJUSTMENT: Trigger consensus only for high-novelty or cross-domain items
        if novelty <= 0.7 and not is_cross_domain:
            logger.debug(f"Consensus Bypassed (Routine Task): Novelty {novelty:.2f}")
            return True

        logger.info(f"Consensus Triggered: Novelty {novelty:.2f}, CrossDomain {is_cross_domain}")

        # 1. Gather Votes
        # Planner: usually APPROVE
        # Coder: Feasibility check (Mock for now)
        coder_vote = self._get_coder_vote(proposal)

        # Critic: Alignment check
        critique = await self.critic.evaluate_proposal(proposal, context=context)

        votes = [planner_vote, coder_vote, critique.vote]
        approve_count = votes.count(Vote.APPROVE)
        reject_count = votes.count(Vote.REJECT)

        total_votes = len(votes)

        # 2. Log Results
        if self.chronicler and reject_count > 0:
            # self.chronicler.log_dissent(...)
            logger.info(f"Dissent registered: {critique.reasoning}")

        # 3. Determine Consensus (2/3 Majority)
        threshold = 2/3 * total_votes
        passed = approve_count >= threshold

        status = "PASSED" if passed else "REJECTED"
        logger.info(f"Consensus {status}: {approve_count}/{total_votes} votes (Critic: {critique.vote.value})")

        return passed

    def _get_coder_vote(self, proposal: dict[str, Any]) -> Vote:
        """Get feasibility vote from Coder agent (Simulated)."""
        # In real system, query CoderAgent
        # Assume valid code for now
        return Vote.APPROVE
