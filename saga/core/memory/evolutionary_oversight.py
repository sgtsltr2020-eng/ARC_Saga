"""
Evolutionary Oversight - Sovereign Memory Optimizer
====================================================

NumPy-based RL optimizer for memory retrieval quality.
Implements a lightweight MLP to predict utility scores and
refine graph edge weights based on task feedback.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 3 - Evolutionary Oversight
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """A single feedback event for the replay buffer."""
    event_id: str
    task_id: str
    context_vector: list[float]
    retrieval_path: list[str]
    confidence: float
    reward: float  # 0.0 = failure, 1.0 = success
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "context_vector": self.context_vector,
            "retrieval_path": self.retrieval_path,
            "confidence": self.confidence,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackEvent":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class SovereignOptimizer:
    """
    NumPy-based RL optimizer for SAGA's sovereign memory.

    Features:
    - 2-layer MLP (Input → Hidden → Output) for utility scoring
    - JSON-based replay buffer for batch training
    - Graph weight evolution with L2 regularization
    - 3σ validation to prevent model collapse
    - Xavier/Glorot weight initialization
    - Weight decay for natural forgetting

    Architecture:
        Input (feature_dim) → Hidden (hidden_dim) → Output (1)
        Activation: Sigmoid
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        l2_lambda: float = 0.001,
        weight_decay: float = 0.995,
        batch_size: int = 10,
        buffer_path: str | Path | None = None,
        state_path: str | Path | None = None
    ):
        """Initialize the SovereignOptimizer."""
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Paths for persistence
        self.buffer_path = Path(buffer_path) if buffer_path else None
        self.state_path = Path(state_path) if state_path else None

        # Initialize weights with Xavier/Glorot
        self._init_weights()

        # Replay buffer
        self.replay_buffer: list[FeedbackEvent] = []

        # Statistics for 3σ validation
        self.weight_change_history: list[float] = []
        self.training_count = 0

        # Cold storage for soft pruning
        self.cold_storage: list[dict[str, Any]] = []

        logger.info(f"SovereignOptimizer initialized: {feature_dim}→{hidden_dim}→1")

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        # Layer 1: Input → Hidden
        scale1 = np.sqrt(2.0 / (self.feature_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.feature_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros((1, self.hidden_dim))

        # Layer 2: Hidden → Output
        scale2 = np.sqrt(2.0 / (self.hidden_dim + 1))
        self.W2 = np.random.randn(self.hidden_dim, 1) * scale2
        self.b2 = np.zeros((1, 1))

        logger.debug("Weights initialized with Xavier/Glorot")

    # --- Activation Functions ---

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))."""
        s = SovereignOptimizer.sigmoid(x)
        return s * (1 - s)

    # --- Forward Pass ---

    def predict_utility(self, context_vector: np.ndarray) -> float:
        """
        Predict utility score for a context vector.

        Args:
            context_vector: Feature vector (1D or 2D array)

        Returns:
            Utility score between 0.0 and 1.0
        """
        x = np.atleast_2d(context_vector)

        # Pad or truncate to feature_dim
        if x.shape[1] < self.feature_dim:
            x = np.pad(x, ((0, 0), (0, self.feature_dim - x.shape[1])))
        elif x.shape[1] > self.feature_dim:
            x = x[:, :self.feature_dim]

        # Forward pass
        z1 = x @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        output = self.sigmoid(z2)

        return float(output[0, 0])

    # --- Feedback & Replay Buffer ---

    def record_feedback(
        self,
        task_id: str,
        context_vector: list[float],
        retrieval_path: list[str],
        confidence: float,
        success: bool
    ) -> None:
        """
        Record a feedback event for later training.

        Args:
            task_id: Identifier for the task
            context_vector: The context features used
            retrieval_path: Graph nodes traversed
            confidence: Retrieval confidence score
            success: Whether the task succeeded
        """
        event = FeedbackEvent(
            event_id=f"fb_{datetime.utcnow().timestamp()}",
            task_id=task_id,
            context_vector=context_vector,
            retrieval_path=retrieval_path,
            confidence=confidence,
            reward=1.0 if success else 0.0
        )

        self.replay_buffer.append(event)
        logger.debug(f"Recorded feedback: {event.event_id} (reward={event.reward})")

        # Check if buffer is full for training
        if len(self.replay_buffer) >= self.batch_size:
            self._train_batch()

        # Persist buffer
        self._save_buffer()

    def _train_batch(self) -> dict[str, Any]:
        """
        Train the MLP on a mini-batch from the replay buffer.

        Returns:
            Training metrics including loss and weight changes
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"status": "insufficient_data"}

        # Prepare batch
        batch = self.replay_buffer[:self.batch_size]
        self.replay_buffer = self.replay_buffer[self.batch_size:]

        X = np.array([e.context_vector for e in batch])
        y = np.array([[e.reward] for e in batch])

        # Pad/truncate features
        if X.shape[1] < self.feature_dim:
            X = np.pad(X, ((0, 0), (0, self.feature_dim - X.shape[1])))
        elif X.shape[1] > self.feature_dim:
            X = X[:, :self.feature_dim]

        # Forward pass
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)

        # Compute loss (MSE + L2 regularization)
        mse_loss = np.mean((a2 - y) ** 2)
        l2_loss = self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        total_loss = mse_loss + l2_loss

        # Backward pass
        m = X.shape[0]

        # Output layer gradients
        dz2 = (a2 - y) * self.sigmoid_derivative(z2)
        dW2 = (a1.T @ dz2) / m + 2 * self.l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(z1)
        dW1 = (X.T @ dz1) / m + 2 * self.l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Calculate weight change magnitude for 3σ validation
        weight_change = np.sqrt(np.sum(dW1 ** 2) + np.sum(dW2 ** 2))

        # 3σ Validation
        if not self._validate_weight_change(weight_change):
            logger.warning(
                f"POTENTIAL_MODEL_COLLAPSE: Weight change {weight_change:.4f} "
                f"exceeds 3σ threshold. Update paused for validation."
            )
            return {
                "status": "paused",
                "reason": "3sigma_violation",
                "weight_change": float(weight_change),
                "loss": float(total_loss)
            }

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        # Apply weight decay (natural forgetting)
        self.W1 *= self.weight_decay
        self.W2 *= self.weight_decay

        self.training_count += 1
        self.weight_change_history.append(float(weight_change))

        logger.info(f"Training batch complete: loss={total_loss:.4f}, Δw={weight_change:.4f}")

        # Save state
        self._save_state()

        return {
            "status": "success",
            "loss": float(total_loss),
            "mse_loss": float(mse_loss),
            "l2_loss": float(l2_loss),
            "weight_change": float(weight_change),
            "training_count": self.training_count
        }

    def _validate_weight_change(self, change: float) -> bool:
        """
        Validate weight change using 3σ rule.

        Returns:
            True if change is within acceptable bounds
        """
        if len(self.weight_change_history) < 5:
            # Not enough history, accept
            return True

        mean = np.mean(self.weight_change_history)
        std = np.std(self.weight_change_history)

        if std < 1e-6:
            # Avoid division by zero
            return True

        threshold = mean + 3 * std
        return change <= threshold

    # --- Graph Weight Evolution ---

    def update_graph_weights(
        self,
        graph: Any,  # RepoGraph
        retrieval_path: list[str],
        reward: float
    ) -> dict[str, Any]:
        """
        Update edge weights in the RepoGraph based on reward.

        Args:
            graph: The RepoGraph instance
            retrieval_path: List of node IDs in the path
            reward: Reward signal (0.0 to 1.0)

        Returns:
            Update statistics
        """
        if len(retrieval_path) < 2:
            return {"status": "path_too_short"}

        updates = []

        for i in range(len(retrieval_path) - 1):
            source = retrieval_path[i]
            target = retrieval_path[i + 1]

            # Get current edge data
            if not graph.graph.has_edge(source, target):
                continue

            current_weight = graph.graph[source][target].get("weight", 1.0)

            # Calculate weight update with L2 regularization
            delta = self.learning_rate * (reward - 0.5)  # Center around 0.5
            new_weight = current_weight + delta

            # Apply L2 regularization
            new_weight *= (1 - self.l2_lambda)

            # Clamp to valid range
            new_weight = np.clip(new_weight, 0.1, 2.0)

            # Update
            graph.graph[source][target]["weight"] = float(new_weight)
            updates.append({
                "edge": (source, target),
                "old_weight": current_weight,
                "new_weight": float(new_weight)
            })

        logger.debug(f"Updated {len(updates)} edge weights")
        return {"status": "success", "updates": updates}

    def apply_weight_decay_to_graph(self, graph: Any) -> int:
        """
        Apply weight decay to all edges (natural forgetting).

        Returns:
            Number of edges decayed
        """
        count = 0
        for u, v, data in graph.graph.edges(data=True):
            if "weight" in data:
                data["weight"] *= self.weight_decay
                count += 1

        logger.debug(f"Applied weight decay to {count} edges")
        return count

    # --- Self-Editing & Audit ---

    def periodic_self_audit(
        self,
        graph: Any,  # RepoGraph
        stale_threshold: float = 0.3,
        min_age_days: int = 7
    ) -> dict[str, Any]:
        """
        Scan for stale edges and tag drift.

        Args:
            graph: The RepoGraph instance
            stale_threshold: Weight below which edge is considered stale
            min_age_days: Minimum age before considering for pruning

        Returns:
            Audit results with stale items
        """
        stale_edges: list[tuple[str, str, float]] = []

        for u, v, data in graph.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight < stale_threshold:
                stale_edges.append((u, v, weight))

        logger.info(f"Audit found {len(stale_edges)} stale edges")

        return {
            "stale_edges": stale_edges,
            "total_edges": graph.edge_count,
            "stale_percentage": len(stale_edges) / max(graph.edge_count, 1) * 100
        }

    def soft_prune_to_cold_storage(
        self,
        graph: Any,
        edges_to_prune: list[tuple[str, str]]
    ) -> int:
        """
        Move stale edges to cold storage instead of deleting.

        Returns:
            Number of edges moved to cold storage
        """
        count = 0
        for source, target in edges_to_prune:
            if graph.graph.has_edge(source, target):
                edge_data = dict(graph.graph[source][target])
                self.cold_storage.append({
                    "source": source,
                    "target": target,
                    "data": edge_data,
                    "archived_at": datetime.utcnow().isoformat()
                })
                graph.graph.remove_edge(source, target)
                count += 1

        logger.info(f"Moved {count} edges to cold storage")
        return count

    # --- Persistence with ECC Hashing ---

    def _compute_state_hash(self) -> str:
        """Compute ECC hash of optimizer state for integrity."""
        state_bytes = self.W1.tobytes() + self.W2.tobytes()
        state_bytes += self.b1.tobytes() + self.b2.tobytes()
        return hashlib.sha256(state_bytes).hexdigest()

    def _save_state(self) -> None:
        """Save optimizer state with ECC hash."""
        if not self.state_path:
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "training_count": self.training_count,
            "weight_change_history": self.weight_change_history[-100:],
            "cold_storage": self.cold_storage[-1000:],
            "hash": self._compute_state_hash(),
            "saved_at": datetime.utcnow().isoformat()
        }

        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug(f"State saved with hash: {state['hash'][:16]}...")

    def load_state(self) -> bool:
        """
        Load optimizer state with integrity verification.

        Returns:
            True if state loaded and verified successfully
        """
        if not self.state_path or not self.state_path.exists():
            return False

        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))

            self.W1 = np.array(state["W1"])
            self.b1 = np.array(state["b1"])
            self.W2 = np.array(state["W2"])
            self.b2 = np.array(state["b2"])
            self.training_count = state.get("training_count", 0)
            self.weight_change_history = state.get("weight_change_history", [])
            self.cold_storage = state.get("cold_storage", [])

            # Verify hash
            expected_hash = state.get("hash", "")
            actual_hash = self._compute_state_hash()

            if expected_hash and expected_hash != actual_hash:
                logger.error("STATE_CORRUPTION_DETECTED: Hash mismatch!")
                return False

            logger.info(f"State loaded: {self.training_count} training iterations")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def _save_buffer(self) -> None:
        """Save replay buffer to disk."""
        if not self.buffer_path:
            return

        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

        buffer_data = [e.to_dict() for e in self.replay_buffer]
        self.buffer_path.write_text(json.dumps(buffer_data, indent=2), encoding="utf-8")

    def load_buffer(self) -> int:
        """
        Load replay buffer from disk.

        Returns:
            Number of events loaded
        """
        if not self.buffer_path or not self.buffer_path.exists():
            return 0

        try:
            data = json.loads(self.buffer_path.read_text(encoding="utf-8"))
            self.replay_buffer = [FeedbackEvent.from_dict(e) for e in data]
            return len(self.replay_buffer)
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")
            return 0

    # --- Synthesis Placeholder ---

    # --- Information-Gain Rewards (Uncanny Triad) ---

    def calculate_novelty_bonus(self, context_vector: list[float], history_vectors: list[list[float]]) -> float:
        """
        Calculate novelty bonus based on cosine distance to history.

        Args:
            context_vector: Current state vector
            history_vectors: List of past state vectors

        Returns:
            Novelty bonus (0.0 to 2.0+)
        """
        if not history_vectors:
            return 2.0  # First experience is maximally novel

        # Convert to numpy
        vec = np.array(context_vector)
        hist = np.array(history_vectors)

        # Pad if dimensions feature_dim mismatch (defensive)
        if vec.shape[0] != hist.shape[1]:
             # Just return base novelty if dims mismatch to avoid crash
             return 0.5

        # Normalize
        norm_vec = np.linalg.norm(vec)
        norm_hist = np.linalg.norm(hist, axis=1)

        if norm_vec == 0:
            return 0.0

        # Cosine similarity
        sims = np.dot(hist, vec) / (norm_hist * norm_vec + 1e-9)

        # Distance = 1 - Max Similarity
        min_dist = 1.0 - np.max(sims) if len(sims) > 0 else 1.0

        # Reward logic: Base +2.0 if distance > 0.6
        if min_dist > 0.6:
            return 2.0 + (min_dist - 0.6)  # Linear boost beyond threshold

        return max(0.0, min_dist)

    def _get_lm_judge_score(self, proposal_summary: str) -> float:
        """
        Get qualitative score from LM Judge (Mock/Placeholder).
        In production, this calls the LLM with a rubric.

        Rubric:
        - Originality (0-10) -> 0.4 weight
        - Criticality (0-10) -> 0.4 weight
        - Feasibility (0-10) -> 0.2 weight
        """
        # Placeholder heuristic: Length/complexity proxy for now
        # Real impl would await self.llm_client.evaluate(...)
        return 0.5  # Neutral default

    def synthesis_utility(
        self,
        context_vector: list[float],
        history_vectors: list[list[float]],
        proposal_summary: str = ""
    ) -> float:
        """
        Calculate Information-Gain Utility with Heuristic Pre-Filter.

        Logic:
        1. Base Utility (from model/heuristic)
        2. Novelty Bonus (Vector Distance)
        3. If Novelty > Threshold:
           - Check additional heuristics (keyword diversity)
           - Only then trigger expensive LM-Judge (simulated here)
        """
        # 1. Base Utility (Predicted)
        base_utility = self.predict_utility(np.array(context_vector))

        # 2. Novelty Bonus
        novelty_bonus = self.calculate_novelty_bonus(context_vector, history_vectors)

        lm_bonus = 0.0

        # HEURISTIC PRE-FILTER: Only escalate to LM Judge if genuine potential detected
        # Threshold: >2.2 means high distance (>0.7) + base interest
        if novelty_bonus > 1.5 and proposal_summary:
            # Quick check for strong keywords before calling LLM
            strong_signals = ["breakthrough", "novel", "critical", "unexpected", "synthesis"]
            signal_score = sum(1 for s in strong_signals if s in proposal_summary.lower())

            # Escalate if vector novelty is extreme OR moderate novelty + strong signals
            should_escalate = (novelty_bonus > 2.0) or (novelty_bonus > 1.0 and signal_score >= 1)

            if should_escalate:
                logger.info(f"Escalating to LM Judge (Novelty: {novelty_bonus:.2f}, Signals: {signal_score})")
                lm_score = self._get_lm_judge_score(proposal_summary)
                lm_bonus = lm_score * 1.0
            else:
                logger.debug("Novelty detected but pre-filter skipped LM Judge.")

        total_utility = base_utility + novelty_bonus + lm_bonus

        if total_utility > 2.5:
            logger.info(f"EXPLOSIVE REWARD: Synthesis Utility {total_utility:.2f} (Novelty: {novelty_bonus:.2f})")

        return total_utility

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "training_count": self.training_count,
            "buffer_size": len(self.replay_buffer),
            "cold_storage_size": len(self.cold_storage),
            "weight_history_size": len(self.weight_change_history),
            "weight_norm": float(np.sqrt(np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)))
        }
