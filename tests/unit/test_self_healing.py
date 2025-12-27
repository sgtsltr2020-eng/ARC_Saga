"""
Tests for Self-Healing Feedback Loop

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
"""

from unittest.mock import Mock

import pytest

from saga.core.memory.self_healing import (
    SelfHealingFeedback,
    SignalType,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_optimizer():
    """Mock SovereignOptimizer."""
    optimizer = Mock()
    optimizer.update_graph_weights.return_value = {"status": "success", "updates": [{"edge": ("a", "b")}]}
    return optimizer

@pytest.fixture
def mock_graph():
    """Mock RepoGraph with real NetworkX graph."""
    import networkx as nx
    repo_graph = Mock()
    repo_graph.graph = nx.DiGraph()

    # Add some test edges
    repo_graph.graph.add_edge("n1", "n2", weight=1.0)
    repo_graph.graph.add_edge("n2", "n3", weight=1.0)
    repo_graph.graph.add_edge("n3", "n4", weight=1.0)

    return repo_graph

@pytest.fixture
def self_healing(mock_optimizer, mock_graph):
    """Create SelfHealingFeedback instance."""
    return SelfHealingFeedback(
        optimizer=mock_optimizer,
        graph=mock_graph,
        normalize_by_path_length=False  # Disable for simpler testing unless needed
    )

# ═══════════════════════════════════════════════════════════════
# GRADUATED SIGNAL TESTS
# ═══════════════════════════════════════════════════════════════

def test_user_correction_minor(self_healing):
    """Test minor user correction signal (-0.3)."""
    signal = self_healing.on_user_correction(["n1", "n2"], severity="minor")

    assert signal.signal_type == SignalType.USER_CORRECTION_MINOR
    assert signal.severity == -0.3
    assert signal.applied is True
    assert self_healing.stats.negative_signals == 1

def test_user_correction_major(self_healing):
    """Test major user correction signal (-0.7)."""
    signal = self_healing.on_user_correction(["n1", "n2"], severity="major")

    assert signal.severity == -0.7

def test_user_correction_rejected(self_healing):
    """Test rejected user correction signal (-1.0)."""
    signal = self_healing.on_user_correction(["n1", "n2"], severity="rejected")

    assert signal.severity == -1.0

def test_task_failure(self_healing):
    """Test task failure signal (-0.7)."""
    signal = self_healing.on_task_failure(["n1", "n2"])

    assert signal.signal_type == SignalType.TASK_FAILURE
    assert signal.severity == -0.7

def test_simulation_break(self_healing):
    """Test simulation break signal (-1.0)."""
    signal = self_healing.on_simulation_break(["n1", "n2"])

    assert signal.signal_type == SignalType.SIMULATION_BREAK
    assert signal.severity == -1.0

def test_success_boost(self_healing):
    """Test utility-based success boost."""
    signal = self_healing.on_success(["n1", "n2"], utility_score=0.95)

    assert signal.signal_type == SignalType.SUCCESS_CRITICAL
    assert signal.severity == 1.0
    assert self_healing.stats.positive_signals == 1

# ═══════════════════════════════════════════════════════════════
# PATH NORMALIZATION TESTS
# ═══════════════════════════════════════════════════════════════

def test_path_normalization(mock_optimizer, mock_graph):
    """Test that signal is normalized by path length."""
    sh = SelfHealingFeedback(mock_optimizer, mock_graph, normalize_by_path_length=True)

    # Path length 4 (3 edges)
    # Severity -0.3 / 3 = -0.1 per edge
    sh.on_user_correction(["n1", "n2", "n3", "n4"], severity="minor")

    # Check that optimizer was called with normalized reward
    # Reward = (-0.1 + 1) / 2 = 0.45
    call_args = mock_optimizer.update_graph_weights.call_args
    reward = call_args[0][2]

    assert reward == pytest.approx(0.45, abs=0.01)

# ═══════════════════════════════════════════════════════════════
# PERMANENT AVOIDANCE TESTS
# ═══════════════════════════════════════════════════════════════

def test_permanent_avoidance_threshold(self_healing, mock_graph):
    """Test that extremely low weight edges are flagged for avoidance."""
    # Setup graph with very low weight edge
    mock_graph.graph.add_edge("n1", "n2", weight=0.04)

    self_healing._check_permanent_avoidance(["n1", "n2"])

    assert "n1->n2" in self_healing._permanent_avoidances

def test_is_path_avoided(self_healing):
    """Test path avoidance checking."""
    self_healing._permanent_avoidances.add("n1->n2")

    assert self_healing.is_path_avoided(["n1", "n2"]) is True
    assert self_healing.is_path_avoided(["n2", "n3"]) is False
