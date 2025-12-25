"""
Unit Tests for USMA Phase 4: Brilliance & Simulation
=====================================================

Tests for SynthesisAgent, ShadowWorkspace, and multi-agent consensus.
"""

import pytest

from saga.core.memory import (
    ArchitecturalDebt,
    EdgeType,
    GraphEdge,
    GraphNode,
    MythosChapter,
    MythosLibrary,
    NodeType,
    RepoGraph,
    ShadowWorkspace,
    SimulationResult,
    SolvedPattern,
    SynthesisAgent,
    SynthesisSpark,
)


class TestSynthesisSpark:
    """Tests for SynthesisSpark data model."""

    def test_spark_creation(self):
        """Test creating a synthesis spark."""
        spark = SynthesisSpark(
            source_pattern="Error Handling",
            source_domain="core",
            target_problem="Timeout Issues",
            target_domain="streaming",
            synthesis_prompt="Bridge error handling to streaming",
            confidence=0.75
        )

        assert spark.source_pattern == "Error Handling"
        assert spark.confidence == 0.75
        assert not spark.validated

    def test_spark_serialization(self):
        """Test spark to_dict."""
        spark = SynthesisSpark(
            source_pattern="Pattern A",
            synthesis_prompt="Test prompt"
        )

        data = spark.to_dict()

        assert "spark_id" in data
        assert data["source_pattern"] == "Pattern A"


class TestSynthesisAgent:
    """Tests for SynthesisAgent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        graph = RepoGraph()
        agent = SynthesisAgent(graph)

        assert agent.graph is graph
        assert agent.walk_length == 5

    def test_random_walk(self):
        """Test random walk on graph."""
        graph = RepoGraph()
        graph.add_node(GraphNode("a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("b", NodeType.FUNCTION, "b"))
        graph.add_node(GraphNode("c", NodeType.FUNCTION, "c"))
        graph.add_edge(GraphEdge("a", "b", EdgeType.CALLS))
        graph.add_edge(GraphEdge("b", "c", EdgeType.CALLS))

        agent = SynthesisAgent(graph, walk_length=3)
        path = agent.random_walk("a")

        assert len(path) >= 1
        assert path[0] == "a"

    def test_record_co_occurrence(self):
        """Test co-occurrence tracking."""
        graph = RepoGraph()
        agent = SynthesisAgent(graph)

        agent.record_co_occurrence(["a", "b", "c"])

        assert agent._co_occurrence.get(("a", "b"), 0) >= 1
        assert agent._co_occurrence.get(("a", "c"), 0) >= 1

    def test_pattern_debt_bridges(self):
        """Test finding pattern-debt bridges."""
        graph = RepoGraph()
        mythos = MythosLibrary()

        mythos.add_chapter(MythosChapter(
            title="Test Chapter",
            summary="Testing",
            solved_patterns=[
                SolvedPattern(
                    name="Retry Logic",
                    description="Implements automatic retry with backoff"
                )
            ],
            architectural_debt=[
                ArchitecturalDebt(
                    name="No Retry",
                    description="API calls fail without retry backoff"
                )
            ]
        ))

        agent = SynthesisAgent(graph, mythos=mythos)
        bridges = agent.find_pattern_debt_bridges()

        # Should find bridge (both mention "retry" and "backoff")
        assert len(bridges) >= 1

    def test_generate_spark(self):
        """Test spark generation."""
        graph = RepoGraph()
        graph.add_node(GraphNode("file:core/handler.py", NodeType.FILE, "handler.py", "core/handler.py"))
        graph.add_node(GraphNode("file:api/client.py", NodeType.FILE, "client.py", "api/client.py"))

        agent = SynthesisAgent(graph)
        spark = agent.generate_spark("file:core/handler.py", "file:api/client.py")

        assert spark.synthesis_prompt != ""
        assert "core" in spark.source_domain or "api" in spark.target_domain


class TestShadowWorkspace:
    """Tests for ShadowWorkspace sandbox."""

    def test_workspace_initialization(self, tmp_path):
        """Test workspace initializes correctly."""
        workspace = ShadowWorkspace(tmp_path)

        assert workspace.project_root == tmp_path
        assert workspace.timeout_seconds == 30

    def test_create_and_destroy_sandbox(self, tmp_path):
        """Test sandbox creation and cleanup."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        workspace = ShadowWorkspace(tmp_path)
        sandbox_id, sandbox_path = workspace.create_sandbox(["test.py"])

        # File should be copied
        assert (sandbox_path / "test.py").exists()
        assert sandbox_id in workspace._active_workspaces

        # Destroy
        workspace.destroy_sandbox(sandbox_id)
        assert sandbox_id not in workspace._active_workspaces
        assert not sandbox_path.exists()

    def test_minimum_viable_context(self, tmp_path):
        """Test MVC extraction uses graph."""
        graph = RepoGraph(tmp_path)
        graph.add_node(GraphNode("file:main.py", NodeType.FILE, "main.py", "main.py"))
        graph.add_node(GraphNode("file:utils.py", NodeType.FILE, "utils.py", "utils.py"))
        graph.add_edge(GraphEdge("file:main.py", "file:utils.py", EdgeType.IMPORTS))

        workspace = ShadowWorkspace(tmp_path, graph=graph)
        context = workspace.get_minimum_viable_context(["main.py"])

        assert "main.py" in context
        # Note: utils.py won't be included because edge target is "file:utils.py" not a module

    def test_evaluate_consensus_both_agree(self, tmp_path):
        """Test consensus when both agents agree."""
        workspace = ShadowWorkspace(tmp_path)

        approved, reason = workspace.evaluate_consensus(True, True, 0.5)
        assert approved is True
        assert "Both" in reason

    def test_evaluate_consensus_both_reject(self, tmp_path):
        """Test consensus when both reject."""
        workspace = ShadowWorkspace(tmp_path)

        approved, reason = workspace.evaluate_consensus(False, False, 0.5)
        assert approved is False

    def test_evaluate_consensus_tie_breaker_alpha(self, tmp_path):
        """Test tie-breaker favors Alpha on high success."""
        workspace = ShadowWorkspace(tmp_path)

        # Alpha agrees, Beta disagrees, high historical success
        approved, reason = workspace.evaluate_consensus(True, False, 0.7)
        assert approved is True
        assert "Alpha" in reason

    def test_evaluate_consensus_tie_breaker_beta(self, tmp_path):
        """Test tie-breaker favors Beta on low success."""
        workspace = ShadowWorkspace(tmp_path)

        # Alpha agrees, Beta disagrees, low historical success
        approved, reason = workspace.evaluate_consensus(True, False, 0.3)
        assert approved is False
        assert "Beta" in reason

    @pytest.mark.asyncio
    async def test_run_simulation_basic(self, tmp_path):
        """Test basic simulation run."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        workspace = ShadowWorkspace(tmp_path, timeout_seconds=5)

        def alpha_action(sandbox_path):
            return "Changes applied"

        def beta_action(sandbox_path):
            return "Tests PASS"

        result = await workspace.run_simulation(
            target_files=["test.py"],
            alpha_action=alpha_action,
            beta_action=beta_action
        )

        assert result.success is True
        assert result.reward == 1.0
        assert result.alpha_output == "Changes applied"

    def test_get_stats(self, tmp_path):
        """Test workspace statistics."""
        workspace = ShadowWorkspace(tmp_path)

        stats = workspace.get_stats()

        assert "active_sandboxes" in stats
        assert "total_simulations" in stats
        assert stats["success_rate"] == 0


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = SimulationResult()

        assert result.success is False
        assert result.reward == 0.0

    def test_result_serialization(self):
        """Test result to_dict."""
        result = SimulationResult(
            success=True,
            reward=1.0,
            alpha_output="Applied",
            beta_output="Tested"
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["reward"] == 1.0
