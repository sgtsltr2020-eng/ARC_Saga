"""
Unit Tests for Docker Shadow Trials
====================================

Tests for Phase 8 Stage D: DockerSandbox with network isolation,
Ghost Mount Strategy, and multi-stage validation.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Stage D
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saga.core.mae.governor import SimulationFailureType
from saga.core.memory.simulation import DockerSandbox, SimulationResult

# ============================================================
# Test 1: SimulationResult with FQL Fields
# ============================================================

class TestSimulationResultFQLFields:
    """Test SimulationResult includes Phase 8 FQL fields."""

    def test_result_has_fql_validated_field(self):
        """SimulationResult should have fql_validated field."""
        result = SimulationResult()
        assert hasattr(result, "fql_validated")
        assert result.fql_validated is False

    def test_result_has_compliance_score(self):
        """SimulationResult should have compliance_score field."""
        result = SimulationResult()
        assert hasattr(result, "compliance_score")
        assert result.compliance_score == 0.0

    def test_result_has_failure_type(self):
        """SimulationResult should have failure_type field."""
        result = SimulationResult()
        assert hasattr(result, "failure_type")
        assert result.failure_type is None

    def test_result_has_docker_exit_code(self):
        """SimulationResult should have docker_exit_code field."""
        result = SimulationResult()
        assert hasattr(result, "docker_exit_code")
        assert result.docker_exit_code is None

    def test_result_has_spark_id(self):
        """SimulationResult should have spark_id for provenance."""
        result = SimulationResult(spark_id="test-spark-123")
        assert result.spark_id == "test-spark-123"

    def test_to_dict_includes_new_fields(self):
        """to_dict() should include all new Phase 8 fields."""
        result = SimulationResult(
            fql_validated=True,
            compliance_score=0.85,
            failure_type=SimulationFailureType.STATIC,
            docker_exit_code=1,
            spark_id="spark-456"
        )
        data = result.to_dict()

        assert data["fql_validated"] is True
        assert data["compliance_score"] == 0.85
        assert data["failure_type"] == "STATIC"
        assert data["docker_exit_code"] == 1
        assert data["spark_id"] == "spark-456"


# ============================================================
# Test 2: DockerSandbox Initialization
# ============================================================

class TestDockerSandboxInit:
    """Test DockerSandbox initialization."""

    def test_init_with_defaults(self):
        """DockerSandbox should initialize with default settings."""
        sandbox = DockerSandbox(project_root="/tmp/test")

        assert sandbox.timeout_seconds == 30
        assert sandbox.memory_limit == "512m"

    def test_init_with_custom_timeout(self):
        """DockerSandbox should accept custom timeout."""
        sandbox = DockerSandbox(
            project_root="/tmp/test",
            timeout_seconds=60
        )

        assert sandbox.timeout_seconds == 60

    def test_network_mode_is_none(self):
        """Network mode should be 'none' for isolation."""
        assert DockerSandbox.NETWORK_MODE == "none"

    def test_docker_image_is_slim(self):
        """Docker image should be slim for speed."""
        assert "slim" in DockerSandbox.DOCKER_IMAGE


# ============================================================
# Test 3: Docker Availability Check
# ============================================================

class TestDockerAvailability:
    """Test Docker availability detection."""

    def test_docker_not_found(self):
        """Should handle Docker not installed gracefully."""
        sandbox = DockerSandbox(project_root="/tmp/test")
        sandbox._docker_available = None  # Reset cache

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            available = sandbox.is_docker_available()

        assert available is False

    def test_docker_available(self):
        """Should detect Docker when available."""
        sandbox = DockerSandbox(project_root="/tmp/test")
        sandbox._docker_available = None  # Reset cache

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            available = sandbox.is_docker_available()

        assert available is True


# ============================================================
# Test 4: Static FQL Check (Fast-Fail)
# ============================================================

class TestStaticFQLCheck:
    """Test run_static_fql_check() for pre-flight validation."""

    @pytest.mark.asyncio
    async def test_valid_code_passes(self):
        """Valid Python code should pass static check."""
        sandbox = DockerSandbox(project_root="/tmp/test")

        # Mock successful ruff check
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            passed, error = await sandbox.run_static_fql_check(
                "def hello(): return 'Hello'", "test.py"
            )

        assert passed is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_ruff_not_installed_passes(self):
        """If ruff not installed, should pass-through."""
        sandbox = DockerSandbox(project_root="/tmp/test")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ruff not found")

            passed, error = await sandbox.run_static_fql_check("code", "test.py")

        # Should pass-through when ruff unavailable
        assert passed is True


# ============================================================
# Test 5: Subprocess Fallback
# ============================================================

class TestSubprocessFallback:
    """Test fallback to subprocess when Docker unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_used_when_no_docker(self):
        """Should use subprocess fallback when Docker unavailable."""
        sandbox = DockerSandbox(project_root="/tmp/test")
        sandbox._docker_available = False  # Force fallback

        result = SimulationResult()

        with patch.object(
            sandbox, "_run_subprocess_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            mock_fallback.return_value = SimulationResult(success=True)

            await sandbox.run_docker_trial(
                code_path=Path("/tmp/test"),
                test_command=["python", "-c", "pass"]
            )

            mock_fallback.assert_called_once()


# ============================================================
# Test 6: Network Isolation
# ============================================================

class TestNetworkIsolation:
    """Test that Docker runs with network=none."""

    @pytest.mark.asyncio
    async def test_docker_command_has_network_none(self):
        """Docker command should include --network=none."""
        sandbox = DockerSandbox(project_root="/tmp/test")
        sandbox._docker_available = True

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_exec.return_value = mock_process

            await sandbox.run_docker_trial(
                code_path=Path("/tmp/test"),
                test_command=["python", "-c", "pass"]
            )

            # Check that network=none is in the command
            call_args = mock_exec.call_args[0]
            assert any("--network=none" in str(arg) for arg in call_args)


# ============================================================
# Test 7: Timeout Enforcement
# ============================================================

class TestTimeoutEnforcement:
    """Test timeout kills container."""

    @pytest.mark.asyncio
    async def test_timeout_sets_error(self):
        """Timeout should set error in result."""
        sandbox = DockerSandbox(project_root="/tmp/test", timeout_seconds=1)
        sandbox._docker_available = True

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.kill = MagicMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            result = await sandbox.run_docker_trial(
                code_path=Path("/tmp/test"),
                test_command=["python", "-c", "pass"]
            )

            assert "TIMEOUT" in result.error
            assert result.reward == -0.5


# ============================================================
# Test 8: Multi-Stage Validation
# ============================================================

class TestMultiStageValidation:
    """Test run_simulation_with_compliance() pipeline."""

    @pytest.mark.asyncio
    async def test_static_failure_stops_early(self):
        """Static failure should prevent Docker trial."""
        sandbox = DockerSandbox(project_root="/tmp/test")

        with patch.object(
            sandbox, "run_static_fql_check", new_callable=AsyncMock
        ) as mock_static:
            mock_static.return_value = (False, "Syntax error")

            with patch.object(
                sandbox, "run_docker_trial", new_callable=AsyncMock
            ) as mock_docker:
                result = await sandbox.run_simulation_with_compliance(
                    code_content="invalid code",
                    file_path="test.py"
                )

                # Docker should not be called
                mock_docker.assert_not_called()

                # Result should indicate static failure
                assert result.success is False
                assert result.failure_type == SimulationFailureType.STATIC

    @pytest.mark.asyncio
    async def test_static_success_runs_docker(self):
        """Static success should proceed to Docker trial."""
        sandbox = DockerSandbox(project_root="/tmp/test")

        with patch.object(
            sandbox, "run_static_fql_check", new_callable=AsyncMock
        ) as mock_static:
            mock_static.return_value = (True, "")

            with patch.object(
                sandbox, "run_docker_trial", new_callable=AsyncMock
            ) as mock_docker:
                mock_docker.return_value = SimulationResult(success=True)

                result = await sandbox.run_simulation_with_compliance(
                    code_content="valid code",
                    file_path="test.py"
                )

                # Docker should be called
                mock_docker.assert_called_once()

                # Result should indicate FQL validated
                assert result.fql_validated is True
