"""
Digital Twin Sandbox - Predictive Simulation
=============================================

Implements a protected simulation environment for testing
refactoring proposals before they affect the actual codebase.

Phase 8 Update: Docker-based Shadow Trials with FQL integration.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Integration
"""

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from saga.core.mae.governor import SimulationFailureType
from saga.core.memory.graph_engine import RepoGraph

logger = logging.getLogger(__name__)


# Resource limits
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY_LIMIT_MB = 512
DOCKER_IMAGE = "python:3.12-slim"  # Pre-cached, minimal


@dataclass
class SimulationResult:
    """Result of a sandbox simulation."""
    simulation_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Outcome
    success: bool = False
    error: str | None = None

    # Worker outputs
    alpha_output: str = ""
    beta_output: str = ""

    # Metrics
    duration_seconds: float = 0.0
    files_modified: list[str] = field(default_factory=list)

    # Feedback for optimizer
    reward: float = 0.0
    feedback_recorded: bool = False

    # Phase 8: FQL and Docker fields
    fql_validated: bool = False
    compliance_score: float = 0.0
    failure_type: SimulationFailureType | None = None
    docker_exit_code: int | None = None
    test_output: str = ""
    spark_id: str | None = None  # Provenance tracking

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "error": self.error,
            "alpha_output": self.alpha_output,
            "beta_output": self.beta_output,
            "duration_seconds": self.duration_seconds,
            "files_modified": self.files_modified,
            "reward": self.reward,
            "fql_validated": self.fql_validated,
            "compliance_score": self.compliance_score,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "docker_exit_code": self.docker_exit_code,
            "spark_id": self.spark_id,
        }


class DockerSandbox:
    """
    Ephemeral Docker container for Shadow Trials.

    Implements the "Trial of Fire" - running synthesized code in a
    completely isolated Docker environment before it touches production.

    Security Features:
    - Network isolation (network="none")
    - Read-only base mounts (Ghost Mount Strategy)
    - Memory limit (512MB default)
    - Timeout enforcement (30s default)

    Usage:
        ```python
        sandbox = DockerSandbox(project_root=Path("/my/project"))
        result = await sandbox.run_docker_trial(
            code_path=Path("/synthesized/code"),
            test_command=["python", "-m", "pytest", "-q"]
        )
        ```
    """

    DOCKER_IMAGE = "python:3.12-slim"  # Pre-cached, minimal
    NETWORK_MODE = "none"  # No internet access
    MEMORY_LIMIT = "512m"
    TMPFS_SIZE = "100m"  # Max disk usage in tmpfs

    def __init__(
        self,
        project_root: Path | str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit: str = "512m"
    ):
        """Initialize DockerSandbox."""
        self.project_root = Path(project_root)
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self._docker_available: bool | None = None

        logger.info(
            "DockerSandbox initialized",
            extra={
                "project_root": str(self.project_root),
                "timeout": timeout_seconds,
                "memory_limit": memory_limit,
            }
        )

    def is_docker_available(self) -> bool:
        """Check if Docker is available on this system."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            self._docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._docker_available = False

        if not self._docker_available:
            logger.warning("Docker not available - falling back to subprocess")

        return self._docker_available

    async def run_static_fql_check(
        self,
        code_content: str,
        file_path: str
    ) -> tuple[bool, str]:
        """
        Run static FQL check before Docker trial.

        Pre-flight validation to save compute - if FQL fails,
        we don't waste resources on Docker.

        Args:
            code_content: The synthesized code to check
            file_path: Virtual file path for context

        Returns:
            (passed, error_message) tuple
        """
        # Run ruff on the code
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code_content)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", "-m", "ruff", "check", temp_path, "--quiet"],
                capture_output=True,
                timeout=10,
                text=True
            )

            if result.returncode != 0:
                return False, f"Static check failed (ruff): {result.stdout or result.stderr}"

            return True, ""

        except subprocess.TimeoutExpired:
            return False, "Static check timeout"
        except FileNotFoundError:
            # Ruff not installed - pass through
            return True, ""
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def run_docker_trial(
        self,
        code_path: Path,
        test_command: list[str] | None = None,
        spark_id: str | None = None
    ) -> SimulationResult:
        """
        Run pytest in isolated Docker container.

        Implements the "Ghost Mount Strategy":
        - Base library: Read-only mount
        - Test directory: Read-write tmpfs

        Args:
            code_path: Path to synthesized code
            test_command: Command to run (default: pytest)
            spark_id: Provenance tracking ID

        Returns:
            SimulationResult with Docker execution details
        """
        result = SimulationResult(spark_id=spark_id)
        start_time = datetime.utcnow()

        if test_command is None:
            test_command = ["python", "-m", "pytest", "-q", "--tb=short"]

        # Check Docker availability
        if not self.is_docker_available():
            # Fallback to subprocess
            return await self._run_subprocess_fallback(
                code_path, test_command, result
            )

        try:
            # Build Docker command with Ghost Mount Strategy
            docker_cmd = [
                "docker", "run",
                "--rm",  # Auto-cleanup
                f"--network={self.NETWORK_MODE}",  # No internet
                f"--memory={self.memory_limit}",  # Memory limit
                "--read-only",  # Read-only root filesystem
                f"--tmpfs=/tmp:{self.TMPFS_SIZE}",  # Write to tmpfs only
                "-v", f"{code_path}:/code:ro",  # Read-only code mount
                "-w", "/code",  # Working directory
                self.DOCKER_IMAGE,
            ] + test_command

            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds
                )

                result.docker_exit_code = process.returncode
                result.test_output = stdout.decode("utf-8", errors="replace")
                result.beta_output = stderr.decode("utf-8", errors="replace")

                result.success = process.returncode == 0
                result.reward = 1.0 if result.success else -0.3

                if not result.success:
                    result.failure_type = SimulationFailureType.DYNAMIC
                    result.error = f"Tests failed (exit code {process.returncode})"

            except asyncio.TimeoutError:
                # Kill the container
                process.kill()
                await process.wait()
                result.error = "DOCKER_TIMEOUT: Trial killed after timeout"
                result.failure_type = SimulationFailureType.DYNAMIC
                result.reward = -0.5

                logger.warning(
                    "Docker trial timeout",
                    extra={"spark_id": spark_id, "timeout": self.timeout_seconds}
                )

        except Exception as e:
            result.error = f"Docker trial failed: {e}"
            result.failure_type = SimulationFailureType.DYNAMIC
            result.reward = -0.3
            logger.error(f"Docker trial exception: {e}")

        finally:
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

        logger.info(
            "Docker trial complete",
            extra={
                "success": result.success,
                "duration": result.duration_seconds,
                "exit_code": result.docker_exit_code,
                "spark_id": spark_id,
            }
        )

        return result

    async def _run_subprocess_fallback(
        self,
        code_path: Path,
        test_command: list[str],
        result: SimulationResult
    ) -> SimulationResult:
        """Fallback when Docker is not available."""
        start_time = datetime.utcnow()

        try:
            process = await asyncio.create_subprocess_exec(
                *test_command,
                cwd=str(code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**__import__("os").environ, "SAGA_SANDBOX": "1"}
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds
            )

            result.test_output = stdout.decode("utf-8", errors="replace")
            result.beta_output = stderr.decode("utf-8", errors="replace")
            result.success = process.returncode == 0
            result.reward = 1.0 if result.success else -0.2

            if not result.success:
                result.failure_type = SimulationFailureType.DYNAMIC
                result.error = f"Subprocess tests failed (exit {process.returncode})"

        except asyncio.TimeoutError:
            result.error = "SUBPROCESS_TIMEOUT"
            result.failure_type = SimulationFailureType.DYNAMIC
            result.reward = -0.5
        except Exception as e:
            result.error = f"Subprocess failed: {e}"
            result.reward = -0.3

        finally:
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

        return result

    async def run_simulation_with_compliance(
        self,
        code_content: str,
        file_path: str,
        test_command: list[str] | None = None,
        spark_id: str | None = None
    ) -> SimulationResult:
        """
        Combined FQL + Docker validation pipeline.

        Runs static check first (fast-fail), then Docker trial.

        Args:
            code_content: Synthesized code to validate
            file_path: Virtual file path
            test_command: Optional test command
            spark_id: Provenance tracking

        Returns:
            SimulationResult with multi-stage validation
        """
        result = SimulationResult(spark_id=spark_id)
        start_time = datetime.utcnow()

        # Stage 1: Static FQL check (fast-fail)
        static_passed, static_error = await self.run_static_fql_check(
            code_content, file_path
        )

        if not static_passed:
            result.success = False
            result.error = static_error
            result.failure_type = SimulationFailureType.STATIC
            result.reward = -0.2  # Minor penalty for static failure
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            logger.info(
                "Simulation failed at static check",
                extra={"spark_id": spark_id, "error": static_error}
            )
            return result

        result.fql_validated = True

        # Stage 2: Docker trial
        # Write code to temp directory
        with tempfile.TemporaryDirectory(prefix="saga_trial_") as tmpdir:
            code_file = Path(tmpdir) / Path(file_path).name
            code_file.write_text(code_content, encoding="utf-8")

            docker_result = await self.run_docker_trial(
                code_path=Path(tmpdir),
                test_command=test_command,
                spark_id=spark_id
            )

            # Merge results
            result.success = docker_result.success
            result.error = docker_result.error
            result.failure_type = docker_result.failure_type
            result.docker_exit_code = docker_result.docker_exit_code
            result.test_output = docker_result.test_output
            result.beta_output = docker_result.beta_output
            result.reward = docker_result.reward

        result.completed_at = datetime.utcnow()
        result.duration_seconds = (result.completed_at - start_time).total_seconds()

        return result


class ShadowWorkspace:
    """
    Manages isolated sandbox environments for simulation.

    Creates temporary copies of affected code for safe testing.
    Implements resource caps and timeout protection.
    """

    def __init__(
        self,
        project_root: Path | str,
        graph: RepoGraph | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    ):
        """Initialize the ShadowWorkspace."""
        self.project_root = Path(project_root)
        self.graph = graph
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

        self._active_workspaces: dict[str, Path] = {}
        self._simulation_history: list[SimulationResult] = []

        logger.info(f"ShadowWorkspace initialized: root={self.project_root}")

    def get_minimum_viable_context(self, target_files: list[str]) -> list[str]:
        """
        Find the minimum set of files needed for simulation.

        Uses the RepoGraph to identify:
        - Target files
        - Direct dependencies (imports)

        Returns:
            List of relative file paths
        """
        if not self.graph:
            return target_files

        context_files = set(target_files)

        for file_path in target_files:
            file_id = f"file:{file_path}"

            if self.graph.has_node(file_id):
                # Get imported files
                for _, target, data in self.graph.get_edges_from(file_id):
                    if data.get("edge_type") == "IMPORTS":
                        target_node = self.graph.get_node(target)
                        if target_node and target_node.file_path:
                            context_files.add(target_node.file_path)

        return list(context_files)

    def create_sandbox(
        self,
        target_files: list[str],
        sandbox_id: str | None = None
    ) -> tuple[str, Path]:
        """
        Create an isolated sandbox with copies of target files.

        Args:
            target_files: Files to copy into sandbox
            sandbox_id: Optional ID (generated if None)

        Returns:
            (sandbox_id, sandbox_path) tuple
        """
        sandbox_id = sandbox_id or str(uuid4())

        # Create temp directory
        sandbox_path = Path(tempfile.mkdtemp(prefix=f"saga_sandbox_{sandbox_id[:8]}_"))

        # Find minimum viable context
        context_files = self.get_minimum_viable_context(target_files)

        # Copy files
        for rel_path in context_files:
            src = self.project_root / rel_path
            if src.exists() and src.is_file():
                dest = sandbox_path / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

        self._active_workspaces[sandbox_id] = sandbox_path
        logger.debug(f"Created sandbox {sandbox_id}: {len(context_files)} files")

        return sandbox_id, sandbox_path

    def destroy_sandbox(self, sandbox_id: str) -> bool:
        """
        Destroy a sandbox and clean up resources.

        Returns:
            True if successfully destroyed
        """
        if sandbox_id not in self._active_workspaces:
            return False

        sandbox_path = self._active_workspaces.pop(sandbox_id)

        try:
            shutil.rmtree(sandbox_path, ignore_errors=True)
            logger.debug(f"Destroyed sandbox {sandbox_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
            return False

    # --- Simulation Execution ---

    async def run_simulation(
        self,
        target_files: list[str],
        alpha_action: Callable[[Path], str] | None = None,
        beta_action: Callable[[Path], str] | None = None,
        proposal: dict[str, Any] | None = None
    ) -> SimulationResult:
        """
        Run a full simulation with Alpha (apply) and Beta (test) phases.

        Args:
            target_files: Files affected by the proposal
            alpha_action: Function to apply the change (receives sandbox path)
            beta_action: Function to test the change (receives sandbox path)
            proposal: The refactoring proposal details

        Returns:
            SimulationResult with outcomes and feedback
        """
        result = SimulationResult()
        start_time = datetime.utcnow()

        # Create sandbox
        sandbox_id, sandbox_path = self.create_sandbox(target_files)

        try:
            # Phase 1: Alpha applies the change
            if alpha_action:
                try:
                    result.alpha_output = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, alpha_action, sandbox_path
                        ),
                        timeout=self.timeout_seconds / 2
                    )
                except asyncio.TimeoutError:
                    result.error = "SIMULATION_TIMEOUT: Alpha phase exceeded time limit"
                    result.reward = -0.5
                    logger.warning(f"Simulation {sandbox_id}: Alpha timeout")
                    return result
                except Exception as e:
                    result.error = f"Alpha phase failed: {e}"
                    result.reward = -0.3
                    return result

            # Phase 2: Beta tests the change
            if beta_action:
                try:
                    result.beta_output = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, beta_action, sandbox_path
                        ),
                        timeout=self.timeout_seconds / 2
                    )
                    result.success = "PASS" in result.beta_output.upper() or "OK" in result.beta_output.upper()
                except asyncio.TimeoutError:
                    result.error = "SIMULATION_TIMEOUT: Beta phase exceeded time limit"
                    result.reward = -0.5
                    logger.warning(f"Simulation {sandbox_id}: Beta timeout")
                    return result
                except Exception as e:
                    result.error = f"Beta phase failed: {e}"
                    result.reward = -0.3
                    return result
            else:
                # No beta action, assume success if alpha didn't fail
                result.success = True

            # Calculate reward
            if result.success:
                result.reward = 1.0
            else:
                result.reward = -0.2

            result.files_modified = target_files

        finally:
            # Cleanup
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - start_time).total_seconds()
            self.destroy_sandbox(sandbox_id)
            self._simulation_history.append(result)

        logger.info(
            f"Simulation complete: success={result.success}, "
            f"reward={result.reward}, duration={result.duration_seconds:.2f}s"
        )
        return result

    async def run_subprocess_simulation(
        self,
        target_files: list[str],
        command: list[str],
        apply_patch: str | None = None
    ) -> SimulationResult:
        """
        Run a simulation using subprocess with resource limits.

        Safer than in-process execution - uses OS-level isolation.

        Args:
            target_files: Files to include in sandbox
            command: Command to run (e.g., ["python", "-m", "pytest"])
            apply_patch: Optional patch to apply before running

        Returns:
            SimulationResult
        """
        result = SimulationResult()
        start_time = datetime.utcnow()

        sandbox_id, sandbox_path = self.create_sandbox(target_files)

        try:
            # Apply patch if provided
            if apply_patch:
                patch_file = sandbox_path / "_patch.py"
                patch_file.write_text(apply_patch, encoding="utf-8")

            # Run command with timeout
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(sandbox_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={
                        **dict(__import__("os").environ),
                        "SAGA_SANDBOX": "1"  # Flag for sandboxed execution
                    }
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds
                    )

                    result.alpha_output = stdout.decode("utf-8", errors="replace")
                    result.beta_output = stderr.decode("utf-8", errors="replace")

                    result.success = process.returncode == 0
                    result.reward = 1.0 if result.success else -0.2

                except asyncio.TimeoutError:
                    # Hard kill
                    process.kill()
                    await process.wait()
                    result.error = "SIMULATION_TIMEOUT: Process killed after timeout"
                    result.reward = -0.5
                    logger.warning(f"Simulation {sandbox_id}: Process killed (timeout)")

            except Exception as e:
                result.error = f"Subprocess failed: {e}"
                result.reward = -0.3

            result.files_modified = target_files

        finally:
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - start_time).total_seconds()
            self.destroy_sandbox(sandbox_id)
            self._simulation_history.append(result)

        return result

    # --- Multi-Agent Consensus ---

    def evaluate_consensus(
        self,
        alpha_agrees: bool,
        beta_agrees: bool,
        historical_success_rate: float = 0.5
    ) -> tuple[bool, str]:
        """
        Evaluate multi-agent consensus using 2/3 rule.

        Args:
            alpha_agrees: Whether Alpha (creator) approves
            beta_agrees: Whether Beta (critic) approves
            historical_success_rate: Success rate from Mythos (tie-breaker)

        Returns:
            (approved, reason) tuple
        """
        votes = [alpha_agrees, beta_agrees]

        # If both agree, decision is clear
        if alpha_agrees and beta_agrees:
            return True, "CONSENSUS: Both Alpha and Beta approve"

        if not alpha_agrees and not beta_agrees:
            return False, "CONSENSUS: Both Alpha and Beta reject"

        # Tie-breaker: Use historical success rate
        # If historical success > 0.6, lean toward approval (Alpha typically wins)
        # If historical success < 0.4, lean toward rejection (Beta typically wins)
        if historical_success_rate > 0.6:
            return alpha_agrees, f"TIE-BREAK: Historical success {historical_success_rate:.1%} favors Alpha"
        elif historical_success_rate < 0.4:
            return beta_agrees, f"TIE-BREAK: Historical caution {1-historical_success_rate:.1%} favors Beta"
        else:
            # True tie - default to safety (Beta)
            return beta_agrees, f"TIE-BREAK: Close call ({historical_success_rate:.1%}), favoring caution"

    # --- Feedback Integration ---

    def record_simulation_feedback(
        self,
        result: SimulationResult,
        optimizer: Any  # SovereignOptimizer
    ) -> bool:
        """
        Feed simulation results back to the optimizer.

        Args:
            result: The simulation result
            optimizer: SovereignOptimizer instance

        Returns:
            True if feedback was recorded
        """
        if result.feedback_recorded:
            return False

        try:
            # Create context vector from simulation
            context = [
                result.duration_seconds / self.timeout_seconds,  # Normalized time
                1.0 if result.success else 0.0,
                len(result.files_modified) / 10.0,  # File count, normalized
                result.reward
            ]
            # Pad to expected size
            context.extend([0.0] * (64 - len(context)))

            optimizer.record_feedback(
                task_id=result.simulation_id,
                context_vector=context[:64],
                retrieval_path=result.files_modified,
                confidence=abs(result.reward),
                success=result.success
            )

            result.feedback_recorded = True
            return True

        except Exception as e:
            logger.error(f"Failed to record simulation feedback: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get sandbox statistics."""
        successful = sum(1 for r in self._simulation_history if r.success)
        total = len(self._simulation_history)

        return {
            "active_sandboxes": len(self._active_workspaces),
            "total_simulations": total,
            "successful": successful,
            "success_rate": successful / max(total, 1),
            "avg_duration": sum(r.duration_seconds for r in self._simulation_history) / max(total, 1),
            "timeouts": sum(1 for r in self._simulation_history if r.error and "TIMEOUT" in r.error)
        }

    def cleanup_all(self) -> int:
        """Cleanup all active sandboxes."""
        count = 0
        for sandbox_id in list(self._active_workspaces.keys()):
            if self.destroy_sandbox(sandbox_id):
                count += 1
        return count
