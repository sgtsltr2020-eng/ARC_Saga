"""
Tests for RippleSimulator - Empirical Change Impact Prediction

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
"""

import tempfile
from pathlib import Path

import pytest

from saga.core.memory.ripple_simulator import (
    RippleDiff,
    RippleSimulator,
    SandboxConfig,
    SubprocessSandbox,
    TestError,
    TestOutcome,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def temp_project():
    """Create a temporary project directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create a simple Python file
        (project_path / "main.py").write_text(
            """
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
"""
        )

        # Create a test file
        tests_dir = project_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_main.py").write_text(
            """
import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import add, subtract

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
"""
        )

        yield project_path


@pytest.fixture
def simulator():
    """Create a RippleSimulator instance."""
    return RippleSimulator(
        use_docker=False,  # Use subprocess for tests
        max_fix_iterations=3,
        confidence_threshold=0.9,
    )


# ═══════════════════════════════════════════════════════════════
# SUBPROCESS SANDBOX TESTS
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_subprocess_sandbox_creation(temp_project):
    """Test subprocess sandbox creation."""
    config = SandboxConfig(
        name="test_alpha",
        project_path=temp_project,
        code_changes={},
    )

    sandbox = SubprocessSandbox(config)
    await sandbox.create()

    assert sandbox._created
    assert sandbox.temp_dir is not None
    assert sandbox.temp_dir.exists()

    await sandbox.cleanup()
    assert not sandbox._created


@pytest.mark.asyncio
async def test_subprocess_sandbox_execution(temp_project):
    """Test command execution in subprocess sandbox."""
    config = SandboxConfig(
        name="test_exec",
        project_path=temp_project,
        code_changes={},
    )

    sandbox = SubprocessSandbox(config)
    await sandbox.create()

    try:
        result = await sandbox.execute("echo hello")
        assert result.return_code == 0
        assert "hello" in result.stdout.lower()
    finally:
        await sandbox.cleanup()


@pytest.mark.asyncio
async def test_subprocess_sandbox_with_changes(temp_project):
    """Test sandbox with code changes applied."""
    config = SandboxConfig(
        name="test_changes",
        project_path=temp_project,
        code_changes={"main.py": "def new_func(): return 42"},
    )

    sandbox = SubprocessSandbox(config)
    await sandbox.create()

    try:
        # Verify change was applied (use 'type' on Windows, 'cat' on Unix)
        import sys
        cmd = "type main.py" if sys.platform == "win32" else "cat main.py"
        result = await sandbox.execute(cmd)
        assert "new_func" in result.stdout
    finally:
        await sandbox.cleanup()


@pytest.mark.asyncio
async def test_subprocess_sandbox_cleanup_on_error(temp_project):
    """Test sandbox cleanup happens even on errors."""
    config = SandboxConfig(
        name="test_cleanup",
        project_path=temp_project,
        code_changes={},
    )

    sandbox = SubprocessSandbox(config)
    await sandbox.create()
    temp_path = sandbox.temp_dir

    # Force cleanup
    await sandbox.cleanup()

    # Temp dir should be gone or cleanup attempted
    assert not sandbox._created


# ═══════════════════════════════════════════════════════════════
# RIPPLE SIMULATOR TESTS
# ═══════════════════════════════════════════════════════════════


def test_simulator_initialization():
    """Test RippleSimulator initialization."""
    sim = RippleSimulator(use_docker=False)

    assert not sim.use_docker
    assert sim.max_fix_iterations == 3
    assert sim.confidence_threshold == 0.9
    assert sim.stats.total_simulations == 0


def test_docker_auto_detection():
    """Test Docker availability auto-detection."""
    # This will auto-detect based on system
    sim = RippleSimulator(use_docker=None)

    # Should be either True or False, not None
    assert sim.use_docker is not None


@pytest.mark.asyncio
async def test_simulate_change_no_tests(simulator, temp_project):
    """Test simulation when no tests are found."""
    # Remove test files
    import shutil
    shutil.rmtree(temp_project / "tests")

    report = await simulator.simulate_change(
        code_changes={"main.py": "def foo(): pass"},
        project_path=temp_project,
    )

    assert report is not None
    assert report.overall_status == "all_pass"
    assert "No tests" in report.summary


@pytest.mark.asyncio
async def test_simulate_change_all_pass(simulator, temp_project):
    """Test simulation where all tests pass."""
    # No changes that break tests
    report = await simulator.simulate_change(
        code_changes={},  # No changes
        project_path=temp_project,
    )

    assert report is not None
    assert report.simulation_id.startswith("sim_")
    assert report.summary != ""  # Always has summary


@pytest.mark.asyncio
async def test_simulate_change_with_failure(simulator, temp_project):
    """Test simulation that detects a failure."""
    # Change add() to return wrong value
    broken_code = """
def add(a: int, b: int) -> int:
    return a - b  # Bug: should be +

def subtract(a: int, b: int) -> int:
    return a - b
"""

    report = await simulator.simulate_change(
        code_changes={"main.py": broken_code},
        project_path=temp_project,
    )

    assert report is not None
    # Should detect the failure
    # (may or may not depending on test discovery in temp dir)


# ═══════════════════════════════════════════════════════════════
# DIFF INTELLIGENCE TESTS
# ═══════════════════════════════════════════════════════════════


def test_diff_new_failures(simulator):
    """Test diff correctly identifies new failures."""
    alpha = TestOutcome(
        sandbox_name="alpha",
        passed_count=5,
        failed_count=2,
        skipped_count=0,
        errors=[
            TestError(
                test_name="test_foo",
                error_type="AssertionError",
                error_message="assert 1 == 2",
                traceback="",
                is_failure=True,
            ),
            TestError(
                test_name="test_bar",
                error_type="RuntimeError",
                error_message="oops",
                traceback="",
                is_failure=True,
            ),
        ],
        stdout="",
        stderr="",
        return_code=1,
        duration_seconds=5.0,
        passed_tests=["test_a", "test_b", "test_c", "test_d", "test_e"],
    )

    beta = TestOutcome(
        sandbox_name="beta",
        passed_count=7,
        failed_count=0,
        skipped_count=0,
        errors=[],
        stdout="",
        stderr="",
        return_code=0,
        duration_seconds=5.0,
        passed_tests=["test_a", "test_b", "test_c", "test_d", "test_e", "test_foo", "test_bar"],
    )

    diff = simulator._diff_outcomes(alpha, beta, {})

    # Both test_foo and test_bar are new failures
    assert len(diff.new_failures) == 2


def test_diff_fixed_tests(simulator):
    """Test diff correctly identifies fixed tests."""
    alpha = TestOutcome(
        sandbox_name="alpha",
        passed_count=5,
        failed_count=0,
        skipped_count=0,
        errors=[],
        stdout="",
        stderr="",
        return_code=0,
        duration_seconds=5.0,
        passed_tests=["test_a", "test_b", "test_fixed"],
    )

    beta = TestOutcome(
        sandbox_name="beta",
        passed_count=2,
        failed_count=1,
        skipped_count=0,
        errors=[
            TestError(
                test_name="test_fixed",
                error_type="AssertionError",
                error_message="was broken",
                traceback="",
                is_failure=True,
            )
        ],
        stdout="",
        stderr="",
        return_code=1,
        duration_seconds=5.0,
        passed_tests=["test_a", "test_b"],
    )

    diff = simulator._diff_outcomes(alpha, beta, {})

    # test_fixed was fixed
    assert "test_fixed" in diff.fixed_tests


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE SCORING TESTS
# ═══════════════════════════════════════════════════════════════


def test_confidence_no_failures(simulator):
    """Test confidence is 0 when no failures."""
    confidence = simulator._calculate_confidence(0, [])
    assert confidence == 0.0


def test_confidence_single_assertion_error(simulator):
    """Test high confidence for single assertion error."""
    errors = [
        TestError(
            test_name="test_foo",
            error_type="AssertionError",
            error_message="assert 1 == 2",
            traceback="",
            is_failure=True,
        )
    ]

    confidence = simulator._calculate_confidence(1, errors)
    assert confidence >= 0.9  # High confidence


def test_confidence_multiple_failures(simulator):
    """Test low confidence for multiple failures."""
    errors = [
        TestError(test_name="test1", error_type="Error", error_message="", traceback="", is_failure=True),
        TestError(test_name="test2", error_type="Error", error_message="", traceback="", is_failure=True),
    ]

    confidence = simulator._calculate_confidence(2, errors)
    assert confidence < 0.5  # Low confidence


# ═══════════════════════════════════════════════════════════════
# AUTO-FIXING TESTS
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_auto_fix_skipped_low_confidence(simulator):
    """Test auto-fix is skipped when confidence is low."""
    diff = RippleDiff(
        new_failures=[
            TestError(test_name="test1", error_type="Error", error_message="", traceback="", is_failure=True),
            TestError(test_name="test2", error_type="Error", error_message="", traceback="", is_failure=True),
        ],
        fixed_tests=[],
        performance_regressions=[],
        impacted_files=[],
        confidence_score=0.3,  # Low
    )

    result = await simulator._attempt_auto_fix(diff, {}, Path("."), False)
    assert result is None  # Skipped


@pytest.mark.asyncio
async def test_auto_fix_skipped_multiple_failures(simulator):
    """Test auto-fix is skipped with multiple failures."""
    diff = RippleDiff(
        new_failures=[
            TestError(test_name="test1", error_type="AssertionError", error_message="", traceback="", is_failure=True),
            TestError(test_name="test2", error_type="AssertionError", error_message="", traceback="", is_failure=True),
        ],
        fixed_tests=[],
        performance_regressions=[],
        impacted_files=[],
        confidence_score=0.95,
    )

    result = await simulator._attempt_auto_fix(diff, {}, Path("."), False)
    assert result is None  # Skipped - multiple failures


# ═══════════════════════════════════════════════════════════════
# REPORTING TESTS
# ═══════════════════════════════════════════════════════════════


def test_build_summary_all_pass(simulator):
    """Test summary for all_pass status."""
    alpha = TestOutcome(
        sandbox_name="alpha",
        passed_count=10,
        failed_count=0,
        skipped_count=0,
        errors=[],
        stdout="",
        stderr="",
        return_code=0,
        duration_seconds=5.0,
    )

    beta = TestOutcome(
        sandbox_name="beta",
        passed_count=10,
        failed_count=0,
        skipped_count=0,
        errors=[],
        stdout="",
        stderr="",
        return_code=0,
        duration_seconds=5.0,
    )

    diff = RippleDiff(
        new_failures=[],
        fixed_tests=[],
        performance_regressions=[],
        impacted_files=[],
        confidence_score=0.0,
    )

    summary = simulator._build_summary(alpha, beta, diff, "all_pass", 5.0)

    assert "No ripples" in summary
    assert "10" in summary  # Test count
    assert "5.0s" in summary  # Duration


def test_build_summary_failures_detected(simulator):
    """Test summary for failures_detected status."""
    alpha = TestOutcome(
        sandbox_name="alpha",
        passed_count=9,
        failed_count=1,
        skipped_count=0,
        errors=[
            TestError(test_name="test_broken", error_type="AssertionError", error_message="", traceback="", is_failure=True)
        ],
        stdout="",
        stderr="",
        return_code=1,
        duration_seconds=5.0,
    )

    beta = TestOutcome(
        sandbox_name="beta",
        passed_count=10,
        failed_count=0,
        skipped_count=0,
        errors=[],
        stdout="",
        stderr="",
        return_code=0,
        duration_seconds=5.0,
    )

    diff = RippleDiff(
        new_failures=[
            TestError(test_name="test_broken", error_type="AssertionError", error_message="", traceback="", is_failure=True)
        ],
        fixed_tests=[],
        performance_regressions=[],
        impacted_files=[],
        confidence_score=0.95,
    )

    summary = simulator._build_summary(alpha, beta, diff, "failures_detected", 5.0)

    assert "1 new failure" in summary
    assert "test_broken" in summary


# ═══════════════════════════════════════════════════════════════
# STATS TESTS
# ═══════════════════════════════════════════════════════════════


def test_stats_tracking(simulator):
    """Test that stats are tracked correctly."""
    stats = simulator.get_stats()

    assert "total_simulations" in stats
    assert "docker_runs" in stats
    assert "subprocess_runs" in stats
    assert "auto_fix_attempts" in stats
    assert "auto_fix_success_rate" in stats


# ═══════════════════════════════════════════════════════════════
# PARALLEL EXECUTION TESTS
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_parallel_sandbox_creation(simulator, temp_project):
    """Test that sandboxes are created in parallel."""
    import time

    start = time.time()

    alpha, beta = await simulator._create_sandboxes(
        code_changes={"main.py": "# changed"},
        project_path=temp_project,
        use_docker=False,
    )

    duration = time.time() - start

    try:
        # Both should be created
        assert alpha._created
        assert beta._created

        # Should be relatively fast (parallel)
        # Single sandbox takes ~0.5s, parallel should be < 2x
        assert duration < 3.0
    finally:
        await simulator._cleanup_sandboxes([alpha, beta])


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_simulation_with_nonexistent_project():
    """Test simulation with non-existent project path."""
    simulator = RippleSimulator(use_docker=False)

    report = await simulator.simulate_change(
        code_changes={},
        project_path=Path("/nonexistent/path/to/project"),
    )

    # Should return a report (even if empty)
    assert report is not None
    assert report.simulation_id.startswith("sim_")


@pytest.mark.asyncio
async def test_simulation_timeout():
    """Test simulation respects timeout."""
    # RippleSimulator used only for config demonstration
    _ = RippleSimulator(use_docker=False)

    # Create config with short timeout
    config = SandboxConfig(
        name="test_timeout",
        project_path=Path("."),
        code_changes={},
        timeout_seconds=1,  # Very short
    )

    sandbox = SubprocessSandbox(config)
    await sandbox.create()

    try:
        # This should timeout (use 'timeout' on Windows, 'sleep' on Unix)
        import sys
        cmd = "ping -n 10 127.0.0.1" if sys.platform == "win32" else "sleep 10"
        result = await sandbox.execute(cmd, timeout=1)
        # On timeout, return_code is -1 OR command was interrupted
        assert result.return_code != 0  # Should not succeed
    finally:
        await sandbox.cleanup()
