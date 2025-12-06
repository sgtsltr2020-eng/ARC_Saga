"""
Unit tests for orchestrator admin module.

Tests verify:
1. QualityGateResult creation and validation
2. QualityGateFailure exception handling
3. SubprocessExecutor tool execution
4. QualityGateManager orchestration
5. Timeout handling
6. Result parsing

Coverage target: 98%+
"""

from __future__ import annotations

import asyncio
from datetime import timezone
from unittest.mock import AsyncMock, patch

import pytest

from arc_saga.orchestrator.admin import (
    IQualityGateExecutor,
    QualityGateFailure,
    QualityGateManager,
    QualityGateResult,
    QualityGateType,
    SubprocessExecutor,
)


class TestQualityGateType:
    """Tests for QualityGateType enum."""

    def test_all_types_have_string_values(self) -> None:
        """Test all gate types have lowercase string values."""
        assert QualityGateType.MYPY.value == "mypy"
        assert QualityGateType.PYTEST.value == "pytest"
        assert QualityGateType.PYLINT.value == "pylint"
        assert QualityGateType.BANDIT.value == "bandit"


class TestQualityGateResult:
    """Tests for QualityGateResult dataclass."""

    def test_create_passed_result(self) -> None:
        """Test creating a passed quality gate result."""
        result = QualityGateResult(
            gate_type=QualityGateType.MYPY,
            passed=True,
            execution_time_ms=1500,
            threshold="0 errors",
            actual_value="0 errors",
        )

        assert result.gate_type == QualityGateType.MYPY
        assert result.passed is True
        assert result.execution_time_ms == 1500
        assert result.threshold == "0 errors"
        assert result.actual_value == "0 errors"
        assert result.failures == ()
        assert result.suggestions == ()
        assert result.executed_at.tzinfo == timezone.utc

    def test_create_failed_result(self) -> None:
        """Test creating a failed quality gate result."""
        result = QualityGateResult(
            gate_type=QualityGateType.PYTEST,
            passed=False,
            execution_time_ms=5000,
            threshold="95% coverage",
            actual_value="87% coverage",
            failures=("Coverage 87% below threshold",),
            suggestions=("Add more tests",),
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert len(result.suggestions) == 1

    def test_result_with_raw_output(self) -> None:
        """Test result with raw output."""
        result = QualityGateResult(
            gate_type=QualityGateType.PYLINT,
            passed=True,
            execution_time_ms=2000,
            threshold="8.0/10",
            actual_value="9.5/10",
            raw_output="Your code has been rated at 9.5/10",
        )

        assert result.raw_output == "Your code has been rated at 9.5/10"


class TestQualityGateFailure:
    """Tests for QualityGateFailure exception."""

    def test_failure_with_single_failed_gate(self) -> None:
        """Test failure exception with single failed gate."""
        results = [
            QualityGateResult(
                gate_type=QualityGateType.MYPY,
                passed=True,
                execution_time_ms=100,
                threshold="0",
                actual_value="0",
            ),
            QualityGateResult(
                gate_type=QualityGateType.PYTEST,
                passed=False,
                execution_time_ms=100,
                threshold="95%",
                actual_value="80%",
            ),
        ]

        error = QualityGateFailure("Tests failed", results)

        assert error.results == results
        assert error.failed_gates == ["pytest"]
        assert "pytest" in str(error)

    def test_failure_with_multiple_failed_gates(self) -> None:
        """Test failure exception with multiple failed gates."""
        results = [
            QualityGateResult(
                gate_type=QualityGateType.MYPY,
                passed=False,
                execution_time_ms=100,
                threshold="0",
                actual_value="5",
            ),
            QualityGateResult(
                gate_type=QualityGateType.PYLINT,
                passed=False,
                execution_time_ms=100,
                threshold="8.0",
                actual_value="6.5",
            ),
        ]

        error = QualityGateFailure("Multiple failures", results)

        assert len(error.failed_gates) == 2
        assert "mypy" in error.failed_gates
        assert "pylint" in error.failed_gates


class TestSubprocessExecutorInit:
    """Tests for SubprocessExecutor initialization."""

    def test_init_with_default_timeout(self) -> None:
        """Test initialization with default timeout."""
        executor = SubprocessExecutor()

        assert executor.timeout_seconds == 300

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        executor = SubprocessExecutor(timeout_seconds=120)

        assert executor.timeout_seconds == 120

    def test_init_with_invalid_timeout_raises_error(self) -> None:
        """Test initialization with invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SubprocessExecutor(timeout_seconds=0)

        with pytest.raises(ValueError, match="must be positive"):
            SubprocessExecutor(timeout_seconds=-1)


class TestSubprocessExecutorMypy:
    """Tests for SubprocessExecutor mypy execution."""

    @pytest.mark.asyncio
    async def test_run_mypy_success(self) -> None:
        """Test running mypy with successful result."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"Success: no issues found",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_mypy("test_module")

            assert result.gate_type == QualityGateType.MYPY
            assert result.passed is True
            assert result.threshold == "0 errors"

    @pytest.mark.asyncio
    async def test_run_mypy_with_errors(self) -> None:
        """Test running mypy with errors."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"module.py:10: error: Missing return statement\nFound 1 error in 1 file",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_mypy("test_module")

            assert result.passed is False
            assert "1 errors" in result.actual_value
            assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_run_mypy_timeout(self) -> None:
        """Test mypy execution timeout."""
        executor = SubprocessExecutor(timeout_seconds=1)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_exec.return_value = mock_process

            result = await executor.run_mypy("test_module")

            assert result.passed is False
            assert "TIMEOUT" in result.actual_value
            assert "timed out" in result.failures[0]

    @pytest.mark.asyncio
    async def test_run_mypy_exception(self) -> None:
        """Test mypy execution exception handling."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("mypy not found")

            result = await executor.run_mypy("test_module")

            assert result.passed is False
            assert "ERROR" in result.actual_value
            assert "mypy not found" in result.failures[0]


class TestSubprocessExecutorPytest:
    """Tests for SubprocessExecutor pytest execution."""

    @pytest.mark.asyncio
    async def test_run_pytest_success(self) -> None:
        """Test running pytest with successful result."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"TOTAL 100 0 100%\n10 passed",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pytest("test_module")

            assert result.gate_type == QualityGateType.PYTEST
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_pytest_low_coverage(self) -> None:
        """Test running pytest with low coverage."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"TOTAL 100 20 80%\n5 passed",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pytest("test_module")

            assert result.passed is False
            assert "80%" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pytest_with_failures(self) -> None:
        """Test running pytest with test failures."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"TOTAL 100 0 100%\n8 passed, 2 failed",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pytest("test_module")

            assert result.passed is False
            assert "2 failed" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pytest_timeout(self) -> None:
        """Test pytest execution timeout."""
        executor = SubprocessExecutor(timeout_seconds=1)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_exec.return_value = mock_process

            result = await executor.run_pytest("test_module")

            assert result.passed is False
            assert "TIMEOUT" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pytest_exception(self) -> None:
        """Test pytest execution exception handling."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = RuntimeError("pytest error")

            result = await executor.run_pytest("test_module")

            assert result.passed is False
            assert "ERROR" in result.actual_value


class TestSubprocessExecutorPylint:
    """Tests for SubprocessExecutor pylint execution."""

    @pytest.mark.asyncio
    async def test_run_pylint_success(self) -> None:
        """Test running pylint with good score."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"Your code has been rated at 9.50/10",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pylint("test_module")

            assert result.gate_type == QualityGateType.PYLINT
            assert result.passed is True
            assert "9.50" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pylint_low_score(self) -> None:
        """Test running pylint with low score."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"Your code has been rated at 6.50/10\n5 convention, 3 warning",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pylint("test_module")

            assert result.passed is False
            assert "6.50" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pylint_timeout(self) -> None:
        """Test pylint execution timeout."""
        executor = SubprocessExecutor(timeout_seconds=1)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_exec.return_value = mock_process

            result = await executor.run_pylint("test_module")

            assert result.passed is False
            assert "TIMEOUT" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_pylint_exception(self) -> None:
        """Test pylint execution exception handling."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("pylint error")

            result = await executor.run_pylint("test_module")

            assert result.passed is False
            assert "ERROR" in result.actual_value


class TestSubprocessExecutorBandit:
    """Tests for SubprocessExecutor bandit execution."""

    @pytest.mark.asyncio
    async def test_run_bandit_success(self) -> None:
        """Test running bandit with no issues."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"High: 0\nMedium: 0\nLow: 0",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_bandit("test_module")

            assert result.gate_type == QualityGateType.BANDIT
            assert result.passed is True
            assert "0 issues" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_bandit_with_issues(self) -> None:
        """Test running bandit with security issues."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"High: 2\nMedium: 3\nLow: 1",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_bandit("test_module")

            assert result.passed is False
            assert "6 issues" in result.actual_value
            assert "2 HIGH" in result.failures[0]

    @pytest.mark.asyncio
    async def test_run_bandit_timeout(self) -> None:
        """Test bandit execution timeout."""
        executor = SubprocessExecutor(timeout_seconds=1)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_exec.return_value = mock_process

            result = await executor.run_bandit("test_module")

            assert result.passed is False
            assert "TIMEOUT" in result.actual_value

    @pytest.mark.asyncio
    async def test_run_bandit_exception(self) -> None:
        """Test bandit execution exception handling."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = RuntimeError("bandit error")

            result = await executor.run_bandit("test_module")

            assert result.passed is False
            assert "ERROR" in result.actual_value


class MockExecutor(IQualityGateExecutor):
    """Mock executor for testing QualityGateManager."""

    def __init__(
        self,
        mypy_pass: bool = True,
        pytest_pass: bool = True,
        pylint_pass: bool = True,
        bandit_pass: bool = True,
    ) -> None:
        self.mypy_pass = mypy_pass
        self.pytest_pass = pytest_pass
        self.pylint_pass = pylint_pass
        self.bandit_pass = bandit_pass

    async def run_mypy(self, target: str) -> QualityGateResult:
        return QualityGateResult(
            gate_type=QualityGateType.MYPY,
            passed=self.mypy_pass,
            execution_time_ms=100,
            threshold="0 errors",
            actual_value="0 errors" if self.mypy_pass else "5 errors",
        )

    async def run_pytest(self, target: str) -> QualityGateResult:
        return QualityGateResult(
            gate_type=QualityGateType.PYTEST,
            passed=self.pytest_pass,
            execution_time_ms=200,
            threshold="95%",
            actual_value="98%" if self.pytest_pass else "80%",
        )

    async def run_pylint(self, target: str) -> QualityGateResult:
        return QualityGateResult(
            gate_type=QualityGateType.PYLINT,
            passed=self.pylint_pass,
            execution_time_ms=150,
            threshold="8.0/10",
            actual_value="9.0/10" if self.pylint_pass else "6.5/10",
        )

    async def run_bandit(self, target: str) -> QualityGateResult:
        return QualityGateResult(
            gate_type=QualityGateType.BANDIT,
            passed=self.bandit_pass,
            execution_time_ms=50,
            threshold="0 issues",
            actual_value="0 issues" if self.bandit_pass else "3 issues",
        )


class TestQualityGateManager:
    """Tests for QualityGateManager."""

    @pytest.mark.asyncio
    async def test_enforce_all_gates_pass(self) -> None:
        """Test enforcing gates when all pass."""
        executor = MockExecutor()
        manager = QualityGateManager(executor)

        results = await manager.enforce_quality_gates("test_module")

        assert len(results) == 4
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_enforce_gates_with_failure_raises_error(self) -> None:
        """Test enforcing gates raises error on failure."""
        executor = MockExecutor(pytest_pass=False)
        manager = QualityGateManager(executor)

        with pytest.raises(QualityGateFailure) as exc_info:
            await manager.enforce_quality_gates("test_module")

        assert "pytest" in exc_info.value.failed_gates

    @pytest.mark.asyncio
    async def test_enforce_gates_without_fail_on_error(self) -> None:
        """Test enforcing gates without raising on failure."""
        executor = MockExecutor(pylint_pass=False)
        manager = QualityGateManager(executor)

        results = await manager.enforce_quality_gates(
            "test_module",
            fail_on_error=False,
        )

        assert len(results) == 4
        assert sum(1 for r in results if r.passed) == 3

    @pytest.mark.asyncio
    async def test_run_single_gate_mypy(self) -> None:
        """Test running single mypy gate."""
        executor = MockExecutor()
        manager = QualityGateManager(executor)

        result = await manager.run_gate(QualityGateType.MYPY, "test_module")

        assert result.gate_type == QualityGateType.MYPY
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_single_gate_pytest(self) -> None:
        """Test running single pytest gate."""
        executor = MockExecutor()
        manager = QualityGateManager(executor)

        result = await manager.run_gate(QualityGateType.PYTEST, "test_module")

        assert result.gate_type == QualityGateType.PYTEST

    @pytest.mark.asyncio
    async def test_run_single_gate_pylint(self) -> None:
        """Test running single pylint gate."""
        executor = MockExecutor()
        manager = QualityGateManager(executor)

        result = await manager.run_gate(QualityGateType.PYLINT, "test_module")

        assert result.gate_type == QualityGateType.PYLINT

    @pytest.mark.asyncio
    async def test_run_single_gate_bandit(self) -> None:
        """Test running single bandit gate."""
        executor = MockExecutor()
        manager = QualityGateManager(executor)

        result = await manager.run_gate(QualityGateType.BANDIT, "test_module")

        assert result.gate_type == QualityGateType.BANDIT

    @pytest.mark.asyncio
    async def test_multiple_gate_failures(self) -> None:
        """Test enforcing with multiple gate failures."""
        executor = MockExecutor(
            mypy_pass=False,
            pytest_pass=False,
            pylint_pass=True,
            bandit_pass=False,
        )
        manager = QualityGateManager(executor)

        with pytest.raises(QualityGateFailure) as exc_info:
            await manager.enforce_quality_gates("test_module")

        assert len(exc_info.value.failed_gates) == 3
        assert "mypy" in exc_info.value.failed_gates
        assert "pytest" in exc_info.value.failed_gates
        assert "bandit" in exc_info.value.failed_gates


class TestIQualityGateExecutorProtocol:
    """Tests for IQualityGateExecutor protocol."""

    def test_subprocess_executor_implements_protocol(self) -> None:
        """Test SubprocessExecutor implements IQualityGateExecutor."""
        executor = SubprocessExecutor()

        assert isinstance(executor, IQualityGateExecutor)

    def test_mock_executor_implements_protocol(self) -> None:
        """Test MockExecutor implements IQualityGateExecutor."""
        executor = MockExecutor()

        assert isinstance(executor, IQualityGateExecutor)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_output_handling(self) -> None:
        """Test handling of empty tool output."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_exec.return_value = mock_process

            result = await executor.run_mypy("test_module")

            # Should handle empty output gracefully
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_negative_score_pylint(self) -> None:
        """Test handling negative pylint score."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"Your code has been rated at -5.00/10",
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_pylint("test_module")

            assert result.passed is False
            assert "-5.00" in result.actual_value

    @pytest.mark.asyncio
    async def test_unicode_output_handling(self) -> None:
        """Test handling of unicode in tool output."""
        executor = SubprocessExecutor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                "Unicode: 日本語 🚀".encode("utf-8"),
                b"",
            )
            mock_exec.return_value = mock_process

            result = await executor.run_mypy("test_module")

            # Should handle unicode gracefully
            assert result is not None

