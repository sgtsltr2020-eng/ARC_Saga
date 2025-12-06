"""
Quality Gate Administration Module.

Provides quality gate enforcement for CI/CD integration using
strategy pattern for flexible execution modes.

Key components:
- IQualityGateExecutor protocol for execution strategies
- SubprocessExecutor for local tool execution
- CIIntegrationExecutor for CI/CD API integration
- QualityGateManager for orchestrating gate checks

Quality thresholds (from decision_catalog.md):
- mypy: --strict, 0 errors
- pytest: 95%+ coverage
- pylint: >=8.0 score
- bandit: 0 issues

Example:
    >>> from arc_saga.orchestrator.admin import (
    ...     QualityGateManager, SubprocessExecutor
    ... )
    >>>
    >>> executor = SubprocessExecutor()
    >>> manager = QualityGateManager(executor)
    >>> results = await manager.enforce_quality_gates("arc_saga/")
"""

from __future__ import annotations

import asyncio
import re
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from ..error_instrumentation import log_with_context


class QualityGateType(str, Enum):
    """
    Types of quality gates.

    Attributes:
        MYPY: Static type checking
        PYTEST: Unit tests with coverage
        PYLINT: Code quality analysis
        BANDIT: Security vulnerability scanning
    """

    MYPY = "mypy"
    PYTEST = "pytest"
    PYLINT = "pylint"
    BANDIT = "bandit"


@dataclass(frozen=True)
class QualityGateResult:
    """
    Result of a quality gate check.

    Attributes:
        gate_type: Type of quality gate
        passed: Whether the gate passed
        execution_time_ms: Time taken to execute
        threshold: Required threshold value
        actual_value: Actual measured value
        failures: List of specific failure messages
        suggestions: List of improvement suggestions
        raw_output: Raw output from the tool
        executed_at: Timestamp of execution
    """

    gate_type: QualityGateType
    passed: bool
    execution_time_ms: int
    threshold: str
    actual_value: str
    failures: tuple[str, ...] = field(default_factory=tuple)
    suggestions: tuple[str, ...] = field(default_factory=tuple)
    raw_output: str = ""
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QualityGateFailure(Exception):
    """
    Exception raised when quality gates fail.

    Attributes:
        results: List of QualityGateResult for all executed gates
        failed_gates: Names of gates that failed
    """

    def __init__(
        self,
        message: str,
        results: list[QualityGateResult],
    ) -> None:
        """
        Initialize QualityGateFailure.

        Args:
            message: Error message
            results: List of gate results
        """
        self.results = results
        self.failed_gates = [r.gate_type.value for r in results if not r.passed]
        super().__init__(
            f"Quality gates failed: {', '.join(self.failed_gates)}. {message}"
        )


@runtime_checkable
class IQualityGateExecutor(Protocol):
    """
    Protocol for quality gate executor implementations.

    Defines the contract for executing quality gate checks.
    Implementations can use subprocess, CI APIs, or other methods.
    """

    @abstractmethod
    async def run_mypy(self, target: str) -> QualityGateResult:
        """
        Run mypy type checking.

        Args:
            target: Path to check

        Returns:
            QualityGateResult with mypy results
        """
        ...

    @abstractmethod
    async def run_pytest(self, target: str) -> QualityGateResult:
        """
        Run pytest with coverage.

        Args:
            target: Path to test

        Returns:
            QualityGateResult with pytest results
        """
        ...

    @abstractmethod
    async def run_pylint(self, target: str) -> QualityGateResult:
        """
        Run pylint code analysis.

        Args:
            target: Path to analyze

        Returns:
            QualityGateResult with pylint results
        """
        ...

    @abstractmethod
    async def run_bandit(self, target: str) -> QualityGateResult:
        """
        Run bandit security scanning.

        Args:
            target: Path to scan

        Returns:
            QualityGateResult with bandit results
        """
        ...


class SubprocessExecutor(IQualityGateExecutor):
    """
    Execute quality gates via subprocess.

    Uses asyncio.create_subprocess_exec for non-blocking execution.
    Includes timeout support to prevent hanging processes.

    Attributes:
        timeout_seconds: Maximum execution time per gate (default: 300)

    Example:
        >>> executor = SubprocessExecutor(timeout_seconds=120)
        >>> result = await executor.run_mypy("arc_saga/")
    """

    def __init__(self, timeout_seconds: int = 300) -> None:
        """
        Initialize SubprocessExecutor.

        Args:
            timeout_seconds: Maximum execution time per gate
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.timeout_seconds = timeout_seconds

        log_with_context(
            "info",
            "subprocess_executor_initialized",
            timeout_seconds=timeout_seconds,
        )

    async def run_mypy(self, target: str) -> QualityGateResult:
        """
        Run mypy --strict on target.

        Args:
            target: Path to type check

        Returns:
            QualityGateResult indicating pass/fail with error details
        """
        start_time = time.perf_counter()

        log_with_context(
            "info",
            "quality_gate_started",
            gate_type="mypy",
            target=target,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "mypy",
                "--strict",
                target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            combined_output = output + error_output

            # Parse mypy output for error count
            error_count = 0
            failures: list[str] = []

            # Count errors from output
            error_match = re.search(r"Found (\d+) error", combined_output)
            if error_match:
                error_count = int(error_match.group(1))

            # Extract specific error lines
            for line in combined_output.split("\n"):
                if ": error:" in line:
                    failures.append(line.strip())

            passed = process.returncode == 0 and error_count == 0

            result = QualityGateResult(
                gate_type=QualityGateType.MYPY,
                passed=passed,
                execution_time_ms=execution_time_ms,
                threshold="0 errors",
                actual_value=f"{error_count} errors",
                failures=tuple(failures[:10]),  # Limit to first 10
                suggestions=(
                    ("Run 'mypy --strict' locally to see all errors",)
                    if not passed
                    else ()
                ),
                raw_output=combined_output[:5000],  # Limit output size
            )

            log_with_context(
                "info" if passed else "warning",
                "quality_gate_completed",
                gate_type="mypy",
                passed=passed,
                error_count=error_count,
                execution_time_ms=execution_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_timeout",
                gate_type="mypy",
                timeout_seconds=self.timeout_seconds,
            )

            return QualityGateResult(
                gate_type=QualityGateType.MYPY,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="0 errors",
                actual_value="TIMEOUT",
                failures=(f"Execution timed out after {self.timeout_seconds}s",),
                suggestions=("Check for infinite loops or slow type checking",),
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_error",
                gate_type="mypy",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            return QualityGateResult(
                gate_type=QualityGateType.MYPY,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="0 errors",
                actual_value="ERROR",
                failures=(str(e),),
                suggestions=("Ensure mypy is installed: pip install mypy",),
            )

    async def run_pytest(self, target: str) -> QualityGateResult:
        """
        Run pytest with coverage on target.

        Args:
            target: Path to test

        Returns:
            QualityGateResult indicating pass/fail with coverage details
        """
        start_time = time.perf_counter()

        log_with_context(
            "info",
            "quality_gate_started",
            gate_type="pytest",
            target=target,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pytest",
                f"--cov={target}",
                "--cov-report=term-missing",
                "--cov-fail-under=95",
                "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            combined_output = output + error_output

            # Parse coverage percentage
            coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", combined_output)
            coverage = int(coverage_match.group(1)) if coverage_match else 0

            # Parse test results
            passed_tests = 0
            failed_tests = 0
            test_match = re.search(r"(\d+) passed", combined_output)
            if test_match:
                passed_tests = int(test_match.group(1))
            fail_match = re.search(r"(\d+) failed", combined_output)
            if fail_match:
                failed_tests = int(fail_match.group(1))

            passed = process.returncode == 0 and coverage >= 95

            failures: list[str] = []
            if coverage < 95:
                failures.append(f"Coverage {coverage}% below 95% threshold")
            if failed_tests > 0:
                failures.append(f"{failed_tests} test(s) failed")

            result = QualityGateResult(
                gate_type=QualityGateType.PYTEST,
                passed=passed,
                execution_time_ms=execution_time_ms,
                threshold="95% coverage",
                actual_value=f"{coverage}% coverage, {passed_tests} passed, {failed_tests} failed",
                failures=tuple(failures),
                suggestions=(
                    ("Add more tests to increase coverage",) if coverage < 95 else ()
                ),
                raw_output=combined_output[:5000],
            )

            log_with_context(
                "info" if passed else "warning",
                "quality_gate_completed",
                gate_type="pytest",
                passed=passed,
                coverage=coverage,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                execution_time_ms=execution_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_timeout",
                gate_type="pytest",
                timeout_seconds=self.timeout_seconds,
            )

            return QualityGateResult(
                gate_type=QualityGateType.PYTEST,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="95% coverage",
                actual_value="TIMEOUT",
                failures=(f"Execution timed out after {self.timeout_seconds}s",),
                suggestions=("Check for slow or hanging tests",),
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_error",
                gate_type="pytest",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            return QualityGateResult(
                gate_type=QualityGateType.PYTEST,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="95% coverage",
                actual_value="ERROR",
                failures=(str(e),),
                suggestions=("Ensure pytest-cov is installed",),
            )

    async def run_pylint(self, target: str) -> QualityGateResult:
        """
        Run pylint code analysis on target.

        Args:
            target: Path to analyze

        Returns:
            QualityGateResult with pylint score and issues
        """
        start_time = time.perf_counter()

        log_with_context(
            "info",
            "quality_gate_started",
            gate_type="pylint",
            target=target,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pylint",
                target,
                "--output-format=text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            combined_output = output + error_output

            # Parse pylint score
            score = 0.0
            score_match = re.search(
                r"Your code has been rated at (-?[\d.]+)/10", combined_output
            )
            if score_match:
                score = float(score_match.group(1))

            passed = score >= 8.0

            # Extract issue summary
            failures: list[str] = []
            if not passed:
                failures.append(f"Score {score:.2f}/10 below 8.0 threshold")

            # Count issue types
            for issue_type in ["convention", "refactor", "warning", "error"]:
                count_match = re.search(
                    rf"(\d+) {issue_type}", combined_output, re.IGNORECASE
                )
                if count_match and int(count_match.group(1)) > 0:
                    count = int(count_match.group(1))
                    failures.append(f"{count} {issue_type}(s)")

            result = QualityGateResult(
                gate_type=QualityGateType.PYLINT,
                passed=passed,
                execution_time_ms=execution_time_ms,
                threshold="8.0/10",
                actual_value=f"{score:.2f}/10",
                failures=tuple(failures[:10]),
                suggestions=(
                    ("Run 'pylint --help-msg=<msg-id>' for fix suggestions",)
                    if not passed
                    else ()
                ),
                raw_output=combined_output[:5000],
            )

            log_with_context(
                "info" if passed else "warning",
                "quality_gate_completed",
                gate_type="pylint",
                passed=passed,
                score=score,
                execution_time_ms=execution_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_timeout",
                gate_type="pylint",
                timeout_seconds=self.timeout_seconds,
            )

            return QualityGateResult(
                gate_type=QualityGateType.PYLINT,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="8.0/10",
                actual_value="TIMEOUT",
                failures=(f"Execution timed out after {self.timeout_seconds}s",),
                suggestions=(),
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_error",
                gate_type="pylint",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            return QualityGateResult(
                gate_type=QualityGateType.PYLINT,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="8.0/10",
                actual_value="ERROR",
                failures=(str(e),),
                suggestions=("Ensure pylint is installed: pip install pylint",),
            )

    async def run_bandit(self, target: str) -> QualityGateResult:
        """
        Run bandit security scanning on target.

        Args:
            target: Path to scan

        Returns:
            QualityGateResult with security findings
        """
        start_time = time.perf_counter()

        log_with_context(
            "info",
            "quality_gate_started",
            gate_type="bandit",
            target=target,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "bandit",
                "-r",
                target,
                "-f",
                "txt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            combined_output = output + error_output

            # Parse issue counts
            high_issues = 0
            medium_issues = 0
            low_issues = 0

            high_match = re.search(r"High:\s+(\d+)", combined_output)
            if high_match:
                high_issues = int(high_match.group(1))

            medium_match = re.search(r"Medium:\s+(\d+)", combined_output)
            if medium_match:
                medium_issues = int(medium_match.group(1))

            low_match = re.search(r"Low:\s+(\d+)", combined_output)
            if low_match:
                low_issues = int(low_match.group(1))

            total_issues = high_issues + medium_issues + low_issues
            passed = total_issues == 0

            failures: list[str] = []
            if high_issues > 0:
                failures.append(f"{high_issues} HIGH severity issue(s)")
            if medium_issues > 0:
                failures.append(f"{medium_issues} MEDIUM severity issue(s)")
            if low_issues > 0:
                failures.append(f"{low_issues} LOW severity issue(s)")

            result = QualityGateResult(
                gate_type=QualityGateType.BANDIT,
                passed=passed,
                execution_time_ms=execution_time_ms,
                threshold="0 issues",
                actual_value=f"{total_issues} issues ({high_issues}H/{medium_issues}M/{low_issues}L)",
                failures=tuple(failures),
                suggestions=(
                    ("Run 'bandit -r <path> --format=json' for detailed report",)
                    if not passed
                    else ()
                ),
                raw_output=combined_output[:5000],
            )

            log_with_context(
                "info" if passed else "warning",
                "quality_gate_completed",
                gate_type="bandit",
                passed=passed,
                total_issues=total_issues,
                high=high_issues,
                medium=medium_issues,
                low=low_issues,
                execution_time_ms=execution_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_timeout",
                gate_type="bandit",
                timeout_seconds=self.timeout_seconds,
            )

            return QualityGateResult(
                gate_type=QualityGateType.BANDIT,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="0 issues",
                actual_value="TIMEOUT",
                failures=(f"Execution timed out after {self.timeout_seconds}s",),
                suggestions=(),
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            log_with_context(
                "error",
                "quality_gate_error",
                gate_type="bandit",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            return QualityGateResult(
                gate_type=QualityGateType.BANDIT,
                passed=False,
                execution_time_ms=execution_time_ms,
                threshold="0 issues",
                actual_value="ERROR",
                failures=(str(e),),
                suggestions=("Ensure bandit is installed: pip install bandit",),
            )


class QualityGateManager:
    """
    Manager for orchestrating quality gate checks.

    Uses strategy pattern to delegate execution to an IQualityGateExecutor.
    Runs all gates and optionally blocks on failures.

    Attributes:
        executor: The quality gate executor implementation

    Example:
        >>> executor = SubprocessExecutor()
        >>> manager = QualityGateManager(executor)
        >>> results = await manager.enforce_quality_gates("arc_saga/")
    """

    def __init__(self, executor: IQualityGateExecutor) -> None:
        """
        Initialize QualityGateManager.

        Args:
            executor: Quality gate executor implementation
        """
        self._executor = executor

        log_with_context(
            "info",
            "quality_gate_manager_initialized",
            executor_type=type(executor).__name__,
        )

    async def enforce_quality_gates(
        self,
        target: str,
        fail_on_error: bool = True,
    ) -> list[QualityGateResult]:
        """
        Run all quality gates on target.

        Args:
            target: Path to check
            fail_on_error: Whether to raise exception on failure

        Returns:
            List of QualityGateResult for all gates

        Raises:
            QualityGateFailure: If any gate fails and fail_on_error is True
        """
        log_with_context(
            "info",
            "quality_gates_enforcement_started",
            target=target,
            fail_on_error=fail_on_error,
        )

        start_time = time.perf_counter()

        # Run all gates
        results = [
            await self._executor.run_mypy(target),
            await self._executor.run_pytest(target),
            await self._executor.run_pylint(target),
            await self._executor.run_bandit(target),
        ]

        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        all_passed = failed_count == 0

        log_with_context(
            "info" if all_passed else "warning",
            "quality_gates_enforcement_completed",
            target=target,
            all_passed=all_passed,
            passed_count=passed_count,
            failed_count=failed_count,
            total_time_ms=total_time_ms,
        )

        if not all_passed and fail_on_error:
            raise QualityGateFailure(
                f"{failed_count} of {len(results)} gates failed",
                results,
            )

        return results

    async def run_gate(
        self,
        gate_type: QualityGateType,
        target: str,
    ) -> QualityGateResult:
        """
        Run a single quality gate.

        Args:
            gate_type: Type of gate to run
            target: Path to check

        Returns:
            QualityGateResult for the specified gate
        """
        if gate_type == QualityGateType.MYPY:
            return await self._executor.run_mypy(target)
        elif gate_type == QualityGateType.PYTEST:
            return await self._executor.run_pytest(target)
        elif gate_type == QualityGateType.PYLINT:
            return await self._executor.run_pylint(target)
        else:  # BANDIT
            return await self._executor.run_bandit(target)
