"""
Task Verifier - Multi-Layer Verification System
===============================================

Validates that 'completed' tasks actually have working code on the filesystem.
Provides progressive checks from simple existence to complex Mimiry validation.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 4.1 - Task Verification
"""

import ast
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from saga.core.task import Task

# Mimiry import skipped to avoid circular dependency if possible, or use TYPE_CHECKING
# Assuming Mimiry is passed as an object.

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of a task verification process."""
    task_id: str
    verified: bool
    verification_level: Literal["none", "exists", "syntax", "import", "tests", "mimiry"]
    issues: list[str] = field(default_factory=list)
    files_checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "verified": self.verified,
            "verification_level": self.verification_level,
            "issues": self.issues,
            "files_checked": self.files_checked
        }

class TaskVerifier:
    """
    Validates completed tasks against the filesystem.
    """

    def __init__(self, project_root: str = ".", mimiry: Optional[Any] = None) -> None:
        """
        Initialize TaskVerifier.

        Args:
            project_root: Root directory of the project.
            mimiry: Optional Mimiry instance for Level 5 validation.
        """
        self.project_root = Path(project_root)
        self.mimiry = mimiry

    async def verify_task(
        self,
        task: Task,
        level: Literal["exists", "syntax", "import", "tests", "mimiry"] = "syntax",
        code_files: Optional[dict[str, str]] = None,
        test_files: Optional[dict[str, str]] = None
    ) -> VerificationResult:
        """
        Verify a task at the specified level.

        Args:
            task: Task to verify.
            level: Verification strictness level.
            code_files: Optional map of generated code files {path: content}.
            test_files: Optional map of generated test files {path: content}.

        Returns:
            VerificationResult.
        """
        logger.info(
            "Verifying task",
            extra={"task_id": task.id, "level": level}
        )

        if task.status != "done":
            return VerificationResult(
                task_id=task.id,
                verified=False,
                verification_level=level,
                issues=["Task not done"]
            )

        # Identify files to check
        file_paths: list[str] = []
        if code_files:
            file_paths.extend(code_files.keys())
        else:
            file_paths.extend(self._infer_file_paths(task))

        # Add test files if present
        if test_files:
            file_paths.extend(test_files.keys())

        # Logic: Progressive checks
        # If level is 'exists', run _check_file_existence
        # If level is 'syntax', run exists + syntax
        # ...

        levels = ["exists", "syntax", "import", "tests", "mimiry"]
        try:
            target_idx = levels.index(level)
        except ValueError:
            target_idx = 1 # Default to syntax if invalid

        issues: list[str] = []

        # Level 1: Existence
        if target_idx >= 0:
            issues.extend(await self._check_file_existence(file_paths))
            if issues: return self._make_result(task.id, False, level, issues, file_paths)

        # Level 2: Syntax
        if target_idx >= 1:
            # Only check .py files for syntax
            py_files = [f for f in file_paths if f.endswith('.py')]
            issues.extend(await self._check_syntax(py_files, code_files))
            if issues: return self._make_result(task.id, False, level, issues, file_paths)

        # Level 3: Import
        if target_idx >= 2:
            py_files = [f for f in file_paths if f.endswith('.py') and "tests/" not in f]
            issues.extend(await self._check_imports(py_files))
            if issues: return self._make_result(task.id, False, level, issues, file_paths)

        # Level 4: Tests
        if target_idx >= 3:
            if test_files:
                issues.extend(await self._check_tests(test_files))
                if issues: return self._make_result(task.id, False, level, issues, file_paths)
            elif target_idx == 3 and not test_files:
                # If specifically asked for tests but none provided
                # Only fail if task implies tests needed? For now, we allow pass if no tests generated
                pass

        # Level 5: Mimiry
        if target_idx >= 4:
            if code_files:
                issues.extend(await self._check_mimiry(code_files, task))
                if issues: return self._make_result(task.id, False, level, issues, file_paths)

        return self._make_result(task.id, True, level, [], file_paths)

    def _make_result(
        self,
        task_id: str,
        verified: bool,
        level: str,
        issues: list[str],
        files: list[str]
    ) -> VerificationResult:
        if not verified:
             logger.warning(
                 "Verification failed",
                 extra={"task_id": task_id, "issues": issues}
             )
        return VerificationResult(
            task_id=task_id,
            verified=verified,
            verification_level=level, # type: ignore
            issues=issues,
            files_checked=files
        )

    async def _check_file_existence(self, file_paths: list[str]) -> list[str]:
        issues = []
        for path in file_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                issues.append(f"File does not exist: {path}")
            elif full_path.stat().st_size == 0:
                issues.append(f"File is empty: {path}")
        return issues

    async def _check_syntax(
        self,
        file_paths: list[str],
        code_files: Optional[dict[str, str]]
    ) -> list[str]:
        issues = []
        for path in file_paths:
            content = ""
            if code_files and path in code_files:
                content = code_files[path]
            else:
                full_path = self.project_root / path
                try:
                    content = full_path.read_text(encoding='utf-8')
                except Exception as e:
                    issues.append(f"Could not read {path}: {e}")
                    continue

            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(f"Syntax error in {path}:{e.lineno}: {e.msg}")
            except Exception as e:
                issues.append(f"Parse error in {path}: {e}")
        return issues

    async def _check_imports(self, file_paths: list[str]) -> list[str]:
        issues = []
        for path in file_paths:
            # Convert foo/bar.py -> foo.bar
            # Need to handle relative paths from project_root
            if path.endswith(".py"):
                module_name = path.replace("/", ".").replace("\\", ".").replace(".py", "")

                try:
                    # Run clean python process to check import
                    result = subprocess.run(
                        ["python", "-c", f"import {module_name}"],
                        capture_output=True,
                        text=True,
                        cwd=str(self.project_root),
                        timeout=5
                    )
                    if result.returncode != 0:
                        issues.append(f"Import failed for {path}: {result.stderr.strip()}")
                except subprocess.TimeoutExpired:
                    issues.append(f"Import check timed out for {path}")
                except Exception as e:
                    issues.append(f"Import check error for {path}: {e}")
        return issues

    async def _check_tests(self, test_files: dict[str, str]) -> list[str]:
        issues = []
        if not test_files:
            return ["No test files generated"]

        for path in test_files.keys():
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", path, "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                    timeout=30
                )
                if result.returncode != 0:
                    issues.append(f"Tests failed in {path}: {result.stdout}")
            except subprocess.TimeoutExpired:
                issues.append(f"Test run timed out for {path}")
            except Exception as e:
                issues.append(f"Test run error for {path}: {e}")
        return issues

    async def _check_mimiry(self, code_files: dict[str, str], task: Task) -> list[str]:
        if not self.mimiry:
            return ["Mimiry not available"]

        issues = []
        for path, content in code_files.items():
            try:
                # Assuming measure_against_ideal signature
                violations = await self.mimiry.measure_against_ideal(
                    code=content,
                    domain="general", # Infer or get from codex
                    trace_id=task.trace_id
                )
                if violations:
                    issues.append(f"Mimiry violations in {path}: {violations}")
            except Exception as e:
                issues.append(f"Mimiry check failed for {path}: {e}")
        return issues

    def _infer_file_paths(self, task: Task) -> list[str]:
        # Try to guess
        if task.self_check_result and "file_path" in task.self_check_result:
             val = task.self_check_result["file_path"]
             if isinstance(val, str): return [val]
             if isinstance(val, list): return val

        # TODO: Better inference based on description or convention
        return []

    async def audit_completed_tasks(
        self,
        tasks: list[Task],
        level: Literal["exists", "syntax", "import", "tests", "mimiry"] = "syntax"
    ) -> list[VerificationResult]:
        """
        Audit a list of tasks.
        """
        results = []
        verified_count = 0
        for task in tasks:
            if task.status != "done":
                continue

            result = await self.verify_task(task, level=level)
            results.append(result)
            if result.verified:
                verified_count += 1

        logger.info(
            "Task audit complete",
            extra={
                "total": len(tasks),
                "verified": verified_count,
                "failed": len(tasks) - verified_count
            }
        )
        return results
