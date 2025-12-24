"""
TaskVerifier Tests
==================

Tests for multi-layer verification system.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""


import pytest

from saga.core.task import Task
from saga.core.task_verifier import TaskVerifier


@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    # Create saga/api directory
    (project_root / "saga" / "api").mkdir(parents=True)
    (project_root / "tests").mkdir()

    return project_root


@pytest.fixture
def task_verifier(temp_project):
    """Provide TaskVerifier instance."""
    return TaskVerifier(project_root=str(temp_project))


@pytest.mark.asyncio
async def test_verify_missing_file(task_verifier, temp_project):
    """Test verification fails when file missing."""
    task = Task(
        id="task-001",
        description="Test task",
        weight="simple",
        status="done"
    )

    result = await task_verifier.verify_task(
        task=task,
        level="exists",
        code_files={"saga/api/users.py": "def test(): pass"}
    )

    assert not result.verified
    assert "does not exist" in result.issues[0]


@pytest.mark.asyncio
async def test_verify_syntax_valid(task_verifier, temp_project):
    """Test verification passes for valid syntax."""
    # Write valid Python file
    file_path = temp_project / "saga" / "api" / "users.py"
    file_path.write_text("def get_user(): return 'test'")

    task = Task(
        id="task-001",
        description="Test task",
        weight="simple",
        status="done"
    )

    result = await task_verifier.verify_task(
        task=task,
        level="syntax",
        code_files={"saga/api/users.py": "def get_user(): return 'test'"}
    )

    assert result.verified is True
    assert len(result.issues) == 0


@pytest.mark.asyncio
async def test_verify_syntax_invalid(task_verifier, temp_project):
    """Test verification fails for invalid syntax."""
    # Write invalid Python
    file_path = temp_project / "saga" / "api" / "users.py"
    file_path.write_text("def get_user( return 'test'")  # Missing closing paren

    task = Task(
        id="task-001",
        description="Test task",
        weight="simple",
        status="done"
    )

    result = await task_verifier.verify_task(
        task=task,
        level="syntax",
        code_files={"saga/api/users.py": "def get_user( return 'test'"}
    )

    assert not result.verified
    assert "Syntax error" in result.issues[0]


@pytest.mark.asyncio
async def test_verify_task_not_done(task_verifier):
    """Test verification skips tasks not marked done."""
    task = Task(
        id="task-001",
        description="Test task",
        weight="simple",
        status="pending"  # Not done
    )

    result = await task_verifier.verify_task(task, level="exists")

    assert not result.verified
    assert "not done" in result.issues[0].lower()


@pytest.mark.asyncio
async def test_audit_completed_tasks(task_verifier, temp_project):
    """Test auditing multiple tasks."""
    # Create valid file
    file1 = temp_project / "saga" / "api" / "users.py"
    file1.write_text("def get_user(): pass")

    task1 = Task(id="task-001", description="Test 1", weight="simple", status="done")
    task2 = Task(id="task-002", description="Test 2", weight="simple", status="pending")
    task3 = Task(id="task-003", description="Test 3", weight="simple", status="done")

    results = await task_verifier.audit_completed_tasks(
        tasks=[task1, task2, task3],
        level="exists"
    )

    # Should only check task1 and task3 (status=done)
    assert len(results) == 2
    task_ids = [r.task_id for r in results]
    assert "task-001" in task_ids
    assert "task-003" in task_ids
