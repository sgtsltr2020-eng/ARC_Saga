import json
from unittest.mock import patch

import pytest

from saga.core.warden import Warden


@pytest.fixture
def mock_root(tmp_path):
    # Setup .saga directory and index
    saga_dir = tmp_path / ".saga"
    saga_dir.mkdir()
    index_path = saga_dir / "sagacodex_index.json"
    index_data = {
        "rules": [
             {
                "id": "45",
                "title": "Minimize Diff Surface",
                "category": "Process",
                "tags": ["refactoring", "tests"],
                "checklist_item": "Rule 45 Checklist Item"
            }
        ]
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f)
    return str(tmp_path)

@pytest.mark.asyncio
async def test_warden_checklist_generation(mock_root):
    # Mocking dependencies to avoid complex setup
    with patch("saga.core.warden.get_codex_manager"), \
         patch("saga.core.warden.SagaConstitution"), \
         patch("saga.core.warden.Mimiry"):

        warden = Warden(project_root=mock_root)

        # 1. Test standard task
        # "test" keyword should trigger Rule 45 (tagged "tests")
        checklist_tests = warden.build_checklist_for_task("Fix logic in test_foo.py", "simple")
        assert "Rule 45 Checklist Item" in checklist_tests
        assert "Consult Mimiry if uncertain" in checklist_tests

        # 2. Test irrelevant task
        checklist_other = warden.build_checklist_for_task("Update documentation", "simple")
        assert "Rule 45 Checklist Item" not in checklist_other

@pytest.mark.asyncio
async def test_warden_decompose_uses_rule_45(mock_root):
    with patch("saga.core.warden.get_codex_manager"), \
         patch("saga.core.warden.SagaConstitution"), \
         patch("saga.core.warden.Mimiry"):

        warden = Warden(project_root=mock_root)

        # Decompose a refactor request
        tasks = await warden.decompose_into_tasks(
            "Refactor the user tests",
            {"budget": 10},
            "trace-123"
        )

        assert len(tasks) > 0
        task = tasks[0]
        # Should have Rule 45 item because of "refactor" and "tests"
        assert any("Rule 45" in item for item in task.checklist)
