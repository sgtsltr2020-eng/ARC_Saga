import json
from unittest.mock import MagicMock, patch

import pytest

from saga.core.mimiry import Mimiry


@pytest.fixture
def mock_root(tmp_path):
    saga_dir = tmp_path / ".saga"
    saga_dir.mkdir()
    index_path = saga_dir / "sagacodex_index.json"
    index_data = {
        "rules": [
             {
                "id": "45",
                "title": "Minimize Diff Surface",
                "category": "Process",
                "severity": "WARNING",
                "tags": ["refactoring", "tests"],
                "description": "Do not rewrite files unnecessarily.",
                "checklist_item": "Keep diffs small"
            }
        ]
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f)
    return str(tmp_path)

@pytest.mark.asyncio
async def test_mimiry_uses_codex_index_for_rule_45(mock_root):
    # Mock CodexManager to return empty results so we fallback to Index
    mock_manager = MagicMock()
    mock_profile = MagicMock()
    mock_profile.search_standards.return_value = [] # No primary results
    mock_manager.get_current_profile.return_value = mock_profile

    with patch("saga.core.mimiry.get_codex_manager", return_value=mock_manager), \
         patch("saga.core.mimiry.CodexIndexClient") as MockClient:

        # Setup mock client behavior
        mock_client_instance = MockClient.return_value
        mock_client_instance.get_rule.return_value = {
            "id": "45",
            "title": "Minimize Diff Surface",
            "description": "Do not rewrite files unnecessarily.",
            "severity": "WARNING",
            "checklist_item": "Keep diffs small"
        }
        mock_client_instance.find_rules.return_value = []

        mimiry = Mimiry()
        # Overwrite client with configured mock just to be sure (since Init creates new one)
        mimiry.codex_client = mock_client_instance

        # Consult about refactoring tests
        response = await mimiry.consult_on_discrepancy(
            question="I want to rewrite the test file completely.",
            context={}
        )

        # Should cite Rule 45
        assert 45 in response.cited_rules
        assert "Minimize Diff Surface" in response.canonical_answer
        assert "Keep diffs small" in response.canonical_answer

@pytest.mark.asyncio
async def test_mimiry_uses_index_rules_if_relevant(mock_root):
    mock_manager = MagicMock()
    mock_profile = MagicMock()
    mock_profile.search_standards.return_value = []
    mock_manager.get_current_profile.return_value = mock_profile

    with patch("saga.core.mimiry.get_codex_manager", return_value=mock_manager), \
         patch("saga.core.mimiry.CodexIndexClient") as MockClient:

         mimiry = Mimiry()
         mimiry.codex_client.find_rules.return_value = []
         mimiry.codex_client.get_rule.return_value = None # Rule 45 not found

         response = await mimiry.consult_on_discrepancy(
             question="What is the meaning of life?",
             context={}
         )

         assert "not align with cataloged SagaCodex" in response.canonical_answer
