import json

import pytest

from saga.core.codex_index_client import CodexIndexClient


@pytest.fixture
def mock_index_file(tmp_path):
    index_data = {
        "rules": [
            {
                "id": "1",
                "title": "Rule 1",
                "category": "Security",
                "tags": ["auth"],
                "affected_artifacts": ["backend"],
                "checklist_item": "Check auth"
            },
            {
                "id": "2",
                "title": "Rule 2",
                "category": "Testing",
                "tags": ["tests"],
                "affected_artifacts": ["tests"],
                "checklist_item": "Write tests"
            },
             {
                "id": "45",
                "title": "Minimize Diff Surface",
                "category": "Process",
                "tags": ["refactoring", "tests"],
                "checklist_item": "Do not rewrite entire files"
            }
        ]
    }
    index_path = tmp_path / "sagacodex_index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f)
    return index_path

def test_load_index(mock_index_file):
    client = CodexIndexClient(mock_index_file)
    assert client.get_rule("1") is not None
    assert client.get_rule("99") is None

def test_find_rules_by_tag(mock_index_file):
    client = CodexIndexClient(mock_index_file)
    rules = client.find_rules(tags=["auth"])
    assert len(rules) == 1
    assert rules[0]["id"] == "1"

def test_find_rules_by_category(mock_index_file):
    client = CodexIndexClient(mock_index_file)
    rules = client.find_rules(category="Testing")
    assert len(rules) == 1
    assert rules[0]["id"] == "2"

def test_get_checklist_for_task(mock_index_file):
    client = CodexIndexClient(mock_index_file)
    checklist = client.get_checklist_for_task(["tests"])
    assert "Write tests" in checklist
    assert "Do not rewrite entire files" in checklist
