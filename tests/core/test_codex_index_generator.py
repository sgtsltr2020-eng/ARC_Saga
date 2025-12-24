import json

import pytest

from saga.core.codex_index import CodexIndexGenerator
from saga.core.onboarding import OnboardingConfig, OnboardingService

# Sample Markdown Content with Rule 1 and Rule 45
SAMPLE_CODEX_MD = """
## CORE PYTHON STANDARDS

### 1. Type Safety Required

**Rule**: Every function signature must have complete type hints.
**Severity**: CRITICAL
**Tags**: types, mypy, python
**Checklist Item**: "All functions have complete type hints."

...

## PROCESS STANDARDS

### 45. Minimize Diff Surface for Safe Fixes

**Rule**: Minimal changes for static-analysis-only fixes (mypy, lint)
**Why**: Large diffs in tests mask regressions.
**Tags**: refactoring, tests, mypy, diff-size
**Checklist Item**: "When fixing lint/mypy-only issues, prefer adding annotations."
**Detection Hint**: Large diff in a test file.
"""

@pytest.fixture
def temp_workspace(tmp_path):
    return tmp_path

def test_parse_simple_rules(temp_workspace):
    """Test parsing of a simple rule from markdown."""
    md_path = temp_workspace / "codex.md"
    md_path.write_text(SAMPLE_CODEX_MD, encoding="utf-8")

    out_path = temp_workspace / "index.json"
    generator = CodexIndexGenerator(md_path, out_path)
    rules = generator.parse_markdown()

    assert len(rules) == 2

    r1 = next(r for r in rules if r.id == "1")
    assert r1.title == "Type Safety Required"
    assert r1.category == "Core Python"
    assert r1.severity == "CRITICAL" # Explicitly in MD

    r45 = next(r for r in rules if r.id == "45")
    assert r45.title == "Minimize Diff Surface for Safe Fixes"

def test_builds_index_with_rule_45_present(temp_workspace):
    """Verify Rule 45 specific fields and overrides."""
    # Create MD file without explicit severity for Rule 45 to test override
    MINIMAL_MD = """
## PROCESS STANDARDS

### 45. Minimize Diff Surface for Safe Fixes
**Rule**: Minimal changes.
"""
    # Note: Tags/Checklist missing in MD, so Generator should backfill them.

    md_path = temp_workspace / "codex_minimal.md"
    md_path.write_text(MINIMAL_MD, encoding="utf-8")

    out_path = temp_workspace / "index.json"
    generator = CodexIndexGenerator(md_path, out_path)
    rules = generator.parse_markdown()

    r45 = rules[0]
    assert r45.id == "45"
    assert r45.severity == "WARNING" # Should be overridden from CRITICAL/Default
    assert r45.category == "Process" # Should be overridden/cleaned from "Process Standards"
    assert "mypy" in r45.tags # Should be backfilled
    assert "When fixing lint/mypy-only issues" in r45.checklist_item

    # Check JSON Generation
    generator.write_index()
    assert out_path.exists()

    with open(out_path) as f:
        data = json.load(f)
        assert data["version"] == "1.0.0"
        assert len(data["rules"]) == 1
        assert data["rules"][0]["id"] == "45"

def test_onboarding_uses_real_index_generator(temp_workspace):
    """Test that onboarding generates the index file from source."""
    # Setup: Mock the repo structure inside temp_workspace
    # Onboarding expects: self.root / "saga" / "config" / "sagacodex_python_fastapi.md"

    config_dir = temp_workspace / "saga" / "config"
    config_dir.mkdir(parents=True)

    source_md = config_dir / "sagacodex_python_fastapi.md"
    source_md.write_text(SAMPLE_CODEX_MD, encoding="utf-8")

    # Run Onboarding
    config = OnboardingConfig(project_root=temp_workspace, force=True)
    service = OnboardingService(config)
    result = service.run()

    # Assert
    expected_index = temp_workspace / ".saga" / "sagacodex_index.json"
    assert expected_index.exists()
    assert expected_index in result.created

    with open(expected_index) as f:
        data = json.load(f)
        ids = [r["id"] for r in data["rules"]]
        assert "45" in ids
        assert "1" in ids
