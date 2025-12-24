"""
Tests for SAGA Onboarding Flow
==============================

Verifies correct file generation and logic for saga init.
"""

from pathlib import Path

import pytest
import yaml  # type: ignore

from saga.core.onboarding import OnboardingConfig, OnboardingService


@pytest.fixture
def test_project(tmp_path: Path) -> Path:
    """Create a temporary project environment."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Simulate a Python FastAPI project
    (project_dir / "pyproject.toml").write_text('[tool.poetry.dependencies]\nfastapi = "^0.100.0"')
    (project_dir / "app").mkdir()
    (project_dir / "app" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")

    return project_dir


class TestOnboardingService:

    def test_happy_path_fastapi(self, test_project: Path) -> None:
        """Test default onboarding for a FastAPI project."""
        config = OnboardingConfig(
            project_root=test_project,
            non_interactive=True,
            strictness="standard"
        )

        service = OnboardingService(config)
        result = service.run()

        # Verify files created
        saga_dir = test_project / ".saga"
        assert saga_dir.exists()
        assert (saga_dir / "config.yaml").exists()
        assert (saga_dir / "lorebook.db").exists()
        assert (saga_dir / "sagacodex_index.json").exists()
        assert (saga_dir / "logs").exists()

        # Verify .cursorrules
        assert (test_project / ".cursorrules").exists()
        assert "SAGA Governance Rules" in (test_project / ".cursorrules").read_text()

        # Verify README
        assert (test_project / "README.md").exists()
        assert "Powered by SAGA" in (test_project / "README.md").read_text()

        # Verify profile detection logic in config
        with open(saga_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
            assert cfg["project"]["framework"] == "fastapi"
            assert cfg["governance"]["profile"] == "codex_python_fastapi_v1"
            assert cfg["codex"]["rules_index"].endswith("sagacodex_index.json")

    def test_idempotency(self, test_project: Path) -> None:
        """Test that running twice doesn't break things."""
        config = OnboardingConfig(
            project_root=test_project,
            non_interactive=True
        )
        service = OnboardingService(config)

        # First run
        res1 = service.run()
        assert len(res1.created) > 0

        # Second run
        res2 = service.run()

        # Should skip existing files
        assert len(res2.created) == 0
        assert len(res2.skipped) > 0

        # .cursorrules shouldn't have duplicate content
        content = (test_project / ".cursorrules").read_text()
        assert content.count("SAGA Governance Rules") == 1

    def test_overwrite_force(self, test_project: Path) -> None:
        """Test that --yes/force overwrites config."""
        config = OnboardingConfig(
            project_root=test_project,
            non_interactive=True,
            force=True
        )
        service = OnboardingService(config)

        # Run once
        service.run()

        # Modify config manually
        cfg_path = test_project / ".saga" / "config.yaml"
        cfg_path.write_text("modified: true")

        # Run again with force
        service.run()

        # Should be reset (basic check)
        content = cfg_path.read_text()
        assert "version: '1.0'" in content or "version: 1.0" in content

    def test_project_detection(self, test_project: Path) -> None:
        """Test heuristics for project detection."""
        # Setup specific structure
        (test_project / "pyproject.toml").write_text("django")

        config = OnboardingConfig(
            project_root=test_project,
            non_interactive=True
        )
        service = OnboardingService(config)

        profile = service.detect_project_profile()
        assert profile.framework == "django"
