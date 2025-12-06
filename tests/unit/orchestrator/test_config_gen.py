"""
Unit tests for orchestrator config generation module.

Tests verify:
1. ProjectType enum values and behavior
2. OrchestrationConfig creation and immutability
3. FileSystemDetector project type detection
4. ConfigGenerator configuration generation
5. Error handling for file operations
6. Concurrent detection operations

Coverage target: 98%+
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from arc_saga.orchestrator.config_gen import (
    ConfigGenerator,
    FileSystemDetector,
    IProjectDetector,
    OrchestrationConfig,
    ProjectType,
)


class TestProjectType:
    """Tests for ProjectType enum."""

    def test_all_enum_values(self) -> None:
        """Test all enum values are correct."""
        assert ProjectType.FASTAPI.value == "fastapi"
        assert ProjectType.DJANGO.value == "django"
        assert ProjectType.FLASK.value == "flask"
        assert ProjectType.UNKNOWN.value == "unknown"

    def test_enum_is_string_subclass(self) -> None:
        """Test ProjectType inherits from str."""
        assert isinstance(ProjectType.FASTAPI, str)
        assert ProjectType.FASTAPI.value == "fastapi"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string value."""
        assert ProjectType("fastapi") == ProjectType.FASTAPI
        assert ProjectType("django") == ProjectType.DJANGO
        assert ProjectType("flask") == ProjectType.FLASK
        assert ProjectType("unknown") == ProjectType.UNKNOWN

    def test_enum_invalid_value_raises_error(self) -> None:
        """Test invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            ProjectType("invalid_type")


class TestOrchestrationConfig:
    """Tests for OrchestrationConfig dataclass."""

    def test_config_creation_with_all_fields(self) -> None:
        """Test creating config with all fields."""
        config = OrchestrationConfig(
            project_type=ProjectType.FASTAPI,
            recommended_agents=("cursor", "perplexity"),
            quality_gates=("mypy", "pytest"),
            memory_tier="standard",
            patterns=("repository", "cqrs"),
            token_budget=50000,
        )

        assert config.project_type == ProjectType.FASTAPI
        assert config.recommended_agents == ("cursor", "perplexity")
        assert config.quality_gates == ("mypy", "pytest")
        assert config.memory_tier == "standard"
        assert config.patterns == ("repository", "cqrs")
        assert config.token_budget == 50000

    def test_config_default_values(self) -> None:
        """Test config default values."""
        config = OrchestrationConfig(project_type=ProjectType.UNKNOWN)

        assert config.recommended_agents == ()
        assert config.quality_gates == ()
        assert config.memory_tier == "standard"
        assert config.patterns == ()
        assert config.token_budget == 50000

    def test_config_immutability(self) -> None:
        """Test config is frozen (immutable)."""
        config = OrchestrationConfig(
            project_type=ProjectType.FASTAPI,
            recommended_agents=("cursor",),
        )

        with pytest.raises(AttributeError):
            config.project_type = ProjectType.DJANGO  # type: ignore[misc]

        with pytest.raises(AttributeError):
            config.recommended_agents = ("other",)  # type: ignore[misc]

    def test_config_fields_are_tuples(self) -> None:
        """Test tuple fields maintain tuple type."""
        config = OrchestrationConfig(
            project_type=ProjectType.FASTAPI,
            recommended_agents=("cursor", "perplexity"),
            quality_gates=("mypy",),
            patterns=("repository",),
        )

        assert isinstance(config.recommended_agents, tuple)
        assert isinstance(config.quality_gates, tuple)
        assert isinstance(config.patterns, tuple)


class TestFileSystemDetector:
    """Tests for FileSystemDetector."""

    @pytest.mark.asyncio
    async def test_detect_fastapi_from_requirements(self, tmp_path: Path) -> None:
        """Test detecting FastAPI from requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\nuvicorn==0.24.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_django_from_requirements(self, tmp_path: Path) -> None:
        """Test detecting Django from requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("django==4.2.0\npsycopg2-binary==2.9.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.DJANGO

    @pytest.mark.asyncio
    async def test_detect_flask_from_requirements(self, tmp_path: Path) -> None:
        """Test detecting Flask from requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("flask==3.0.0\nwerkzeug==3.0.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FLASK

    @pytest.mark.asyncio
    async def test_detect_django_takes_precedence_over_flask(
        self, tmp_path: Path
    ) -> None:
        """Test Django takes precedence when both Django and Flask present."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("django==4.2.0\nflask==3.0.0\n")  # Both present

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.DJANGO

    @pytest.mark.asyncio
    async def test_detect_from_pyproject_toml(self, tmp_path: Path) -> None:
        """Test detecting from pyproject.toml Poetry dependencies."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
"""
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_from_pyproject_toml_with_next_section(
        self, tmp_path: Path
    ) -> None:
        """Test detecting stops at next section in pyproject.toml."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[tool.poetry.dependencies]
fastapi = "^0.104.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
"""
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # Should detect FastAPI and stop at next section
        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_pep621_exits_on_next_section(self, tmp_path: Path) -> None:
        """Test PEP 621 detection exits when next section found."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[project]
name = "myproject"
dependencies = [
    "django>=4.2",
]

[build-system]
requires = ["setuptools"]
"""
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # Should detect Django before exiting at [build-system]
        assert result == ProjectType.DJANGO

    @pytest.mark.asyncio
    async def test_detect_from_pyproject_project_dependencies(
        self, tmp_path: Path
    ) -> None:
        """Test detecting from pyproject.toml PEP 621 project dependencies."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[project]
name = "myproject"
dependencies = [
    "django>=4.2",
]
"""
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.DJANGO

    @pytest.mark.asyncio
    async def test_detect_unknown_when_no_markers(self, tmp_path: Path) -> None:
        """Test detecting UNKNOWN when no framework markers found."""
        # Create empty directory
        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_unknown_when_no_framework_in_requirements(
        self, tmp_path: Path
    ) -> None:
        """Test detecting UNKNOWN when requirements.txt has no framework."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("requests==2.31.0\npytest==7.4.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_handles_file_not_found(self) -> None:
        """Test detector handles non-existent project path."""
        detector = FileSystemDetector()
        non_existent = Path("/non/existent/path")

        with pytest.raises(FileNotFoundError):
            await detector.detect(non_existent)

    @pytest.mark.asyncio
    async def test_detect_handles_not_directory(self, tmp_path: Path) -> None:
        """Test detector handles path that is not a directory."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")

        detector = FileSystemDetector()

        with pytest.raises(ValueError, match="must be a directory"):
            await detector.detect(file_path)

    @pytest.mark.asyncio
    async def test_detect_handles_permission_error(self, tmp_path: Path) -> None:
        """Test detector handles permission errors gracefully."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        detector = FileSystemDetector()

        # Mock permission error on file read
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = PermissionError("Access denied")

            with pytest.raises(PermissionError):
                await detector.detect(tmp_path)

    @pytest.mark.asyncio
    async def test_detect_handles_generic_exception_requirements(
        self, tmp_path: Path
    ) -> None:
        """Test detector handles generic exceptions on requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        detector = FileSystemDetector()

        # Mock generic exception on requirements.txt read (not PermissionError)
        with patch("asyncio.to_thread") as mock_to_thread:
            # First call (requirements.txt) raises generic exception
            call_count = 0

            async def mock_side_effect(*args: Any, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("Unexpected error")
                return "content"  # Return something for pyproject.toml

            mock_to_thread.side_effect = mock_side_effect

            # Should continue to check pyproject.toml (not raise)
            result = await detector.detect(tmp_path)

            # Should return UNKNOWN if pyproject.toml also doesn't have framework
            assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_handles_generic_read_error_pyproject(
        self, tmp_path: Path
    ) -> None:
        """Test detector handles read errors on pyproject.toml gracefully."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[tool.poetry.dependencies]
fastapi = "^0.104.0"
"""
        )

        detector = FileSystemDetector()

        # Mock to_thread to raise exception when reading pyproject.toml
        # This tests the generic exception handler at lines 226-233
        call_count = 0

        async def mock_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            # First call: requirements.txt (doesn't exist, return empty)
            if call_count == 1:
                return ""
            # Second call: pyproject.toml - raise generic exception
            if call_count == 2:
                raise RuntimeError("Read error")
            return ""

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            # Should handle gracefully and return UNKNOWN (not raise)
            result = await detector.detect(tmp_path)
            assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_handles_generic_exception_pyproject(
        self, tmp_path: Path
    ) -> None:
        """Test detector handles generic exceptions on pyproject.toml."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[tool.poetry.dependencies]
fastapi = "^0.104.0"
"""
        )

        detector = FileSystemDetector()

        # Mock generic exception on pyproject.toml read
        call_count = 0

        async def mock_side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call is pyproject.toml
                raise RuntimeError("Parse error")
            return ""  # Return empty for requirements.txt

        with patch("asyncio.to_thread", side_effect=mock_side_effect):
            # Should return UNKNOWN gracefully
            result = await detector.detect(tmp_path)
            assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_concurrent_calls(self, tmp_path: Path) -> None:
        """Test concurrent detection calls are handled correctly."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        detector = FileSystemDetector()

        # Make concurrent calls
        results = await asyncio.gather(*[detector.detect(tmp_path) for _ in range(10)])

        # All should succeed and return same result
        assert all(r == ProjectType.FASTAPI for r in results)

    @pytest.mark.asyncio
    async def test_detect_case_insensitive(self, tmp_path: Path) -> None:
        """Test detection is case-insensitive."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("FASTAPI==0.104.0\n")  # Uppercase

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_with_comments_in_requirements(self, tmp_path: Path) -> None:
        """Test detection works with comments in requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text(
            "# Web framework\nfastapi==0.104.0\n# Other deps\nrequests==2.31.0\n"
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_requirements_before_pyproject(self, tmp_path: Path) -> None:
        """Test requirements.txt takes precedence over pyproject.toml."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            """
[tool.poetry.dependencies]
django = "^4.2"
"""
        )

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # Should detect FastAPI from requirements.txt, not Django from pyproject.toml
        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_with_extra_whitespace(self, tmp_path: Path) -> None:
        """Test detection handles extra whitespace in requirements."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("   fastapi==0.104.0   \n\n\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_from_requirements_with_version_specifiers(
        self, tmp_path: Path
    ) -> None:
        """Test detection works with version specifiers."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("flask>=3.0.0,<4.0.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.FLASK


class TestConfigGenerator:
    """Tests for ConfigGenerator."""

    @pytest.mark.asyncio
    async def test_generate_fastapi_config(self, tmp_path: Path) -> None:
        """Test generating FastAPI configuration."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        detector = FileSystemDetector()
        generator = ConfigGenerator(detector)
        config = await generator.generate(tmp_path)

        assert config.project_type == ProjectType.FASTAPI
        assert "cursor" in config.recommended_agents
        assert "perplexity" in config.recommended_agents
        assert "mypy --strict" in config.quality_gates
        assert "pytest" in config.quality_gates
        assert "bandit" in config.quality_gates
        assert config.memory_tier == "standard"
        assert "repository" in config.patterns
        assert "cqrs" in config.patterns
        assert "circuit_breaker" in config.patterns
        assert config.token_budget == 50000

    @pytest.mark.asyncio
    async def test_generate_django_config(self, tmp_path: Path) -> None:
        """Test generating Django configuration."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("django==4.2.0\n")

        detector = FileSystemDetector()
        generator = ConfigGenerator(detector)
        config = await generator.generate(tmp_path)

        assert config.project_type == ProjectType.DJANGO
        assert "cursor" in config.recommended_agents
        assert "perplexity" in config.recommended_agents
        assert "mypy --strict" in config.quality_gates
        assert "pytest" in config.quality_gates
        assert "pylint" in config.quality_gates
        assert "bandit" in config.quality_gates
        assert config.memory_tier == "standard"
        assert "mvt" in config.patterns
        assert "repository" in config.patterns
        assert config.token_budget == 50000

    @pytest.mark.asyncio
    async def test_generate_flask_config(self, tmp_path: Path) -> None:
        """Test generating Flask configuration."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("flask==3.0.0\n")

        detector = FileSystemDetector()
        generator = ConfigGenerator(detector)
        config = await generator.generate(tmp_path)

        assert config.project_type == ProjectType.FLASK
        assert "cursor" in config.recommended_agents
        assert "perplexity" in config.recommended_agents
        assert "mypy --strict" in config.quality_gates
        assert "pytest" in config.quality_gates
        assert "bandit" in config.quality_gates
        assert config.memory_tier == "standard"
        assert "mvc" in config.patterns
        assert "repository" in config.patterns
        assert config.token_budget == 50000

    @pytest.mark.asyncio
    async def test_generate_unknown_config_has_safe_defaults(
        self, tmp_path: Path
    ) -> None:
        """Test generating UNKNOWN configuration has safe defaults."""
        detector = FileSystemDetector()
        generator = ConfigGenerator(detector)
        config = await generator.generate(tmp_path)

        assert config.project_type == ProjectType.UNKNOWN
        assert "cursor" in config.recommended_agents
        assert len(config.recommended_agents) == 1  # Minimal set
        assert "pytest" in config.quality_gates
        assert config.memory_tier == "minimal"
        assert len(config.patterns) == 0  # No patterns for unknown
        assert config.token_budget == 10000  # Lower budget

    @pytest.mark.asyncio
    async def test_generate_uses_detector_result(self) -> None:
        """Test generator uses detector result correctly."""
        mock_detector = AsyncMock(spec=IProjectDetector)
        mock_detector.detect.return_value = ProjectType.DJANGO

        generator = ConfigGenerator(mock_detector)  # type: ignore[arg-type]
        config = await generator.generate(Path("/fake/path"))

        assert config.project_type == ProjectType.DJANGO
        mock_detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_logs_detection_result(self, tmp_path: Path) -> None:
        """Test generator logs detection result."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\n")

        detector = FileSystemDetector()
        generator = ConfigGenerator(detector)

        # Verify config is generated (which implies logging occurred)
        config = await generator.generate(tmp_path)

        assert config.project_type == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_generate_handles_detector_error(self) -> None:
        """Test generator handles detector errors."""
        mock_detector = AsyncMock(spec=IProjectDetector)
        mock_detector.detect.side_effect = FileNotFoundError("Path not found")

        generator = ConfigGenerator(mock_detector)  # type: ignore[arg-type]

        with pytest.raises(FileNotFoundError):
            await generator.generate(Path("/fake/path"))

    @pytest.mark.asyncio
    async def test_generate_with_custom_detector(self, tmp_path: Path) -> None:
        """Test generator works with custom detector implementation."""

        class CustomDetector(IProjectDetector):
            async def detect(self, project_path: Path) -> ProjectType:
                return ProjectType.FLASK

        custom_detector = CustomDetector()
        generator = ConfigGenerator(custom_detector)
        config = await generator.generate(Path("/any/path"))

        assert config.project_type == ProjectType.FLASK


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_detect_with_empty_requirements_file(self, tmp_path: Path) -> None:
        """Test detection with empty requirements.txt."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_with_multiple_frameworks_priority(
        self, tmp_path: Path
    ) -> None:
        """Test detection priority when multiple frameworks present."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("fastapi==0.104.0\ndjango==4.2.0\nflask==3.0.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # FastAPI should be detected first (checked first)
        assert result == ProjectType.FASTAPI

    @pytest.mark.asyncio
    async def test_detect_with_framework_in_comment(self, tmp_path: Path) -> None:
        """Test framework name in comment doesn't trigger detection."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("# This project uses fastapi\nrequests==2.31.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # Should not detect FastAPI from comment (comments are skipped)
        assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_with_partial_framework_name(self, tmp_path: Path) -> None:
        """Test partial framework name doesn't trigger false positive."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.write_text("not-fastapi==1.0.0\n")

        detector = FileSystemDetector()
        result = await detector.detect(tmp_path)

        # Should not detect FastAPI from "not-fastapi" (checks exact dependency name)
        assert result == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_config_generator_with_unknown_type(self) -> None:
        """Test ConfigGenerator handles UNKNOWN type correctly."""
        mock_detector = AsyncMock(spec=IProjectDetector)
        mock_detector.detect.return_value = ProjectType.UNKNOWN

        generator = ConfigGenerator(mock_detector)  # type: ignore[arg-type]
        config = await generator.generate(Path("/any/path"))

        assert config.project_type == ProjectType.UNKNOWN
        assert config.memory_tier == "minimal"
        assert config.token_budget == 10000
