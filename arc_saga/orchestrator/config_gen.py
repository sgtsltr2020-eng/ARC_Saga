"""
Auto-Configuration Generator for Orchestrator.

Automatically detects project type and generates orchestration configuration.
Follows Phase 2 patterns from decision_catalog.md.

This module provides:
- Project type detection (FastAPI, Django, Flask)
- Automatic configuration generation based on project structure
- Protocol-based detector interface for extensibility

Example:
    >>> from arc_saga.orchestrator.config_gen import (
    ...     ConfigGenerator, FileSystemDetector
    ... )
    >>> from pathlib import Path
    >>>
    >>> detector = FileSystemDetector()
    >>> generator = ConfigGenerator(detector)
    >>> config = await generator.generate(Path("my_project"))
    >>> print(config.project_type)  # FASTAPI
    >>> print(config.recommended_agents)  # ['cursor', 'perplexity']
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..error_instrumentation import log_with_context


class ProjectType(str, Enum):
    """
    Supported project types for auto-configuration.

    Attributes:
        FASTAPI: FastAPI web framework projects
        DJANGO: Django web framework projects
        FLASK: Flask web framework projects
        UNKNOWN: Unknown or unsupported project type
    """

    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OrchestrationConfig:
    """
    Orchestration configuration for a project.

    Immutable configuration generated based on project type.
    Contains recommended agents, quality gates, memory settings,
    workflow patterns, and token budget.

    Attributes:
        project_type: Detected or specified project type
        recommended_agents: List of recommended AI agents for this project
        quality_gates: List of quality gate commands to enforce
        memory_tier: Recommended memory tier (minimal, standard, enhanced, etc.)
        patterns: List of architectural patterns to apply
        token_budget: Recommended token budget per operation

    Example:
        >>> config = OrchestrationConfig(
        ...     project_type=ProjectType.FASTAPI,
        ...     recommended_agents=["cursor", "perplexity"],
        ...     quality_gates=["mypy --strict", "pytest", "bandit"],
        ...     memory_tier="standard",
        ...     patterns=["repository", "cqrs"],
        ...     token_budget=50000,
        ... )
    """

    project_type: ProjectType
    recommended_agents: tuple[str, ...] = field(default_factory=tuple)
    quality_gates: tuple[str, ...] = field(default_factory=tuple)
    memory_tier: str = "standard"
    patterns: tuple[str, ...] = field(default_factory=tuple)
    token_budget: int = 50000


@runtime_checkable
class IProjectDetector(Protocol):
    """
    Protocol for project type detection.

    Defines the contract for detecting project types from file system.
    Implementations can use various strategies (file scanning, config parsing, etc.).
    """

    async def detect(self, project_path: Path) -> ProjectType:
        """
        Detect project type from project path.

        Args:
            project_path: Root directory of the project

        Returns:
            Detected ProjectType (UNKNOWN if cannot determine)
        """
        ...


class FileSystemDetector(IProjectDetector):
    """
    File system-based project type detector.

    Scans project files to identify framework:
    - Reads requirements.txt for dependencies
    - Falls back to pyproject.toml for Poetry projects
    - Returns UNKNOWN if no markers found

    Uses asyncio.to_thread for non-blocking file I/O operations.

    Example:
        >>> detector = FileSystemDetector()
        >>> project_type = await detector.detect(Path("my_fastapi_project"))
        >>> print(project_type)  # ProjectType.FASTAPI
    """

    async def detect(self, project_path: Path) -> ProjectType:
        """
        Detect project type by scanning files.

        Args:
            project_path: Root directory of the project

        Returns:
            Detected ProjectType or UNKNOWN if not found

        Raises:
            FileNotFoundError: If project_path does not exist
            PermissionError: If cannot read project files
        """
        if not project_path.exists():
            log_with_context(
                "warning",
                "project_path_not_found",
                project_path=str(project_path),
            )
            raise FileNotFoundError(f"Project path does not exist: {project_path}")

        if not project_path.is_dir():
            log_with_context(
                "warning",
                "project_path_not_directory",
                project_path=str(project_path),
            )
            raise ValueError(f"Project path must be a directory: {project_path}")

        log_with_context(
            "info",
            "project_detection_started",
            project_path=str(project_path),
        )

        try:
            # Try requirements.txt first
            requirements_path = project_path / "requirements.txt"
            if requirements_path.exists():
                try:
                    content = await asyncio.to_thread(
                        requirements_path.read_text, encoding="utf-8"
                    )
                    detected = self._detect_from_content(content)
                    if detected != ProjectType.UNKNOWN:
                        log_with_context(
                            "info",
                            "project_detected_from_requirements",
                            project_type=detected.value,
                            project_path=str(project_path),
                        )
                        return detected
                except PermissionError as e:
                    log_with_context(
                        "error",
                        "project_detection_permission_error",
                        file=str(requirements_path),
                        error_message=str(e),
                    )
                    raise
                except Exception as e:
                    log_with_context(
                        "warning",
                        "project_detection_read_error",
                        file=str(requirements_path),
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

            # Fallback to pyproject.toml
            pyproject_path = project_path / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    content = await asyncio.to_thread(
                        pyproject_path.read_text, encoding="utf-8"
                    )
                    detected = self._detect_from_pyproject(content)
                    if detected != ProjectType.UNKNOWN:
                        log_with_context(
                            "info",
                            "project_detected_from_pyproject",
                            project_type=detected.value,
                            project_path=str(project_path),
                        )
                        return detected
                except PermissionError as e:
                    log_with_context(
                        "error",
                        "project_detection_permission_error",
                        file=str(pyproject_path),
                        error_message=str(e),
                    )
                    raise
                except Exception as e:
                    log_with_context(
                        "warning",
                        "project_detection_read_error",
                        file=str(pyproject_path),
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

            log_with_context(
                "info",
                "project_type_unknown",
                project_path=str(project_path),
            )

            return ProjectType.UNKNOWN

        except Exception as e:
            log_with_context(
                "error",
                "project_detection_failed",
                project_path=str(project_path),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def _detect_from_content(self, content: str) -> ProjectType:
        """
        Detect project type from file content.

        Args:
            content: File content to analyze

        Returns:
            Detected ProjectType or UNKNOWN
        """
        lines = content.split("\n")
        dependencies: list[str] = []

        # Parse lines, skip comments and extract dependencies
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Remove quotes, version specifiers and whitespace
            # Handle formats like "fastapi==0.104.0", "fastapi>=0.104", fastapi~=0.104
            dep_name = (
                line.replace('"', "")
                .replace("'", "")
                .split("=")[0]
                .split(">")[0]
                .split("<")[0]
                .split("~")[0]
                .split(",")[0]
                .strip()
            )
            if dep_name:
                dependencies.append(dep_name.lower())

        # Check for FastAPI (exact match, not substring)
        if "fastapi" in dependencies:
            return ProjectType.FASTAPI

        # Check for Django (need to distinguish from Flask)
        if "django" in dependencies:
            # If both django and flask present, prioritize django
            return ProjectType.DJANGO

        # Check for Flask (only if django not present)
        if "flask" in dependencies:
            return ProjectType.FLASK

        return ProjectType.UNKNOWN

    def _detect_from_pyproject(self, content: str) -> ProjectType:
        """
        Detect project type from pyproject.toml content.

        Parses Poetry dependencies section for framework markers.

        Args:
            content: pyproject.toml content

        Returns:
            Detected ProjectType or UNKNOWN
        """
        # Look for [tool.poetry.dependencies] section
        if "[tool.poetry.dependencies]" in content.lower():
            # Extract dependencies section
            lines = content.split("\n")
            in_dependencies = False
            dependencies_lines = []

            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith("[tool.poetry.dependencies]"):
                    in_dependencies = True
                    continue
                if in_dependencies:
                    if line.strip().startswith("[") and line.strip().endswith("]"):
                        break  # Next section
                    if line.strip() and not line.strip().startswith("#"):
                        dependencies_lines.append(line)

            dependencies_text = "\n".join(dependencies_lines)
            return self._detect_from_content(dependencies_text)

        # Also check [project.dependencies] for PEP 621 projects
        if "[project]" in content.lower():
            # For PEP 621, extract all lines in [project] section that might contain dependencies
            lines = content.split("\n")
            in_project_section = False
            dependencies_found: list[str] = []

            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()

                # Enter [project] or [project.dependencies] section
                if line_lower in ("[project]", "[project.dependencies]"):
                    in_project_section = True
                    continue

                # Exit project section on next top-level section (not [project.*])
                if (
                    in_project_section
                    and line_stripped.startswith("[")
                    and not line_lower.startswith("[project")
                ):
                    break

                # Collect all lines in project section that contain dependencies keyword or values
                if in_project_section:
                    # Include lines with dependencies keyword or lines with quoted dependency names
                    if (
                        "dependencies" in line_lower
                        or '"django' in line_lower
                        or "'django" in line_lower
                        or '"fastapi' in line_lower
                        or "'fastapi" in line_lower
                        or '"flask' in line_lower
                        or "'flask" in line_lower
                    ):
                        dependencies_found.append(line_stripped)

            # Parse found dependencies
            if dependencies_found:
                deps_text = "\n".join(dependencies_found)
                # Use the existing content parser to extract dependency names
                return self._detect_from_content(deps_text)

        return ProjectType.UNKNOWN


class ConfigGenerator:
    """
    Generates orchestration configuration based on project type.

    Uses a project detector to identify the project type,
    then maps it to a predefined OrchestrationConfig.

    Attributes:
        detector: Project type detector implementation

    Example:
        >>> detector = FileSystemDetector()
        >>> generator = ConfigGenerator(detector)
        >>> config = await generator.generate(Path("my_project"))
        >>> print(config.recommended_agents)  # ['cursor', 'perplexity']
    """

    def __init__(self, detector: IProjectDetector) -> None:
        """
        Initialize ConfigGenerator.

        Args:
            detector: Project type detector to use
        """
        self._detector = detector

        log_with_context(
            "info",
            "config_generator_initialized",
            detector_type=type(detector).__name__,
        )

    async def generate(self, project_path: Path) -> OrchestrationConfig:
        """
        Generate orchestration configuration for project.

        Args:
            project_path: Root directory of the project

        Returns:
            OrchestrationConfig with recommended settings

        Raises:
            FileNotFoundError: If project_path does not exist
            PermissionError: If cannot read project files
        """
        log_with_context(
            "info",
            "config_generation_started",
            project_path=str(project_path),
        )

        # Detect project type
        project_type = await self._detector.detect(project_path)

        log_with_context(
            "info",
            "project_type_detected",
            project_type=project_type.value,
            project_path=str(project_path),
        )

        # Generate configuration based on type
        config = self._generate_for_type(project_type)

        log_with_context(
            "info",
            "config_generation_completed",
            project_type=project_type.value,
            agents_count=len(config.recommended_agents),
            quality_gates_count=len(config.quality_gates),
            project_path=str(project_path),
        )

        return config

    def _generate_for_type(self, project_type: ProjectType) -> OrchestrationConfig:
        """
        Generate configuration for specific project type.

        Args:
            project_type: Type of project to configure

        Returns:
            OrchestrationConfig with type-specific recommendations
        """
        if project_type == ProjectType.FASTAPI:
            return OrchestrationConfig(
                project_type=ProjectType.FASTAPI,
                recommended_agents=("cursor", "perplexity"),
                quality_gates=(
                    "mypy --strict",
                    "pytest",
                    "bandit",
                ),
                memory_tier="standard",
                patterns=("repository", "cqrs", "circuit_breaker"),
                token_budget=50000,
            )

        elif project_type == ProjectType.DJANGO:
            return OrchestrationConfig(
                project_type=ProjectType.DJANGO,
                recommended_agents=("cursor", "perplexity"),
                quality_gates=(
                    "mypy --strict",
                    "pytest",
                    "pylint",
                    "bandit",
                ),
                memory_tier="standard",
                patterns=("mvt", "repository"),
                token_budget=50000,
            )

        elif project_type == ProjectType.FLASK:
            return OrchestrationConfig(
                project_type=ProjectType.FLASK,
                recommended_agents=("cursor", "perplexity"),
                quality_gates=(
                    "mypy --strict",
                    "pytest",
                    "bandit",
                ),
                memory_tier="standard",
                patterns=("mvc", "repository"),
                token_budget=50000,
            )

        else:  # UNKNOWN
            return OrchestrationConfig(
                project_type=ProjectType.UNKNOWN,
                recommended_agents=("cursor",),
                quality_gates=("pytest",),
                memory_tier="minimal",
                patterns=(),
                token_budget=10000,
            )
