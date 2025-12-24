"""
SAGA Onboarding Service
=======================

Implements the business logic for initializing new SAGA projects.
Adheres to the specification in docs/Onboarding_Flow_v1.md.

Responsibilities:
- Detect project context (language, framework, tests)
- Bootstrap governance configuration
- Generate required files (.saga directory, .cursorrules, README)
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class OnboardingConfig:
    """Configuration for the onboarding process."""
    project_root: Path
    non_interactive: bool = False
    profile: Optional[str] = None  # e.g., "python-fastapi"
    strictness: str = "standard"   # "relaxed", "standard", "faang"
    force: bool = False            # Overwrite existing files


@dataclass
class ProjectProfile:
    """Detected information about the project."""
    name: str = "unknown-project"
    language: str = "python"
    framework: str = "fastapi"     # Default assumption for now
    test_runner: str = "pytest"
    strictness: str = "standard"
    root_dir: Path = Path(".")


@dataclass
class GovernanceBootstrapResult:
    """Result of global governance selection."""
    profile_name: str
    codex_index_path: Path
    token_budget_mode: str
    strictness: str


@dataclass
class FileGenerationResult:
    """Summary of files created or modified."""
    created: list[Path]
    modified: list[Path]
    skipped: list[Path]


class OnboardingService:
    """Service to handle SAGA project initialization."""

    def __init__(self, config: OnboardingConfig):
        self.config = config
        self.root = config.project_root

    def run(self) -> FileGenerationResult:
        """Execute the full onboarding flow."""
        logger.info(f"Starting SAGA onboarding in {self.root}")

        # 1. Discovery
        profile = self.detect_project_profile()

        # 2. Bootstrap Governance
        gov_result = self.bootstrap_governance(profile)

        # 3. Generate Files
        files_result = self.generate_files(profile, gov_result)

        return files_result

    def detect_project_profile(self) -> ProjectProfile:
        """Analyze project structure to detect language and framework.

        See Onboarding_Flow_v1.md Section 3.1.
        """
        name = self.root.name if self.root.name else "project"

        # Simple heuristics for MVP
        language = "python"
        framework = "fastapi"
        test_runner = "pytest"

        # Check files
        if (self.root / "package.json").exists():
            language = "javascript"
            # Detecting js frameworks is future work

        # Determine framework in Python
        if (self.root / "pyproject.toml").exists():
            content = (self.root / "pyproject.toml").read_text()
            if "django" in content:
                framework = "django"
            elif "flask" in content:
                framework = "flask"

        # If strictness provided in config, use it
        strictness = self.config.strictness

        return ProjectProfile(
            name=name,
            language=language,
            framework=framework,
            test_runner=test_runner,
            strictness=strictness,
            root_dir=self.root
        )

    def bootstrap_governance(
        self, profile: ProjectProfile
    ) -> GovernanceBootstrapResult:
        """Select governance settings based on profile.

        See Onboarding_Flow_v1.md Section 3.2.
        """
        # Select budget mode based on strictness
        budget_mode = "balanced"
        if profile.strictness == "relaxed":
            budget_mode = "fast"
        elif profile.strictness == "faang":
            budget_mode = "einstein"

        codex_profile = f"codex_{profile.language}_{profile.framework}_v1"

        return GovernanceBootstrapResult(
            profile_name=codex_profile,
            codex_index_path=Path(".saga/sagacodex_index.json"),
            token_budget_mode=budget_mode,
            strictness=profile.strictness
        )

    def generate_files(
        self, profile: ProjectProfile, gov: GovernanceBootstrapResult
    ) -> FileGenerationResult:
        """Generate or patch required SAGA files.

        See Onboarding_Flow_v1.md Section 3.3.
        """
        created: list[Path] = []
        modified: list[Path] = []
        skipped: list[Path] = []

        saga_dir = self.root / ".saga"
        saga_dir.mkdir(exist_ok=True)

        # 1. .saga/config.yaml
        config_path = saga_dir / "config.yaml"
        if not config_path.exists() or self.config.force:
            config_data = {
                "version": "1.0",
                "project": {
                    "name": profile.name,
                    "framework": profile.framework
                },
                "governance": {
                    "profile": gov.profile_name,
                    "strictness": gov.strictness,
                    "budget_mode": gov.token_budget_mode
                },
                "codex": {
                    "rules_index": str(gov.codex_index_path)
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            created.append(config_path)
        else:
            skipped.append(config_path)

        # 2. .saga/lorebook.db (Empty SQLite)
        lorebook_path = saga_dir / "lorebook.db"
        if not lorebook_path.exists() or self.config.force:
            # Create empty file
            conn = sqlite3.connect(str(lorebook_path))
            conn.close()
            created.append(lorebook_path)
        else:
            skipped.append(lorebook_path)

        # 3. .saga/sagacodex_index.json (Generated from Source)
        index_path = saga_dir / "sagacodex_index.json"

        # Determine source path - assume inside project structure for MVP
        codex_source = self.root / "saga" / "config" / "sagacodex_python_fastapi.md"

        if not index_path.exists() or self.config.force:
            # Generate real index
            from saga.core.codex_index import CodexIndexGenerator

            # If source is missing (e.g. running outside repo), we might want to fallback or warn.
            # For this MVP task, we assume source exists as per prompt context.
            generator = CodexIndexGenerator(codex_source, index_path)
            try:
                generator.write_index()
                created.append(index_path)
            except Exception as e:
                logger.error(f"Failed to generate Codex Index: {e}")
                # Fallback to empty stub if generation fails to avoid blocking onboarding
                import json
                index_data = {
                    "version": "1.0.0",
                    "language": profile.language.capitalize(),
                    "framework": profile.framework,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "rules": []
                }
                with open(index_path, "w") as f:
                    json.dump(index_data, f, indent=2)
                created.append(index_path)
        else:
            skipped.append(index_path)

        # 4. .saga/logs/ directory
        logs_dir = saga_dir / "logs"
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
            created.append(logs_dir)

        # 5. .cursorrules (Append/Create)
        cursorrules_path = self.root / ".cursorrules"
        cursor_content = (
            "\n# SAGA Governance Rules\n"
            "# ---------------------\n"
            "# 1. ALWAYS consult .saga/sagacodex_index.json for style guidelines.\n"
            "# 2. Critical changes (file deletion, auth) require SAGA Review.\n"
            "# 3. Run 'saga resolve' if you encounter conflicts.\n"
        )

        if not cursorrules_path.exists():
            cursorrules_path.write_text(cursor_content)
            created.append(cursorrules_path)
        else:
            # Check if already patched
            current = cursorrules_path.read_text()
            if "SAGA Governance Rules" not in current:
                with open(cursorrules_path, "a") as f:
                    f.write(cursor_content)
                modified.append(cursorrules_path)
            else:
                skipped.append(cursorrules_path)

        # 6. README.md (Append/Create)
        readme_path = self.root / "README.md"
        badge_content = (
            "\n\n## Powered by SAGA\n\n"
            "This project is governed by [SAGA](https://github.com/arc-saga/saga).\n"
            "- **Run Server**: `saga serve`\n"
            "- **Onboarding**: `saga init`\n"
        )

        if not readme_path.exists():
            readme_path.write_text("# Project\n" + badge_content)
            created.append(readme_path)
        else:
            current = readme_path.read_text()
            if "Powered by SAGA" not in current:
                with open(readme_path, "a") as f:
                    f.write(badge_content)
                modified.append(readme_path)
            else:
                skipped.append(readme_path)

        return FileGenerationResult(created, modified, skipped)
