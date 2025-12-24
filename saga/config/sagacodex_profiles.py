"""
SagaCodex Profile Manager
=========================

Manages language and framework-specific rule profiles.

MVP (Phase 2): Python/FastAPI hardcoded
Future (Phase 3+): TypeScript/React, Go, Rust, etc.

Architecture:
- Each profile is a collection of standards, patterns, anti-patterns
- Profiles are versioned for compatibility tracking
- Profiles can inherit from base profiles (e.g., TypeScript extends JavaScript)
- User can select profile or SAGA auto-detects from project structure

Author: ARC SAGA Development Team
Date: December 14, 2025
Status: Phase 2 Week 1 - Foundation
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LanguageProfile(Enum):
    """Supported language/framework profiles."""
    PYTHON_FASTAPI = "python_fastapi"
    PYTHON_DJANGO = "python_django"  # Future
    PYTHON_FLASK = "python_flask"  # Future
    TYPESCRIPT_REACT = "typescript_react"  # Future
    TYPESCRIPT_NEXTJS = "typescript_nextjs"  # Future
    GO = "go"  # Future
    RUST = "rust"  # Future


@dataclass
class CodeStandard:
    """
    A single code quality standard.

    Example:
        CodeStandard(
            rule_number=3,
            name="Type Safety Required",
            description="All public functions must have type hints",
            applies_to=["python"],
            tool="mypy --strict",
            example_correct="def process(data: dict) -> str: ...",
            example_wrong="def process(data): ...",
            rationale="Catches bugs at development time",
            enforcement="CI blocks PRs with mypy errors"
        )
    """
    rule_number: int
    name: str
    description: str
    applies_to: list[str]  # ["python", "typescript", etc.]
    tool: Optional[str] = None
    example_correct: Optional[str] = None
    example_wrong: Optional[str] = None
    rationale: Optional[str] = None
    enforcement: Optional[str] = None
    severity: str = "WARNING"  # CRITICAL | WARNING | INFO

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_number": self.rule_number,
            "name": self.name,
            "description": self.description,
            "applies_to": self.applies_to,
            "tool": self.tool,
            "example_correct": self.example_correct,
            "example_wrong": self.example_wrong,
            "rationale": self.rationale,
            "enforcement": self.enforcement,
            "severity": self.severity,
        }


@dataclass
class AntiPattern:
    r"""
    A pattern to avoid.

    Example:
        AntiPattern(
            name="Print Statements in Production",
            pattern=r'print\(',
            why_bad="Not structured, not parseable, not traceable",
            use_instead="logger.info() with structured context",
            severity="WARNING"
        )
    """
    name: str
    pattern: str  # Regex pattern to detect
    why_bad: str
    use_instead: str
    severity: str = "WARNING"


@dataclass
class ElitePattern:
    """
    Elite app pattern to emulate.

    Example:
        ElitePattern(
            name="Optimistic UI Updates",
            source="Linear",
            description="Update UI immediately, rollback if server fails",
            code_example="...",
            when_to_use="User actions that change state",
            benefits=["Feels instant", "Better UX", "Users forgive slow backends"]
        )
    """
    name: str
    source: str  # "Linear", "Figma", "Stripe", etc.
    description: str
    code_example: Optional[str] = None
    when_to_use: Optional[str] = None
    benefits: list[str] = field(default_factory=list)


@dataclass
class SagaCodex:
    """
    A complete set of standards for a language/framework.

    Contains:
    - Core standards (type safety, error handling, logging, etc.)
    - Language-specific patterns (async/await, dependency injection, etc.)
    - Anti-patterns to avoid
    - Elite patterns to emulate
    - Testing requirements
    - Performance guidelines
    """
    language: str
    framework: Optional[str]
    version: str
    standards: list[CodeStandard]
    anti_patterns: list[AntiPattern]
    elite_patterns: list[ElitePattern]
    testing_requirements: dict[str, Any]
    performance_guidelines: dict[str, Any]

    def get_standard(self, rule_number: int) -> Optional[CodeStandard]:
        """Get standard by rule number."""
        for standard in self.standards:
            if standard.rule_number == rule_number:
                return standard
        return None

    def get_standards_by_severity(self, severity: str) -> list[CodeStandard]:
        """Get all standards of a given severity."""
        return [s for s in self.standards if s.severity == severity]

    def search_standards(self, query: str) -> list[CodeStandard]:
        """Search standards by keyword."""
        query_lower = query.lower()
        results = []

        for standard in self.standards:
            if (query_lower in standard.name.lower() or
                query_lower in standard.description.lower()):
                results.append(standard)

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "standards": [s.to_dict() for s in self.standards],
            "anti_patterns": [
                {
                    "name": ap.name,
                    "pattern": ap.pattern,
                    "why_bad": ap.why_bad,
                    "use_instead": ap.use_instead,
                    "severity": ap.severity,
                }
                for ap in self.anti_patterns
            ],
            "elite_patterns": [
                {
                    "name": ep.name,
                    "source": ep.source,
                    "description": ep.description,
                    "code_example": ep.code_example,
                    "when_to_use": ep.when_to_use,
                    "benefits": ep.benefits,
                }
                for ep in self.elite_patterns
            ],
            "testing_requirements": self.testing_requirements,
            "performance_guidelines": self.performance_guidelines,
        }


class SagaCodexManager:
    """
    Manages language-specific SagaCodex profiles.

    MVP: Python/FastAPI hardcoded
    Future: Load from config files, support multiple profiles

    Usage:
        manager = SagaCodexManager()
        codex = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

        # Search for a standard
        standards = codex.search_standards("type safety")

        # Check anti-patterns
        violations = manager.check_code(code, LanguageProfile.PYTHON_FASTAPI)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize SagaCodex manager.

        Args:
            config_dir: Directory containing profile configs (future use)
        """
        self.config_dir = config_dir or Path(__file__).parent
        self._profiles: dict[LanguageProfile, SagaCodex] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load all available profiles."""
        # MVP: Only Python/FastAPI
        self._profiles[LanguageProfile.PYTHON_FASTAPI] = self._create_python_fastapi_profile()

        logger.info(
            "SagaCodex profiles loaded",
            extra={"profiles": list(self._profiles.keys())}
        )

    def _create_python_fastapi_profile(self) -> SagaCodex:
        """
        Create Python/FastAPI profile (MVP).

        This is SAGA's expert knowledge for building herself.
        Standards derived from Phase 1 achievements and FAANG practices.
        """
        standards = [
            CodeStandard(
                rule_number=1,
                name="Type Safety Required",
                description="All public functions must have complete type hints",
                applies_to=["python"],
                tool="mypy --strict",
                example_correct=(
                    "def process_request(user_id: str, data: Dict[str, Any]) -> Response:\n"
                    "    ..."
                ),
                example_wrong="def process_request(user_id, data):\n    ...",
                rationale="Python's dynamic typing causes runtime errors. Types catch bugs at dev time.",
                enforcement="CI blocks PRs with mypy errors",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=2,
                name="Async for I/O Operations",
                description="Use async/await for database, API calls, file I/O",
                applies_to=["python"],
                tool="None (pattern review)",
                example_correct=(
                    "async def get_user(db: AsyncSession, user_id: str) -> User:\n"
                    "    result = await db.execute(select(User).where(User.id == user_id))\n"
                    "    return result.scalar_one_or_none()"
                ),
                example_wrong=(
                    "def get_user(db: Session, user_id: str) -> User:\n"
                    "    return db.query(User).filter(User.id == user_id).first()"
                ),
                rationale="FastAPI is async-first. Blocking event loop kills performance.",
                enforcement="Code review + pattern detection",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=3,
                name="Custom Exceptions with Context",
                description="Never silent failures, never bare except:",
                applies_to=["python"],
                tool="pylint, bandit",
                example_correct=(
                    "class UserNotFoundError(Exception):\n"
                    "    def __init__(self, user_id: str):\n"
                    "        self.user_id = user_id\n"
                    "        super().__init__(f'User {user_id} not found')"
                ),
                example_wrong="try:\n    ...\nexcept:\n    pass",
                rationale="Silent failures hide bugs. Context enables debugging.",
                enforcement="Linter blocks bare except",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=4,
                name="Structured Logging Only",
                description="All logs include trace_id, operation, outcome",
                applies_to=["python"],
                tool="None (pattern review)",
                example_correct=(
                    "logger.info(\n"
                    "    'User created',\n"
                    "    extra={'user_id': user.id, 'trace_id': get_trace_id()}\n"
                    ")"
                ),
                example_wrong="print(f'User created: {user.id}')",
                rationale="Print statements not parseable, not traceable.",
                enforcement="Linter detects print() in production paths",
                severity="WARNING"
            ),
            CodeStandard(
                rule_number=5,
                name="Pydantic Validation for API Inputs",
                description="All API inputs/outputs use Pydantic models",
                applies_to=["python"],
                tool="FastAPI automatic validation",
                example_correct=(
                    "class UserCreate(BaseModel):\n"
                    "    email: EmailStr\n"
                    "    password: str = Field(min_length=8)"
                ),
                example_wrong="@app.post('/users/')\nasync def create_user(request: dict): ...",
                rationale="Automatic validation, documentation, type safety.",
                enforcement="Code review",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=6,
                name="Dependency Injection via FastAPI Depends",
                description="Use Depends() for database, auth, config",
                applies_to=["python"],
                tool="FastAPI pattern",
                example_correct=(
                    "@app.post('/users/')\n"
                    "async def create_user(\n"
                    "    user: UserCreate,\n"
                    "    db: AsyncSession = Depends(get_db)\n"
                    "): ..."
                ),
                example_wrong="db = create_engine(DATABASE_URL)\n@app.post('/users/')\nasync def create_user(user: dict): ...",
                rationale="Testable, mockable, clean separation of concerns.",
                enforcement="Code review",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=7,
                name="99% Test Coverage",
                description="All modules achieve 99% coverage (Phase 1 standard)",
                applies_to=["python"],
                tool="pytest-cov",
                example_correct="pytest --cov=saga --cov-report=term-missing --cov-fail-under=99",
                example_wrong="# No tests",
                rationale="Proven in Phase 1. High coverage catches regressions.",
                enforcement="CI blocks PRs below 99%",
                severity="CRITICAL"
            ),
            CodeStandard(
                rule_number=8,
                name="Event Sourcing for State Changes",
                description="Emit events for all state changes (Phase 1 pattern)",
                applies_to=["python"],
                tool="saga.core.events",
                example_correct=(
                    "event = Event(\n"
                    "    type=EventType.USER_CREATED,\n"
                    "    trace_id=trace_id,\n"
                    "    payload={'user_id': user.id}\n"
                    ")\n"
                    "await event_store.append(event)"
                ),
                example_wrong="# State change without event",
                rationale="Enables audit trail, rollback, debugging.",
                enforcement="Code review",
                severity="WARNING"
            ),
            CodeStandard(
                rule_number=9,
                name="Circuit Breaker for External APIs",
                description="All external API calls protected (Phase 1 pattern)",
                applies_to=["python"],
                tool="saga.resilience.circuit_breaker",
                example_correct=(
                    "@circuit_breaker\n"
                    "async def call_external_api(data: dict) -> Response:\n"
                    "    ..."
                ),
                example_wrong="async def call_external_api(data: dict):\n    # No protection",
                rationale="Prevents cascade failures, enables graceful degradation.",
                enforcement="Code review",
                severity="WARNING"
            ),
            CodeStandard(
                rule_number=10,
                name="trace_id Required for All Operations",
                description="Every operation has trace_id for debugging",
                applies_to=["python"],
                tool="contextvars",
                example_correct="trace_id = str(uuid.uuid4())\ntrace_id_var.set(trace_id)",
                example_wrong="# No trace_id",
                rationale="Enables distributed tracing, debugging production issues.",
                enforcement="Code review",
                severity="WARNING"
            ),
        ]

        anti_patterns = [
            AntiPattern(
                name="Print Statements in Production",
                pattern=r'print\(',
                why_bad="Not structured, not parseable, not traceable",
                use_instead="logger.info() with structured context",
                severity="WARNING"
            ),
            AntiPattern(
                name="Bare Except Clauses",
                pattern=r'except\s*:',
                why_bad="Catches KeyboardInterrupt, SystemExit, hides bugs",
                use_instead="except SpecificException:",
                severity="CRITICAL"
            ),
            AntiPattern(
                name="Synchronous Database Calls",
                pattern=r'db\.query\(',
                why_bad="Blocks FastAPI event loop, kills performance",
                use_instead="await db.execute(select(...))",
                severity="CRITICAL"
            ),
            AntiPattern(
                name="Global Database Connection",
                pattern=r'db\s*=\s*create_engine',
                why_bad="Not testable, no lifecycle management, connection leaks",
                use_instead="Depends(get_db) for session per request",
                severity="CRITICAL"
            ),
            AntiPattern(
                name="Type Hints with 'Any' Everywhere",
                pattern=r':\s*Any\b',
                why_bad="Defeats purpose of type checking",
                use_instead="Specific types: Dict[str, int], List[User], etc.",
                severity="WARNING"
            ),
        ]

        elite_patterns = [
            ElitePattern(
                name="Optimistic UI Updates",
                source="Linear",
                description="Update UI immediately, rollback if backend fails",
                code_example=(
                    "# Update UI state\n"
                    "set_task_completed(task_id)\n"
                    "# Send to backend\n"
                    "try:\n"
                    "    await api.complete_task(task_id)\n"
                    "except Exception:\n"
                    "    set_task_pending(task_id)  # Rollback"
                ),
                when_to_use="User actions that change state",
                benefits=["Feels instant", "Better UX", "Users forgive slow backends"]
            ),
            ElitePattern(
                name="Prefetch on Hover",
                source="Vercel",
                description="Prefetch destination on link hover",
                code_example=(
                    "<Link\n"
                    "    href='/dashboard'\n"
                    "    onMouseEnter={() => prefetch('/dashboard')}\n"
                    ">\n"
                    "    Dashboard\n"
                    "</Link>"
                ),
                when_to_use="Navigation-heavy apps",
                benefits=["Instant navigation", "Perceived performance", "Simple implementation"]
            ),
            ElitePattern(
                name="Skeleton Screens",
                source="Linear, Figma",
                description="Show content structure while loading",
                code_example=(
                    "if loading:\n"
                    "    return <UserCardSkeleton />  # Shows shape of content\n"
                    "return <UserCard user={user} />"
                ),
                when_to_use="Any async data loading",
                benefits=["Better than spinners", "Shows progress", "Feels faster"]
            ),
        ]

        testing_requirements = {
            "coverage_minimum": 99,
            "tool": "pytest-cov",
            "async_support": True,
            "fixtures_required": True,
            "integration_tests": True,
            "example": (
                "@pytest.mark.asyncio\n"
                "async def test_create_user(client: AsyncClient):\n"
                "    response = await client.post('/users/', json={...})\n"
                "    assert response.status_code == 201"
            )
        }

        performance_guidelines = {
            "api_latency_p99": "< 100ms",
            "database_queries": "Use indexes on all WHERE/JOIN fields",
            "n_plus_one": "Use selectinload() or joinedload()",
            "background_tasks": "Operations > 100ms use BackgroundTasks",
            "caching": "Cache expensive operations with TTL",
        }

        return SagaCodex(
            language="Python",
            framework="FastAPI",
            version="1.0.0",
            standards=standards,
            anti_patterns=anti_patterns,
            elite_patterns=elite_patterns,
            testing_requirements=testing_requirements,
            performance_guidelines=performance_guidelines,
        )

    def get_profile(self, profile: LanguageProfile) -> SagaCodex:
        """
        Get SagaCodex for a language/framework profile.

        Args:
            profile: Language profile enum

        Returns:
            SagaCodex instance

        Raises:
            NotImplementedError: If profile not yet supported

        Example:
            >>> manager = SagaCodexManager()
            >>> codex = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)
            >>> print(codex.language)
            Python
        """
        if profile not in self._profiles:
            raise NotImplementedError(
                f"Profile {profile.value} not yet implemented. "
                f"Available: {[p.value for p in self._profiles.keys()]}"
            )

        return self._profiles[profile]

    def get_current_profile(self) -> SagaCodex:
        """
        Get current profile (MVP: always Python/FastAPI).

        Future: Will detect from project structure or user config.
        """
        return self.get_profile(LanguageProfile.PYTHON_FASTAPI)

    def check_code(
        self,
        code: str,
        profile: LanguageProfile
    ) -> list[dict[str, str]]:
        """
        Check code against anti-patterns.

        Args:
            code: Code to check
            profile: Language profile to use

        Returns:
            List of violations with details

        Example:
            >>> violations = manager.check_code("print('hello')", LanguageProfile.PYTHON_FASTAPI)
            >>> print(violations[0]['name'])
            Print Statements in Production
        """
        import re

        codex = self.get_profile(profile)
        violations = []

        for anti_pattern in codex.anti_patterns:
            if re.search(anti_pattern.pattern, code):
                violations.append({
                    "name": anti_pattern.name,
                    "pattern": anti_pattern.pattern,
                    "why_bad": anti_pattern.why_bad,
                    "use_instead": anti_pattern.use_instead,
                    "severity": anti_pattern.severity,
                })

        if violations:
            logger.warning(
                "Anti-patterns detected",
                extra={"count": len(violations), "profile": profile.value}
            )

        return violations

    def export_profile(self, profile: LanguageProfile, output_path: Path) -> None:
        """
        Export profile to JSON for external tools.

        Args:
            profile: Profile to export
            output_path: Path to write JSON file

        Example:
            >>> manager.export_profile(
            ...     LanguageProfile.PYTHON_FASTAPI,
            ...     Path("sagacodex_python_fastapi.json")
            ... )
        """
        codex = self.get_profile(profile)

        with open(output_path, 'w') as f:
            json.dump(codex.to_dict(), f, indent=2)

        logger.info(
            "Profile exported",
            extra={"profile": profile.value, "path": str(output_path)}
        )


# Singleton instance for easy access
_manager: Optional[SagaCodexManager] = None

def get_codex_manager() -> SagaCodexManager:
    """Get singleton SagaCodex manager instance."""
    global _manager
    if _manager is None:
        _manager = SagaCodexManager()
    return _manager


# Export main classes
__all__ = [
    "SagaCodexManager",
    "SagaCodex",
    "CodeStandard",
    "AntiPattern",
    "ElitePattern",
    "LanguageProfile",
    "get_codex_manager",
]
