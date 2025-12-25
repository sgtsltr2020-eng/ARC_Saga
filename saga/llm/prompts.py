"""
Prompt Templates - SagaCodex-Enforcing System
==============================================

Builds prompts that force LLMs to generate code per SAGA standards.
Injects SagaCodex rules, LoreBook patterns, and task context.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3B - Agent Execution Framework
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from saga.config.sagacodex_profiles import SagaCodex
from saga.core.lorebook import Pattern
from saga.core.task import Task

logger = logging.getLogger(__name__)


@dataclass
class ExtractedCode:
    """Extracted code from LLM response."""

    production_code: str
    test_code: str
    file_path: str
    test_path: str
    rationale: str


class PromptBuilder:
    """
    Builds SagaCodex-enforcing prompts for coding agents.

    Architecture:
        - System prompt: SagaCodex rules + LoreBook patterns
        - Task prompt: Specific request + context + examples
        - Output format: Structured code blocks

    Usage:
        builder = PromptBuilder(sagacodex_profile)

        messages = builder.build_messages(
            task=task,
            lorebook_patterns=patterns,
            project_context=context
        )

        response = await llm.chat(messages)
        code = builder.extract_code(response)
    """

    def __init__(self, sagacodex_profile: SagaCodex):
        """Initialize with SagaCodex profile."""
        self.sagacodex = sagacodex_profile

    def build_messages(
        self,
        task: Task,
        lorebook_patterns: Optional[list[Pattern]] = None,
        project_context: Optional[dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Build complete message list for LLM.

        Args:
            task: Task to execute
            lorebook_patterns: Learned patterns from LoreBook
            project_context: Current project structure
            system_prompt_override: Optional override for the system prompt

        Returns:
            List of chat messages (system + user)
        """
        lorebook_patterns = lorebook_patterns or []
        project_context = project_context or {}

        # Build system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = self._build_system_prompt(lorebook_patterns, project_context)

        # Build task prompt
        task_prompt = self._build_task_prompt(task)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]

    def _build_system_prompt(
        self,
        lorebook_patterns: list[Pattern],
        project_context: dict[str, Any]
    ) -> str:
        """Build system prompt with SagaCodex rules."""

        # Get SagaCodex rules
        rules = self._format_sagacodex_rules()

        # Format LoreBook patterns
        patterns_text = self._format_lorebook_patterns(lorebook_patterns)

        # Format project context
        context_text = self._format_project_context(project_context)

        system_prompt = f"""You are an elite coding agent in SAGA, generating production-grade Python code.

## MANDATORY REQUIREMENTS (SagaCodex)

You MUST follow these 15 standards. Code that violates these will be rejected:

{rules}

## PROJECT CONTEXT

Framework: {project_context.get('framework', 'FastAPI')}
Python Version: {project_context.get('python_version', '3.12+')}
Project Structure:
{context_text}

## LEARNED PATTERNS (LoreBook)

These patterns have proven successful in THIS project:
{patterns_text}

## OUTPUT FORMAT

Generate code in this EXACT format:

```saga/path/to/file.py
<production code here>
```

```tests/test_file.py
<test code here>
```

**Rationale:** <Brief explanation of approach and design decisions>

CRITICAL RULES
ALL functions must have type hints (mypy --strict compliant)

Use async/await for ANY I/O (DB, API, filesystem)

Use Pydantic BaseModel for ALL data structures

Use logger.info(extra={{}}) for logging (NO print statements)

Include docstrings (Napoleon format: Args, Returns, Raises)

Tests must use pytest-asyncio and achieve 99%+ coverage

Handle errors with custom exceptions (inherit from base)

Use dependency injection (FastAPI Depends())

Your code will be validated by Mimiry against these standards. Violations = rejection.
"""
        return system_prompt

    def _format_sagacodex_rules(self) -> str:
        """Format SagaCodex rules as numbered list."""
        rules_text = []

        # Core 15 standards
        standards = [
            "**Type Safety**: Full type hints (mypy --strict). All parameters, returns, variables typed.",
            "**Async I/O**: Use async/await for DB, API, filesystem, external calls. Never blocking I/O.",
            "**Pydantic Models**: All inputs/outputs are Pydantic BaseModel. Validate at boundaries.",
            "**Structured Logging**: logger.info(extra={{...}}). NO print(). Include context (user_id, request_id).",
            "**Error Handling**: Custom exceptions inheriting from base. Include context. Never bare except.",
            "**Dependency Injection**: Use FastAPI Depends() or similar. No global state.",
            "**Tests First**: pytest-asyncio tests BEFORE implementation. 99%+ coverage required.",
            "**Docstrings**: Napoleon format (Args, Returns, Raises). All public functions.",
            "**Security**: Input validation, SQL injection prevention, XSS protection, CSRF tokens.",
            "**Database**: Async SQLAlchemy 2.0+ or async Prisma. Connection pooling. Transactions.",
            "**API Design**: RESTful conventions. OpenAPI/Swagger. Versioning (/v1/). Pagination.",
            "**Performance**: Database indices. N+1 query prevention. Caching (async Redis).",
            "**Idempotency**: POST/PATCH/DELETE operations idempotent. Idempotency keys for critical ops.",
            "**Observability**: Structured logs, request tracing (trace_id), metrics (latency, errors).",
            "**File Naming**: snake_case files. Descriptive names. Modular structure (api/, core/, models/)."
        ]

        for i, standard in enumerate(standards, 1):
            rules_text.append(f"{i}. {standard}")

        return "\\n".join(rules_text)

    def _format_lorebook_patterns(self, patterns: list[Pattern]) -> str:
        """Format LoreBook patterns."""
        if not patterns:
            return "No patterns yet (this is a new project)."

        pattern_lines = []
        for pattern in patterns[:5]:  # Top 5 patterns
            pattern_lines.append(
                f"- {pattern.description} (Success rate: {pattern.success_rate:.0%})"
            )

        return "\\n".join(pattern_lines)

    def _format_project_context(self, context: dict[str, Any]) -> str:
        """Format project structure."""
        lines = []

        if context.get("existing_models"):
            lines.append(f"  Existing Models: {', '.join(context['existing_models'][:5])}")

        if context.get("existing_apis"):
            lines.append(f"  Existing APIs: {', '.join(context['existing_apis'][:5])}")

        if context.get("database"):
            lines.append(f"  Database: {context['database']}")

        if not lines:
            lines.append("  (New project, no existing structure)")

        return "\\n".join(lines)

    def _build_task_prompt(self, task: Task) -> str:
        """Build task-specific prompt."""

        # Determine task type
        task_type = self._infer_task_type(task)

        # Add examples for task type
        example = self._get_example_for_task_type(task_type)

        # Get metadata with fallback for tasks without metadata attr
        metadata = getattr(task, 'metadata', {}) or {}

        task_prompt = f"""## TASK
{task.description}

Requirements:
{self._format_task_requirements(task)}

Expected Files:

Production: saga/{metadata.get('module', 'api')}/{metadata.get('file', 'handler')}.py

Tests: tests/test_{metadata.get('file', 'handler')}.py

EXAMPLE (Similar Task)
{example}

Now generate the code for the task above, following ALL SagaCodex standards.
"""
        return task_prompt

    def _infer_task_type(self, task: Task) -> str:
        """Infer task type from description."""
        desc_lower = task.description.lower()

        if "endpoint" in desc_lower or "api" in desc_lower or any(method in desc_lower for method in ["get", "post", "put", "delete", "patch"]):
            return "api_endpoint"
        elif "model" in desc_lower or "schema" in desc_lower:
            return "data_model"
        elif "database" in desc_lower or "migration" in desc_lower:
            return "database"
        elif "auth" in desc_lower or "login" in desc_lower or "jwt" in desc_lower:
            return "authentication"
        else:
            return "general"

    def _get_example_for_task_type(self, task_type: str) -> str:
        """Get few-shot example for task type."""

        examples = {
            "api_endpoint": """
```saga/api/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from saga.core.db import get_db
from saga.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/users", tags=["users"])

class UserResponse(BaseModel):
    id: int
    email: str
    name: str

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    \"\"\"
    Retrieve user by ID.

    Args:
        user_id: User ID
        db: Database session

    Returns:
        User data

    Raises:
        HTTPException: 404 if user not found
    \"\"\"
    logger.info("Fetching user", extra={"user_id": user_id})

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name
    )
```

```tests/test_users.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_get_user_success(client: AsyncClient, test_user):
    response = await client.get(f"/v1/users/{test_user.id}")
    assert response.status_code == 200
    assert response.json()["email"] == test_user.email

@pytest.mark.asyncio
async def test_get_user_not_found(client: AsyncClient):
    response = await client.get("/v1/users/99999")
    assert response.status_code == 404
```

**Rationale:** RESTful design, async DB access, Pydantic validation, structured logging, 404 handling.
""",

            "data_model": """
```saga/models/user.py
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    \"\"\"User creation schema.\"\"\"
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1, max_length=100)

    @validator('password')
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v

class User(BaseModel):
    \"\"\"User model.\"\"\"
    id: int
    email: EmailStr
    name: str
    created_at: datetime

    class Config:
        orm_mode = True
```

```tests/test_models.py
import pytest
from pydantic import ValidationError
# Assuming imports of models

def test_user_create_valid():
    user = UserCreate(
        email="test@example.com",
        password="Secret123",
        name="Test User"
    )
    assert user.email == "test@example.com"

def test_user_create_weak_password():
    with pytest.raises(ValidationError):
        UserCreate(
            email="test@example.com",
            password="weak",
            name="Test"
        )
```

**Rationale:** Pydantic validation, strong typing, email validation, password strength checks.
"""
        }

        return examples.get(task_type, examples["api_endpoint"])

    def _format_task_requirements(self, task: Task) -> str:
        """Format task requirements as bullet list."""
        requirements = []

        # Add task-specific requirements (with fallback for tasks without metadata)
        metadata = getattr(task, 'metadata', None) or {}
        if metadata:
            if metadata.get("input_model"):
                requirements.append(f"- Input: {metadata['input_model']}")
            if metadata.get("output_model"):
                requirements.append(f"- Output: {metadata['output_model']}")
            if metadata.get("database_tables"):
                requirements.append(f"- Tables: {metadata['database_tables']}")

        # Default requirements
        if not requirements:
            requirements.append("- Follow SagaCodex standards")
            requirements.append("- Include comprehensive tests")

        return "\\n".join(requirements)

    def extract_code(self, llm_response: str) -> ExtractedCode:
        """
        Extract code blocks from LLM response.

        Args:
            llm_response: Raw LLM response text

        Returns:
            ExtractedCode with production + test code

        Raises:
            ValueError: If code blocks not found or malformed
        """
        # Extract production code
        # Pattern match: ```filename.py ... ```
        prod_match = re.search(
            r"```(\S+)\n(.*?)```",
            llm_response,
            re.DOTALL
        )

        if not prod_match:
            # Try looser match if filename in code block tag
            prod_match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL)
            if not prod_match:
                raise ValueError("Production code block not found in LLM response")
            # If default python block, assume generic name if not found
            file_path = "saga/generated_code.py"
            production_code = prod_match.group(1).strip()
        else:
            file_path = prod_match.group(1).strip()
            production_code = prod_match.group(2).strip()
            # If formatting was ```python:path``` cleanup
            if ":" in file_path:
                file_path = file_path.split(":")[-1]

        # Extract test code (look for tests/ or test_)
        # We search again globally to find the OTHER block

        blocks = list(re.finditer(r"```(\S*)\n(.*?)```", llm_response, re.DOTALL))
        test_code = ""
        test_path = ""

        for match in blocks:
            tag = match.group(1).strip()
            content = match.group(2).strip()

            if "test" in tag or "test_" in content[:100]:
                test_path = tag if tag and "test" in tag else f"tests/test_{file_path.split('/')[-1]}"
                test_code = content
                if ":" in test_path:
                    test_path = test_path.split(":")[-1]
                break

        if not test_code:
             # Just set empty if not mandatory or raise? The prompt requires it.
             # Let's be lenient for extraction but strict on validation
             test_code = ""
             test_path = ""

        # Extract rationale
        rationale_match = re.search(
            r"\*\*Rationale:\*\*\s*(.+?)(?:\n\n|$)",
            llm_response,
            re.DOTALL
        )

        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"

        logger.info(
            "Code extracted from LLM response",
            extra={
                "file_path": file_path,
                "test_path": test_path,
                "production_lines": len(production_code.split("\\n")),
                "test_lines": len(test_code.split("\\n"))
            }
        )

        return ExtractedCode(
            production_code=production_code,
            test_code=test_code,
            file_path=file_path,
            test_path=test_path,
            rationale=rationale
        )
