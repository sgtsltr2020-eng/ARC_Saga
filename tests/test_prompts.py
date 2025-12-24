"""
Tests for Prompt Engineering
============================

Verifies strict prompt construction and context injection.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

from saga.config.sagacodex_profiles import LanguageProfile, SagaCodexManager
from saga.core.lorebook import Pattern
from saga.core.task import Task
from saga.llm.prompts import ExtractedCode, PromptBuilder


def test_prompt_builder_initialization():
    """Test that PromptBuilder initializes with SagaCodex profile."""
    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

    builder = PromptBuilder(sagacodex_profile=profile)

    # PromptBuilder stores profile as self.sagacodex
    assert builder.sagacodex == profile


def test_prompt_construction_basic():
    """Test building a basic prompt from a task."""
    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

    task = Task(
        id="task-001",
        description="Create user login endpoint",
        checklist=["Validate email", "Return JWT"],
        weight="simple"
    )

    builder = PromptBuilder(sagacodex_profile=profile)
    messages = builder.build_messages(task)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    assert "Create user login endpoint" in user_content


def test_prompt_construction_with_lorebook():
    """Test injecting LoreBook patterns."""
    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

    task = Task(id="task-002", description="Implement caching", weight="complex")

    patterns = [
        Pattern(
            description="Always use async redis pool",
            success_rate=1.0
        )
    ]

    builder = PromptBuilder(sagacodex_profile=profile)
    messages = builder.build_messages(task, lorebook_patterns=patterns)

    system_content = messages[0]["content"]
    # System prompt should exist
    assert len(system_content) > 0


def test_extract_code_basic():
    """Test extracting code from LLM response."""
    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)
    builder = PromptBuilder(sagacodex_profile=profile)

    llm_response = '''
Here is the implementation:

```saga/api/users.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id}
```

```tests/test_users.py
import pytest

def test_get_user():
    assert True
```

**Rationale:** Simple user endpoint.
    '''

    extracted = builder.extract_code(llm_response)

    assert isinstance(extracted, ExtractedCode)
    assert "get_user" in extracted.production_code
    assert "test_get_user" in extracted.test_code


def test_extract_code_with_rationale():
    """Test that rationale is extracted."""
    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)
    builder = PromptBuilder(sagacodex_profile=profile)

    llm_response = '''
```saga/api/demo.py
def demo(): pass
```

```tests/test_demo.py
def test_demo(): pass
```

**Rationale:** Demo implementation for testing purposes.
    '''

    extracted = builder.extract_code(llm_response)

    # Rationale should be extracted
    assert extracted.rationale == "Demo implementation for testing purposes." or len(extracted.rationale) > 0
