"""
Coding Agent Tests
==================

Tests CodingAgent's code generation and extraction logic.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from saga.agents.coder import AgentOutput, CodingAgent
from saga.config.sagacodex_profiles import LanguageProfile, SagaCodexManager
from saga.core.task import Task
from saga.llm.client import LLMResponse, Provider
from saga.llm.prompts import ExtractedCode


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


def test_coding_agent_initialization():
    """Test CodingAgent initializes with required dependencies."""
    mock_llm = AsyncMock()
    mock_lorebook = AsyncMock()
    mock_mimiry = MagicMock()

    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

    agent = CodingAgent(
        llm_client=mock_llm,
        lorebook=mock_lorebook,
        mimiry=mock_mimiry,
        sagacodex_profile=profile,
        agent_name="TestAgent"
    )

    assert agent.llm_client == mock_llm
    assert agent.lorebook == mock_lorebook


def test_coding_agent_solve_task():
    """Test CodingAgent generates code for a task."""

    async def _test():
        # Setup mocks
        mock_llm = AsyncMock()
        mock_llm_response = LLMResponse(
            text="""
```saga/api/hello.py
def hello():
    return "world"
```

```tests/test_hello.py
def test_hello():
    assert hello() == "world"
```

**Rationale:** Simple demo endpoint.
            """,
            provider=Provider.OPENAI,
            model="gpt-4-turbo",
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            estimated_cost=0.03
        )
        mock_llm.chat.return_value = mock_llm_response

        mock_lorebook = AsyncMock()
        mock_lorebook.get_relevant_decisions = AsyncMock(return_value=[])
        mock_lorebook.get_project_patterns = AsyncMock(return_value=[])

        mock_mimiry = MagicMock()
        mock_mimiry_response = MagicMock()
        mock_mimiry_response.severity = "OK"
        mock_mimiry_response.violations_detected = []
        mock_mimiry.consult_on_discrepancy = AsyncMock(return_value=mock_mimiry_response)

        manager = SagaCodexManager()
        profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

        # Create agent - no project_root param
        agent = CodingAgent(
            llm_client=mock_llm,
            lorebook=mock_lorebook,
            mimiry=mock_mimiry,
            sagacodex_profile=profile,
            agent_name="TestAgent"
        )

        # Create task
        task = Task(
            id="task-001",
            description="Create hello world endpoint",
            weight="simple",
            budget_allocation=1.0
        )

        # Execute - project_root is arg to solve_task
        result = await agent.solve_task(task, project_root="/tmp/test")

        # Verify
        assert isinstance(result, AgentOutput)
        assert result.task_id == "task-001"

    run_async(_test())


def test_coding_agent_extract_code():
    """Test code extraction from LLM response."""
    mock_llm = AsyncMock()
    mock_lorebook = AsyncMock()
    mock_mimiry = MagicMock()

    manager = SagaCodexManager()
    profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

    agent = CodingAgent(
        llm_client=mock_llm,
        lorebook=mock_lorebook,
        mimiry=mock_mimiry,
        sagacodex_profile=profile,
        agent_name="TestAgent"
    )

    llm_response = """
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

**Rationale:** User API endpoint.
    """

    # Extract code using agent's prompt builder
    extracted = agent.prompt_builder.extract_code(llm_response)

    assert isinstance(extracted, ExtractedCode)
    assert "get_user" in extracted.production_code
    assert "test_get_user" in extracted.test_code


def test_coding_agent_error_handling():
    """Test CodingAgent handles LLM errors gracefully."""

    async def _test():
        # Mock LLM that raises error
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = Exception("API Error")

        mock_lorebook = AsyncMock()
        mock_lorebook.get_relevant_decisions = AsyncMock(return_value=[])
        mock_lorebook.get_project_patterns = AsyncMock(return_value=[])

        mock_mimiry = MagicMock()

        manager = SagaCodexManager()
        profile = manager.get_profile(LanguageProfile.PYTHON_FASTAPI)

        agent = CodingAgent(
            llm_client=mock_llm,
            lorebook=mock_lorebook,
            mimiry=mock_mimiry,
            sagacodex_profile=profile,
            agent_name="TestAgent"
        )

        task = Task(
            id="task-error",
            description="This will fail",
            weight="simple",
            budget_allocation=1.0
        )

        result = await agent.solve_task(task, project_root="/tmp/test")

        # Should return error result, not crash
        assert isinstance(result, AgentOutput)
        assert result.task_id == "task-error"
        assert result.status == "failed"

    run_async(_test())
