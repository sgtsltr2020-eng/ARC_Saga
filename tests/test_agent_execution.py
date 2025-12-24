"""
Agent Execution Integration Tests
=================================

Tests Warden -> CodingAgent -> LLM flow.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from saga.core.task import Task
from saga.core.warden import Warden
from saga.llm.client import LLMResponse, Provider


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


def test_warden_execute_task_integration(tmp_path: Path):
    """
    Test full flow: Warden -> CodingAgent -> LLMClient (Mock) -> Output
    """
    async def _test():
        project_root = str(tmp_path / "test_project")
        (tmp_path / "test_project").mkdir()

        # 1. Setup Mock LLM Client
        mock_llm_client = AsyncMock()
        mock_response = LLMResponse(
            text="""
Here is the code:
```saga/api/demo.py
def hello():
    return "world"
```

```tests/test_demo.py
def test_hello():
    assert hello() == "world"
```

**Rationale:** Simple demo.
""",
            provider=Provider.OPENAI,
            model="gpt-4-turbo",
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            estimated_cost=0.03
        )
        mock_llm_client.chat.return_value = mock_response
        mock_llm_client.initialize = AsyncMock()

        # 2. Setup Warden with mocked client
        warden = Warden(project_root=project_root)
        warden.llm_client = mock_llm_client

        # Mock other dependencies
        warden.lorebook = AsyncMock()
        warden.lorebook.initialize = AsyncMock()
        warden.lorebook.get_project_patterns = AsyncMock(return_value=[])
        warden.lorebook.record_outcome = AsyncMock()

        # Initialize Warden
        await warden.initialize()

        # 3. Create Task
        task = Task(
            id="task-123",
            description="Create a demo hello world API.",
            weight="simple",
            budget_allocation=1.0
        )

        # 4. Execute
        results = await warden.execute_task(task)

        # 5. Verify Results
        assert len(results) == 1
        result = results[0]

        assert result["agent"] == "CodingAgent"
        assert "code" in result

        await warden.task_store.close()

    run_async(_test())


def test_warden_execute_task_failure_handling(tmp_path: Path):
    """Test handling of LLM failure."""
    async def _test():
        project_root = str(tmp_path / "test_project")
        (tmp_path / "test_project").mkdir()

        mock_llm_client = AsyncMock()
        mock_llm_client.chat.side_effect = Exception("API Error")
        mock_llm_client.initialize = AsyncMock()

        warden = Warden(project_root=project_root)
        warden.llm_client = mock_llm_client
        warden.lorebook = AsyncMock()
        warden.lorebook.initialize = AsyncMock()

        await warden.initialize()

        task = Task(
            id="task-fail",
            description="Fail this task",
            weight="simple",
            budget_allocation=1.0
        )

        results = await warden.execute_task(task)

        assert len(results) == 1
        result = results[0]

        assert result["agent"] == "CodingAgent"
        assert result["code"] == {}

        await warden.task_store.close()

    run_async(_test())
