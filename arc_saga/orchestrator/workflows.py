"""
Workflow Builder Module.

Provides a fluent interface for constructing workflows
and ensuring task consistency.
"""

from __future__ import annotations

from typing import Any, TypeVar

from arc_saga.orchestrator.core import WorkflowPattern
from arc_saga.orchestrator.types import (
    AIProvider,
    AITaskInput,
    ResponseMode,
    Task,
)

T = TypeVar("T")


class WorkflowBuilder:
    """
    Builder for constructing valid workflows.
    
    Example:
        >>> builder = WorkflowBuilder()
        >>> builder.set_pattern(WorkflowPattern.SEQUENTIAL)
        >>> builder.add_ai_task("Hello", "gpt-4")
        >>> pattern, tasks = builder.build()
    """

    def __init__(self) -> None:
        self._pattern: WorkflowPattern = WorkflowPattern.SEQUENTIAL
        self._tasks: list[Task[Any]] = []
        self._default_timeout: int = 30000

    def set_pattern(self, pattern: WorkflowPattern) -> WorkflowBuilder:
        """Set the execution pattern."""
        self._pattern = pattern
        return self

    def set_default_timeout(self, timeout_ms: int) -> WorkflowBuilder:
        """Set default timeout for subsequent tasks."""
        if timeout_ms <= 0:
            raise ValueError("Timeout must be positive")
        self._default_timeout = timeout_ms
        return self

    def add_task(
        self,
        operation: str,
        input_data: Any,
        timeout_ms: int | None = None,
        priority: int = 0,
    ) -> WorkflowBuilder:
        """Add a generic task to the workflow."""
        task = Task(
            operation=operation,
            input_data=input_data,
            timeout_ms=timeout_ms or self._default_timeout,
            priority=priority,
        )
        self._tasks.append(task)
        return self

    def add_ai_task(
        self,
        prompt: str,
        model: str,
        provider: AIProvider = AIProvider.OPENAI,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        priority: int = 0,
        response_mode: ResponseMode = ResponseMode.COMPLETE,
    ) -> WorkflowBuilder:
        """Add an AI generation task."""
        input_data = AITaskInput(
            prompt=prompt,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        task = Task(
            operation="ai_completion",  # Standard operation name
            input_data=input_data,
            timeout_ms=self._default_timeout,
            priority=priority,
            response_mode=response_mode,
        )
        self._tasks.append(task)
        return self

    def build(self) -> tuple[WorkflowPattern, list[Task[Any]]]:
        """
        Build and return the workflow configuration.
        
        Returns:
            Tuple of (execution_pattern, list_of_tasks)
            
        Raises:
            ValueError: If no tasks have been added
        """
        if not self._tasks:
            raise ValueError("Cannot build workflow with no tasks")
            
        return self._pattern, list(self._tasks)
