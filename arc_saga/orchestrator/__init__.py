"""
ARC SAGA Orchestrator Module.

Event-driven CQRS orchestration for AI workflow management.

This module provides:
- Generic Task[T] and Result[R] types for workflow operations
- Event sourcing with IEventStore protocol and implementations
- Orchestrator class for workflow execution and policy enforcement
- Quality gate management for CI/CD integration

Example:
    >>> from arc_saga.orchestrator import Orchestrator, AITask, AITaskInput
    >>> from arc_saga.orchestrator import InMemoryEventStore, WorkflowPattern
    >>>
    >>> event_store = InMemoryEventStore()
    >>> orchestrator = Orchestrator(event_store)
    >>> task = AITask(operation="chat", input_data=AITaskInput(...))
    >>> results = await orchestrator.execute_workflow(
    ...     WorkflowPattern.SEQUENTIAL, [task]
    ... )
"""

from __future__ import annotations

# Admin
from .admin import (
    IQualityGateExecutor,
    QualityGateFailure,
    QualityGateManager,
    QualityGateResult,
    QualityGateType,
    SubprocessExecutor,
)

# Config Generation
from .config_gen import (
    ConfigGenerator,
    FileSystemDetector,
    IProjectDetector,
    OrchestrationConfig,
    ProjectType,
)

# Core
from .core import (
    Command,
    OperationContext,
    Orchestrator,
    OrchestratorError,
    Policy,
    PolicyResult,
    PolicyViolationError,
    WorkflowError,
    WorkflowPattern,
)

# Events
from .events import (
    EventStoreError,
    IEventStore,
    InMemoryEventStore,
    OperationLoggedEvent,
    OrchestratorEvent,
    PolicyEnforcedEvent,
    SQLiteEventStore,
    TaskExecutedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)

# Types
from .types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    AITaskInput,
    Result,
    Task,
    TaskStatus,
)

__all__ = [
    # Types
    "Task",
    "Result",
    "AITaskInput",
    "AIResultOutput",
    "AITask",
    "AIResult",
    "AIProvider",
    "TaskStatus",
    # Events
    "IEventStore",
    "SQLiteEventStore",
    "InMemoryEventStore",
    "OrchestratorEvent",
    "WorkflowStartedEvent",
    "WorkflowCompletedEvent",
    "TaskExecutedEvent",
    "PolicyEnforcedEvent",
    "OperationLoggedEvent",
    "EventStoreError",
    # Core
    "Orchestrator",
    "WorkflowPattern",
    "Policy",
    "Command",
    "PolicyResult",
    "OperationContext",
    "OrchestratorError",
    "WorkflowError",
    "PolicyViolationError",
    # Admin
    "QualityGateManager",
    "IQualityGateExecutor",
    "SubprocessExecutor",
    "QualityGateResult",
    "QualityGateFailure",
    "QualityGateType",
    # Config Generation
    "ProjectType",
    "OrchestrationConfig",
    "IProjectDetector",
    "FileSystemDetector",
    "ConfigGenerator",
]
