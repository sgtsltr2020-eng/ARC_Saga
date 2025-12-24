"""
Orchestrator Type Definitions.

Generic immutable dataclasses for workflow operations with AI-specific extensions.
Follows event-driven CQRS pattern from decision_catalog.md.

This module provides:
- Generic Task[T] and Result[R] types for any operation
- AI-specific extensions: AITaskInput, AIResultOutput
- Type aliases: AITask, AIResult for common AI workflows

Example:
    >>> from saga.orchestrator.types import AITask, AITaskInput, AIProvider
    >>>
    >>> task = AITask(
    ...     operation="chat_completion",
    ...     input_data=AITaskInput(
    ...         prompt="What is the capital of France?",
    ...         model="gpt-4",
    ...         provider=AIProvider.OPENAI,
    ...         max_tokens=100,
    ...     ),
    ... )
    >>> print(f"Task {task.id} created at {task.created_at}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

# Type variables for generic Task and Result
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Output type


class AIProvider(str, Enum):
    """
    Supported AI providers for orchestration.

    Extends the base Provider enum with additional providers
    specific to AI orchestration workflows.

    Attributes:
        OPENAI: OpenAI API (GPT models)
        ANTHROPIC: Anthropic API (Claude models)
        GOOGLE: Google AI (Gemini models)
        PERPLEXITY: Perplexity API
        GROQ: Groq API (fast inference)
        LOCAL: Local models (Ollama, vLLM, etc.)
        CUSTOM: Custom provider implementation
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    PERPLEXITY = "perplexity"
    GROQ = "groq"
    LOCAL = "local"
    CUSTOM = "custom"
    COPILOT_CHAT = "copilot_chat"


class TaskStatus(str, Enum):
    """
    Task execution status.

    Attributes:
        PENDING: Task created but not started
        RUNNING: Task currently executing
        COMPLETED: Task finished successfully
        FAILED: Task finished with error
        CANCELLED: Task was cancelled before completion
        TIMEOUT: Task exceeded timeout limit
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResponseMode(str, Enum):
    """
    Response delivery mode for AI tasks.

    Controls whether responses are streamed incrementally or returned
    as a complete result after all tokens are generated.

    Attributes:
        STREAMING: Yield tokens as they arrive (real-time)
        COMPLETE: Wait for full response, return once (default)
    """

    STREAMING = "streaming"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Task(Generic[T]):
    """
    Generic immutable task for workflow operations.

    A Task represents a unit of work to be executed by the orchestrator.
    It is generic over input type T, allowing type-safe task definitions.

    Attributes:
        operation: Name of the operation to perform (e.g., "chat_completion")
        input_data: Strongly-typed input data for the operation
        id: Unique task identifier (auto-generated UUID)
        created_at: Task creation timestamp (UTC)
        timeout_ms: Maximum execution time in milliseconds (default: 30000)
        priority: Task priority (higher = more urgent, default: 0)
        metadata: Additional task metadata (default: empty dict)

    Example:
        >>> task = Task[dict](
        ...     operation="process_data",
        ...     input_data={"key": "value"},
        ...     timeout_ms=5000,
        ... )

    Raises:
        ValueError: If operation is empty or timeout_ms is non-positive
    """

    operation: str
    input_data: T
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_ms: int = 30000
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    response_mode: ResponseMode = ResponseMode.COMPLETE

    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.operation or not self.operation.strip():
            raise ValueError("Task operation cannot be empty")
        if self.timeout_ms <= 0:
            raise ValueError("Task timeout_ms must be positive")
        if self.priority < 0:
            raise ValueError("Task priority cannot be negative")


@dataclass(frozen=True)
class Result(Generic[R]):
    """
    Generic immutable result from task execution.

    A Result represents the outcome of a Task execution, containing
    either success data or error information.

    Attributes:
        task_id: ID of the task that produced this result
        success: Whether the task completed successfully
        output_data: Result data if successful (None on failure)
        error: Error message if failed (None on success)
        error_type: Error type/class name if failed
        completed_at: Completion timestamp (UTC)
        duration_ms: Execution duration in milliseconds
        status: Final task status

    Example:
        >>> result = Result[str](
        ...     task_id="abc-123",
        ...     success=True,
        ...     output_data="Task completed",
        ...     duration_ms=150,
        ... )

    Raises:
        ValueError: If task_id is empty or duration_ms is negative
    """

    task_id: str
    success: bool
    output_data: Optional[R] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0
    status: TaskStatus = TaskStatus.COMPLETED
    stream_available: bool = False

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if not self.task_id or not self.task_id.strip():
            raise ValueError("Result task_id cannot be empty")
        if self.duration_ms < 0:
            raise ValueError("Result duration_ms cannot be negative")
        # Validate consistency
        if self.success and self.error:
            raise ValueError("Successful result cannot have error message")
        if not self.success and self.output_data is not None:
            raise ValueError("Failed result should not have output_data")


@dataclass(frozen=True)
class AITaskInput:
    """
    AI-specific task input for LLM operations.

    Contains all parameters needed to execute an AI model request.

    Attributes:
        prompt: The input prompt/query for the model
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        provider: AI provider enum value
        max_tokens: Maximum tokens in response (default: 1000)
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        system_prompt: Optional system/context prompt
        stop_sequences: Optional list of stop sequences
        top_p: Optional nucleus sampling parameter
        frequency_penalty: Optional frequency penalty (-2.0 to 2.0)
        presence_penalty: Optional presence penalty (-2.0 to 2.0)

    Example:
        >>> input_data = AITaskInput(
        ...     prompt="Explain quantum computing",
        ...     model="gpt-4-turbo",
        ...     provider=AIProvider.OPENAI,
        ...     max_tokens=500,
        ...     temperature=0.5,
        ... )

    Raises:
        ValueError: If prompt is empty, max_tokens non-positive,
                   or temperature out of range
    """

    prompt: str
    model: str
    provider: AIProvider
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    stop_sequences: tuple[str, ...] = field(default_factory=tuple)
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate AI task input after initialization."""
        if not self.prompt or not self.prompt.strip():
            raise ValueError("AITaskInput prompt cannot be empty")
        if not self.model or not self.model.strip():
            raise ValueError("AITaskInput model cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("AITaskInput max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("AITaskInput temperature must be between 0.0 and 2.0")
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("AITaskInput top_p must be between 0.0 and 1.0")
        if self.frequency_penalty is not None:
            if not -2.0 <= self.frequency_penalty <= 2.0:
                raise ValueError(
                    "AITaskInput frequency_penalty must be between -2.0 and 2.0"
                )
        if self.presence_penalty is not None:
            if not -2.0 <= self.presence_penalty <= 2.0:
                raise ValueError(
                    "AITaskInput presence_penalty must be between -2.0 and 2.0"
                )


@dataclass(frozen=True)
class AIResultOutput:
    """
    AI-specific result output from LLM operations.

    Contains the response and usage metrics from an AI model request.

    Attributes:
        response: The generated text response
        tokens_used: Total tokens consumed (prompt + completion)
        prompt_tokens: Tokens in the input prompt
        completion_tokens: Tokens in the generated response
        provider: AI provider that generated the response
        model: Model that generated the response
        cost_usd: Estimated cost in USD (as Decimal for precision)
        finish_reason: Why generation stopped (e.g., "stop", "length")
        latency_ms: Model inference latency in milliseconds

    Example:
        >>> output = AIResultOutput(
        ...     response="Paris is the capital of France.",
        ...     tokens_used=25,
        ...     prompt_tokens=10,
        ...     completion_tokens=15,
        ...     provider=AIProvider.OPENAI,
        ...     model="gpt-4",
        ...     cost_usd=Decimal("0.0015"),
        ... )

    Raises:
        ValueError: If response is empty or token counts are negative
    """

    response: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    provider: AIProvider
    model: str
    cost_usd: Decimal = Decimal("0.0")
    finish_reason: Optional[str] = None
    latency_ms: int = 0

    def __post_init__(self) -> None:
        """Validate AI result output after initialization."""
        # Response can be empty string for some edge cases (streaming start)
        if self.response is None:
            raise ValueError("AIResultOutput response cannot be None")
        if self.tokens_used < 0:
            raise ValueError("AIResultOutput tokens_used cannot be negative")
        if self.prompt_tokens < 0:
            raise ValueError("AIResultOutput prompt_tokens cannot be negative")
        if self.completion_tokens < 0:
            raise ValueError("AIResultOutput completion_tokens cannot be negative")
        if self.cost_usd < Decimal("0.0"):
            raise ValueError("AIResultOutput cost_usd cannot be negative")
        if self.latency_ms < 0:
            raise ValueError("AIResultOutput latency_ms cannot be negative")
        # Validate token consistency
        if self.tokens_used != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "AIResultOutput tokens_used must equal "
                "prompt_tokens + completion_tokens"
            )


# Type aliases for common AI workflow types
AITask = Task[AITaskInput]
AIResult = Result[AIResultOutput]
