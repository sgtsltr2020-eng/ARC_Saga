"""
Unit tests for orchestrator type definitions.

Tests verify:
1. Task[T] generic creation and validation
2. Result[R] generic creation and validation
3. AITaskInput validation (prompt, model, temperature, etc.)
4. AIResultOutput validation (response, tokens, cost)
5. Type aliases AITask and AIResult
6. Immutability (frozen dataclasses)
7. Edge cases and boundary conditions

Coverage target: 100%
"""

from __future__ import annotations

from datetime import timezone
from decimal import Decimal
from typing import Any

import pytest

from arc_saga.orchestrator.types import (
    AIProvider,
    AIResultOutput,
    AITask,
    AITaskInput,
    AIResult,
    Result,
    Task,
    TaskStatus,
)


class TestAIProviderEnum:
    """Tests for AIProvider enum."""

    def test_all_providers_have_string_values(self) -> None:
        """Test all providers have lowercase string values."""
        assert AIProvider.OPENAI.value == "openai"
        assert AIProvider.ANTHROPIC.value == "anthropic"
        assert AIProvider.GOOGLE.value == "google"
        assert AIProvider.PERPLEXITY.value == "perplexity"
        assert AIProvider.GROQ.value == "groq"
        assert AIProvider.LOCAL.value == "local"
        assert AIProvider.CUSTOM.value == "custom"

    def test_provider_is_string_subclass(self) -> None:
        """Test AIProvider inherits from str for serialization."""
        assert isinstance(AIProvider.OPENAI, str)
        assert AIProvider.OPENAI.value == "openai"

    def test_provider_from_string(self) -> None:
        """Test creating provider from string value."""
        assert AIProvider("openai") == AIProvider.OPENAI
        assert AIProvider("anthropic") == AIProvider.ANTHROPIC

    def test_provider_invalid_value_raises_error(self) -> None:
        """Test invalid provider value raises ValueError."""
        with pytest.raises(ValueError):
            AIProvider("invalid_provider")


class TestTaskStatusEnum:
    """Tests for TaskStatus enum."""

    def test_all_statuses_have_string_values(self) -> None:
        """Test all statuses have lowercase string values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.TIMEOUT.value == "timeout"

    def test_status_is_string_subclass(self) -> None:
        """Test TaskStatus inherits from str."""
        assert isinstance(TaskStatus.PENDING, str)


class TestTaskGeneric:
    """Tests for generic Task[T] dataclass."""

    def test_create_task_with_dict_input(self) -> None:
        """Test creating task with dict input type."""
        task: Task[dict[str, Any]] = Task(
            operation="process",
            input_data={"key": "value"},
        )

        assert task.operation == "process"
        assert task.input_data == {"key": "value"}
        assert task.timeout_ms == 30000  # Default
        assert task.priority == 0  # Default
        assert task.metadata == {}  # Default
        assert len(task.id) == 36  # UUID format

    def test_create_task_with_string_input(self) -> None:
        """Test creating task with string input type."""
        task: Task[str] = Task(
            operation="echo",
            input_data="hello world",
        )

        assert task.input_data == "hello world"

    def test_create_task_with_custom_id(self) -> None:
        """Test creating task with custom ID."""
        task: Task[str] = Task(
            operation="test",
            input_data="data",
            id="custom-id-123",
        )

        assert task.id == "custom-id-123"

    def test_create_task_with_custom_timeout(self) -> None:
        """Test creating task with custom timeout."""
        task: Task[str] = Task(
            operation="long_task",
            input_data="data",
            timeout_ms=60000,
        )

        assert task.timeout_ms == 60000

    def test_create_task_with_priority(self) -> None:
        """Test creating task with priority."""
        task: Task[str] = Task(
            operation="urgent",
            input_data="data",
            priority=10,
        )

        assert task.priority == 10

    def test_create_task_with_metadata(self) -> None:
        """Test creating task with metadata."""
        metadata = {"source": "api", "user_id": "123"}
        task: Task[str] = Task(
            operation="test",
            input_data="data",
            metadata=metadata,
        )

        assert task.metadata == metadata

    def test_task_created_at_is_utc(self) -> None:
        """Test task created_at is UTC timezone."""
        task: Task[str] = Task(operation="test", input_data="data")

        assert task.created_at.tzinfo == timezone.utc

    def test_task_empty_operation_raises_error(self) -> None:
        """Test empty operation raises ValueError."""
        with pytest.raises(ValueError, match="operation cannot be empty"):
            Task(operation="", input_data="data")

    def test_task_whitespace_operation_raises_error(self) -> None:
        """Test whitespace-only operation raises ValueError."""
        with pytest.raises(ValueError, match="operation cannot be empty"):
            Task(operation="   ", input_data="data")

    def test_task_zero_timeout_raises_error(self) -> None:
        """Test zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            Task(operation="test", input_data="data", timeout_ms=0)

    def test_task_negative_timeout_raises_error(self) -> None:
        """Test negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            Task(operation="test", input_data="data", timeout_ms=-1000)

    def test_task_negative_priority_raises_error(self) -> None:
        """Test negative priority raises ValueError."""
        with pytest.raises(ValueError, match="priority cannot be negative"):
            Task(operation="test", input_data="data", priority=-1)

    def test_task_is_immutable(self) -> None:
        """Test task is frozen (immutable)."""
        task: Task[str] = Task(operation="test", input_data="data")

        with pytest.raises(AttributeError):
            task.operation = "modified"  # type: ignore[misc]


class TestResultGeneric:
    """Tests for generic Result[R] dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating successful result."""
        result: Result[str] = Result(
            task_id="task-123",
            success=True,
            output_data="completed",
            duration_ms=150,
        )

        assert result.task_id == "task-123"
        assert result.success is True
        assert result.output_data == "completed"
        assert result.error is None
        assert result.duration_ms == 150
        assert result.status == TaskStatus.COMPLETED

    def test_create_failed_result(self) -> None:
        """Test creating failed result."""
        result: Result[str] = Result(
            task_id="task-456",
            success=False,
            error="Connection timeout",
            error_type="TimeoutError",
            duration_ms=5000,
            status=TaskStatus.FAILED,
        )

        assert result.success is False
        assert result.output_data is None
        assert result.error == "Connection timeout"
        assert result.error_type == "TimeoutError"
        assert result.status == TaskStatus.FAILED

    def test_create_result_with_dict_output(self) -> None:
        """Test creating result with dict output type."""
        result: Result[dict[str, Any]] = Result(
            task_id="task-789",
            success=True,
            output_data={"items": [1, 2, 3]},
            duration_ms=100,
        )

        assert result.output_data == {"items": [1, 2, 3]}

    def test_result_completed_at_is_utc(self) -> None:
        """Test result completed_at is UTC timezone."""
        result: Result[str] = Result(
            task_id="task-123",
            success=True,
            output_data="done",
        )

        assert result.completed_at.tzinfo == timezone.utc

    def test_result_empty_task_id_raises_error(self) -> None:
        """Test empty task_id raises ValueError."""
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            Result(task_id="", success=True, output_data="data")

    def test_result_whitespace_task_id_raises_error(self) -> None:
        """Test whitespace-only task_id raises ValueError."""
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            Result(task_id="   ", success=True, output_data="data")

    def test_result_negative_duration_raises_error(self) -> None:
        """Test negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_ms cannot be negative"):
            Result(
                task_id="task-123",
                success=True,
                output_data="data",
                duration_ms=-100,
            )

    def test_result_success_with_error_raises_error(self) -> None:
        """Test successful result with error raises ValueError."""
        with pytest.raises(ValueError, match="Successful result cannot have error"):
            Result(
                task_id="task-123",
                success=True,
                output_data="data",
                error="This should not be here",
            )

    def test_result_failed_with_output_raises_error(self) -> None:
        """Test failed result with output_data raises ValueError."""
        with pytest.raises(
            ValueError, match="Failed result should not have output_data"
        ):
            Result(
                task_id="task-123",
                success=False,
                output_data="This should not be here",
                error="Error occurred",
            )

    def test_result_is_immutable(self) -> None:
        """Test result is frozen (immutable)."""
        result: Result[str] = Result(
            task_id="task-123",
            success=True,
            output_data="done",
        )

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


class TestAITaskInput:
    """Tests for AITaskInput dataclass."""

    def test_create_valid_ai_task_input(self) -> None:
        """Test creating valid AI task input."""
        input_data = AITaskInput(
            prompt="What is 2+2?",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        )

        assert input_data.prompt == "What is 2+2?"
        assert input_data.model == "gpt-4"
        assert input_data.provider == AIProvider.OPENAI
        assert input_data.max_tokens == 1000  # Default
        assert input_data.temperature == 0.7  # Default
        assert input_data.system_prompt is None
        assert input_data.stop_sequences == ()

    def test_create_ai_task_input_with_all_params(self) -> None:
        """Test creating AI task input with all parameters."""
        input_data = AITaskInput(
            prompt="Explain quantum computing",
            model="claude-3-opus",
            provider=AIProvider.ANTHROPIC,
            max_tokens=2000,
            temperature=0.5,
            system_prompt="You are a physics professor",
            stop_sequences=("END", "STOP"),
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        assert input_data.max_tokens == 2000
        assert input_data.temperature == 0.5
        assert input_data.system_prompt == "You are a physics professor"
        assert input_data.stop_sequences == ("END", "STOP")
        assert input_data.top_p == 0.9
        assert input_data.frequency_penalty == 0.5
        assert input_data.presence_penalty == 0.3

    def test_ai_task_input_empty_prompt_raises_error(self) -> None:
        """Test empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            AITaskInput(
                prompt="",
                model="gpt-4",
                provider=AIProvider.OPENAI,
            )

    def test_ai_task_input_whitespace_prompt_raises_error(self) -> None:
        """Test whitespace-only prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            AITaskInput(
                prompt="   ",
                model="gpt-4",
                provider=AIProvider.OPENAI,
            )

    def test_ai_task_input_empty_model_raises_error(self) -> None:
        """Test empty model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            AITaskInput(
                prompt="Test",
                model="",
                provider=AIProvider.OPENAI,
            )

    def test_ai_task_input_zero_max_tokens_raises_error(self) -> None:
        """Test zero max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=0,
            )

    def test_ai_task_input_negative_max_tokens_raises_error(self) -> None:
        """Test negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=-100,
            )

    def test_ai_task_input_temperature_below_zero_raises_error(self) -> None:
        """Test temperature below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                temperature=-0.1,
            )

    def test_ai_task_input_temperature_above_two_raises_error(self) -> None:
        """Test temperature above 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                temperature=2.1,
            )

    def test_ai_task_input_temperature_at_boundaries(self) -> None:
        """Test temperature at boundary values (0.0 and 2.0)."""
        # Temperature 0.0 should work
        input_zero = AITaskInput(
            prompt="Test",
            model="gpt-4",
            provider=AIProvider.OPENAI,
            temperature=0.0,
        )
        assert input_zero.temperature == 0.0

        # Temperature 2.0 should work
        input_two = AITaskInput(
            prompt="Test",
            model="gpt-4",
            provider=AIProvider.OPENAI,
            temperature=2.0,
        )
        assert input_two.temperature == 2.0

    def test_ai_task_input_top_p_out_of_range_raises_error(self) -> None:
        """Test top_p out of range raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                top_p=1.5,
            )

    def test_ai_task_input_top_p_negative_raises_error(self) -> None:
        """Test negative top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                top_p=-0.1,
            )

    def test_ai_task_input_frequency_penalty_out_of_range(self) -> None:
        """Test frequency_penalty out of range raises ValueError."""
        with pytest.raises(
            ValueError, match="frequency_penalty must be between -2.0 and 2.0"
        ):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                frequency_penalty=2.5,
            )

    def test_ai_task_input_presence_penalty_out_of_range(self) -> None:
        """Test presence_penalty out of range raises ValueError."""
        with pytest.raises(
            ValueError, match="presence_penalty must be between -2.0 and 2.0"
        ):
            AITaskInput(
                prompt="Test",
                model="gpt-4",
                provider=AIProvider.OPENAI,
                presence_penalty=-2.5,
            )

    def test_ai_task_input_is_immutable(self) -> None:
        """Test AITaskInput is frozen (immutable)."""
        input_data = AITaskInput(
            prompt="Test",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        )

        with pytest.raises(AttributeError):
            input_data.prompt = "Modified"  # type: ignore[misc]


class TestAIResultOutput:
    """Tests for AIResultOutput dataclass."""

    def test_create_valid_ai_result_output(self) -> None:
        """Test creating valid AI result output."""
        output = AIResultOutput(
            response="Paris is the capital of France.",
            tokens_used=25,
            prompt_tokens=10,
            completion_tokens=15,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )

        assert output.response == "Paris is the capital of France."
        assert output.tokens_used == 25
        assert output.prompt_tokens == 10
        assert output.completion_tokens == 15
        assert output.provider == AIProvider.OPENAI
        assert output.model == "gpt-4"
        assert output.cost_usd == Decimal("0.0")
        assert output.finish_reason is None
        assert output.latency_ms == 0

    def test_create_ai_result_output_with_all_params(self) -> None:
        """Test creating AI result output with all parameters."""
        output = AIResultOutput(
            response="Detailed response here...",
            tokens_used=500,
            prompt_tokens=100,
            completion_tokens=400,
            provider=AIProvider.ANTHROPIC,
            model="claude-3-opus",
            cost_usd=Decimal("0.0150"),
            finish_reason="stop",
            latency_ms=1500,
        )

        assert output.cost_usd == Decimal("0.0150")
        assert output.finish_reason == "stop"
        assert output.latency_ms == 1500

    def test_ai_result_output_empty_response_allowed(self) -> None:
        """Test empty response string is allowed (streaming start)."""
        output = AIResultOutput(
            response="",
            tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )

        assert output.response == ""

    def test_ai_result_output_none_response_raises_error(self) -> None:
        """Test None response raises ValueError."""
        with pytest.raises(ValueError, match="response cannot be None"):
            AIResultOutput(
                response=None,  # type: ignore[arg-type]
                tokens_used=10,
                prompt_tokens=5,
                completion_tokens=5,
                provider=AIProvider.OPENAI,
                model="gpt-4",
            )

    def test_ai_result_output_negative_tokens_raises_error(self) -> None:
        """Test negative tokens_used raises ValueError."""
        with pytest.raises(ValueError, match="tokens_used cannot be negative"):
            AIResultOutput(
                response="Test",
                tokens_used=-10,
                prompt_tokens=5,
                completion_tokens=5,
                provider=AIProvider.OPENAI,
                model="gpt-4",
            )

    def test_ai_result_output_negative_prompt_tokens_raises_error(self) -> None:
        """Test negative prompt_tokens raises ValueError."""
        with pytest.raises(ValueError, match="prompt_tokens cannot be negative"):
            AIResultOutput(
                response="Test",
                tokens_used=10,
                prompt_tokens=-5,
                completion_tokens=15,
                provider=AIProvider.OPENAI,
                model="gpt-4",
            )

    def test_ai_result_output_negative_completion_tokens_raises_error(
        self,
    ) -> None:
        """Test negative completion_tokens raises ValueError."""
        with pytest.raises(ValueError, match="completion_tokens cannot be negative"):
            AIResultOutput(
                response="Test",
                tokens_used=10,
                prompt_tokens=15,
                completion_tokens=-5,
                provider=AIProvider.OPENAI,
                model="gpt-4",
            )

    def test_ai_result_output_negative_cost_raises_error(self) -> None:
        """Test negative cost_usd raises ValueError."""
        with pytest.raises(ValueError, match="cost_usd cannot be negative"):
            AIResultOutput(
                response="Test",
                tokens_used=10,
                prompt_tokens=5,
                completion_tokens=5,
                provider=AIProvider.OPENAI,
                model="gpt-4",
                cost_usd=Decimal("-0.01"),
            )

    def test_ai_result_output_negative_latency_raises_error(self) -> None:
        """Test negative latency_ms raises ValueError."""
        with pytest.raises(ValueError, match="latency_ms cannot be negative"):
            AIResultOutput(
                response="Test",
                tokens_used=10,
                prompt_tokens=5,
                completion_tokens=5,
                provider=AIProvider.OPENAI,
                model="gpt-4",
                latency_ms=-100,
            )

    def test_ai_result_output_token_mismatch_raises_error(self) -> None:
        """Test token count mismatch raises ValueError."""
        with pytest.raises(
            ValueError,
            match="tokens_used must equal prompt_tokens \\+ completion_tokens",
        ):
            AIResultOutput(
                response="Test",
                tokens_used=100,  # Doesn't match 10 + 15
                prompt_tokens=10,
                completion_tokens=15,
                provider=AIProvider.OPENAI,
                model="gpt-4",
            )

    def test_ai_result_output_is_immutable(self) -> None:
        """Test AIResultOutput is frozen (immutable)."""
        output = AIResultOutput(
            response="Test",
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )

        with pytest.raises(AttributeError):
            output.response = "Modified"  # type: ignore[misc]


class TestTypeAliases:
    """Tests for type aliases AITask and AIResult."""

    def test_ai_task_alias_works(self) -> None:
        """Test AITask type alias creates Task[AITaskInput]."""
        input_data = AITaskInput(
            prompt="Test prompt",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        )
        task: AITask = Task(
            operation="chat_completion",
            input_data=input_data,
        )

        assert task.operation == "chat_completion"
        assert task.input_data.prompt == "Test prompt"
        assert task.input_data.provider == AIProvider.OPENAI

    def test_ai_result_alias_works(self) -> None:
        """Test AIResult type alias creates Result[AIResultOutput]."""
        output_data = AIResultOutput(
            response="Test response",
            tokens_used=20,
            prompt_tokens=10,
            completion_tokens=10,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )
        result: AIResult = Result(
            task_id="task-123",
            success=True,
            output_data=output_data,
            duration_ms=150,
        )

        assert result.success is True
        assert result.output_data is not None
        assert result.output_data.response == "Test response"
        assert result.output_data.tokens_used == 20


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_task_with_none_input_data(self) -> None:
        """Test task can have None as input_data."""
        task: Task[None] = Task(
            operation="ping",
            input_data=None,
        )

        assert task.input_data is None

    def test_task_with_complex_nested_input(self) -> None:
        """Test task with complex nested input structure."""
        complex_input: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                },
            },
            "list": [{"a": 1}, {"b": 2}],
        }
        task: Task[dict[str, Any]] = Task(
            operation="process",
            input_data=complex_input,
        )

        assert task.input_data["level1"]["level2"]["level3"] == [1, 2, 3]

    def test_result_with_zero_duration(self) -> None:
        """Test result with zero duration is valid."""
        result: Result[str] = Result(
            task_id="task-123",
            success=True,
            output_data="instant",
            duration_ms=0,
        )

        assert result.duration_ms == 0

    def test_ai_task_input_with_unicode_prompt(self) -> None:
        """Test AI task input handles unicode properly."""
        input_data = AITaskInput(
            prompt="Translate: こんにちは 🌍 مرحبا",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        )

        assert "こんにちは" in input_data.prompt
        assert "🌍" in input_data.prompt

    def test_ai_result_output_with_unicode_response(self) -> None:
        """Test AI result output handles unicode properly."""
        output = AIResultOutput(
            response="Translation: Hello 世界 🌏",
            tokens_used=15,
            prompt_tokens=10,
            completion_tokens=5,
            provider=AIProvider.OPENAI,
            model="gpt-4",
        )

        assert "世界" in output.response
        assert "🌏" in output.response

    def test_ai_task_input_very_long_prompt(self) -> None:
        """Test AI task input with very long prompt."""
        long_prompt = "Test " * 10000  # 50000 characters
        input_data = AITaskInput(
            prompt=long_prompt,
            model="gpt-4",
            provider=AIProvider.OPENAI,
        )

        assert len(input_data.prompt) == 50000

    def test_task_minimum_valid_timeout(self) -> None:
        """Test task with minimum valid timeout (1ms)."""
        task: Task[str] = Task(
            operation="quick",
            input_data="data",
            timeout_ms=1,
        )

        assert task.timeout_ms == 1

    def test_task_very_large_timeout(self) -> None:
        """Test task with very large timeout."""
        task: Task[str] = Task(
            operation="long",
            input_data="data",
            timeout_ms=86400000,  # 24 hours in ms
        )

        assert task.timeout_ms == 86400000

    def test_ai_result_output_zero_cost(self) -> None:
        """Test AI result with zero cost (free tier)."""
        output = AIResultOutput(
            response="Free response",
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            provider=AIProvider.LOCAL,
            model="llama-3",
            cost_usd=Decimal("0.0"),
        )

        assert output.cost_usd == Decimal("0.0")

    def test_ai_result_output_precise_cost(self) -> None:
        """Test AI result with precise decimal cost."""
        output = AIResultOutput(
            response="Response",
            tokens_used=1000,
            prompt_tokens=500,
            completion_tokens=500,
            provider=AIProvider.OPENAI,
            model="gpt-4",
            cost_usd=Decimal("0.0000001"),  # Very precise
        )

        assert output.cost_usd == Decimal("0.0000001")
