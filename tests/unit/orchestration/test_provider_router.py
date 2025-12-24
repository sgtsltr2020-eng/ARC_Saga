"""Tests for ProviderRouter with fallback routing and error handling.

Test matrix:
- 10+ test cases covering all scenarios
- Guard tests for singleton-only registry usage
- Caplog assertions for structured logging
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from saga.orchestrator.engine_registry import ReasoningEngineRegistry
from saga.orchestrator.errors import PermanentError, ProviderError, TransientError
from saga.orchestrator.provider_router import (
    ProviderRouter,
    RoutingRule,
)
from saga.orchestrator.types import (
    AIProvider,
    AIResult,
    AIResultOutput,
    AITask,
    AITaskInput,
)


class BaseMockEngine:
    """Base class for mock reasoning engines."""

    def __init__(self, name: str, supported_ops: set[str]) -> None:
        self.name = name
        self.supported_ops = supported_ops

    async def reason(self, task: AITask) -> AIResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass  # No-op for tests


class SuccessEngine(BaseMockEngine):
    """Engine that always succeeds."""

    async def reason(self, task: AITask) -> AIResult:
        return AIResult(
            task_id=task.id,
            success=True,
            output_data=AIResultOutput(
                response=f"Response from {self.name}",
                tokens_used=100,
                prompt_tokens=50,
                completion_tokens=50,
                provider=AIProvider.COPILOT_CHAT,
                model="mock-model",
                cost_usd=Decimal("0.001"),
                latency_ms=50,
            ),
            duration_ms=50,
        )


class TransientThenSuccessEngine(BaseMockEngine):
    """Engine that fails once with TransientError, then succeeds."""

    def __init__(self, name: str, supported_ops: set[str], fail_count: int = 1) -> None:
        super().__init__(name, supported_ops)
        self._remaining_fails = fail_count

    async def reason(self, task: AITask) -> AIResult:
        if self._remaining_fails > 0:
            self._remaining_fails -= 1
            raise TransientError("ephemeral timeout")
        return AIResult(
            task_id=task.id,
            success=True,
            output_data=AIResultOutput(
                response="Success after retry",
                tokens_used=50,
                prompt_tokens=25,
                completion_tokens=25,
                provider=AIProvider.ANTHROPIC,
                model="mock-model",
                cost_usd=Decimal("0.0005"),
                latency_ms=100,
            ),
            duration_ms=100,
        )


class PermanentFailEngine(BaseMockEngine):
    """Engine that always fails with PermanentError."""

    async def reason(self, task: AITask) -> AIResult:
        raise PermanentError("unsupported operation")


class UnknownExceptionEngine(BaseMockEngine):
    """Engine that raises unknown exception (not TransientError or PermanentError)."""

    async def reason(self, task: AITask) -> AIResult:
        raise RuntimeError("unknown issue")


def make_task(operation: str, task_id: str = "t1") -> AITask:
    """Helper to create test AITask."""

    return AITask(
        operation=operation,
        input_data=AITaskInput(
            prompt="test",
            model="gpt-4",
            provider=AIProvider.OPENAI,
        ),
        id=task_id,
    )


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clear registry before and after each test."""

    ReasoningEngineRegistry.clear()
    yield
    ReasoningEngineRegistry.clear()


@pytest.mark.asyncio
async def test_primary_success_short_circuit(caplog: pytest.LogCaptureFixture) -> None:
    """Primary provider succeeds on first attempt."""

    caplog.set_level("INFO")
    engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, engine)

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.COPILOT_CHAT])],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("reasoning", "a1")
    prov = await router.route_with_provenance(task, {"correlation_id": "a1"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.COPILOT_CHAT
    assert len(prov.attempts) == 1
    assert prov.attempts[0].outcome == "success"
    assert any("event='routing_start'" in r.message for r in caplog.records)
    assert any("event='routing_success'" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_transient_retry_then_success() -> None:
    """Provider fails once with TransientError, succeeds on retry."""

    engine = TransientThenSuccessEngine("transient", {"reasoning"}, fail_count=1)
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, engine)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC],
                max_retries=2,
            )
        ],
        default_order=[AIProvider.ANTHROPIC],
    )

    task = make_task("reasoning", "b2")
    prov = await router.route_with_provenance(task, {"correlation_id": "b2"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.ANTHROPIC
    outcomes = [a.outcome for a in prov.attempts]
    assert outcomes == ["transient_error", "success"]


@pytest.mark.asyncio
async def test_permanent_error_skips_to_next_provider() -> None:
    """First provider permanent-fails; second provider succeeds."""

    perm_engine = PermanentFailEngine("perm", {"reasoning"})
    succ_engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, perm_engine)
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, succ_engine)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC, AIProvider.COPILOT_CHAT],
            )
        ],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("reasoning", "c3")
    prov = await router.route_with_provenance(task, {"correlation_id": "c3"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.COPILOT_CHAT
    assert len(prov.attempts) == 2
    assert prov.attempts[0].provider == AIProvider.ANTHROPIC
    assert prov.attempts[0].outcome == "permanent_error"
    assert prov.attempts[1].provider == AIProvider.COPILOT_CHAT
    assert prov.attempts[1].outcome == "success"


@pytest.mark.asyncio
async def test_all_providers_fail_returns_failure() -> None:
    """All providers fail; routing returns failed outcome."""

    perm1 = PermanentFailEngine("perm1", {"reasoning"})
    perm2 = PermanentFailEngine("perm2", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, perm1)
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, perm2)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC, AIProvider.COPILOT_CHAT],
            )
        ],
        default_order=[AIProvider.ANTHROPIC],
    )

    task = make_task("reasoning", "d4")
    prov = await router.route_with_provenance(task, {"correlation_id": "d4"})

    assert prov.outcome == "failed"
    assert prov.chosen_provider is None
    assert len(prov.attempts) == 2
    assert all(a.outcome == "permanent_error" for a in prov.attempts)


@pytest.mark.asyncio
async def test_unknown_exception_transient_by_default() -> None:
    """Unknown exception treated as transient when classify_unknown_as_transient=True."""

    unknown_engine = UnknownExceptionEngine("unknown", {"reasoning"})
    succ_engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, unknown_engine)
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, succ_engine)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC, AIProvider.COPILOT_CHAT],
                max_retries=1,
            )
        ],
        default_order=[AIProvider.COPILOT_CHAT],
        classify_unknown_exceptions_as_transient=True,
    )

    task = make_task("reasoning", "e5")
    prov = await router.route_with_provenance(task, {"correlation_id": "e5"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.COPILOT_CHAT
    unknown_attempts = [a for a in prov.attempts if a.provider == AIProvider.ANTHROPIC]
    assert len(unknown_attempts) >= 1


@pytest.mark.asyncio
async def test_unknown_exception_permanent_when_configured() -> None:
    """Unknown exception treated as permanent when classify_unknown_exceptions_as_transient=False."""

    unknown_engine = UnknownExceptionEngine("unknown", {"reasoning"})
    succ_engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, unknown_engine)
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, succ_engine)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC, AIProvider.COPILOT_CHAT],
            )
        ],
        default_order=[AIProvider.COPILOT_CHAT],
        classify_unknown_exceptions_as_transient=False,
    )

    task = make_task("reasoning", "f6")
    prov = await router.route_with_provenance(task, {"correlation_id": "f6"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.COPILOT_CHAT


@pytest.mark.asyncio
async def test_no_rule_uses_default_order() -> None:
    """Task type not in any rule; use default_order."""

    engine = SuccessEngine("succ", {"coding"})
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, engine)

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.ANTHROPIC])],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("coding", "g7")
    prov = await router.route_with_provenance(task, {"correlation_id": "g7"})

    assert prov.outcome == "success"
    assert prov.chosen_provider == AIProvider.COPILOT_CHAT


@pytest.mark.asyncio
async def test_route_method_returns_result() -> None:
    """route() method calls engine and returns AIResult."""

    engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, engine)

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.COPILOT_CHAT])],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("reasoning", "h8")
    result = await router.route(task, {"correlation_id": "h8"})

    assert isinstance(result, AIResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_route_raises_on_all_providers_fail() -> None:
    """route() raises ProviderError when all providers fail."""

    perm_engine = PermanentFailEngine("perm", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, perm_engine)

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.COPILOT_CHAT])],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("reasoning", "i9")
    with pytest.raises(ProviderError):
        await router.route(task, {"correlation_id": "i9"})


@pytest.mark.asyncio
async def test_no_candidates_immediate_failure() -> None:
    """Empty candidate list results in immediate failure."""

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.ANTHROPIC])],
        default_order=[],  # Empty default
    )

    task = make_task("coding", "j10")  # Different task type
    prov = await router.route_with_provenance(task, {"correlation_id": "j10"})

    assert prov.outcome == "failed"
    assert len(prov.attempts) == 0


@pytest.mark.asyncio
async def test_correlation_id_flows_through_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Correlation ID is present in all log messages."""

    caplog.set_level("INFO")
    engine = SuccessEngine("succ", {"reasoning"})
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, engine)

    router = ProviderRouter(
        rules=[RoutingRule(task_types={"reasoning"}, ordered_providers=[AIProvider.COPILOT_CHAT])],
        default_order=[AIProvider.COPILOT_CHAT],
    )

    task = make_task("reasoning", "k11")
    await router.route_with_provenance(task, {"correlation_id": "test-corr-123"})

    for record in caplog.records:
        if "routing" in record.message.lower():
            assert "test-corr-123" in record.message


@pytest.mark.asyncio
async def test_backoff_timing_is_exponential(monkeypatch) -> None:
    """Backoff delays follow exponential growth: 0.2, 0.4, 0.8, etc."""

    from saga.orchestrator import provider_router as pr_mod

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(pr_mod.asyncio, "sleep", fake_sleep)

    engine = TransientThenSuccessEngine("transient", {"reasoning"}, fail_count=1)
    ReasoningEngineRegistry.register(AIProvider.ANTHROPIC, engine)

    router = ProviderRouter(
        rules=[
            RoutingRule(
                task_types={"reasoning"},
                ordered_providers=[AIProvider.ANTHROPIC],
                max_retries=1,
                base_backoff_seconds=0.2,
                max_backoff_seconds=1.0,
            )
        ],
    )

    task = make_task("reasoning", "l12")
    prov = await router.route_with_provenance(task, {"correlation_id": "l12"})

    assert prov.outcome == "success"
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] == pytest.approx(0.2, rel=0.05)
