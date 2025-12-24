"""Provider routing with deterministic fallback and structured provenance.

Routes AI tasks across multiple reasoning engines (providers) with:
- Automatic fallback chain execution
- Exponential backoff on transient errors
- Full provenance tracking (all attempts, outcomes, timings)
- Structured logging with correlation IDs

Example:
    >>> from saga.orchestrator.provider_router import ProviderRouter, RoutingRule
    >>> from saga.orchestrator.types import AIProvider, AITask, AITaskInput
    >>>
    >>> task = AITask(
    ...     operation="chat_completion",
    ...     input_data=AITaskInput(
    ...         prompt="Hello",
    ...         model="gpt-4",
    ...         provider=AIProvider.OPENAI,
    ...     ),
    ... )
    >>>
    >>> router = ProviderRouter(
    ...     rules=[
    ...         RoutingRule(
    ...             task_types={"chat_completion"},
    ...             ordered_providers=[AIProvider.COPILOT_CHAT, AIProvider.ANTHROPIC],
    ...         )
    ...     ],
    ...     default_order=[AIProvider.COPILOT_CHAT],
    ... )
    >>>
    >>> prov = await router.route_with_provenance(task, {"correlation_id": "abc-123"})
    >>> if prov.outcome == "success":
    ...     result = await router.route(task, {"correlation_id": "abc-123"})
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from saga.exceptions.integration_exceptions import (
    AuthenticationError,
    InputValidationError,
    RateLimitError,
)
from saga.exceptions.integration_exceptions import (
    TransientError as EngineTransientError,
)
from saga.orchestrator.cost_optimizer import (
    CostOptimizer,
    reorder_candidates_with_optimizer,
)
from saga.orchestrator.engine_registry import ReasoningEngineRegistry
from saga.orchestrator.errors import PermanentError, ProviderError, TransientError
from saga.orchestrator.protocols import IEngineRegistry, IReasoningEngine
from saga.orchestrator.types import AIProvider, AIResult, AITask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingRule:
    """Rule mapping task types to ordered provider chain and retry policy.

    Attributes:
        task_types: Set of task operation names this rule applies to.
        ordered_providers: List of AIProvider values in fallback order.
        max_retries: Max retry attempts on TransientError (default: 2).
        base_backoff_seconds: Initial backoff delay in seconds (default: 0.2).
        max_backoff_seconds: Maximum backoff cap in seconds (default: 1.0).

    Example:
        >>> RoutingRule(
        ...     task_types={"reasoning", "coding", "analysis"},
        ...     ordered_providers=[AIProvider.COPILOT_CHAT, AIProvider.ANTHROPIC],
        ...     max_retries=3,
        ...     base_backoff_seconds=0.1,
        ...     max_backoff_seconds=2.0,
        ... )
    """

    task_types: set[str]
    ordered_providers: list[AIProvider]
    max_retries: int = 2
    base_backoff_seconds: float = 0.2
    max_backoff_seconds: float = 1.0


@dataclass
class AttemptRecord:
    """Single attempt against a provider.

    Attributes:
        provider: AIProvider that was attempted.
        attempt_index: 1-based attempt number for this provider.
        started_at: time.perf_counter() at attempt start.
        finished_at: time.perf_counter() at attempt end.
        outcome: One of "success", "transient_error", "permanent_error", "exception".
        error_type: Exception class name if outcome is error/exception.
        error_message: str(exception) if outcome is error/exception.
    """

    provider: AIProvider
    attempt_index: int
    started_at: float
    finished_at: float
    outcome: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RouteProvenance:
    """Full provenance for a routed task.

    Attributes:
        task_type: Task operation name (from task.operation).
        selected_rule: RoutingRule that matched (None if using default_order).
        candidate_providers: List of providers considered in order.
        attempts: List of all AttemptRecord from all attempts.
        outcome: "success" or "failed".
        chosen_provider: AIProvider that succeeded (None if failed).
        total_duration_seconds: Total elapsed time for all attempts.
        final_error_type: Exception class name if outcome="failed".
        final_error_message: str(exception) if outcome="failed".
    """

    task_type: str
    selected_rule: Optional[RoutingRule]
    candidate_providers: list[AIProvider]
    attempts: list[AttemptRecord]
    outcome: str
    chosen_provider: Optional[AIProvider]
    total_duration_seconds: float
    final_error_type: Optional[str] = None
    final_error_message: Optional[str] = None


class ProviderRouter:
    """Routes tasks to engines via ReasoningEngineRegistry with deterministic fallback.

    Features:
    - Automatic fallback chain execution
    - Exponential backoff on transient errors
    - Configurable unknown exception classification
    - Full provenance tracking (all attempts, outcomes, timings)
    - Structured INFO/WARNING/ERROR logging with correlation IDs

    Design:
    - Registry is read-only dependency (accessed via classmethods only)
    - Router state is immutable (rules, default_order frozen at construction)
    - Each route() call is independent with no side effects on router state

    Example:
        >>> router = ProviderRouter(
        ...     rules=[
        ...         RoutingRule(
        ...             task_types={"chat_completion"},
        ...             ordered_providers=[AIProvider.COPILOT_CHAT, AIProvider.ANTHROPIC],
        ...         )
        ...     ],
        ...     default_order=[AIProvider.COPILOT_CHAT],
        ... )
        >>>
        >>> # Option 1: Get full provenance
        >>> prov = await router.route_with_provenance(task, context)
        >>> print(f"Routing outcome: {prov.outcome}")
        >>> print(f"Chosen provider: {prov.chosen_provider}")
        >>>
        >>> # Option 2: Get result directly
        >>> result = await router.route(task, context)
    """

    def __init__(
        self,
        rules: Sequence[RoutingRule],
        default_order: Optional[Sequence[AIProvider]] = None,
        classify_unknown_exceptions_as_transient: bool = True,
        optimizer: CostOptimizer | None = None,
        registry: IEngineRegistry | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            rules: List of RoutingRule mapping task types to fallback chains.
            default_order: Fallback providers when no rule matches task type.
            classify_unknown_exceptions_as_transient: If True, unknown exceptions
                are retried; if False, treated as permanent and provider skipped.
            optimizer: Optional CostOptimizer for token optimization.
            registry: Optional registry instance. Falls back to global ReasoningEngineRegistry if None.
        """

        self._rules = tuple(rules)
        self._default_order = tuple(default_order) if default_order else tuple()
        self._classify_unknown_as_transient = classify_unknown_exceptions_as_transient
        self._optimizer = optimizer
        # Use provided registry or fall back to static global registry wrapper
        self._registry = registry or ReasoningEngineRegistry

    def get_candidate_providers(self, task_type: str) -> list[AIProvider]:
        """Return ordered providers for the given task type.

        Searches RoutingRule for matching task_type. Falls back to default_order.

        Args:
            task_type: Task operation name to route for.

        Returns:
            List of AIProvider values in fallback order.
        """

        for rule in self._rules:
            if task_type in rule.task_types:
                return list(rule.ordered_providers)
        return list(self._default_order)

    async def route(
        self, task: AITask, context: Optional[Mapping[str, Any]] = None
    ) -> AIResult:
        """Route the task and return the engine result.

        Executes route_with_provenance() to determine best provider, then calls
        that provider's engine.reason() method to get the result.

        Args:
            task: AITask to execute.
            context: Context dict (should contain 'correlation_id' for logging).

        Returns:
            AIResult from the chosen engine's reason() call.

        Raises:
            ProviderError: If routing fails (all providers exhausted or no candidates).
        """

        context = context or {}
        prov = await self.route_with_provenance(task, context)
        if prov.outcome == "success":
            chosen = prov.chosen_provider
            assert chosen is not None
            engine = self._get_engine(chosen)
            return await self._reason(engine, task)
        msg = prov.final_error_message or "Routing failed"
        raise ProviderError(msg)

    async def route_with_provenance(
        self, task: AITask, context: Mapping[str, Any]
    ) -> RouteProvenance:
        """Route the task and return full provenance.

        Executes the fallback chain, attempting each provider in order until one
        succeeds. Returns complete record of all attempts, timings, and outcomes.

        Does NOT execute the chosen engine; only routes and returns provenance.
        Use route() if you need the actual result.

        Args:
            task: AITask to route.
            context: Context dict (should contain 'correlation_id' for logging).

        Returns:
            RouteProvenance with outcome, chosen_provider, all attempts, timings.
        """

        start = time.perf_counter()
        task_type = task.operation
        correlation_id = str(context.get("correlation_id") or task.id or "")
        candidate = self.get_candidate_providers(task_type)
        candidate = reorder_candidates_with_optimizer(self._optimizer, task, candidate)
        selected_rule = self._find_rule(task_type)
        attempts: list[AttemptRecord] = []

        if not candidate:
            msg = f"No routing rule or default order for task_type='{task_type}'"
            logger.error(
                "event='routing_failed' task_type='%s' correlation_id='%s' reason='%s'",
                task_type,
                correlation_id,
                msg,
            )
            end = time.perf_counter()
            return RouteProvenance(
                task_type=task_type,
                selected_rule=selected_rule,
                candidate_providers=[],
                attempts=[],
                outcome="failed",
                chosen_provider=None,
                total_duration_seconds=end - start,
                final_error_type="PermanentError",
                final_error_message=msg,
            )

        logger.info(
            "event='routing_start' task_type='%s' correlation_id='%s' providers=%s",
            task_type,
            correlation_id,
            [p.value for p in candidate],
        )

        for provider in candidate:
            engine = self._get_engine(provider)
            max_retries, base_backoff, max_backoff = self._retry_policy(selected_rule)
            attempt_idx = 0

            while True:
                attempt_idx += 1
                started_at = time.perf_counter()
                try:
                    _ = await self._reason(engine, task)
                    finished_at = time.perf_counter()
                    attempts.append(
                        AttemptRecord(
                            provider=provider,
                            attempt_index=attempt_idx,
                            started_at=started_at,
                            finished_at=finished_at,
                            outcome="success",
                        )
                    )
                    end = time.perf_counter()
                    logger.info(
                        "event='routing_success' task_type='%s' correlation_id='%s' provider='%s' attempts=%d latency_ms=%.2f",
                        task_type,
                        correlation_id,
                        provider.value,
                        attempt_idx,
                        (end - start) * 1000.0,
                    )
                    return RouteProvenance(
                        task_type=task_type,
                        selected_rule=selected_rule,
                        candidate_providers=list(candidate),
                        attempts=attempts,
                        outcome="success",
                        chosen_provider=provider,
                        total_duration_seconds=end - start,
                    )
                except PermanentError as pe:
                    finished_at = time.perf_counter()
                    attempts.append(
                        AttemptRecord(
                            provider=provider,
                            attempt_index=attempt_idx,
                            started_at=started_at,
                            finished_at=finished_at,
                            outcome="permanent_error",
                            error_type=type(pe).__name__,
                            error_message=str(pe),
                        )
                    )
                    logger.error(
                        "event='engine_permanent_error' task_type='%s' correlation_id='%s' provider='%s' attempt=%d msg='%s'",
                        task_type,
                        correlation_id,
                        provider.value,
                        attempt_idx,
                        pe,
                    )
                    break
                except TransientError as te:
                    finished_at = time.perf_counter()
                    attempts.append(
                        AttemptRecord(
                            provider=provider,
                            attempt_index=attempt_idx,
                            started_at=started_at,
                            finished_at=finished_at,
                            outcome="transient_error",
                            error_type=type(te).__name__,
                            error_message=str(te),
                        )
                    )
                    logger.warning(
                        "event='engine_transient_error' task_type='%s' correlation_id='%s' provider='%s' attempt=%d msg='%s'",
                        task_type,
                        correlation_id,
                        provider.value,
                        attempt_idx,
                        te,
                    )
                    if attempt_idx > max_retries:
                        break
                    await asyncio.sleep(self._compute_backoff(attempt_idx, base_backoff, max_backoff))
                    continue
                except Exception as ex:
                    finished_at = time.perf_counter()
                    classify_transient = self._classify_unknown_as_transient
                    attempts.append(
                        AttemptRecord(
                            provider=provider,
                            attempt_index=attempt_idx,
                            started_at=started_at,
                            finished_at=finished_at,
                            outcome="exception",
                            error_type="TransientError" if classify_transient else "PermanentError",
                            error_message=str(ex),
                        )
                    )
                    level = logger.warning if classify_transient else logger.error
                    level(
                        "event='engine_unknown_exception' task_type='%s' correlation_id='%s' provider='%s' attempt=%d transient=%s msg='%s'",
                        task_type,
                        correlation_id,
                        provider.value,
                        attempt_idx,
                        str(classify_transient).lower(),
                        ex,
                    )
                    if classify_transient and attempt_idx <= max_retries:
                        await asyncio.sleep(self._compute_backoff(attempt_idx, base_backoff, max_backoff))
                        continue
                    break

        end = time.perf_counter()
        final_type = attempts[-1].error_type if attempts else "PermanentError"
        final_msg = attempts[-1].error_message if attempts else "No candidate providers available"
        tried = [a.provider.value for a in attempts]
        logger.error(
            "event='routing_failed' task_type='%s' correlation_id='%s' tried=%s attempts=%d latency_ms=%.2f final_error_type='%s'",
            task_type,
            correlation_id,
            tried,
            len(attempts),
            (end - start) * 1000.0,
            final_type,
        )
        return RouteProvenance(
            task_type=task_type,
            selected_rule=selected_rule,
            candidate_providers=list(candidate),
            attempts=attempts,
            outcome="failed",
            chosen_provider=None,
            total_duration_seconds=end - start,
            final_error_type=final_type,
            final_error_message=final_msg,
        )

    def _find_rule(self, task_type: str) -> Optional[RoutingRule]:
        """Find the RoutingRule for a task type."""

        for rule in self._rules:
            if task_type in rule.task_types:
                return rule
        return None

    def _retry_policy(self, rule: Optional[RoutingRule]) -> tuple[int, float, float]:
        """Extract retry policy from rule or return defaults."""

        if rule is None:
            return 2, 0.2, 1.0
        return rule.max_retries, rule.base_backoff_seconds, rule.max_backoff_seconds

    @staticmethod
    def _compute_backoff(attempt_index: int, base: float, max_backoff: float) -> float:
        """Compute exponential backoff: base * (2 ^ (attempt - 1)), capped at max."""

        delay = base * (2 ** (attempt_index - 1))
        return min(delay, max_backoff)

    def _get_engine(self, provider: AIProvider) -> IReasoningEngine:
        """Get engine from registry.

        Raises:
            PermanentError: If provider not found in registry.
        """

        try:
            engine = self._registry.get(provider)
        except Exception as ex:
            logger.error("event='registry_error' provider='%s' msg='%s'", provider.value, ex)
            raise PermanentError(f"Registry access failed for {provider.value}: {ex}")
        if engine is None:
            logger.error("event='engine_missing' provider='%s'", provider.value)
            raise PermanentError(f"Engine not found for provider={provider.value}")
        return engine

    @staticmethod
    async def _reason(engine: IReasoningEngine, task: AITask) -> AIResult:
        """Call engine.reason(task) and map engine exceptions to routing exceptions.

        Maps engine exceptions from saga.exceptions.integration_exceptions to
        routing exceptions from saga.orchestrator.errors.

        For now, we ignore streaming and just get the first result.
        Full streaming support deferred to Component 4.
        """

        try:
            result = await engine.reason(task)
            # If result is an async generator (streaming), consume first token
            if hasattr(result, "__anext__"):
                try:
                    await result.__anext__()  # Validate streaming works; don't use token
                except StopAsyncIteration:
                    pass  # Empty stream
            return result  # type: ignore[return-value]
        except (AuthenticationError, InputValidationError) as e:
            # Permanent errors: don't retry
            raise PermanentError(str(e)) from e
        except (RateLimitError, EngineTransientError, TimeoutError) as e:
            # Transient errors: retry with backoff
            raise TransientError(str(e)) from e
        # Other exceptions pass through for classify_unknown_exceptions_as_transient handling
