"""Cost-aware optimizer decorator for ProviderRouter.

Ranks providers by weighted cost/latency/quality, enforces budget limits,
and caches scores for sub-millisecond selection on low-spec hardware.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Iterable, Tuple

from saga.error_instrumentation import log_with_context
from saga.orchestrator.cost_models import (
    CostProfileRegistry,
    CostSettings,
    CostWeights,
    score_provider,
)
from saga.orchestrator.errors import BudgetExceededRoutingError
from saga.orchestrator.types import AIProvider, AITask
from saga.validators import _require

try:  # Optional observability metrics
    from prometheus_client import Counter  # type: ignore
except Exception:  # pragma: no cover
    Counter = None  # type: ignore

_COST_TOTAL_USD = (
    Counter("arc_saga_cost_total_usd", "Total estimated USD cost", ["provider"])
    if Counter
    else None
)
_TIER_ESCALATIONS = (
    Counter("arc_saga_tier_escalations_total", "Budget or tier escalations", ["reason"])
    if Counter
    else None
)


class SelectionStrategy(str, Enum):
    CHEAPEST = "CHEAPEST"
    FASTEST = "FASTEST"
    BALANCED = "BALANCED"


@dataclass
class BudgetConfig:
    enforce_hard_limits: bool
    max_usd: Decimal


class CostOptimizer:
    """Singleton-style optimizer that ranks providers before routing."""

    _instance: "CostOptimizer" | None = None

    def __new__(cls, settings: CostSettings | None = None) -> "CostOptimizer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: CostSettings | None = None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._settings = settings or CostSettings()  # env validated here
        self._strategy = SelectionStrategy(self._settings.strategy)
        self._weights = self._derive_weights(self._strategy, self._settings.parsed_weights())
        self._budget = BudgetConfig(
            enforce_hard_limits=self._settings.enforce_hard_limits,
            max_usd=self._settings.max_usd,
        )
        self._cache_size = self._settings.cache_size
        self._cache: OrderedDict[tuple[AIProvider, int, SelectionStrategy], Decimal] = OrderedDict()
        self._disabled = self._settings.disable
        self._initialized = True

    def is_disabled(self) -> bool:
        return self._disabled

    def rank_providers(self, task: AITask, providers: Iterable[AIProvider]) -> list[AIProvider]:
        """Return providers ordered by score (lower first)."""

        if self._disabled:
            return list(providers)

        start = time.perf_counter()
        est_tokens = self._estimate_tokens(task)
        self._enforce_budget(est_tokens)

        scores: list[tuple[AIProvider, Decimal]] = []
        for provider in providers:
            profile = CostProfileRegistry.get(provider)
            score = self._score_with_cache(provider, profile, est_tokens)
            scores.append((provider, score))

        scores.sort(key=lambda item: item[1])
        ordered = [provider for provider, _ in scores]

        log_with_context(
            "info",
            "cost_optimizer_ranked",
            providers=[p.value for p in ordered],
            strategy=self._strategy.value,
            est_tokens=est_tokens,
            duration_ms=round((time.perf_counter() - start) * 1000, 3),
        )
        return ordered

    def _score_with_cache(
        self,
        provider: AIProvider,
        profile,
        est_tokens: int,
    ) -> Decimal:
        key = (provider, est_tokens, self._strategy)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        score = score_provider(profile, est_tokens, self._weights.as_tuple())
        self._cache[key] = score
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return score

    def _derive_weights(self, strategy: SelectionStrategy, base: CostWeights) -> CostWeights:
        if strategy == SelectionStrategy.FASTEST:
            return CostWeights(cost=Decimal("0.2"), latency=Decimal("0.6"), quality=Decimal("0.2"))
        if strategy == SelectionStrategy.CHEAPEST:
            return CostWeights(cost=Decimal("0.6"), latency=Decimal("0.2"), quality=Decimal("0.2"))
        return base

    def _estimate_tokens(self, task: AITask) -> int:
        prompt = task.input_data.prompt
        est = int(len(prompt) * 1.5) + 100
        return max(est, 0)

    def _enforce_budget(self, est_tokens: int) -> None:
        profile_openai = CostProfileRegistry.get(AIProvider.OPENAI)
        est_cost = (profile_openai.cost_per_1k * Decimal(est_tokens)) / Decimal("1000")

        if _COST_TOTAL_USD:
            _COST_TOTAL_USD.labels(provider=AIProvider.OPENAI.value).inc(float(est_cost))

        if est_tokens > 1_000_000 and self._budget.enforce_hard_limits:
            if _TIER_ESCALATIONS:
                _TIER_ESCALATIONS.labels(reason="token_overflow").inc()
            raise BudgetExceededRoutingError("Estimated tokens exceed 1M hard cap")

        if self._budget.enforce_hard_limits and est_cost > self._budget.max_usd:
            if _TIER_ESCALATIONS:
                _TIER_ESCALATIONS.labels(reason="budget_limit").inc()
            raise BudgetExceededRoutingError(
                f"Estimated cost {est_cost} exceeds max ${self._budget.max_usd}"
            )


def reorder_candidates_with_optimizer(
    optimizer: CostOptimizer | None, task: AITask, candidates: list[AIProvider]
) -> list[AIProvider]:
    """Helper to reorder ProviderRouter candidates if optimizer is provided."""

    if optimizer is None or optimizer.is_disabled():
        return candidates
    _require(bool(candidates), "Candidate providers required for cost optimization")
    return optimizer.rank_providers(task, candidates)
