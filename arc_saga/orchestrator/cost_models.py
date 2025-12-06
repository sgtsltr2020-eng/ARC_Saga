"""Cost scoring models and helpers for provider selection.

Implements cost profiles, environment-driven weighting, and scoring helpers
used by CostOptimizer (Phase 2.4 Component 4). All math uses Decimal for
precision and avoids network calls to keep low-spec perf (<1ms scoring).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, Tuple

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from arc_saga.error_instrumentation import log_with_context
from arc_saga.orchestrator.types import AIProvider
from arc_saga.validators import _require


class CostWeights(BaseModel):
    """Weighted tuple for scoring components."""

    cost: Decimal = Field(default=Decimal("0.4"))
    latency: Decimal = Field(default=Decimal("0.3"))
    quality: Decimal = Field(default=Decimal("0.3"))

    @field_validator("cost", "latency", "quality")
    @classmethod
    def _non_negative(cls, value: Decimal) -> Decimal:
        _require(value >= Decimal("0"), "Weight values must be non-negative")
        return value

    @field_validator("quality")
    @classmethod
    def _weights_not_all_zero(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        data = info.data
        if "cost" in data and "latency" in data:
            total = data["cost"] + data["latency"] + value
            _require(
                total > Decimal("0"), "At least one weight must be greater than zero"
            )
        return value

    @classmethod
    def from_tuple(cls, weights: Tuple[Decimal, Decimal, Decimal]) -> CostWeights:
        cost, latency, quality = weights
        return cls(cost=cost, latency=latency, quality=quality)

    def as_tuple(self) -> Tuple[Decimal, Decimal, Decimal]:
        return (self.cost, self.latency, self.quality)


class CostSettings(BaseSettings):
    """Environment-driven knobs for cost optimization."""

    model_config = SettingsConfigDict(env_prefix="SAGA_COST_", extra="ignore")

    strategy: str = Field(default="CHEAPEST")
    weights: str = Field(default="0.4,0.3,0.3")
    max_usd: Decimal = Field(default=Decimal("0.10"))
    enforce_hard_limits: bool = Field(default=True)
    cache_size: int = Field(default=10)
    disable: bool = Field(default=False)

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        normalized = value.strip().upper()
        _require(
            normalized in {"CHEAPEST", "FASTEST", "BALANCED"},
            "SAGA_COST_STRATEGY must be CHEAPEST, FASTEST, or BALANCED",
        )
        return normalized

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: str) -> str:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        _require(
            len(parts) == 3, "SAGA_COST_WEIGHTS must have three comma-separated values"
        )
        decimals: list[Decimal] = [Decimal(part) for part in parts]
        total = sum(decimals, Decimal("0"))
        _require(total > Decimal("0"), "SAGA_COST_WEIGHTS must sum to a positive value")
        return value

    @field_validator("cache_size")
    @classmethod
    def _validate_cache_size(cls, value: int) -> int:
        _require(value > 0, "SAGA_COST_CACHE_SIZE must be positive")
        _require(
            value <= 1000, "SAGA_COST_CACHE_SIZE must stay under 1000 to bound RAM"
        )
        return value

    def parsed_weights(self) -> CostWeights:
        parts = [Decimal(p.strip()) for p in self.weights.split(",")]
        return CostWeights.from_tuple(tuple(parts))  # type: ignore[arg-type]


class CostProfile(BaseModel):
    """Static cost/latency/quality profile for an AI provider."""

    provider: AIProvider
    cost_per_1k: Decimal = Field(default=Decimal("0.0"))
    latency_p95_ms: float = Field(default=0.0)
    quality: float = Field(default=0.85)

    @field_validator("cost_per_1k")
    @classmethod
    def _cost_non_negative(cls, value: Decimal) -> Decimal:
        _require(value >= Decimal("0"), "cost_per_1k must be non-negative")
        return value

    @field_validator("latency_p95_ms")
    @classmethod
    def _latency_positive(cls, value: float) -> float:
        _require(value >= 0.0, "latency_p95_ms must be non-negative")
        return value

    @field_validator("quality")
    @classmethod
    def _quality_range(cls, value: float) -> float:
        _require(0.0 <= value <= 1.0, "quality must be between 0 and 1")
        return value


def _default_profiles() -> Dict[AIProvider, CostProfile]:
    return {
        AIProvider.COPILOT_CHAT: CostProfile(
            provider=AIProvider.COPILOT_CHAT,
            cost_per_1k=Decimal("0.002"),
            latency_p95_ms=320.0,
            quality=0.86,
        ),
        AIProvider.ANTHROPIC: CostProfile(
            provider=AIProvider.ANTHROPIC,
            cost_per_1k=Decimal("0.004"),
            latency_p95_ms=420.0,
            quality=0.92,
        ),
        AIProvider.OPENAI: CostProfile(
            provider=AIProvider.OPENAI,
            cost_per_1k=Decimal("0.003"),
            latency_p95_ms=380.0,
            quality=0.90,
        ),
        AIProvider.GOOGLE: CostProfile(
            provider=AIProvider.GOOGLE,
            cost_per_1k=Decimal("0.0025"),
            latency_p95_ms=360.0,
            quality=0.88,
        ),
        AIProvider.PERPLEXITY: CostProfile(
            provider=AIProvider.PERPLEXITY,
            cost_per_1k=Decimal("0.000"),
            latency_p95_ms=410.0,
            quality=0.80,
        ),
        AIProvider.GROQ: CostProfile(
            provider=AIProvider.GROQ,
            cost_per_1k=Decimal("0.001"),
            latency_p95_ms=180.0,
            quality=0.84,
        ),
        AIProvider.LOCAL: CostProfile(
            provider=AIProvider.LOCAL,
            cost_per_1k=Decimal("0.0"),
            latency_p95_ms=250.0,
            quality=0.75,
        ),
        AIProvider.CUSTOM: CostProfile(
            provider=AIProvider.CUSTOM,
            cost_per_1k=Decimal("0.0"),
            latency_p95_ms=500.0,
            quality=0.70,
        ),
    }


class CostProfileRegistry:
    """Mutable singleton cache of provider cost profiles."""

    _profiles: Dict[AIProvider, CostProfile] = _default_profiles()

    @classmethod
    def all(cls) -> Dict[AIProvider, CostProfile]:
        return dict(cls._profiles)

    @classmethod
    def get(cls, provider: AIProvider) -> CostProfile:
        profile = cls._profiles.get(provider)
        _require(
            profile is not None, f"Cost profile missing for provider={provider.value}"
        )
        assert profile is not None
        return profile

    @classmethod
    def update(cls, provider: AIProvider, profile: CostProfile) -> None:
        _require(
            profile.provider == provider,
            "Profile provider mismatch when updating cost registry",
        )
        cls._profiles[provider] = profile
        log_with_context(
            "info",
            "cost_profile_updated",
            provider=provider.value,
            cost_per_1k=str(profile.cost_per_1k),
            latency_p95_ms=profile.latency_p95_ms,
            quality=profile.quality,
        )

    @classmethod
    def providers(cls) -> Iterable[AIProvider]:
        return tuple(cls._profiles.keys())


def _clamp_decimal(value: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    return max(min(value, maximum), minimum)


def score_provider(
    profile: CostProfile,
    est_tokens: int,
    weights: Tuple[Decimal, Decimal, Decimal] | None = None,
) -> Decimal:
    """Compute weighted score for provider selection (lower is better)."""

    _require(est_tokens >= 0, "Estimated tokens must be non-negative")
    weights_model = CostWeights.from_tuple(weights) if weights else CostWeights()

    estimated_cost = (profile.cost_per_1k * Decimal(est_tokens)) / Decimal("1000")
    cost_norm = _clamp_decimal(
        estimated_cost / Decimal("1.0"), Decimal("0"), Decimal("1")
    )
    latency_norm = _clamp_decimal(
        Decimal(str(profile.latency_p95_ms / 1000.0)),
        Decimal("0"),
        Decimal("1"),
    )
    quality_norm = _clamp_decimal(
        Decimal("1") - Decimal(str(profile.quality)),
        Decimal("0"),
        Decimal("1"),
    )

    score = (
        weights_model.cost * cost_norm
        + weights_model.latency * latency_norm
        + weights_model.quality * quality_norm
    )
    quantized = score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    log_with_context(
        "info",
        "cost_score_computed",
        provider=profile.provider.value,
        est_tokens=est_tokens,
        cost_norm=str(cost_norm),
        latency_norm=str(latency_norm),
        quality_norm=str(quality_norm),
        score=str(quantized),
    )

    return quantized
