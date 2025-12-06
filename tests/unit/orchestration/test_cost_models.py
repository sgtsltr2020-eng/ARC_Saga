from decimal import Decimal

import pytest
from fastapi import HTTPException

from arc_saga.orchestrator.cost_models import (
    CostProfile,
    CostProfileRegistry,
    CostSettings,
    CostWeights,
    score_provider,
)
from arc_saga.orchestrator.types import AIProvider


def test_score_provider_balanced_weights():
    profile = CostProfile(
        provider=AIProvider.GROQ,
        cost_per_1k=Decimal("0.001"),
        latency_p95_ms=200.0,
        quality=0.9,
    )

    score = score_provider(profile, est_tokens=500, weights=None)

    assert score >= Decimal("0.0")
    assert score < Decimal("0.5")


def test_score_provider_custom_weights_cost_heavy():
    profile = CostProfile(
        provider=AIProvider.OPENAI,
        cost_per_1k=Decimal("1.0"),
        latency_p95_ms=10.0,
        quality=0.99,
    )
    weight = CostWeights(
        cost=Decimal("0.9"), latency=Decimal("0.05"), quality=Decimal("0.05")
    )

    score = score_provider(profile, est_tokens=1000, weights=weight.as_tuple())

    assert score > Decimal("0.8")


def test_score_provider_rejects_negative_tokens():
    profile = CostProfile(
        provider=AIProvider.OPENAI,
        cost_per_1k=Decimal("0.001"),
        latency_p95_ms=100.0,
        quality=0.9,
    )

    with pytest.raises(HTTPException):
        score_provider(profile, est_tokens=-1)


def test_cost_settings_env_parsing(monkeypatch):
    monkeypatch.setenv("SAGA_COST_STRATEGY", "BALANCED")
    monkeypatch.setenv("SAGA_COST_WEIGHTS", "0.5,0.25,0.25")
    monkeypatch.setenv("SAGA_COST_MAX_USD", "0.20")
    monkeypatch.setenv("SAGA_COST_ENFORCE_HARD_LIMITS", "true")
    monkeypatch.setenv("SAGA_COST_CACHE_SIZE", "8")
    settings = CostSettings()
    parsed = settings.parsed_weights()

    assert settings.strategy == "BALANCED"
    assert parsed.as_tuple() == (
        Decimal("0.5"),
        Decimal("0.25"),
        Decimal("0.25"),
    )
    assert settings.max_usd == Decimal("0.20")
    assert settings.enforce_hard_limits is True
    assert settings.cache_size == 8


def test_cost_profile_registry_update_and_get():
    provider = AIProvider.PERPLEXITY
    profile = CostProfile(
        provider=provider,
        cost_per_1k=Decimal("0.0"),
        latency_p95_ms=300.0,
        quality=0.82,
    )

    CostProfileRegistry.update(provider, profile)
    fetched = CostProfileRegistry.get(provider)

    assert fetched == profile


def test_cost_weights_require_positive_total():
    with pytest.raises(HTTPException):
        CostWeights.from_tuple((Decimal("0"), Decimal("0"), Decimal("0")))
