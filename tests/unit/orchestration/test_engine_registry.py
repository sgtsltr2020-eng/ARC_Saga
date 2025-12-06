"""Tests for ReasoningEngineRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.types import AIProvider


@pytest.fixture
def registry():
    """Clear registry before and after each test (singleton discipline)."""
    ReasoningEngineRegistry.clear()
    yield ReasoningEngineRegistry
    ReasoningEngineRegistry.clear()


@pytest.fixture
def mock_engines():
    """Create mock engines for testing."""
    return {
        AIProvider.COPILOT_CHAT: AsyncMock(spec=IReasoningEngine),
        AIProvider.ANTHROPIC: AsyncMock(spec=IReasoningEngine),
        AIProvider.OPENAI: AsyncMock(spec=IReasoningEngine),
    }


# ============ Registration Tests ============


def test_register_single_engine(registry, mock_engines):
    """Verify single engine registration."""
    engine = mock_engines[AIProvider.COPILOT_CHAT]

    registry.register(AIProvider.COPILOT_CHAT, engine)

    assert registry.has_provider(AIProvider.COPILOT_CHAT)
    assert registry.get(AIProvider.COPILOT_CHAT) == engine


def test_register_multiple_engines(registry, mock_engines):
    """Verify multiple engine registration."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    registry.register(AIProvider.ANTHROPIC, mock_engines[AIProvider.ANTHROPIC])

    assert registry.has_provider(AIProvider.COPILOT_CHAT)
    assert registry.has_provider(AIProvider.ANTHROPIC)
    assert len(registry.list_providers()) == 2


def test_register_duplicate_raises_error(registry, mock_engines):
    """Verify duplicate registration prevented."""
    engine1 = mock_engines[AIProvider.COPILOT_CHAT]
    engine2 = MagicMock(spec=IReasoningEngine)

    registry.register(AIProvider.COPILOT_CHAT, engine1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(AIProvider.COPILOT_CHAT, engine2)


# ============ Retrieval Tests ============


def test_get_registered_engine(registry, mock_engines):
    """Verify retrieving registered engine."""
    engine = mock_engines[AIProvider.COPILOT_CHAT]
    registry.register(AIProvider.COPILOT_CHAT, engine)

    retrieved = registry.get(AIProvider.COPILOT_CHAT)

    assert retrieved == engine


def test_get_unregistered_returns_none(registry):
    """Verify unregistered provider returns None."""
    result = registry.get(AIProvider.ANTHROPIC)

    assert result is None


def test_has_provider_true(registry, mock_engines):
    """Verify has_provider returns True for registered provider."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])

    assert registry.has_provider(AIProvider.COPILOT_CHAT) is True


def test_has_provider_false(registry):
    """Verify has_provider returns False for unregistered provider."""
    assert registry.has_provider(AIProvider.ANTHROPIC) is False


# ============ Unregistration Tests ============


def test_unregister_success(registry, mock_engines):
    """Verify successful engine unregistration."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])

    success = registry.unregister(AIProvider.COPILOT_CHAT)

    assert success is True
    assert registry.get(AIProvider.COPILOT_CHAT) is None


def test_unregister_not_registered_returns_false(registry):
    """Verify unregistering non-existent provider returns False."""
    success = registry.unregister(AIProvider.OPENAI)

    assert success is False


def test_unregister_multiple_leaves_others(registry, mock_engines):
    """Verify unregistering one doesn't affect others."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    registry.register(AIProvider.ANTHROPIC, mock_engines[AIProvider.ANTHROPIC])

    registry.unregister(AIProvider.COPILOT_CHAT)

    assert registry.get(AIProvider.COPILOT_CHAT) is None
    assert registry.get(AIProvider.ANTHROPIC) is not None


# ============ Listing Tests ============


def test_list_providers_empty(registry):
    """Verify empty list when no providers registered."""
    providers = registry.list_providers()

    assert providers == []


def test_list_providers_multiple(registry, mock_engines):
    """Verify multiple providers in list."""
    for provider, engine in mock_engines.items():
        registry.register(provider, engine)

    providers = registry.list_providers()

    assert set(providers) == set(mock_engines.keys())


# ============ Clear Tests ============


def test_clear_empties_registry(registry, mock_engines):
    """Verify clear removes all engines."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    registry.register(AIProvider.ANTHROPIC, mock_engines[AIProvider.ANTHROPIC])

    registry.clear()

    assert len(registry.list_providers()) == 0
    assert registry.get(AIProvider.COPILOT_CHAT) is None


def test_clear_allows_reregistration(registry, mock_engines):
    """Verify can register after clear."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    registry.clear()

    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])

    assert registry.has_provider(AIProvider.COPILOT_CHAT)


# ============ Edge Case Tests ============


def test_singleton_behavior(registry, mock_engines):
    """Verify singleton behavior (shared state)."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])

    engine1 = registry.get(AIProvider.COPILOT_CHAT)
    engine2 = registry.get(AIProvider.COPILOT_CHAT)

    assert engine1 is engine2
    assert engine1 is mock_engines[AIProvider.COPILOT_CHAT]


def test_get_all_returns_copy(registry, mock_engines):
    """Verify get_all returns copy (not reference)."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])

    all_engines = registry.get_all()

    # Modifying returned dict shouldn't affect registry
    all_engines[AIProvider.ANTHROPIC] = mock_engines[AIProvider.ANTHROPIC]

    assert AIProvider.ANTHROPIC not in registry.list_providers()
