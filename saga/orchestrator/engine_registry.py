"""Reasoning engine registry for centralized engine management."""

from __future__ import annotations

from typing import Optional

from saga.error_instrumentation import log_with_context
from saga.orchestrator.protocols import IEngineRegistry, IReasoningEngine
from saga.orchestrator.types import AIProvider


class EngineRegistry(IEngineRegistry):
    """
    Instance-based reasoning engine registry.
    Implements IEngineRegistry for dependency injection.
    """

    def __init__(self) -> None:
        self._engines: dict[AIProvider, IReasoningEngine] = {}

    def register(
        self,
        provider: AIProvider,
        engine: IReasoningEngine,
    ) -> None:
        """
        Register a reasoning engine for a provider.

        Args:
            provider: AIProvider enum value (e.g., AIProvider.COPILOT_CHAT)
            engine: Instance implementing IReasoningEngine protocol

        Raises:
            ValueError: If provider already has a registered engine
        """
        if provider in self._engines:
            raise ValueError(f"Engine already registered for {provider.value}.")

        self._engines[provider] = engine
        log_with_context(
            "info",
            "engine_registered",
            provider=provider.value,
            engine_type=type(engine).__name__,
        )

    def get(self, provider: AIProvider) -> Optional[IReasoningEngine]:
        """
        Retrieve a registered engine for a provider.

        Args:
            provider: AIProvider enum value to look up

        Returns:
            IReasoningEngine instance if registered, None otherwise
        """
        engine = self._engines.get(provider)

        log_with_context(
            "info",
            "engine_retrieved" if engine else "engine_not_found",
            provider=provider.value,
            engine_type=type(engine).__name__ if engine else None,
        )

        return engine

    def unregister(self, provider: AIProvider) -> bool:
        """
        Unregister an engine for a provider.

        Args:
            provider: AIProvider enum value to unregister

        Returns:
            True if engine was unregistered, False if wasn't registered
        """
        if provider in self._engines:
            engine_type = type(self._engines[provider]).__name__
            del self._engines[provider]

            log_with_context(
                "info",
                "engine_unregistered",
                provider=provider.value,
                engine_type=engine_type,
            )
            return True

        log_with_context(
            "info",
            "engine_unregister_failed_not_found",
            provider=provider.value,
        )
        return False

    def list_providers(self) -> list[AIProvider]:
        """
        List all registered providers.

        Returns:
            List of AIProvider enum values that have registered engines
        """
        providers = list(self._engines.keys())

        log_with_context(
            "info",
            "registry_list_providers",
            provider_count=len(providers),
            providers=[p.value for p in providers],
        )

        return providers

    def clear(self) -> None:
        """Clear all registered engines."""
        count = len(self._engines)
        self._engines.clear()

        log_with_context(
            "info",
            "registry_cleared",
            engines_removed=count,
        )


# Global singleton instance (for backward compatibility and simple usage)
# In production, prefer injecting EngineRegistry instance into Orchestrator.
_global_registry = EngineRegistry()


class ReasoningEngineRegistry:
    """
    Static proxy for the global singleton registry.
    Kept for backward compatibility.
    """

    @staticmethod
    def register(provider: AIProvider, engine: IReasoningEngine) -> None:
        return _global_registry.register(provider, engine)

    @staticmethod
    def get(provider: AIProvider) -> Optional[IReasoningEngine]:
        return _global_registry.get(provider)

    @staticmethod
    def unregister(provider: AIProvider) -> bool:
        return _global_registry.unregister(provider)

    @staticmethod
    def list_providers() -> list[AIProvider]:
        return _global_registry.list_providers()

    @staticmethod
    def clear() -> None:
        return _global_registry.clear()

    @staticmethod
    def has_provider(provider: AIProvider) -> bool:
        return _global_registry.get(provider) is not None

    @staticmethod
    def get_all() -> dict[AIProvider, IReasoningEngine]:
        """Return a copy of all registered engines."""
        return dict(_global_registry._engines)
