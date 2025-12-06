"""Reasoning engine registry for centralized engine management."""

from typing import Dict, List, Optional

from arc_saga.error_instrumentation import log_with_context
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.types import AIProvider


class ReasoningEngineRegistry:
    """
    Centralized registry for reasoning engines by provider.

    Singleton pattern: Registry state is class-level and shared across all uses.
    Always call clear() in test fixtures to avoid cross-test contamination.

    Design Notes:
    - Registry only manages engines by provider.
    - Fallback logic belongs in ProviderRouter, not here.
    - Keep design lean: simple mapping with logging.

    Example:
        >>> copilot_engine = CopilotReasoningEngine(...)
        >>> ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, copilot_engine)
        >>>
        >>> engine = ReasoningEngineRegistry.get(AIProvider.COPILOT_CHAT)
        >>> providers = ReasoningEngineRegistry.list_providers()
        >>> ReasoningEngineRegistry.unregister(AIProvider.COPILOT_CHAT)
    """

    # Class-level dictionary (shared across all instances)
    _engines: Dict[AIProvider, IReasoningEngine] = {}

    @classmethod
    def register(
        cls,
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
        if provider in cls._engines:
            raise ValueError(f"Engine already registered for {provider.value}.")

        cls._engines[provider] = engine
        log_with_context(
            "info",
            "engine_registered",
            provider=provider.value,
            engine_type=type(engine).__name__,
        )

    @classmethod
    def get(cls, provider: AIProvider) -> Optional[IReasoningEngine]:
        """
        Retrieve a registered engine for a provider.

        Args:
            provider: AIProvider enum value to look up

        Returns:
            IReasoningEngine instance if registered, None otherwise
        """
        engine = cls._engines.get(provider)

        log_with_context(
            "info",
            "engine_retrieved" if engine else "engine_not_found",
            provider=provider.value,
            engine_type=type(engine).__name__ if engine else None,
        )

        return engine

    @classmethod
    def unregister(cls, provider: AIProvider) -> bool:
        """
        Unregister an engine for a provider.

        Args:
            provider: AIProvider enum value to unregister

        Returns:
            True if engine was unregistered, False if wasn't registered
        """
        if provider in cls._engines:
            engine_type = type(cls._engines[provider]).__name__
            del cls._engines[provider]

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

    @classmethod
    def list_providers(cls) -> List[AIProvider]:
        """
        List all registered providers.

        Returns:
            List of AIProvider enum values that have registered engines
        """
        providers = list(cls._engines.keys())

        log_with_context(
            "info",
            "registry_list_providers",
            provider_count=len(providers),
            providers=[p.value for p in providers],
        )

        return providers

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered engines.

        WARNING: Use only for testing! Clears entire registry.
        Call in test fixtures to avoid singleton contamination.
        """
        count = len(cls._engines)
        cls._engines.clear()

        log_with_context(
            "info",
            "registry_cleared",
            engines_removed=count,
        )

    @classmethod
    def has_provider(cls, provider: AIProvider) -> bool:
        """
        Check if a provider has a registered engine.

        Args:
            provider: AIProvider to check

        Returns:
            True if engine is registered, False otherwise
        """
        return provider in cls._engines

    @classmethod
    def get_all(cls) -> Dict[AIProvider, IReasoningEngine]:
        """
        Get all registered engines (for advanced use).

        Returns:
            Dictionary mapping providers to engines (copy, not reference)
        """
        return dict(cls._engines)  # Return copy to prevent accidental modification
