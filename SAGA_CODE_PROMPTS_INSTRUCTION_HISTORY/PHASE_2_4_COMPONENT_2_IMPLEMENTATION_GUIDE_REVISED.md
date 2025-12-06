# PHASE 2.4 COMPONENT 2: REASONINGENGINEREGISTRY — IMPLEMENTATION GUIDE (REVISED)

**Status:** 🔜 READY FOR IMPLEMENTATION  
**Timeline:** ~6 hours (0.75 days)  
**Based On:** Component 1 ✅ Complete  
**Tests Planned:** 15+  
**Expected Lines:** ~320 (120 production + 200 test)

---

## 🎯 COMPONENT 2 OBJECTIVES

### Primary Objectives
1. Create centralized registry for reasoning engines by provider
2. Support dynamic registration and unregistration
3. Enable engine lookups by AIProvider enum
4. Provide clean integration points for fallback chains (Component 3)
5. Maintain type safety and error handling

### Secondary Objectives
1. Implement singleton pattern cleanly
2. Provide comprehensive logging
3. Support testing with mock engines
4. Enable provider enumeration
5. Include clear() for testing discipline

---

## 🏗️ ARCHITECTURE

### Registry Pattern
```
┌─────────────────────────────────────────┐
│  ReasoningEngineRegistry (Singleton)    │
├─────────────────────────────────────────┤
│  _engines: Dict[AIProvider,             │
│    IReasoningEngine]                    │
├─────────────────────────────────────────┤
│  @classmethod register()                │
│  @classmethod get()                     │
│  @classmethod unregister()              │
│  @classmethod list_providers()          │
│  @classmethod clear()                   │
│  @classmethod has_provider()            │
│  @classmethod get_all()                 │
└─────────────────────────────────────────┘

         ↓
┌─────────────────────────────────────────┐
│  Available Engines (by provider)        │
├─────────────────────────────────────────┤
│  • COPILOT_CHAT → CopilotReasoningEngine│
│  • CLAUDE → ClaudeReasoningEngine (stub)│
│  • GPT4 → GPT4ReasoningEngine (stub)    │
└─────────────────────────────────────────┘
```

### Key Design Notes
- **Singleton Discipline:** Registry state is class-level. Always call `clear()` in test fixtures to avoid cross-test contamination.
- **Fallback Separation:** Registry only manages engines. Fallback chains are handled in ProviderRouter (Component 3). This keeps responsibilities clean.
- **Lean Design:** Avoid unnecessary abstractions — registry is a simple mapping with logging.

---

## 📝 IMPLEMENTATION SPECIFICATION

### File: arc_saga/orchestrator/engine_registry.py

```python
"""Reasoning engine registry for centralized engine management."""

from typing import Dict, List, Optional
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.types import AIProvider
from arc_saga.errorinstrumentation import log_with_context


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
                "engine_unregistered",
                provider=provider.value,
                engine_type=engine_type,
            )
            return True
        
        log_with_context(
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
```

**File Size:** ~120 lines  
**Methods:** 7 public methods  
**Logging Events:** 7 distinct events  
**Type Safety:** Full (no Any types)

---

## 🧪 TEST SPECIFICATION

### File: tests/unit/orchestration/test_engine_registry.py

```python
"""Tests for ReasoningEngineRegistry."""

import pytest
from unittest.mock import MagicMock, AsyncMock
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
        AIProvider.CLAUDE: AsyncMock(spec=IReasoningEngine),
        AIProvider.GPT4: AsyncMock(spec=IReasoningEngine),
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
    registry.register(AIProvider.CLAUDE, mock_engines[AIProvider.CLAUDE])
    
    assert registry.has_provider(AIProvider.COPILOT_CHAT)
    assert registry.has_provider(AIProvider.CLAUDE)
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
    result = registry.get(AIProvider.CLAUDE)
    
    assert result is None


def test_has_provider_true(registry, mock_engines):
    """Verify has_provider returns True for registered provider."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    
    assert registry.has_provider(AIProvider.COPILOT_CHAT) is True


def test_has_provider_false(registry):
    """Verify has_provider returns False for unregistered provider."""
    assert registry.has_provider(AIProvider.CLAUDE) is False


# ============ Unregistration Tests ============

def test_unregister_success(registry, mock_engines):
    """Verify successful engine unregistration."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    
    success = registry.unregister(AIProvider.COPILOT_CHAT)
    
    assert success is True
    assert registry.get(AIProvider.COPILOT_CHAT) is None


def test_unregister_not_registered_returns_false(registry):
    """Verify unregistering non-existent provider returns False."""
    success = registry.unregister(AIProvider.GPT4)
    
    assert success is False


def test_unregister_multiple_leaves_others(registry, mock_engines):
    """Verify unregistering one doesn't affect others."""
    registry.register(AIProvider.COPILOT_CHAT, mock_engines[AIProvider.COPILOT_CHAT])
    registry.register(AIProvider.CLAUDE, mock_engines[AIProvider.CLAUDE])
    
    registry.unregister(AIProvider.COPILOT_CHAT)
    
    assert registry.get(AIProvider.COPILOT_CHAT) is None
    assert registry.get(AIProvider.CLAUDE) is not None


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
    registry.register(AIProvider.CLAUDE, mock_engines[AIProvider.CLAUDE])
    
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
    all_engines[AIProvider.CLAUDE] = mock_engines[AIProvider.CLAUDE]
    
    assert AIProvider.CLAUDE not in registry.list_providers()
```

**File Size:** ~200+ lines  
**Tests:** 15+ individual tests  
**Coverage Target:** 95%+  
**Test Groups:** 7 (Registration, Retrieval, Unregistration, Listing, Clear, Edge Cases)

---

## ✅ IMPLEMENTATION CHECKLIST

### Pre-Implementation
- [ ] Component 1 complete and all tests passing
- [ ] Review this spec carefully
- [ ] Understand singleton pattern and test fixture discipline
- [ ] Create `arc_saga/orchestrator/engine_registry.py`
- [ ] Create `tests/unit/orchestration/test_engine_registry.py`

### Implementation Phase

#### engine_registry.py
- [ ] Implement ReasoningEngineRegistry class
- [ ] Implement all 7 methods (register, get, unregister, list_providers, clear, has_provider, get_all)
- [ ] Add comprehensive docstrings
- [ ] Add logging calls (7 distinct events)
- [ ] Verify imports work

#### test_engine_registry.py
- [ ] Create registry fixture with clear() discipline
- [ ] Create mock_engines fixture
- [ ] Implement all test groups (15+ tests)
- [ ] All tests import correctly

### Quality Gates
- [ ] mypy --strict (0 errors)
  - `mypy --strict arc_saga/orchestrator/engine_registry.py`
- [ ] Tests passing (15+/15+)
  - `pytest tests/unit/orchestration/test_engine_registry.py -v`
- [ ] Phase 2.3 regression (61/61 still passing)
  - `pytest tests/unit/integrations/ -v`
- [ ] Component 1 regression (11/11 still passing)
  - `pytest tests/unit/orchestration/test_response_mode.py -v`
- [ ] Code formatting
  - `black arc_saga/orchestrator/engine_registry.py`
  - `isort arc_saga/orchestrator/engine_registry.py`
- [ ] All tests together (87+/87+)
  - `pytest tests/unit/ -v`

### Post-Implementation Verification
- [ ] Total tests: 87+/87+ all passing
- [ ] Coverage: 95%+ (both components)
- [ ] Type safety: 0 errors
- [ ] No Phase 2.3 regressions (61/61 still passing)
- [ ] Ready for Component 3

---

## 🔗 INTEGRATION POINTS

### Required Imports
```python
from typing import Dict, List, Optional
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.types import AIProvider
from arc_saga.errorinstrumentation import log_with_context
```

### Used By Component 3: ProviderRouter
```python
# Component 3 will use:
engine = ReasoningEngineRegistry.get(task.provider)
all_providers = ReasoningEngineRegistry.list_providers()
```

### Used By Component 5: MultiLLMOrchestrator
```python
# Component 5 will use:
engine = ReasoningEngineRegistry.get(selected_provider)
```

### No Breaking Changes
- No modifications needed to Phase 2.3 files
- CopilotReasoningEngine unchanged
- No changes to protocols or types

---

## 🎯 SUCCESS CRITERIA FOR COMPONENT 2

- [x] ReasoningEngineRegistry fully implemented
- [x] 15+ tests passing (100%)
- [x] Type safety: mypy --strict = 0 errors
- [x] Phase 2.3 tests still passing (61/61)
- [x] Component 1 tests still passing (11/11)
- [x] Combined: 87+/87+ tests passing
- [x] Code formatted (black + isort)
- [x] Docstrings complete
- [x] Logging comprehensive
- [x] Ready for Component 3

---

## ⏱️ TIMELINE

**Component 2 Timeline (~6 hours)**

| Time | Activity |
|------|----------|
| 0-1h | Review spec, understand singleton discipline, create files |
| 1-2h | Implement ReasoningEngineRegistry class (7 methods) |
| 2-3h | Implement test fixtures and first 3 test groups |
| 3-4h | Implement remaining 4 test groups (15+ tests) |
| 4-5h | Run quality gates, fix any issues |
| 5-6h | Code review, merge to feature branch |

---

## 📞 READY TO START?

When you're ready to implement Component 2:

1. Copy the implementation spec above
2. Create `arc_saga/orchestrator/engine_registry.py`
3. Copy test spec above
4. Create `tests/unit/orchestration/test_engine_registry.py`
5. Run tests and quality gates
6. Verify all pass
7. Ready for Component 3

**Component 2 builds directly on Component 1's foundation.**  
**Use the patterns established in Component 1 tests.**  
**Singleton discipline is critical — always clear() in fixtures.**

---

**Component 1 Complete: ✅ Day 1 (Dec 7)**  
**Component 2 Ready: 🔜 Day 2 (Dec 8)**  
**Phase 2.4 Progress:** 1/5 components complete (20%)  

**Let's build Component 2.** 🚀
