# CURSOR INSTRUCTION: PHASE 2.4 COMPONENT 2 IMPLEMENTATION
## ReasoningEngineRegistry — Registry Pattern for Multi-Provider Engine Management

**Start:** Now  
**Duration:** ~6 hours  
**Status:** Copy-paste ready implementation guide  

---

## CRITICAL SETUP

### Prerequisite Verification
```bash
# Verify Component 1 is complete
pytest tests/unit/orchestration/test_response_mode.py -v
# Expected: 11/11 passing

# Verify Phase 2.3 baseline
pytest tests/unit/integrations/ -v
# Expected: 61/61 passing

# Total before Component 2
# Expected: 72/72 tests passing
```

### Files to Create
```
arc_saga/orchestrator/engine_registry.py          (NEW, 120 lines)
tests/unit/orchestration/test_engine_registry.py  (NEW, 200+ lines)
```

### No Files to Modify
- ✅ No changes to Phase 2.3
- ✅ No changes to Component 1
- ✅ No protocol changes
- ✅ Fully backward compatible

---

## IMPLEMENTATION STEP 1: engine_registry.py (120 lines)

### Location
`arc_saga/orchestrator/engine_registry.py`

### Complete Implementation (Copy-Paste Ready)

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

### Verification
```bash
# Check syntax
python -m py_compile arc_saga/orchestrator/engine_registry.py

# Check imports
python -c "from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry; print('✅ Imports OK')"

# Check type safety
mypy --strict arc_saga/orchestrator/engine_registry.py
# Expected: 0 errors
```

---

## IMPLEMENTATION STEP 2: test_engine_registry.py (200+ lines)

### Location
`tests/unit/orchestration/test_engine_registry.py`

### Complete Test Implementation (Copy-Paste Ready)

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

### Verification
```bash
# Check syntax
python -m py_compile tests/unit/orchestration/test_engine_registry.py

# Run tests
pytest tests/unit/orchestration/test_engine_registry.py -v
# Expected: 15/15 passing

# Check test count
pytest tests/unit/orchestration/test_engine_registry.py --collect-only | grep "test_"
# Expected: 15+ tests
```

---

## QUALITY GATES (Run After Implementation)

### Gate 1: Type Safety
```bash
mypy --strict arc_saga/orchestrator/engine_registry.py
# Expected: No errors or warnings (0 total)
```

### Gate 2: Component 2 Tests
```bash
pytest tests/unit/orchestration/test_engine_registry.py -v
# Expected: 15+/15+ passing (100%)
```

### Gate 3: Phase 2.3 Regression
```bash
pytest tests/unit/integrations/ -v
# Expected: 61/61 still passing (no regressions)
```

### Gate 4: Component 1 Regression
```bash
pytest tests/unit/orchestration/test_response_mode.py -v
# Expected: 11/11 still passing (no regressions)
```

### Gate 5: Code Formatting
```bash
black arc_saga/orchestrator/engine_registry.py tests/unit/orchestration/test_engine_registry.py
isort arc_saga/orchestrator/engine_registry.py tests/unit/orchestration/test_engine_registry.py
# Expected: No changes needed (already formatted)
```

### Gate 6: Combined Tests
```bash
pytest tests/unit/orchestration/ -v
# Expected: 26+/26+ passing (11 + 15 = 26 minimum)
```

### Gate 7: All Unit Tests
```bash
pytest tests/unit/ -v
# Expected: 87+/87+ passing (72 existing + 15 new)
```

---

## SUCCESS CHECKLIST

### Pre-Implementation
- [ ] Phase 2.3 baseline verified (61/61)
- [ ] Component 1 verified (11/11)
- [ ] Combined baseline verified (72/72)
- [ ] Understood singleton pattern and fixture discipline
- [ ] Understood fallback separation (handled by Component 3)
- [ ] Understood lean design principle

### File Creation
- [ ] `arc_saga/orchestrator/engine_registry.py` created (120 lines)
- [ ] `tests/unit/orchestration/test_engine_registry.py` created (200+ lines)
- [ ] Imports verified in both files

### Implementation
- [ ] ReasoningEngineRegistry class implemented
- [ ] All 7 methods implemented (register, get, unregister, list_providers, clear, has_provider, get_all)
- [ ] Docstrings complete for all methods
- [ ] Logging calls in place (7 events)
- [ ] All 15+ tests implemented
- [ ] Test fixtures use clear() for singleton discipline

### Quality Gates (All Passing)
- [ ] mypy --strict: 0 errors
- [ ] Component 2 tests: 15+/15+ passing
- [ ] Phase 2.3 regression: 61/61 passing
- [ ] Component 1 regression: 11/11 passing
- [ ] Code formatting: black + isort compliant
- [ ] Combined: 87+/87+ all passing

### Post-Implementation
- [ ] All tests passing
- [ ] No Phase 2.3 regressions
- [ ] Type safety verified
- [ ] Ready to commit to feature branch
- [ ] Ready to start Component 3

---

## 🎯 TIMELINE

| Time | Activity |
|------|----------|
| 0-15min | Review this instruction, understand goals |
| 15-45min | Create files, copy implementations |
| 45-60min | Verify syntax and imports |
| 60-90min | Run Component 2 tests (debug if needed) |
| 90-120min | Run all quality gates |
| 120-180min | Code review, format, final verification |
| 180-360min | Buffer for debugging or refinement |

**Total: ~6 hours (0.75 days)**

---

## 🚀 LAUNCH CHECKLIST

### Before Starting
- [ ] Read this entire instruction file
- [ ] Understood: singleton pattern, test fixtures, fallback separation, lean design
- [ ] Verified Phase 2.3 baseline (61/61 tests passing)
- [ ] Verified Component 1 (11/11 tests passing)

### During Implementation
- [ ] Copy engine_registry.py implementation exactly
- [ ] Copy test implementation exactly
- [ ] Do NOT modify Phase 2.3 or Component 1 files
- [ ] Run quality gates after each major step

### After Implementation
- [ ] All 7 quality gates passing
- [ ] 87+/87+ total tests passing
- [ ] Type safety: 0 errors
- [ ] No regressions detected
- [ ] Ready for Component 3

---

## ⚠️ CRITICAL NOTES

1. **Singleton Discipline:** Always call `clear()` in test fixtures. This is NOT optional.
2. **Fallback Separation:** Registry only manages engines. Fallback logic belongs in ProviderRouter (Component 3).
3. **Lean Design:** Keep it simple. The registry is just a mapping + logging + error handling.
4. **No Breaking Changes:** Do NOT modify Phase 2.3 or Component 1 files.
5. **Copy Exactly:** Use the implementations provided. Do NOT refactor or rearrange.

---

## 📞 INTEGRATION POINTS

### Used By Component 3: ProviderRouter
```python
engine = ReasoningEngineRegistry.get(task.provider)
all_providers = ReasoningEngineRegistry.list_providers()
```

### Used By Component 5: MultiLLMOrchestrator
```python
engine = ReasoningEngineRegistry.get(selected_provider)
```

### Component 2 → Component 3 Flow
```
Component 2 (Registry)
         ↓
    Manages engines by provider
         ↓
Component 3 (Router)
    Uses registry to get engines
    Implements fallback chains
```

---

## ✅ WHEN COMPLETE

- Component 2 is production-ready
- 15+ new tests passing
- Combined 87+ tests passing (72 existing + 15 new)
- Type safety maintained (mypy --strict: 0 errors)
- No Phase 2.3 regressions (61/61 still passing)
- Ready to start Component 3 (ProviderRouter)

---

**Status:** 🔜 Ready to implement  
**Duration:** ~6 hours  
**Complexity:** Medium (singleton pattern, test discipline)  
**Risk:** Low (isolated component, no Phase 2.3 changes)  

**Let's build Component 2.** 🚀
