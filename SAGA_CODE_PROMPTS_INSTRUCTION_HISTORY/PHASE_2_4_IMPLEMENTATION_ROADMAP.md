# PHASE 2.4 IMPLEMENTATION ROADMAP
## Multi-LLM Orchestration: ResponseMode + ProviderRouter + Cost Optimization

**Status:** 🔜 READY FOR IMPLEMENTATION  
**Based On:** Phase 2.3 Foundation (Production Ready)  
**Date:** December 5, 2025  

---

## 🎯 PHASE 2.4 OBJECTIVES

Transform Phase 2.3 (single-provider integration) into a **true orchestrator** that:
- ✅ Routes tasks to optimal providers (Copilot, Claude, GPT-4)
- ✅ Streams or returns complete responses on demand
- ✅ Selects providers based on cost/quality tradeoffs
- ✅ Handles fallbacks and provider failures gracefully
- ✅ Maintains comprehensive logging and observability

---

## 📋 PHASE 2.4 STRUCTURE (5 Major Components)

### Component 1: ResponseMode (Streaming vs Complete)

**Purpose:** Allow clients to request responses as streaming tokens or complete text

**Implementation Plan:**

```python
# Update arc_saga/orchestrator/types.py
class ResponseMode(str, Enum):
    STREAMING = "streaming"      # Yield tokens as they arrive
    COMPLETE = "complete"        # Wait for full response, return once

# Update AITask to include response mode
@dataclass
class AITask:
    id: str
    input_data: AITaskInput
    user_id: str
    response_mode: ResponseMode = ResponseMode.COMPLETE  # Default
    timeout_ms: int = 30000
    # ... existing fields ...

# Update AIResult to indicate streaming eligibility
@dataclass
class AIResult:
    task_id: str
    success: bool
    output_data: Optional[AIResultOutput]
    status: TaskStatus
    duration_ms: int
    stream_available: bool = False  # NEW
    # ... existing fields ...
```

**Streaming Implementation:**

```python
# In CopilotReasoningEngine
async def reason(self, task: AITask) -> Union[AIResult, AsyncGenerator[str, None]]:
    """
    Execute reasoning task.
    
    If task.response_mode == STREAMING:
        Yields token strings as they arrive
    Else:
        Returns AIResult with complete response
    """
    if task.response_mode == ResponseMode.STREAMING:
        return self.reason_streaming(task)
    else:
        return await self.reason_complete(task)

async def reason_complete(self, task: AITask) -> AIResult:
    """Collect all tokens and return AIResult (existing reason() logic)."""
    tokens = []
    async for token in self.reason_streaming(task):
        tokens.append(token)
    
    full_response = "".join(tokens)
    return AIResult(
        task_id=task.id,
        success=True,
        output_data=AIResultOutput(response=full_response, ...),
        status=TaskStatus.COMPLETED,
        duration_ms=...,
        stream_available=True,
    )

async def reason_streaming(self, task: AITask) -> AsyncGenerator[str, None]:
    """Stream tokens as they arrive (already implemented in Phase 2.3)."""
    # existing streaming logic...
```

**Test Strategy:**

```python
# tests/unit/orchestration/test_response_mode.py
@pytest.mark.asyncio
async def test_response_mode_streaming():
    """Verify streaming mode yields tokens."""
    task = AITask(..., response_mode=ResponseMode.STREAMING)
    tokens = []
    async for token in engine.reason(task):
        tokens.append(token)
    assert len(tokens) > 0
    assert "".join(tokens) == expected_full_response

@pytest.mark.asyncio
async def test_response_mode_complete():
    """Verify complete mode returns AIResult."""
    task = AITask(..., response_mode=ResponseMode.COMPLETE)
    result = await engine.reason(task)
    assert isinstance(result, AIResult)
    assert result.success == True
    assert result.output_data.response == expected_full_response
```

**Files to Create/Modify:**

- `arc_saga/orchestrator/types.py` — Add ResponseMode enum, update AITask
- `arc_saga/integrations/copilot_reasoning_engine.py` — Add reason_complete(), update reason()
- `tests/unit/orchestration/test_response_mode.py` — NEW (10+ tests)

---

### Component 2: ReasoningEngineRegistry

**Purpose:** Dynamically register/lookup reasoning engines by provider

**Implementation Plan:**

```python
# Create arc_saga/orchestrator/engine_registry.py

from typing import Dict, Type
from arc_saga.orchestrator.protocols import IReasoningEngine
from arc_saga.orchestrator.types import AIProvider

class ReasoningEngineRegistry:
    """
    Central registry for reasoning engines.
    
    Supports:
    - Register engines for providers
    - Retrieve engines by provider
    - Fallback chain (if Copilot fails, try Claude)
    - Type safety and validation
    """
    
    _engines: Dict[AIProvider, IReasoningEngine] = {}
    
    @classmethod
    def register(
        cls,
        provider: AIProvider,
        engine: IReasoningEngine,
    ) -> None:
        """
        Register an engine for a provider.
        
        Args:
            provider: AIProvider enum value
            engine: Instance implementing IReasoningEngine
        
        Raises:
            ValueError: If provider already registered (no duplicates)
        """
        if provider in cls._engines:
            raise ValueError(f"Engine already registered for {provider}")
        
        cls._engines[provider] = engine
        log_with_context("engine_registered", provider=provider.value)
    
    @classmethod
    def get(cls, provider: AIProvider) -> Optional[IReasoningEngine]:
        """
        Retrieve engine for provider.
        
        Args:
            provider: AIProvider enum value
        
        Returns:
            Engine instance or None if not registered
        """
        engine = cls._engines.get(provider)
        if engine:
            log_with_context("engine_retrieved", provider=provider.value)
        return engine
    
    @classmethod
    def unregister(cls, provider: AIProvider) -> bool:
        """
        Unregister an engine.
        
        Args:
            provider: AIProvider enum value
        
        Returns:
            True if unregistered, False if wasn't registered
        """
        if provider in cls._engines:
            del cls._engines[provider]
            log_with_context("engine_unregistered", provider=provider.value)
            return True
        return False
    
    @classmethod
    def list_providers(cls) -> List[AIProvider]:
        """Return all registered providers."""
        return list(cls._engines.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered engines (for testing)."""
        cls._engines.clear()
        log_with_context("registry_cleared")
```

**Usage Example:**

```python
# At application startup
from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry
from arc_saga.integrations.copilot_reasoning_engine import CopilotReasoningEngine
from arc_saga.orchestrator.types import AIProvider

# Initialize Copilot engine
copilot_engine = CopilotReasoningEngine(
    client_id=os.getenv("COPILOT_CLIENT_ID"),
    client_secret=os.getenv("COPILOT_CLIENT_SECRET"),
    tenant_id=os.getenv("COPILOT_TENANT_ID"),
    token_store=token_store,
)

# Register it
ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, copilot_engine)

# Later, retrieve it
engine = ReasoningEngineRegistry.get(AIProvider.COPILOT_CHAT)
```

**Test Strategy:**

```python
# tests/unit/orchestration/test_engine_registry.py
@pytest.fixture
def registry():
    """Clear registry before each test."""
    ReasoningEngineRegistry.clear()
    yield ReasoningEngineRegistry
    ReasoningEngineRegistry.clear()

def test_register_engine(registry):
    """Verify engine registration."""
    engine = MagicMock(spec=IReasoningEngine)
    registry.register(AIProvider.COPILOT_CHAT, engine)
    assert registry.get(AIProvider.COPILOT_CHAT) == engine

def test_register_duplicate_raises_error(registry):
    """Verify duplicate registration prevented."""
    engine1 = MagicMock(spec=IReasoningEngine)
    engine2 = MagicMock(spec=IReasoningEngine)
    
    registry.register(AIProvider.COPILOT_CHAT, engine1)
    
    with pytest.raises(ValueError, match="already registered"):
        registry.register(AIProvider.COPILOT_CHAT, engine2)

def test_get_unregistered_returns_none(registry):
    """Verify unregistered provider returns None."""
    assert registry.get(AIProvider.CLAUDE) is None

def test_unregister(registry):
    """Verify engine unregistration."""
    engine = MagicMock(spec=IReasoningEngine)
    registry.register(AIProvider.COPILOT_CHAT, engine)
    
    assert registry.unregister(AIProvider.COPILOT_CHAT) == True
    assert registry.get(AIProvider.COPILOT_CHAT) is None
    assert registry.unregister(AIProvider.COPILOT_CHAT) == False

def test_list_providers(registry):
    """Verify provider list."""
    engine1 = MagicMock(spec=IReasoningEngine)
    engine2 = MagicMock(spec=IReasoningEngine)
    
    registry.register(AIProvider.COPILOT_CHAT, engine1)
    registry.register(AIProvider.CLAUDE, engine2)
    
    providers = registry.list_providers()
    assert AIProvider.COPILOT_CHAT in providers
    assert AIProvider.CLAUDE in providers
```

**Files to Create:**

- `arc_saga/orchestrator/engine_registry.py` — NEW (120 lines)
- `tests/unit/orchestration/test_engine_registry.py` — NEW (200 lines)

---

### Component 3: ProviderRouter

**Purpose:** Route AITasks to appropriate provider engines

**Implementation Plan:**

```python
# Create arc_saga/orchestrator/provider_router.py

from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry
from arc_saga.orchestrator.types import AITask, AIResult, AIProvider
from arc_saga.exceptions import *

class ProviderRouter:
    """
    Routes tasks to the appropriate reasoning engine.
    
    Supports:
    - Provider-specific routing
    - Fallback chain (try Copilot, fall back to Claude)
    - Error handling and retry logic
    - Logging and observability
    """
    
    # Default fallback chain
    FALLBACK_CHAIN = {
        AIProvider.COPILOT_CHAT: [AIProvider.CLAUDE, AIProvider.GPT4],
        AIProvider.CLAUDE: [AIProvider.COPILOT_CHAT, AIProvider.GPT4],
        AIProvider.GPT4: [AIProvider.COPILOT_CHAT, AIProvider.CLAUDE],
    }
    
    @classmethod
    async def route(
        cls,
        task: AITask,
        use_fallback: bool = True,
    ) -> Union[AIResult, AsyncGenerator[str, None]]:
        """
        Route task to appropriate engine.
        
        Args:
            task: AITask with provider specified
            use_fallback: Try fallback providers if primary fails
        
        Returns:
            AIResult or async generator of tokens (if streaming)
        
        Raises:
            ValueError: No engine registered for provider
            TransientError: All providers failed and no fallback
        """
        
        # Get primary engine
        engine = ReasoningEngineRegistry.get(task.input_data.provider)
        if not engine:
            raise ValueError(f"No engine registered for {task.input_data.provider}")
        
        log_with_context(
            "routing_start",
            task_id=task.id,
            provider=task.input_data.provider.value,
        )
        
        try:
            # Attempt primary provider
            result = await engine.reason(task)
            log_with_context(
                "routing_success",
                task_id=task.id,
                provider=task.input_data.provider.value,
            )
            return result
        
        except (AuthenticationError, InputValidationError) as e:
            # Permanent errors, don't retry
            log_with_context(
                "routing_failed_permanent",
                task_id=task.id,
                provider=task.input_data.provider.value,
                error=type(e).__name__,
            )
            raise
        
        except (RateLimitError, TransientError, TimeoutError) as e:
            # Transient errors, try fallback chain
            if not use_fallback:
                raise
            
            fallback_providers = cls.FALLBACK_CHAIN.get(
                task.input_data.provider, []
            )
            
            for fallback_provider in fallback_providers:
                log_with_context(
                    "routing_fallback_attempt",
                    task_id=task.id,
                    primary_provider=task.input_data.provider.value,
                    fallback_provider=fallback_provider.value,
                )
                
                fallback_engine = ReasoningEngineRegistry.get(fallback_provider)
                if not fallback_engine:
                    continue
                
                try:
                    # Update task provider for fallback
                    fallback_task = AITask(
                        id=task.id,
                        input_data=AITaskInput(
                            prompt=task.input_data.prompt,
                            system_prompt=task.input_data.system_prompt,
                            max_tokens=task.input_data.max_tokens,
                            temperature=task.input_data.temperature,
                            provider=fallback_provider,  # Changed
                        ),
                        user_id=task.user_id,
                        response_mode=task.response_mode,
                    )
                    
                    result = await fallback_engine.reason(fallback_task)
                    log_with_context(
                        "routing_fallback_success",
                        task_id=task.id,
                        fallback_provider=fallback_provider.value,
                    )
                    return result
                
                except Exception as fallback_error:
                    log_with_context(
                        "routing_fallback_failed",
                        task_id=task.id,
                        fallback_provider=fallback_provider.value,
                        error=type(fallback_error).__name__,
                    )
                    continue
            
            # All providers failed
            log_with_context(
                "routing_all_failed",
                task_id=task.id,
                primary_provider=task.input_data.provider.value,
            )
            raise TransientError(
                f"All providers failed. Primary: {type(e).__name__}"
            )
```

**Test Strategy:**

```python
# tests/unit/orchestration/test_provider_router.py
@pytest.mark.asyncio
async def test_route_to_copilot():
    """Verify routing to Copilot engine."""
    task = AITask(
        id="task1",
        input_data=AITaskInput(
            prompt="Test",
            provider=AIProvider.COPILOT_CHAT,
        ),
        user_id="user1",
    )
    
    mock_engine = AsyncMock(spec=IReasoningEngine)
    mock_engine.reason = AsyncMock(return_value=AIResult(...))
    
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, mock_engine)
    
    result = await ProviderRouter.route(task)
    
    assert result.success == True
    mock_engine.reason.assert_called_once_with(task)

@pytest.mark.asyncio
async def test_route_fallback_on_rate_limit():
    """Verify fallback when primary provider rate-limited."""
    task = AITask(
        id="task1",
        input_data=AITaskInput(
            prompt="Test",
            provider=AIProvider.COPILOT_CHAT,
        ),
        user_id="user1",
    )
    
    copilot_engine = AsyncMock(spec=IReasoningEngine)
    copilot_engine.reason = AsyncMock(
        side_effect=RateLimitError("Rate limited")
    )
    
    claude_engine = AsyncMock(spec=IReasoningEngine)
    claude_engine.reason = AsyncMock(return_value=AIResult(...))
    
    ReasoningEngineRegistry.register(AIProvider.COPILOT_CHAT, copilot_engine)
    ReasoningEngineRegistry.register(AIProvider.CLAUDE, claude_engine)
    
    result = await ProviderRouter.route(task)
    
    # Should have fallen back to Claude
    assert claude_engine.reason.called
```

**Files to Create:**

- `arc_saga/orchestrator/provider_router.py` — NEW (200 lines)
- `tests/unit/orchestration/test_provider_router.py` — NEW (250 lines)

---

### Component 4: CostOptimizer

**Purpose:** Select provider based on cost/quality/latency tradeoffs

**Implementation Plan:**

```python
# Create arc_saga/orchestrator/cost_optimizer.py

from dataclasses import dataclass
from enum import Enum
from arc_saga.orchestrator.types import AIProvider

class OptimizationStrategy(str, Enum):
    CHEAPEST = "cheapest"              # Minimum cost
    FASTEST = "fastest"                # Minimum latency
    BALANCED = "balanced"              # Cost + latency balance
    QUALITY = "quality"                # Highest quality (usually most expensive)

@dataclass
class ProviderMetrics:
    """Metadata for a provider."""
    
    provider: AIProvider
    cost_per_token: float              # USD per token (input + output avg)
    latency_p50_ms: float              # Median latency
    latency_p99_ms: float              # 99th percentile latency
    success_rate: float                # 0.0 - 1.0 (% of requests that succeed)
    quality_score: float               # 0.0 - 10.0 (subjective quality)
    available: bool = True             # Is provider currently available
    
    def cost_score(self) -> float:
        """Lower is better (for sorting)."""
        return self.cost_per_token if self.available else float('inf')
    
    def latency_score(self) -> float:
        """Lower is better."""
        return self.latency_p50_ms if self.available else float('inf')
    
    def quality_score(self) -> float:
        """Higher is better (negate for sorting)."""
        return -self.quality_score if self.available else float('inf')
    
    def balanced_score(self) -> float:
        """
        Combined score: cost + normalized latency + quality.
        Lower is better.
        """
        if not self.available:
            return float('inf')
        
        # Normalize to 0-1 range (example)
        cost_norm = min(self.cost_per_token / 0.01, 1.0)  # Normalize to $0.01
        latency_norm = min(self.latency_p50_ms / 1000, 1.0)  # Normalize to 1s
        quality_norm = 1.0 - (self.quality_score / 10.0)  # Invert (lower is better)
        
        return (0.5 * cost_norm) + (0.3 * latency_norm) + (0.2 * quality_norm)

class CostOptimizer:
    """
    Selects optimal provider based on strategy and metrics.
    
    Example:
        >>> metrics = [
        ...     ProviderMetrics(
        ...         provider=AIProvider.COPILOT_CHAT,
        ...         cost_per_token=0.005,
        ...         latency_p50_ms=150,
        ...         latency_p99_ms=500,
        ...         success_rate=0.99,
        ...         quality_score=9.0,
        ...     ),
        ...     ProviderMetrics(
        ...         provider=AIProvider.CLAUDE,
        ...         cost_per_token=0.003,
        ...         latency_p50_ms=300,
        ...         latency_p99_ms=800,
        ...         success_rate=0.98,
        ...         quality_score=8.5,
        ...     ),
        ... ]
        >>> 
        >>> optimizer = CostOptimizer(metrics)
        >>> 
        >>> # Get cheapest provider
        >>> optimizer.select(strategy=OptimizationStrategy.CHEAPEST)
        AIProvider.CLAUDE
        >>> 
        >>> # Get fastest provider
        >>> optimizer.select(strategy=OptimizationStrategy.FASTEST)
        AIProvider.COPILOT_CHAT
        >>> 
        >>> # Get balanced provider
        >>> optimizer.select(strategy=OptimizationStrategy.BALANCED)
        AIProvider.COPILOT_CHAT  # or CLAUDE depending on scores
    """
    
    def __init__(self, metrics: List[ProviderMetrics]):
        """
        Initialize optimizer with provider metrics.
        
        Args:
            metrics: List of ProviderMetrics for each provider
        """
        self.metrics_by_provider = {m.provider: m for m in metrics}
    
    def select(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ) -> AIProvider:
        """
        Select optimal provider based on strategy.
        
        Args:
            strategy: Optimization strategy
        
        Returns:
            Selected AIProvider
        
        Raises:
            ValueError: No available providers
        """
        if strategy == OptimizationStrategy.CHEAPEST:
            return min(
                self.metrics_by_provider.values(),
                key=lambda m: m.cost_score(),
            ).provider
        
        elif strategy == OptimizationStrategy.FASTEST:
            return min(
                self.metrics_by_provider.values(),
                key=lambda m: m.latency_score(),
            ).provider
        
        elif strategy == OptimizationStrategy.QUALITY:
            return min(
                self.metrics_by_provider.values(),
                key=lambda m: m.quality_score(),
            ).provider
        
        elif strategy == OptimizationStrategy.BALANCED:
            return min(
                self.metrics_by_provider.values(),
                key=lambda m: m.balanced_score(),
            ).provider
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def update_metrics(self, metrics: List[ProviderMetrics]) -> None:
        """Update metrics (e.g., based on recent performance)."""
        self.metrics_by_provider = {m.provider: m for m in metrics}
```

**Test Strategy:**

```python
# tests/unit/orchestration/test_cost_optimizer.py
def test_select_cheapest():
    """Verify cheapest provider selection."""
    metrics = [
        ProviderMetrics(
            provider=AIProvider.COPILOT_CHAT,
            cost_per_token=0.005,
            latency_p50_ms=150,
            latency_p99_ms=500,
            success_rate=0.99,
            quality_score=9.0,
        ),
        ProviderMetrics(
            provider=AIProvider.CLAUDE,
            cost_per_token=0.003,  # Cheaper
            latency_p50_ms=300,
            latency_p99_ms=800,
            success_rate=0.98,
            quality_score=8.5,
        ),
    ]
    
    optimizer = CostOptimizer(metrics)
    selected = optimizer.select(OptimizationStrategy.CHEAPEST)
    
    assert selected == AIProvider.CLAUDE

def test_select_fastest():
    """Verify fastest provider selection."""
    metrics = [
        ProviderMetrics(
            provider=AIProvider.COPILOT_CHAT,
            cost_per_token=0.005,
            latency_p50_ms=150,  # Faster
            latency_p99_ms=500,
            success_rate=0.99,
            quality_score=9.0,
        ),
        ProviderMetrics(
            provider=AIProvider.CLAUDE,
            cost_per_token=0.003,
            latency_p50_ms=300,
            latency_p99_ms=800,
            success_rate=0.98,
            quality_score=8.5,
        ),
    ]
    
    optimizer = CostOptimizer(metrics)
    selected = optimizer.select(OptimizationStrategy.FASTEST)
    
    assert selected == AIProvider.COPILOT_CHAT
```

**Files to Create:**

- `arc_saga/orchestrator/cost_optimizer.py` — NEW (250 lines)
- `tests/unit/orchestration/test_cost_optimizer.py` — NEW (200 lines)

---

### Component 5: MultiLLMOrchestrator

**Purpose:** Coordinate all components into unified orchestrator

**Implementation Plan:**

```python
# Create arc_saga/orchestrator/multi_llm_orchestrator.py

class MultiLLMOrchestrator:
    """
    Unified orchestrator coordinating all Phase 2.4 components.
    
    - Selects provider (registry + router + optimizer)
    - Handles response modes (streaming vs complete)
    - Manages errors and fallbacks
    - Logs everything with context
    """
    
    def __init__(
        self,
        cost_optimizer: CostOptimizer,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        """Initialize with cost optimizer and strategy."""
        self.optimizer = cost_optimizer
        self.strategy = optimization_strategy
    
    async def execute_task(
        self,
        task: AITask,
    ) -> Union[AIResult, AsyncGenerator[str, None]]:
        """
        Execute a task using optimal provider.
        
        Process:
        1. Select provider based on optimizer + strategy
        2. Update task with selected provider
        3. Route to engine via ProviderRouter
        4. Return result or stream
        
        Args:
            task: AITask (provider may be overridden)
        
        Returns:
            AIResult or async generator of tokens
        """
        
        # 1. Select provider
        selected_provider = self.optimizer.select(self.strategy)
        log_with_context(
            "orchestrator_provider_selected",
            task_id=task.id,
            user_id=task.user_id,
            strategy=self.strategy.value,
            selected_provider=selected_provider.value,
        )
        
        # 2. Update task with selected provider
        updated_task = AITask(
            id=task.id,
            input_data=AITaskInput(
                prompt=task.input_data.prompt,
                system_prompt=task.input_data.system_prompt,
                max_tokens=task.input_data.max_tokens,
                temperature=task.input_data.temperature,
                provider=selected_provider,  # Override
            ),
            user_id=task.user_id,
            response_mode=task.response_mode,
        )
        
        # 3. Route to engine
        try:
            result = await ProviderRouter.route(updated_task)
            log_with_context(
                "orchestrator_task_complete",
                task_id=task.id,
                provider=selected_provider.value,
            )
            return result
        
        except Exception as e:
            log_with_context(
                "orchestrator_task_failed",
                task_id=task.id,
                provider=selected_provider.value,
                error=type(e).__name__,
            )
            raise
    
    def update_strategy(self, strategy: OptimizationStrategy) -> None:
        """Update optimization strategy on the fly."""
        self.strategy = strategy
        log_with_context(
            "orchestrator_strategy_updated",
            new_strategy=strategy.value,
        )
```

**Test Strategy:**

```python
# tests/unit/orchestration/test_multi_llm_orchestrator.py
@pytest.mark.asyncio
async def test_orchestrator_selects_optimal_provider():
    """Verify orchestrator selects provider based on strategy."""
    metrics = [
        ProviderMetrics(AIProvider.COPILOT_CHAT, cost_per_token=0.005, ...),
        ProviderMetrics(AIProvider.CLAUDE, cost_per_token=0.003, ...),
    ]
    
    optimizer = CostOptimizer(metrics)
    orchestrator = MultiLLMOrchestrator(
        optimizer,
        strategy=OptimizationStrategy.CHEAPEST,
    )
    
    task = AITask(
        id="task1",
        input_data=AITaskInput(
            prompt="Test",
            provider=AIProvider.COPILOT_CHAT,  # Will be overridden
        ),
        user_id="user1",
    )
    
    # Mock engines
    claude_engine = AsyncMock(spec=IReasoningEngine)
    claude_engine.reason = AsyncMock(return_value=AIResult(...))
    
    ReasoningEngineRegistry.register(AIProvider.CLAUDE, claude_engine)
    
    result = await orchestrator.execute_task(task)
    
    # Should have used Claude (cheapest)
    assert claude_engine.reason.called
```

**Files to Create:**

- `arc_saga/orchestrator/multi_llm_orchestrator.py` — NEW (150 lines)
- `tests/unit/orchestration/test_multi_llm_orchestrator.py` — NEW (200 lines)

---

## 📊 PHASE 2.4 FILE SUMMARY

### New Files (8 total)

| File | Lines | Purpose |
|------|-------|---------|
| `arc_saga/orchestrator/engine_registry.py` | 120 | Engine registration |
| `arc_saga/orchestrator/provider_router.py` | 200 | Task routing |
| `arc_saga/orchestrator/cost_optimizer.py` | 250 | Cost optimization |
| `arc_saga/orchestrator/multi_llm_orchestrator.py` | 150 | Unified orchestrator |
| `tests/unit/orchestration/test_response_mode.py` | 150 | ResponseMode tests |
| `tests/unit/orchestration/test_engine_registry.py` | 200 | Registry tests |
| `tests/unit/orchestration/test_provider_router.py` | 250 | Router tests |
| `tests/unit/orchestration/test_multi_llm_orchestrator.py` | 200 | Orchestrator tests |
| **SUBTOTAL** | **1,520** | **Phase 2.4 code** |

### Modified Files (2 total)

| File | Change |
|------|--------|
| `arc_saga/orchestrator/types.py` | Add ResponseMode enum, update AITask/AIResult |
| `arc_saga/integrations/copilot_reasoning_engine.py` | Add reason_complete(), update reason() |

---

## 🧪 TEST STRATEGY FOR PHASE 2.4

### Coverage Goals
- **Target:** 98%+ coverage (improve from Phase 2.3's 89%)
- **Focus:** Orchestration layer (new components)
- **Existing:** Maintain 100% pass rate on Phase 2.3 tests

### Test Architecture

```python
# tests/unit/orchestration/__init__.py (new directory)

# Fixtures for all tests
@pytest.fixture
def registry():
    ReasoningEngineRegistry.clear()
    yield ReasoningEngineRegistry
    ReasoningEngineRegistry.clear()

@pytest.fixture
def mock_engines():
    return {
        AIProvider.COPILOT_CHAT: AsyncMock(spec=IReasoningEngine),
        AIProvider.CLAUDE: AsyncMock(spec=IReasoningEngine),
        AIProvider.GPT4: AsyncMock(spec=IReasoningEngine),
    }

@pytest.fixture
def optimizer():
    metrics = [
        ProviderMetrics(AIProvider.COPILOT_CHAT, cost_per_token=0.005, ...),
        ProviderMetrics(AIProvider.CLAUDE, cost_per_token=0.003, ...),
        ProviderMetrics(AIProvider.GPT4, cost_per_token=0.010, ...),
    ]
    return CostOptimizer(metrics)
```

### Test Categories

1. **Unit Tests** (exact numbers per component)
   - ResponseMode: 10+ tests
   - Registry: 15+ tests
   - Router: 20+ tests
   - Optimizer: 15+ tests
   - Orchestrator: 15+ tests
   - **Total: 75+ new tests**

2. **Integration Tests** (all components together)
   - End-to-end routing with fallback
   - Cost-based provider selection
   - Streaming vs complete response handling
   - Error scenarios (all providers fail, etc.)
   - **Total: 20+ integration tests**

3. **Performance Tests**
   - Provider selection time (<1ms)
   - Registry lookup time (<0.5ms)
   - Fallback chain execution time (<100ms per provider)

---

## 🔄 DEPENDENCY CHAIN

```
Phase 2.3 (Foundation)
    ↓
ResponseMode (streaming capability)
    ↓
ReasoningEngineRegistry (provider management)
    ↓
ProviderRouter (routing logic)
    ↓
CostOptimizer (intelligent selection)
    ↓
MultiLLMOrchestrator (unified interface)
```

**Implementation Order:** 1 → 2 → 3 → 4 → 5 (sequential, each builds on previous)

---

## 📋 EXECUTION CHECKLIST

### Pre-Implementation
- [x] Phase 2.3 verified and production-ready
- [x] All Phase 2.3 tests passing (61/61)
- [ ] Review Phase 2.4 roadmap (this document)
- [ ] Verify Phase 2.3 verification script runs cleanly

### Implementation Phase
- [ ] Component 1: ResponseMode (stream vs complete)
- [ ] Component 2: ReasoningEngineRegistry
- [ ] Component 3: ProviderRouter
- [ ] Component 4: CostOptimizer
- [ ] Component 5: MultiLLMOrchestrator
- [ ] Integration tests (orchestration layer)
- [ ] Run full test suite (Phase 2.3 + Phase 2.4)

### Quality Gates (Phase 2.4)
- [ ] mypy --strict (0 errors)
- [ ] pylint (8.0+ score)
- [ ] bandit (0 issues)
- [ ] pytest (100% pass rate)
- [ ] coverage (98%+ target)
- [ ] black/isort (formatted)

### Final Verification
- [ ] All 95+ new tests passing
- [ ] All Phase 2.3 tests still passing (61/61)
- [ ] Total coverage maintained at 98%+
- [ ] Type safety verified
- [ ] Security audit passed
- [ ] Documentation complete

---

## 🚀 GO-LIVE CRITERIA

Phase 2.4 is production-ready when:

✅ All 95+ new tests passing (100%)  
✅ Coverage maintained at 98%+  
✅ All quality gates passing (6/6)  
✅ Orchestrator handles 1000+ concurrent tasks  
✅ Provider failover works under load  
✅ Cost optimization delivers 10-30% savings  
✅ Documentation complete  

---

## 📚 DOCUMENTATION

For Phase 2.4, create:

1. **Architecture Guide** — How orchestrator works
2. **Provider Integration Guide** — How to add new providers
3. **API Reference** — Public interfaces
4. **Deployment Guide** — Production setup
5. **Cost Optimization Guide** — How to tune for cost/quality

---

## 🎯 PHASE 2.4 SUCCESS METRICS

| Metric | Target | Verification |
|--------|--------|--------------|
| **New Test Cases** | 95+ | pytest output |
| **Coverage** | 98%+ | --cov report |
| **Type Safety** | 0 errors | mypy output |
| **Code Quality** | 8.0+ | pylint score |
| **Security** | 0 issues | bandit output |
| **Pass Rate** | 100% | pytest final result |
| **Performance** | <1ms selection | timing tests |
| **Scalability** | 1000+ tasks | load test |

---

## 🎬 NEXT STEPS

1. ✅ Verify Phase 2.3 one final time: `.\scripts\verify_phase_2_3.ps1`
2. 🔜 Create branch: `git checkout -b feature/phase-2.4-orchestration`
3. 🔜 Implement components 1-5 in order
4. 🔜 Write tests as you go (TDD)
5. 🔜 Run quality gates between each component
6. 🔜 Final integration test and verification
7. 🔜 Merge to main and tag `v2.4.0`

---

**You've built a rock-solid foundation. Phase 2.4 transforms it into a true orchestrator.** 🚀

**From single-provider integration → multi-LLM orchestration**

**Ready to proceed with confidence.** ✅
