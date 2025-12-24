"""
LLM Client - Multi-Provider Abstraction
========================================

Supports OpenAI, Anthropic, Perplexity with unified interface.
Implements Constitution Rule 2 (multi-provider verification).

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3B - Agent Execution Framework
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from saga.resilience.async_utils import with_retry, with_timeout

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"


@dataclass
class LLMResponse:
    """Response from LLM API."""

    text: str
    provider: Provider
    model: str

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost (USD)
    estimated_cost: float = 0.0

    # Metadata
    latency_ms: float = 0.0
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "text": self.text,
            "provider": self.provider.value,
            "model": self.model,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens
            },
            "cost_usd": self.estimated_cost,
            "latency_ms": self.latency_ms
        }


@dataclass
class VerifiedResponse:
    """Response verified by multiple providers (Rule 2)."""

    canonical_response: LLMResponse
    verification_responses: list[LLMResponse] = field(default_factory=list)
    agreement: bool = True
    disagreement_details: Optional[str] = None
    total_cost: float = 0.0


class LLMClient:
    """
    Multi-provider LLM client with verification support.

    Supports:
        - OpenAI (GPT-4o, GPT-4o-mini)
        - Anthropic (Claude 3.5 Sonnet)
        - Perplexity (Sonar models)

    Implements Constitution Rule 2:
        - Critical tasks use multi-provider verification
        - Detects disagreements, escalates to user

    Usage:
        client = LLMClient(provider="openai")
        await client.initialize()

        # Simple chat
        response = await client.chat([{"role": "user", "content": "Hello"}])

        # Verified chat (critical task)
        verified = await client.chat_with_verification(
            messages=[...],
            task_type="security_audit"
        )
    """

    def __init__(
        self,
        provider: Provider = Provider.OPENAI,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client.

        Args:
            provider: Default provider to use
            api_key: API key (auto-loads from env if not provided)
        """
        self.provider = provider
        self.api_key = api_key or self._load_api_key(provider)

        # HTTP clients (initialized in initialize())
        self.openai_client: Optional[Any] = None
        self.anthropic_client: Optional[Any] = None
        self.perplexity_client: Optional[Any] = None

        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0

        # Pricing (USD per 1k tokens) - December 2025
        self.pricing = {
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
            "sonar": {"prompt": 0.001, "completion": 0.001},
        }

    def _load_api_key(self, provider: Provider) -> str:
        """Load API key from environment."""
        key_map = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.PERPLEXITY: "PERPLEXITY_API_KEY",
        }

        env_var = key_map[provider]
        key = os.getenv(env_var)

        if not key:
            logger.warning(f"{env_var} not set, LLM calls will fail")

        return key or ""

    async def initialize(self) -> None:
        """Initialize HTTP clients for providers."""
        try:
            if self.provider == Provider.OPENAI or self._has_key(Provider.OPENAI):
                try:
                    from openai import AsyncOpenAI
                    if self._has_key(Provider.OPENAI):
                        self.openai_client = AsyncOpenAI(api_key=self._load_api_key(Provider.OPENAI))
                except ImportError:
                    pass

            if self.provider == Provider.ANTHROPIC or self._has_key(Provider.ANTHROPIC):
                try:
                    from anthropic import AsyncAnthropic
                    if self._has_key(Provider.ANTHROPIC):
                        self.anthropic_client = AsyncAnthropic(api_key=self._load_api_key(Provider.ANTHROPIC))
                except ImportError:
                    pass

            # Perplexity uses OpenAI client interface
            if self.provider == Provider.PERPLEXITY or self._has_key(Provider.PERPLEXITY):
                try:
                    from openai import AsyncOpenAI
                    if self._has_key(Provider.PERPLEXITY):
                         self.perplexity_client = AsyncOpenAI(
                             api_key=self._load_api_key(Provider.PERPLEXITY),
                             base_url="https://api.perplexity.ai"
                         )
                except ImportError:
                     pass

            logger.info(f"LLM client initialized (provider: {self.provider.value})")

        except ImportError as e:
            logger.error(f"Failed to import LLM library: {e}")
            raise RuntimeError("Install required: pip install openai anthropic")

    def _has_key(self, provider: Provider) -> bool:
        """Check if API key exists for provider."""
        return bool(self._load_api_key(provider))

    @with_timeout(60)
    @with_retry(max_attempts=3, backoff=2.0)
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        provider: Optional[Provider] = None
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            model: Model name (auto-selects if None)
            temperature: Sampling temperature (0-1)
            max_tokens: Max completion tokens
            provider: Override default provider

        Returns:
            LLMResponse with text, tokens, cost
        """
        provider = provider or self.provider

        if not model:
            model = self._get_default_model(provider)

        start_time = asyncio.get_event_loop().time()

        # Route to provider
        if provider == Provider.OPENAI:
            response = await self._chat_openai(messages, model, temperature, max_tokens)
        elif provider == Provider.ANTHROPIC:
            response = await self._chat_anthropic(messages, model, temperature, max_tokens)
        elif provider == Provider.PERPLEXITY:
            response = await self._chat_perplexity(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Calculate latency
        response.latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Track cost
        self.total_cost += response.estimated_cost
        self.request_count += 1

        logger.info(
            "LLM chat completed",
            extra={
                "provider": provider.value,
                "model": model,
                "tokens": response.total_tokens,
                "cost_usd": response.estimated_cost,
                "latency_ms": response.latency_ms
            }
        )

        return response

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call OpenAI API."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized or key missing")

        completion = await self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        usage = completion.usage
        # Handle cases where usage is None or missing
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        cost = self._calculate_cost(
            model,
            prompt_tokens,
            completion_tokens
        )

        return LLMResponse(
            text=completion.choices[0].message.content or "",
            provider=Provider.OPENAI,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            finish_reason=completion.choices[0].finish_reason
        )

    async def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized or key missing")

        # Convert messages format
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        message = await self.anthropic_client.messages.create(
            model=model,
            system=system_message if system_message else [],
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        cost = self._calculate_cost(
            model,
            message.usage.input_tokens,
            message.usage.output_tokens
        )

        return LLMResponse(
            text=message.content[0].text,
            provider=Provider.ANTHROPIC,
            model=model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
            estimated_cost=cost,
            finish_reason=message.stop_reason or "stop"
        )

    async def _chat_perplexity(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Perplexity API (via OpenAI interface)."""
        if not self.perplexity_client:
            raise RuntimeError("Perplexity client not initialized or key missing")

        completion = await self.perplexity_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        usage = completion.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        cost = self._calculate_cost(
            model,
            prompt_tokens,
            completion_tokens
        )

        return LLMResponse(
            text=completion.choices[0].message.content or "",
            provider=Provider.PERPLEXITY,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            finish_reason=completion.choices[0].finish_reason
        )

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate estimated cost in USD."""
        # Find pricing (use closest match)
        # Sort keys by length desc to ensure "gpt-4o-mini" matches before "gpt-4o"
        pricing = None
        sorted_keys = sorted(self.pricing.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if key in model:
                pricing = self.pricing[key]
                break

        if not pricing:
            logger.warning(f"No pricing for model {model}, using default")
            pricing = {"prompt": 0.001, "completion": 0.002}

        cost = (
            (prompt_tokens / 1000) * pricing["prompt"] +
            (completion_tokens / 1000) * pricing["completion"]
        )

        return round(cost, 6)

    def _get_default_model(self, provider: Provider) -> str:
        """Get default model for provider."""
        defaults = {
            Provider.OPENAI: "gpt-4o-mini",
            Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            Provider.PERPLEXITY: "sonar",
        }
        return defaults[provider]

    async def chat_with_verification(
        self,
        messages: list[dict[str, str]],
        task_type: str,
        providers: Optional[list[Provider]] = None
    ) -> VerifiedResponse:
        """
        Multi-provider verification for critical tasks (Rule 2).

        Args:
            messages: Chat messages
            task_type: Type of task (e.g., "security_audit")
            providers: Providers to use (defaults to available)

        Returns:
            VerifiedResponse with canonical answer + verification
        """
        # Determine providers
        if not providers:
            providers = [self.provider]
            # Add second provider if available
            if self.provider == Provider.OPENAI and self._has_key(Provider.ANTHROPIC):
                providers.append(Provider.ANTHROPIC)
            elif self.provider == Provider.ANTHROPIC and self._has_key(Provider.OPENAI):
                providers.append(Provider.OPENAI)
            elif self.provider != Provider.OPENAI and self._has_key(Provider.OPENAI):
                providers.append(Provider.OPENAI)

        # Ensure we have unique providers
        providers = list(set(providers))

        if len(providers) < 2:
            logger.warning("Verification requested but only 1 provider available. Verification will be trivial.")

        # Call all providers in parallel
        tasks = [
            self.chat(messages, provider=p)
            for p in providers
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter failures
        successful_responses = []
        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                logger.error(f"Provider {providers[i]} failed: {r}")
            elif isinstance(r, LLMResponse):
                successful_responses.append(r)

        if not successful_responses:
            raise RuntimeError("All LLM providers failed")

        # Check agreement
        canonical = successful_responses[0]
        agreement = self._check_agreement([r.text for r in successful_responses])

        total_cost = sum(r.estimated_cost for r in successful_responses)

        if not agreement and len(successful_responses) > 1:
            logger.warning(
                "Multi-provider disagreement detected",
                extra={
                    "task_type": task_type,
                    "providers": [r.provider.value for r in successful_responses],
                    "responses": [r.text[:100] for r in successful_responses]
                }
            )

        return VerifiedResponse(
            canonical_response=canonical,
            verification_responses=successful_responses[1:],
            agreement=agreement,
            disagreement_details=None if agreement else "Providers gave different answers",
            total_cost=total_cost
        )

    def _check_agreement(self, texts: list[str]) -> bool:
        """Check if responses agree (simple heuristic)."""
        if len(texts) < 2:
            return True

        # Simple check: if responses are >80% similar in length and share keywords
        first = texts[0].lower()

        for text in texts[1:]:
            text_lower = text.lower()

            # Length check
            if abs(len(first) - len(text_lower)) / max(len(first), 1) > 0.5:
                return False

            # Keyword overlap check (basic)
            words_first = set(first.split())
            words_text = set(text_lower.split())

            if not words_first or not words_text:
                return True # Empty responses technically agree in emptiness? Or assume agreement if no content.

            overlap = len(words_first & words_text) / len(words_first | words_text)

            if overlap < 0.6:
                return False

        return True

    def get_cost_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "request_count": self.request_count,
            "avg_cost_per_request": round(self.total_cost / max(self.request_count, 1), 4)
        }
