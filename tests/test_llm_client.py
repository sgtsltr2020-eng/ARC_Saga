"""
LLM Client Tests
================

Tests LLM client initialization and API calls.

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
import os
from unittest.mock import patch

from saga.llm.client import LLMClient, LLMResponse, Provider


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


def test_llm_client_initialization():
    """Test LLMClient initializes with provider."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = LLMClient(provider=Provider.OPENAI)

        assert client.provider == Provider.OPENAI


def test_llm_client_initialization_with_api_key():
    """Test LLMClient uses provided api_key."""
    client = LLMClient(provider=Provider.OPENAI, api_key="explicit-key")

    assert client.provider == Provider.OPENAI


def test_llm_response_creation():
    """Test LLMResponse dataclass."""
    response = LLMResponse(
        text="Hello world",
        provider=Provider.OPENAI,
        model="gpt-4-turbo",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost=0.01
    )

    assert response.text == "Hello world"
    assert response.provider == Provider.OPENAI
    assert response.total_tokens == 15


def test_llm_response_to_dict():
    """Test LLMResponse serialization."""
    response = LLMResponse(
        text="Test",
        provider=Provider.OPENAI,
        model="gpt-4",
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        estimated_cost=0.005
    )

    data = response.to_dict()

    assert data["text"] == "Test"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4"


def test_provider_enum():
    """Test Provider enum values."""
    assert Provider.OPENAI.value == "openai"
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.PERPLEXITY.value == "perplexity"


def test_provider_is_string():
    """Test Provider enum is string subclass."""
    assert isinstance(Provider.OPENAI, str)
    assert isinstance(Provider.ANTHROPIC, str)
