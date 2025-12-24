"""
Mimiry Arbitration Scaling (Parallel)
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from saga.core.mimiry import Mimiry, OracleResponse


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_parallel_arbitration_scaling(benchmark):
    """Benchmark resolve_conflict on N agents."""
    mimiry = Mimiry()
    # Mock CPU-bound measurement to avoid external calls
    mimiry.measure_against_ideal = AsyncMock(return_value=OracleResponse(
        question="test", canonical_answer="Ok", severity="ACCEPTABLE", cited_rules=[]
    ))

    async def measure_n_agents(n):
        outputs = [{"agent": f"A{i}", "code": f"mock code {i}"} for i in range(n)]
        start = asyncio.get_event_loop().time()
        await mimiry.resolve_conflict(outputs, {"task": "mock"})
        return asyncio.get_event_loop().time() - start

    for n in [2, 10, 50]:
        time_taken = await measure_n_agents(n) # Benchmark wrapper complexity with async, calling directly for now
        assert time_taken < 2.0  # <2s target
