Testing SAGA
============

Unit: `make test-fast`
Integration: `pytest tests/integration/ -m integration`
Benchmarks: `pytest benchmarks/ -m slow`
Full: `make test` (99% cov enforced)

Always: `python -m pytest` (fixes imports).
