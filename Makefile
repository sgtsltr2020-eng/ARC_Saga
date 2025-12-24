.PHONY: help install test test-fast test-cov lint format docs clean

help:
	@echo "SAGA Development Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make test-fast    - Run tests without coverage"
	@echo "  make test-cov     - Generate HTML coverage report"
	@echo "  make lint         - Run mypy strict type checking"
	@echo "  make format       - Format code with black"
	@echo "  make docs         - Build Sphinx documentation"
	@echo "  make clean        - Remove build artifacts"

install:
	pip install -e ".[dev,docs]"

test:
	pytest tests/ --cov=saga --cov-report=term-missing --cov-fail-under=99 -v

test-fast:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=saga --cov-report=html:htmlcov
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	mypy --strict saga/
	ruff check saga/

format:
	black saga/ tests/
	ruff check --fix saga/ tests/

docs:
	cd docs && make html
	@echo "Documentation built in docs/_build/html/index.html"

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	cd docs && make clean
