"""
SAGA - Systematized Autonomous Generative Assistant
Core orchestration, memory, and quality enforcement
"""

from setuptools import find_packages, setup

setup(
    name="saga",
    version="2.0.0a0",
    description="Systematized Autonomous Generative Assistant - AI Code Quality & Memory Manager",
    author="ARC SAGA Development Team",
    packages=find_packages(exclude=["tests*", "docs*", "archive*"]),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.0.0",
        "sqlalchemy[asyncio]>=2.0.0",
        "asyncpg>=0.28.0",
        "httpx>=0.25.0",
        "networkx>=3.0.0",
        "spacy>=3.0.0",
        "openai>=1.0.0",
        "scikit-learn>=1.3.0",
        "PyMuPDF>=1.23.0",
        "python-docx>=1.1.0",
        "watchdog>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.9.0",
            "ruff>=0.0.290",
            "pylint>=2.17.0",
            "bandit>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
)
