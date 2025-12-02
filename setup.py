"""
ARC Saga Memory Layer Setup
"""

from setuptools import setup, find_packages

setup(
    name="arc_saga",
    version="1.0.0",
    description="Unified memory layer for AI conversations",
    author="Your Name",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.6.4",
        "scikit-learn>=1.3.0",
        "PyMuPDF>=1.23.0",
        "python-docx>=1.1.0",
        "openai>=1.0.0",
        "watchdog>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "autopep8>=2.0.0",
            "autoflake>=2.2.0",
            "flake8>=6.1.0",
        ]
    },
    python_requires=">=3.10",
)