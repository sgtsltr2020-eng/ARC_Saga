"""
Automatically fix all linting issues
"""

import subprocess
import sys

print("🔧 Fixing all linting issues with autopep8...")
print()

# Install autopep8 if needed
try:
    import autopep8
except ImportError:
    print("Installing autopep8...")
    subprocess.run([sys.executable, "-m", "pip", "install", "autopep8"], check=True)

# Fix all Python files recursively
files_to_fix = [
    "arc_saga/",
    "tests/",
    "shared/",
]

for path in files_to_fix:
    print(f"Fixing {path}...")
    subprocess.run([
        sys.executable, "-m", "autopep8",
        "--in-place",
        "--aggressive",
        "--aggressive",
        "--recursive",
        path
    ])

print()
print("✅ All linting issues fixed!")
print("Run 'python -m flake8 arc_saga/' to verify")