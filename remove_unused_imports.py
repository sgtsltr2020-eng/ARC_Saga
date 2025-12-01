"""
Remove unused imports automatically
"""

import subprocess
import sys

print("🔧 Removing unused imports with autoflake...")
print()

try:
    import autoflake
except ImportError:
    print("Installing autoflake...")
    subprocess.run([sys.executable, "-m", "pip", "install", "autoflake"], check=True)

# Remove unused imports
paths = ["arc_saga/", "tests/", "shared/"]

for path in paths:
    print(f"Processing {path}...")
    subprocess.run([
        sys.executable, "-m", "autoflake",
        "--in-place",
        "--remove-unused-variables",
        "--remove-all-unused-imports",
        "--recursive",
        path
    ])

print()
print("✅ Unused imports removed!")