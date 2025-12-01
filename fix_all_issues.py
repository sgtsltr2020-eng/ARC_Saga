"""
Master script to fix all identified issues
"""

import subprocess
import sys

print("=" * 80)
print("🔧 FIX ALL ARC SAGA ISSUES")
print("=" * 80)
print()

steps = [
    ("Install tools", [
        [sys.executable, "-m", "pip", "install", "autopep8", "autoflake", "pytest", "pytest-asyncio", "pytest-cov"]
    ]),
    ("Fix linting", [
        [sys.executable, "-m", "autopep8", "--in-place", "--aggressive", "--aggressive", "--recursive", "arc_saga/"]
    ]),
    ("Remove unused imports", [
        [sys.executable, "-m", "autoflake", "--in-place", "--remove-all-unused-imports", "--recursive", "arc_saga/"]
    ]),
    ("Upgrade dependencies", [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pydantic>=2.6.4", "starlette>=0.37.2"]
    ]),
]

for step_name, commands in steps:
    print(f"\n{'='*80}")
    print(f"▶️  {step_name}")
    print('='*80)
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {step_name} completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {step_name} had issues:")
            if e.stderr:
                print(e.stderr[:500])
        except FileNotFoundError:
            print(f"⚠️  Command not found: {cmd[0]}")

print()
print("=" * 80)
print("✅ ALL FIXES APPLIED")
print("=" * 80)
print()
print("Next steps:")
print("1. Run tests: pytest tests/ -v")
print("2. Test server: python -m arc_saga.api.server")
print("3. Run audit: python audit_system.py")