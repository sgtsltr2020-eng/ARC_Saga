"""
Complete ARC Saga System Audit
Identifies ALL issues, warnings, and incomplete implementations
"""

import sys
import os
from pathlib import Path
import sqlite3
import logging
import subprocess
import platform

# Configure logging
logging.basicConfig(
    filename="audit_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("=" * 80)
print("🔍 ARC SAGA COMPLETE SYSTEM AUDIT")
print("=" * 80)
print()

issues = []
warnings = []
successes = []

# ============================================================================
# 1. FILE STRUCTURE AUDIT
# ============================================================================
print("1. FILE STRUCTURE AUDIT")
print("-" * 80)

expected_files = {
    "Core": [
        "arc_saga/__init__.py",
        "arc_saga/models.py",
        "arc_saga/exceptions.py",
        "arc_saga/logging_config.py",
    ],
    "Storage": [
        "arc_saga/storage/__init__.py",
        "arc_saga/storage/sqlite.py",
    ],
    "API": [
        "arc_saga/api/__init__.py",
        "arc_saga/api/server.py",
    ],
    "Services": [
        "arc_saga/services/__init__.py",
        "arc_saga/services/auto_tagger.py",
        "arc_saga/services/file_processor.py",
    ],
    "Integrations": [
        "arc_saga/integrations/__init__.py",
        "arc_saga/integrations/perplexity_client.py",
    ],
    "Shared": [
        "shared/config.py",
        "shared/utils.py",
    ],
    "Tests": [
        "tests/__init__.py",
    ],
    "Root": [
        "test_server.py",
        "diagnose.py",
    ]
}

for category, files in expected_files.items():
    print(f"\n{category}:")
    for file_path in files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
            successes.append(f"File exists: {file_path}")
        else:
            print(f"  ❌ MISSING: {file_path}")
            issues.append(f"Missing file: {file_path}")

print()

# ============================================================================
# 2. IMPORT VERIFICATION
# ============================================================================
print("2. IMPORT VERIFICATION")
print("-" * 80)

imports_to_check = [
    ("arc_saga.models", ["Message", "MessageRole", "Provider", "File"]),
    ("arc_saga.storage.sqlite", ["SQLiteStorage"]),
    ("arc_saga.exceptions", ["StorageError", "ValidationError"]),
    ("arc_saga.api.server", ["app"]),
    ("arc_saga.services.auto_tagger", ["AutoTagger"]),
    ("arc_saga.services.file_processor", ["FileProcessor"]),
]

for module_name, expected_exports in imports_to_check:
    try:
        module = __import__(module_name, fromlist=expected_exports)
        print(f"\n✅ {module_name}")
        for export in expected_exports:
            if hasattr(module, export):
                print(f"   ✅ {export}")
                successes.append(f"Import OK: {module_name}.{export}")
            else:
                print(f"   ❌ Missing: {export}")
                issues.append(f"Missing export: {module_name}.{export}")
    except Exception as e:
        print(f"\n❌ {module_name}: {e}")
        issues.append(f"Import failed: {module_name} - {str(e)}")

print()

# ============================================================================
# 3. DATABASE AUDIT
# ============================================================================
print("3. DATABASE AUDIT")
print("-" * 80)

db_path = Path.home() / ".arc_saga" / "memory.db"
print(f"Database: {db_path}")

if db_path.exists():
    print(f"✅ Exists ({db_path.stat().st_size} bytes)")
    successes.append(f"Database exists: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        
        print(f"\nTables found: {len(tables)}")
        
        expected_tables = ["messages", "files", "messages_fts", "files_fts"]
        for table in expected_tables:
            if table in tables:
                print(f"  ✅ {table}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"     Rows: {count}")
                
                # Get schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print(f"     Columns: {', '.join([c[1] for c in columns])}")
                
                successes.append(f"Table OK: {table} ({count} rows)")
            else:
                print(f"  ❌ Missing table: {table}")
                issues.append(f"Missing table: {table}")
        
        # Check for unexpected tables
        unexpected = [t for t in tables if t not in expected_tables]
        if unexpected:
            print(f"\n⚠️  Unexpected tables: {unexpected}")
            warnings.append(f"Unexpected tables: {unexpected}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database inspection failed: {e}")
        issues.append(f"Database inspection error: {str(e)}")
else:
    print("❌ Database does not exist")
    issues.append("Database missing")

print()

# ============================================================================
# 4. DEPENDENCY CHECK
# ============================================================================
print("4. DEPENDENCY CHECK")
print("-" * 80)

required_packages = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "sqlite3",  # Built-in
    "scikit-learn",
    "fitz",  # PyMuPDF
    "docx",  # python-docx
    "openai",
    "watchdog",
]

for package in required_packages:
    try:
        if package == "sqlite3":
            import sqlite3
            print(f"✅ {package} (built-in)")
        elif package == "fitz":
            import fitz
            print(f"✅ {package} (PyMuPDF)")
        elif package == "docx":
            import docx
            print(f"✅ {package} (python-docx)")
        else:
            __import__(package)
            print(f"✅ {package}")
        successes.append(f"Dependency OK: {package}")
    except ImportError:
        print(f"❌ Missing: {package}")
        issues.append(f"Missing dependency: {package}")

print()

# ============================================================================
# 5. SERVER FUNCTIONALITY CHECK
# ============================================================================
print("5. SERVER FUNCTIONALITY CHECK")
print("-" * 80)

try:
    import requests
    base_url = "http://localhost:8421"
    
    # Try to connect
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Server is running and responding")
            print(f"   Status: {response.json()}")
            successes.append("Server health check passed")
        else:
            print(f"⚠️  Server responding but with status {response.status_code}")
            warnings.append(f"Server returned {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running (this is OK if you haven't started it)")
        warnings.append("Server not running")
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        issues.append(f"Server error: {str(e)}")
        
except ImportError:
    print("⚠️  requests package not available for server check")
    warnings.append("Cannot check server status")

print()

# ============================================================================
# 6. CODE QUALITY CHECKS
# ============================================================================
print("6. CODE QUALITY CHECKS")
print("-" * 80)

# Check for deprecated warnings
deprecated_patterns = [
    ("arc_saga/api/server.py", "@app.on_event", "Use lifespan handlers instead"),
]

for file_path, pattern, suggestion in deprecated_patterns:
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if pattern in content:
                print(f"⚠️  {file_path}")
                print(f"   Found deprecated: {pattern}")
                print(f"   Suggestion: {suggestion}")
                warnings.append(f"Deprecated code in {file_path}: {pattern}")
            else:
                print(f"✅ {file_path} - No deprecated patterns")
                successes.append(f"Code quality OK: {file_path}")

print()

# ============================================================================
# 7. CONFIGURATION CHECK
# ============================================================================
print("7. CONFIGURATION CHECK")
print("-" * 80)

env_vars_needed = [
    ("PPLX_API_KEY", "Optional - Perplexity API integration"),
]

for var_name, description in env_vars_needed:
    if os.getenv(var_name):
        print(f"✅ {var_name} is set")
        successes.append(f"Env var set: {var_name}")
    else:
        print(f"⚠️  {var_name} not set - {description}")
        warnings.append(f"Env var missing: {var_name}")

print()

# ============================================================================
# 8. TEST COVERAGE
# ============================================================================
print("8. TEST COVERAGE")
print("-" * 80)

test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
print(f"Test files found: {len(test_files)}")

if test_files:
    for test_file in test_files:
        print(f"  ✅ {test_file}")
        successes.append(f"Test file exists: {test_file}")
else:
    print("  ⚠️  No test files found in tests/ directory")
    warnings.append("No unit tests found")

# Check for test_server.py
if Path("test_server.py").exists():
    print("  ✅ test_server.py (integration test)")
    successes.append("Integration test exists")
else:
    print("  ❌ test_server.py missing")
    issues.append("Integration test missing")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("📊 AUDIT SUMMARY")
print("=" * 80)
print()

print(f"✅ Successes: {len(successes)}")
print(f"⚠️  Warnings: {len(warnings)}")
print(f"❌ Issues: {len(issues)}")
print()

if issues:
    print("🔴 CRITICAL ISSUES TO FIX:")
    print("-" * 80)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print()

if warnings:
    print("🟡 WARNINGS (Should Address):")
    print("-" * 80)
    for i, warning in enumerate(warnings, 1):
        print(f"{i}. {warning}")
    print()

# Overall health score
total = len(successes) + len(warnings) + len(issues)
health_score = (len(successes) / total * 100) if total > 0 else 0

print(f"🏥 SYSTEM HEALTH: {health_score:.1f}%")
print()

if health_score >= 90:
    print("✅ System is in excellent condition")
elif health_score >= 75:
    print("⚠️  System is functional but needs attention")
elif health_score >= 50:
    print("🟡 System has significant issues")
else:
    print("🔴 System requires immediate fixes")

print()
print("=" * 80)

# Log audit start
logging.info("Starting ARC Saga System Audit")

# Check Python version
print("Checking Python version...")
python_version = platform.python_version()
if tuple(map(int, python_version.split('.'))) < (3, 10):
    print(f"❌ Python version must be 3.10 or higher (current: {python_version})")
    logging.error(f"Python version too low: {python_version}")
else:
    print(f"✅ Python version: {python_version}")
    logging.info(f"Python version: {python_version}")

# Test coverage analysis
def check_test_coverage():
    print("Running test coverage analysis...")
    try:
        result = subprocess.run(
            ["coverage", "run", "-m", "unittest", "discover"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Tests executed successfully")
            logging.info("Tests executed successfully")
            coverage_result = subprocess.run(
                ["coverage", "report"],
                capture_output=True,
                text=True
            )
            print(coverage_result.stdout)
            logging.info("Test coverage report:\n" + coverage_result.stdout)
        else:
            print("❌ Test execution failed")
            logging.error("Test execution failed:\n" + result.stderr)
    except FileNotFoundError:
        print("⚠️  coverage.py not installed")
        logging.warning("coverage.py not installed")

check_test_coverage()

# Static code analysis
def run_static_analysis():
    print("Running static code analysis...")
    try:
        result = subprocess.run(
            ["flake8", "arc_saga/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ No linting issues found")
            logging.info("No linting issues found")
        else:
            print("⚠️  Linting issues found:")
            print(result.stdout)
            logging.warning("Linting issues:\n" + result.stdout)
    except FileNotFoundError:
        print("⚠️  flake8 not installed")
        logging.warning("flake8 not installed")

run_static_analysis()

# Check for unused dependencies
def check_unused_dependencies():
    print("Checking for unused dependencies...")
    try:
        result = subprocess.run(
            ["pip", "check"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ No dependency issues found")
            logging.info("No dependency issues found")
        else:
            print("⚠️  Dependency issues found:")
            print(result.stdout)
            logging.warning("Dependency issues:\n" + result.stdout)
    except FileNotFoundError:
        print("⚠️  pip not available")
        logging.warning("pip not available")

check_unused_dependencies()

# Check for deprecated code patterns
def check_deprecated_code():
    print("Checking for deprecated code patterns...")
    deprecated_patterns = [
        ("arc_saga/api/server.py", "@app.on_event", "Use lifespan handlers instead"),
    ]
    for file_path, pattern, suggestion in deprecated_patterns:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if pattern in content:
                    print(f"⚠️  {file_path}")
                    print(f"   Found deprecated: {pattern}")
                    print(f"   Suggestion: {suggestion}")
                    logging.warning(f"Deprecated code in {file_path}: {pattern}")
                else:
                    print(f"✅ {file_path} - No deprecated patterns")
                    logging.info(f"Code quality OK: {file_path}")

check_deprecated_code()

# Generate detailed report
def generate_report():
    print("Generating detailed audit report...")
    try:
        with open("audit_log.txt", "r") as log_file:
            print("\n--- Audit Report ---\n")
            print(log_file.read())
    except FileNotFoundError:
        print("❌ Audit log file not found")

generate_report()