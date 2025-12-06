# CRITICAL NEXT ACTIONS - Phase 2.3 Finalization

**Priority Level:** 🔴 IMMEDIATE  
**Deadline:** Complete quality gates within 1 hour  
**Owner:** You (Dr. Alex Chen's framework)

---

## ✅ VERIFICATION CHECKLIST

### Pre-Gate Inspection (5 mins)
- [ ] All 9 code files exist in correct directories
- [ ] No syntax errors in Python files
- [ ] All test files in `tests/unit/integrations/` directory
- [ ] Package `__init__.py` files updated

### Gate 1: Type Safety (CRITICAL)
```bash
mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py
```
**Expected:** 0 errors  
**If fails:** Run with `--show-error-codes` and fix one by one

**Common issues to fix:**
- Missing return type annotations
- Bare `Any` types (add comment justifying)
- Async function signatures
- Optional type handling

### Gate 2: Linting (CRITICAL)
```bash
pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ --exit-zero
```
**Expected:** 8.0+ score  
**If fails:** Fix in this order:
1. Missing docstrings (add to all classes/functions)
2. Import ordering (use `isort`)
3. Line length (break long lines)
4. Naming conventions (snake_case for functions)

### Gate 3: Security (CRITICAL)
```bash
bandit -r arc_saga/integrations/ arc_saga/exceptions/
```
**Expected:** 0 issues  
**If issues found:** Fix immediately—don't skip
- [ ] No hardcoded credentials
- [ ] No secrets in logs
- [ ] No temp file vulnerabilities
- [ ] Proper encryption key handling

### Gate 4: Tests (CRITICAL)
```bash
pytest tests/unit/integrations/ -v --tb=short
```
**Expected:** All tests pass  
**If failures:** 
1. Run failed test with `-vv --pdb`
2. Check mock setup
3. Verify async/await syntax
4. Check fixture dependencies

### Gate 5: Coverage (CRITICAL)
```bash
pytest tests/unit/integrations/ \
  --cov=arc_saga.integrations \
  --cov=arc_saga.exceptions \
  --cov-report=term-missing
```
**Expected:** 98%+ coverage  
**If below threshold:**
- Identify uncovered lines (look for red lines in term-missing)
- Add edge case tests
- Target: 99%+ if possible

### Gate 6: Formatting (CRITICAL)
```bash
black --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
isort --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
```
**Expected:** All files compliant  
**If fails:** Run WITHOUT `--check` to auto-fix:
```bash
black arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
isort arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/
```

---

## 🚨 CRITICAL FIXES IF GATES FAIL

### If mypy fails:
```python
# ❌ WRONG
def get_token(self, user_id) -> str:
    pass

# ✅ CORRECT
def get_token(self, user_id: str) -> str:
    pass
```

### If bandit warns about secrets:
```python
# ❌ WRONG
token_key = os.environ.get("ARC_SAGA_TOKEN_ENCRYPTION_KEY") or "default-key"

# ✅ CORRECT
token_key = os.environ.get("ARC_SAGA_TOKEN_ENCRYPTION_KEY")
if not token_key:
    raise ValueError("ARC_SAGA_TOKEN_ENCRYPTION_KEY not set")
```

### If tests fail:
1. Check mock setup for async functions
2. Verify pytest-asyncio is installed
3. Use `@pytest.mark.asyncio` on all async tests
4. For HTTP mocking, use proper context managers

### If coverage is low:
Add tests for:
- Error paths (exceptions)
- Edge cases (malformed input, empty strings, None)
- Timeout scenarios
- Network failures

---

## 📋 EXECUTION PLAN

### Phase 1: Prepare (2 mins)
```bash
cd arc_saga
git status  # Ensure all files committed
mkdir -p tests/unit/integrations
```

### Phase 2: Run All Gates (15 mins)
Execute in this order:
```bash
# 1. Type check
mypy --strict arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/protocols.py && echo "✅ Type check passed" || echo "❌ Type check FAILED"

# 2. Lint
pylint arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ --exit-zero && echo "✅ Linting passed" || echo "❌ Linting FAILED"

# 3. Security
bandit -r arc_saga/integrations/ arc_saga/exceptions/ -q && echo "✅ Security passed" || echo "❌ Security FAILED"

# 4. Tests
pytest tests/unit/integrations/ -v --tb=short && echo "✅ Tests passed" || echo "❌ Tests FAILED"

# 5. Coverage
pytest tests/unit/integrations/ --cov=arc_saga.integrations --cov=arc_saga.exceptions --cov-report=term-missing && echo "✅ Coverage OK" || echo "❌ Coverage FAILED"

# 6. Format
black --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ && echo "✅ Format OK" || echo "❌ Format needs fixing"
isort --check arc_saga/integrations/ arc_saga/exceptions/ arc_saga/orchestrator/ && echo "✅ Imports OK" || echo "❌ Imports need fixing"
```

### Phase 3: Fix Issues (10-20 mins)
For each failure:
1. Identify root cause
2. Apply fix from "CRITICAL FIXES" section above
3. Re-run that gate
4. Repeat until passing

### Phase 4: Final Verification (5 mins)
```bash
# Run all gates one final time
./scripts/verify_phase_2_3.sh  # (create this script first)
```

---

## 📊 SUCCESS CRITERIA

| Gate | Metric | Status | Evidence |
|------|--------|--------|----------|
| Type Safety | mypy --strict | ✅ Pass | 0 errors in output |
| Code Quality | pylint | ✅ Pass | 8.0+ score |
| Security | bandit | ✅ Pass | 0 issues found |
| Tests | pytest | ✅ Pass | All tests pass |
| Coverage | --cov | ✅ Pass | 98%+ coverage |
| Formatting | black + isort | ✅ Pass | No changes needed |

**All 6 gates must pass before proceeding to Phase 2.4**

---

## 🛠️ TROUBLESHOOTING QUICK REFERENCE

### "ModuleNotFoundError: No module named 'arc_saga'"
**Fix:** 
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "No module named 'pytest_asyncio'"
**Fix:**
```bash
pip install pytest-asyncio
```

### "Type checker errors on async/await"
**Fix:** Ensure return types on async functions:
```python
# ❌ WRONG
async def get_token(self, user_id: str):
    pass

# ✅ CORRECT
async def get_token(self, user_id: str) -> str:
    pass
```

### "Tests not running (collection error)"
**Fix:** Ensure `__init__.py` in test directories:
```bash
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/unit/integrations/__init__.py
```

### "Coverage not including module"
**Fix:** Use full module path in coverage flags:
```bash
pytest --cov=arc_saga.integrations --cov=arc_saga.exceptions
```

---

## 📞 ESCALATION PROTOCOL

If any gate fails after 2 attempts to fix:

1. **Run with verbose output:**
   ```bash
   mypy --strict --show-error-codes <module>
   pylint --disable=all --enable=<error-code> <module>
   pytest -vv --tb=long <test_file>
   ```

2. **Check recent changes:**
   ```bash
   git diff HEAD~1
   ```

3. **Verify dependencies:**
   ```bash
   pip list | grep -E "mypy|pylint|bandit|pytest"
   ```

4. **Rollback if necessary:**
   ```bash
   git checkout <filename>
   ```

---

## 🎯 FINAL CHECKPOINT

**Before declaring Phase 2.3 complete:**

- [ ] All 6 quality gates passing
- [ ] Created `scripts/verify_phase_2_3.sh`
- [ ] Created `docs/PHASE_2_3_COMPLETION.md`
- [ ] Git commit with message: "feat: Phase 2.3 - Copilot Trust Layer ✅"
- [ ] Tag release: `git tag -a v2.3.0 -m "Phase 2.3 Complete"`

**Then:**
- [ ] Archive this checklist
- [ ] Begin Phase 2.4 (ResponseMode + ProviderRouter)
- [ ] Update project roadmap

---

## ⏰ TIME BUDGET

- Pre-Gate Inspection: 5 mins
- Run Quality Gates: 15 mins
- Fix Issues: 10-20 mins
- Final Verification: 5 mins
- **Total: 35-45 minutes max**

**If exceeding 60 minutes on any single gate, escalate.**

---

**🚀 Ready? Execute the plan. Check items off. Report completion.**

**Dr. Alex Chen's Framework demands precision. Execute flawlessly.**
