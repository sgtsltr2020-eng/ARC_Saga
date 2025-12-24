# LoreBook Patterns v1

This document defines reusable patterns and decision templates for the LoreBook system.
Agents should reference these patterns to avoid repeating known mistakes.

## Pattern Catalog

### pattern_minimal_diff_tests_001

- **Name**: Avoid full test rewrites for type-hint-only fixes
- **Summary**: When fixing static analysis issues in working tests, minimize changes to avoid regressions and ease review.
- **Trigger Example**: "mypy reports missing type annotations in tests; tests already pass."
- **Bad Outcome Example**: "Agent rewrites entire test_onboarding_flow.py instead of adding -> None to 5 test functions."
- **Preferred Action**: "Only add required type annotations; preserve existing test structure and logic."
- **Related Rules**: ["45"]
- **Tags**: ["tests", "minimal-diff", "mypy", "refactoring"]
- **Applies When**:
  - File path starts with `tests/`
  - Change is triggered by static check (mypy/lint) only
- **LoreBook Storage Hint**: Set `DebateRecord.should_generalize = True` with tags above.

```json
{
  "id": "pattern_minimal_diff_tests_001",
  "name": "Avoid full test rewrites for type-hint-only fixes",
  "summary": "When fixing static analysis issues in working tests, minimize changes.",
  "trigger_example": "mypy reports missing type annotations in tests; tests already pass.",
  "bad_outcome_example": "Agent rewrites entire test file instead of adding annotations.",
  "preferred_action": "Only add required type annotations; preserve existing structure.",
  "related_rules": ["45"],
  "tags": ["tests", "minimal-diff", "mypy", "refactoring"],
  "applies_when": "File path starts with 'tests/' AND trigger is static check only",
  "lorebook_storage_hint": "DebateRecord.should_generalize = True"
}
```
