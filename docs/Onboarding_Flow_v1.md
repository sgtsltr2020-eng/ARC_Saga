# SAGA Onboarding Flow v1

This document defines the technical design for SAGA's project onboarding process. The onboarding flow ensures that a project is correctly initialized with the necessary configuration, governance structures (Constitution, Codex), and IDE integrations to enable SAGA's agentic capabilities. It runs either on explicit command or implicitly on first execution in an unconfigured repository.

## Triggers

The onboarding process is initiated through one of the following triggers:

### 1. explicit `saga init` CLI command

- **Preconditions:** Docker/Python environment available.
- **Auto-detection:**
  - Root directory determination (searching for `.git`, `pyproject.toml`, `package.json`).
  - Framework detection (e.g., parsing `app/main.py` for `FastAPI`, `django.conf`, `next.config.js`).
  - Dependency manager detection (`poetry.lock`, `requirements.txt`, `yarn.lock`).

### 2. First Run Implicit Trigger

- **Preconditions:** `saga serve` or `saga run` executed in a directory without a `.saga/` configuration subdirectory.
- **Auto-detection:** Same as above.
- **Behavior:** SAGA pauses execution, informs the user that the project is unconfigured, and prompts to enter the interactive onboarding flow.

## Interaction Flow

The interactive flow guides the user through configuration phases.

### 3.1 Project Context Discovery

SAGA builds an in-memory `ProjectProfile` by combining file scan results with user input.

#### Inputs

1.  **Filesystem Scan:**
    - Analyzes standard config files (`pyproject.toml`, `requirements.txt`) to guess language and framework.
    - Scans for existing test directories (`tests/`, `spec/`) to infer test runners.
2.  **User Q&A:**
    - **"Detected Python/FastAPI project. Is this correct?"** (Y/n -> Select Language/Framework)
    - **"Select Governance Strictness Level:"**
      - `Relaxed` (Informational warnings, minimal blocking).
      - `Standard` (Blocks on critical errors, requires type hints).
      - `FAANG` (Strict enforcement, high test coverage, rigid style guides).
    - **"Preferred Test Framework:"** (Auto-selected if detected, e.g., `pytest`).

#### Outputs

- `ProjectProfile` object:
  ```python
  @dataclass
  class ProjectProfile:
      name: str
      language: str  # "python"
      framework: str # "fastapi"
      test_runner: str # "pytest"
      strictness: str # "standard"
      root_dir: Path
  ```

### 3.2 Governance Bootstrap

Based on the `ProjectProfile`, SAGA selects and initializes the appropriate governance structures.

1.  **Select SagaCodex Profile:**
    - Maps `ProjectProfile.framework` to a specific Codex (e.g., `sagacodex_python_fastapi.md`).
    - References `docs/SagaCodex_Index_v1.md` for the machine-readable rule definitions.
2.  **Initialize Policies:**
    - Sets defaults for Token Budget (e.g., low limit for `Relaxed`, higher for `Standard`).
    - Applies `SagaRules` overrides based on `strictness` (e.g., `FAANG` enforces `TriggerSeverity.CRITICAL` for `TriggerType.CODEX` violations).
3.  **Output Generation:**
    - Prepares the `.saga/config.yaml` structure.

#### Example `.saga/config.yaml`:

```yaml
version: 1.0
project:
  name: "arc_saga"
  framework: "fastapi"

governance:
  profile: "codex_python_fastapi_v1"
  strictness: "standard"
  budget_mode: "balanced" # fast vs einstein

codex:
  rules_index: ".saga/sagacodex_index.json"
```

### 3.3 File Generation

SAGA proposes writing the following files. The user is prompted to confirm.

1.  **`.saga/` Directory:**

    - **`.saga/config.yaml`**: (Create) The core project configuration described above.
    - **`.saga/lorebook.db`**: (Create) Initializes an empty SQLite schema for LoreBook persistence.
    - **`.saga/sagacodex_index.json`**: (Create) The materialized Codex rules index specific to this project's profile.
    - **`.saga/logs/`**: (Create) Directory for structured JSON logs.

2.  **Project Metadata:**

    - **`README.md`**: (Append) Adds a "Powered by SAGA" badge and a concise "Development with SAGA" section covering `saga serve` and admin approval workflow.
    - **`requirements.txt` / `pyproject.toml`**: (Patch) Adds `saga-llm` (or equivalent package) ensuring version compatibility.

3.  **`.cursorrules`:**
    - (Create/Patch) SAGA seeds this with prompt instructions for Cursor/AG.
    - **Content Strategy:**
      - Directs code generation requests to cross-reference `.saga/sagacodex_index.json`.
      - Enforces that agents must check for `SagaConstitution` violations before finalizing code.
      - Includes snippet: `"ALWAYS consult .saga/sagacodex_index.json for style guidelines."`

**Conflict Handling:**

- **File exists:** Prompt user to [Overwrite | Backup & Overwrite | Skip].
- **Config merge:** For `pyproject.toml`, uses AST-aware injection to avoid breaking existing syntax.

### 3.4 IDE and API Integration

On success, onboarding outputs connection information for the IDE.

1.  **Local Server API:**

    - Displays: `SAGA Server listening on http://localhost:8000`.
    - Provides health check curl: `curl http://localhost:8000/approval/health`.

2.  **Cursor/AG Suggestions:**
    - "Add this to your IDE custom commands:"
      ```bash
      # Trigger SAGA Review
      curl -X POST http://localhost:8000/task/review -d '{"path": "$ACTIVE_FILE"}'
      ```
    - "To start a new feature with SAGA:"
      ```text
      @SAGA Plan and implement feature 'User Login' adhering to Strict Codex.
      ```

## Non-Interactive (Headless) Mode

For CI/CD and scripted environments, onboarding supports a flag-driven headless mode.

**Command:**

```bash
saga init --non-interactive \
    --framework fastapi \
    --strictness faang \
    --budget-mode fast
```

**Configuration Precedence:**

1.  CLI Flags (`--framework`, `--strictness`).
2.  Environment Variables (`SAGA_FRAMEWORK`, `SAGA_STRICTNESS`).
3.  Auto-detection results.
4.  Hardcoded Defaults (`standard` strictness, `balanced` budget).

## Failure Modes and Recovery

| Failure Scenario             | Behavior                                                 | Recovery                                            |
| :--------------------------- | :------------------------------------------------------- | :-------------------------------------------------- |
| **Write Permission Denied**  | Abort. Error reported to stderr.                         | User must `chmod` or change directories and re-run. |
| **Existing Config Conflict** | Abort (Interactive: Prompt).                             | User backs up old file. Re-run `saga init --force`. |
| **Python Env Missing**       | Warning. Continues but skips dep installation.           | User must manually pip install dependencies later.  |
| **Detection Ambiguity**      | Fallback to interactive prompt (or default in headless). | User explicitly selects framework via CLI args.     |

**Rollback Strategy:**

- Onboarding is atomic per file. If `.saga/config.yaml` creation fails, previous files (e.g. `.cursorrules`) are preserved but the process exits 1.
- Re-running `saga init` is idempotent; it checks for existing valid config before overwriting.

## Future Extensions (v2+)

- **GUI Wizard:** Web-based setup UI served locally for visual configuration.
- **Framework Presets:** Pre-bundled configs for Django, Node.js, React, Go.
- **Org Templates:** Fetching `.saga/config.yaml` templates from a remote URL or private git repo for team standardization.
