n8n-perplexity-integration-phase1-setup
Devlog Format Version: 1.0

## 🔧 Log Meta

**Devlog file:** `devlog_n8n-perplexity-integration_phase1_setup_1.md`  
**Feature:** n8n-perplexity-integration  
**Phase:** phase1  
**Component:** setup  
**Status:** ACTIVE  
**Created (UTC):** 2025-12-14T07:34:37Z  
**RotatedFrom:** null

---

## 📥 INBOX (MAD writes here)

**Rules:**

- Use prefix: `MAD:`
- Keep asks concise; one ask per bullet.
- Only one bullet at a time.
- To start a new phase/component, add ONE bullet like:
  `MAD: START | Feature={{FEATURE}} | Phase={{PHASE}} | Component={{COMPONENT}}`

**Current Request:**

- MAD: START | Feature=n8n-perplexity-integration | Phase=phase1 | Component=setup
- MAD: Configure n8n Local File Trigger to watch C:\Users\sgtsl\PROJECTS\ARC_Saga\DEVLOG folder, detect when I add new INBOX items, call Perplexity API to generate guidance, and append PPX responses back to this devlog file. Include proper Docker volume mounting for Windows paths and error handling for API failures.
- MAD: CURSOR, give me an overview of the project in 50 words or less.

---

## 📥 GUIDANCE (PPX/GEM write here)

**Rules:**

- Use prefixes: `PPX:`, `GEM:`
- Keep guidance concise; one item per bullet.
- Optional: Provide input for Cursor.

_(n8n will append PPX/GEM guidance here)_

---

## 📤 SUMMARY STREAM (AI-friendly, short)

**Rules:**

- Each line = one event.
- Prefix required: `MAD:` | `PPX:` | `GEM:` | `CUR:`
- 1–4 sentences max.
- Be precise. Avoid rambling.

**Event Log:**

- [2025-12-14T07:34:37Z] MAD: Initialized n8n-perplexity-integration / phase1 / setup. Created first devlog file and requested n8n workflow configuration with Perplexity integration.

---

## 🧾 FULL LOG (Expandable Details)

**[2025-12-14T07:34:37Z] MAD**  
Created this devlog file for n8n-perplexity-integration / phase1 / setup.

**Context:**  
Setting up automated workflow where:

1. n8n monitors DEVLOG folder for file changes
2. Extracts new MAD requests from INBOX section
3. Calls Perplexity API with project context
4. Appends PPX guidance back to this file
5. Updates STATE hash to prevent duplicate processing

**Environment:**

- OS: Windows 11
- Project Path: C:\Users\sgtsl\PROJECTS\ARC_Saga
- Docker: Docker Desktop (required for n8n Local File Trigger)
- Tech Stack: Python/FastAPI, n8n self-hosted, Perplexity Sonar API

**Success Criteria:**

- n8n workflow triggers on devlog file save
- Perplexity responds with actionable guidance within 10 seconds
- STATE JSON inbox hash updates after each processing cycle
- No duplicate processing of same INBOX request
- Cursor can read updated devlog and implement guidance

---

## ✅ Cursor Instructions (local behavior)

**Your role as Cursor:**

1. After any meaningful work OR response to MAD, append ONE `CUR:` line under SUMMARY STREAM
2. When PPX/GEM add guidance, respond (when prompted) with:
   - `CUR: Acknowledged [topic]; plan [next steps]; questions [if any].`
3. **Do NOT create new devlog files yourself** — n8n handles rotation
4. If you believe rotation is needed, write:
   - `CUR: ROTATE_REQUEST (reason: soft_limit | hard_limit | phase_complete | component_complete)`

**What you CAN edit:**

- ✅ INBOX section (adding your implementation notes)
- ✅ SUMMARY STREAM (CUR: entries only)
- ✅ Code files in the repository
- ✅ Documentation files

**What you CANNOT edit:**

- ❌ STATE JSON block (n8n owns this)
- ❌ GUIDANCE section (PPX/GEM own this)
- ❌ Log Meta section
- ❌ Creating new devlog files

---

## 🛡️ STATE (n8n replaces ONLY the JSON block below)

```json
{
  "current": true,
  "feature": "n8n-perplexity-integration",
  "phase": "phase1",
  "component": "setup",

  "devlog_index": 1,
  "active_file": "devlog_n8n-perplexity-integration_phase1_setup_1.md",
  "created_utc": "2025-12-14T07:34:37Z",
  "rotated_from": null,

  "word_limit_soft": 1800,
  "word_limit_hard": 2400,

  "hashes": {
    "full_file_sha256": "pending_first_n8n_scan",
    "inbox": null,
    "summary": "initial_hash_pending"
  },

  "summary_line_count": 1,
  "max_summary_lines": 50,

  "rotation_reason": null,
  "next_devlog_index": 2,
  "next_file_name_template": "devlog_n8n-perplexity-integration_phase1_setup_2.md",

  "processing": {
    "in_progress": false,
    "started_utc": null,
    "agent": null
  },

  "last_event_utc": "2025-12-14T07:34:37Z"
}
```

---

## 📋 Integration Checklist

**Docker Setup:**

- [ ] Docker Desktop installed and running
- [ ] n8n container started with volume mount: `-v C:\Users\sgtsl\PROJECTS\ARC_Saga:/data/saga`
- [ ] Access n8n UI at http://localhost:5678

**n8n Workflow Configuration:**

- [ ] Workflow "INBOX Monitor" created
- [ ] Local File Trigger configured for `/data/saga/DEVLOG`
- [ ] Perplexity API credentials added
- [ ] Test trigger by saving this file

**Perplexity API:**

- [ ] API key obtained from https://perplexity.ai/settings/api
- [ ] Test endpoint: `POST https://api.perplexity.ai/chat/completions`
- [ ] Model: `sonar` or `sonar-pro`

**Cursor Configuration:**

- [ ] .cursorrules file references this devlog
- [ ] Cursor can read and parse STATE JSON
- [ ] Test: Ask Cursor to "read active devlog and summarize current task"

---

## 🚀 Next Actions (Priority Order)

1. **MAD:** Verify Docker Desktop is running (`docker ps`)
2. **MAD:** Start n8n with correct volume mount
3. **MAD:** Import n8n workflow JSON (to be provided)
4. **MAD:** Add Perplexity API key to n8n credentials
5. **MAD:** Save this file and check n8n execution log
6. **PPX:** Will provide n8n setup validation steps
7. **CUR:** Will implement first integration test

---

_End of devlog file. Ready for n8n processing._
