{{FEATURE}}-{{PHASE}}-{{COMPONENT}} 

Devlog Format Version: 1.0
🔧 Log Meta

Devlog file: {{DEVLOG_FILE}}
Feature: {{FEATURE}}
Phase: {{PHASE}}
Component: {{COMPONENT}}
Status: ACTIVE
Created (UTC): {{CREATED_UTC}}
RotatedFrom: {{ROTATED_FROM}}


📥 INBOX (MAD writes here)
Rules:

Use prefix: MAD:
Keep asks concise; one ask per bullet.
Only one bullet at a time.
To start a new phase/component, add ONE bullet like:
MAD: START | Feature={{FEATURE}} | Phase={{PHASE}} | Component={{COMPONENT}}




MAD: (write the next request here)



📥 GUIDANCE (PPX/GEM write here)
Rules:

Use prefixes: PPX:, GEM:
Keep guidance concise; one item per bullet.
Optional: Provide input for Cursor.




📤 SUMMARY STREAM (AI-friendly, short)
Rules:

Each line = one event.
Prefix required: MAD: | PPX: | GEM: | CUR:
1–4 sentences max.
Be precise. Avoid rambling.



MAD: Initialized {{FEATURE}} / {{PHASE}} / {{COMPONENT}}.



🧾 FULL LOG (Expandable Details)



[{{CREATED_UTC}}] MAD
Created this devlog file for {{FEATURE}} / {{PHASE}} / {{COMPONENT}}.



✅ Cursor instructions (local behavior)

After any meaningful work OR any meaningful response to MAD, append ONE CUR: line under SUMMARY.
When PPX/GEM add guidance, respond (when prompted) with:
CUR: Acknowledged …; plan …; questions … (if any).

Do not create new devlog files yourself.
If you believe rotation is needed, write:
CUR: ROTATE_REQUEST (reason: soft_limit | hard_limit | phase_complete | component_complete).



🛡️ STATE (n8n replaces ONLY the JSON block below)

JSON{
  "current": true,
  "feature": "{{FEATURE}}",
  "phase": "{{PHASE}}",
  "component": "{{COMPONENT}}",

  "devlog_index": {{DEVLOG_INDEX}},
  "active_file": "{{DEVLOG_FILE}}",
  "created_utc": "{{CREATED_UTC}}",
  "rotated_from": "{{ROTATED_FROM}}",

  "word_limit_soft": 1800,
  "word_limit_hard": 2400,

  "hashes": {
    "full_file_sha256": "initial_value_set_by_n8n",
    "inbox": null,
    "summary": "initial_value"
  },

  "summary_line_count": 0,
  "max_summary_lines": 50,

  "rotation_reason": null,
  "next_devlog_index": {{DEVLOG_INDEX}} + 1,
  "next_file_name_template": "devlog_{{FEATURE}}_{{PHASE}}_{{COMPONENT}}_{{NEXT_INDEX}}.md",

  "processing": {
    "in_progress": false,
    "started_utc": null,
    "agent": null
  },

  "last_event_utc": "{{CREATED_UTC}}"
}