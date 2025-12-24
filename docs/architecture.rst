Architecture
============

Saga (User Interface)
↓
Warden (Delegation)
  ↓ Decompose → TaskGraph (DAG)
  ↓ Agents Execute (parallel ready tasks)
  ↓ Conflicts → Mimiry.resolve_conflict (parallel measure)
  ↓ Enforce Canonical → Approve/Escalate
↓
User Output (verified code)
