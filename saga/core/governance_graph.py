
import logging
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from saga.core.personas import PersonaLibrary, Role

# Setup logging
logger = logging.getLogger(__name__)

# --- State Definition ---

class WardenState(TypedDict):
    """
    The persistent state of the Warden's governance loop.
    Supports Phase 3.0 Parallel Sovereignty.
    """
    # Inputs
    task_input: str
    context: dict[str, Any]
    trace_id: str

    # Workflow Artifacts
    plan: str | None

    # Parallel Worker Outputs
    alpha_output: dict[str, Any] | None  # Coder
    beta_output: dict[str, Any] | None   # SDET

    # Synchronization
    merged_result: dict[str, Any] | None
    conflict_detected: bool

    # Ledger & HitL
    ledger_proposal: dict[str, Any] | None  # The Wisdom Proposal
    user_feedback: str | None

    # Meta-State - Use reducer for parallel-safe updates
    history: Annotated[list[str], operator.add]
    iteration_count: int
    status: Literal[
        "planning",
        "constructing",  # Parallel work
        "syncing",
        "proposing",     # HITL
        "completed",
        "failed"
    ]


# --- Helper Functions ---

def _log_node_entry(node_name: str, state: WardenState):
    """Standardized node entry logging."""
    trace_id = state.get("trace_id", "UNKNOWN_TRACE")
    history_len = len(state.get("history", []))
    iteration = state.get("iteration_count", 0)
    logger.info(f"[{trace_id}] - Node: {node_name} | History: {history_len} | Iter: {iteration}")

def _append_history(state: WardenState, entry: str) -> list[str]:
    """Helper to safely append to history list."""
    current = state.get("history", [])
    return current + [entry]

# --- Node Logic ---

def planner(state: WardenState) -> dict[str, Any]:
    """Generates the initial plan based on task input. Adopt Architect persona logic here."""
    _log_node_entry("PLANNER", state)

    # In real logic, this would use the Architect persona to decompose the task
    return {
        "plan": f"Architect Plan: Implement {state['task_input']}",
        "history": _append_history(state, "Planner: Plan created."),
        "status": "constructing"
    }

def worker_alpha(state: WardenState) -> dict[str, Any]:
    """
    Worker Alpha (Construction/Coder).
    Focuses on implementation logic.
    """
    _log_node_entry("WORKER_ALPHA", state)

    # Simulate Coder Persona work
    persona = PersonaLibrary.get_persona(Role.CODER)

    return {
        "alpha_output": {
            "code": "def hello(): print('Hello')",
            "persona": persona.name
        },
        "history": _append_history(state, "Worker Alpha: Implemented code.")
    }

def worker_beta(state: WardenState) -> dict[str, Any]:
    """
    Worker Beta (Verification/SDET).
    Focuses on writing tests.
    """
    _log_node_entry("WORKER_BETA", state)

    # Simulate SDET Persona work
    persona = PersonaLibrary.get_persona(Role.SDET)

    return {
        "beta_output": {
            "tests": "def test_hello(): assert hello() is None",
            "persona": persona.name
        },
        "history": _append_history(state, "Worker Beta: Wrote tests.")
    }

def synchronizer(state: WardenState) -> dict[str, Any]:
    """
    Merge outputs from Alpha and Beta.
    Detects conflicts (file locking).
    """
    _log_node_entry("SYNCHRONIZER", state)

    alpha = state.get("alpha_output", {})
    beta = state.get("beta_output", {})

    # Simple merge logic
    merged = {
        "code": alpha.get("code"),
        "tests": beta.get("tests")
    }

    return {
        "merged_result": merged,
        "conflict_detected": False, # Mock logic
        "history": _append_history(state, "Synchronizer: Merged Alpha and Beta outputs."),
        "status": "proposing"
    }

def proposal_ledger(state: WardenState) -> dict[str, Any]:
    """
    The Wisdom Proposal.
    Summarizes the "Attempted vs Result" for user approval.
    Includes contextual anchoring with recent lore entries.
    """
    _log_node_entry("PROPOSAL_LEDGER", state)

    merged = state.get("merged_result", {})
    alpha_output = state.get("alpha_output", {})
    beta_output = state.get("beta_output", {})

    # Build proposal summary
    proposal = {
        "summary": f"Implement {state.get('task_input')} with tests.",
        "changes": merged,
        "codex_status": "Compliant",  # Mock - would be set by LLM tagger
        # Contextual anchoring data for LoreEntry generation
        "ledger_content": str(merged),
        "active_personas": ["Architect", "Coder", "SDET"],
        "alpha_output": alpha_output,
        "beta_output": beta_output,
    }

    # Detect persona tension for inline tagging
    tension_detected = False
    alpha_str = str(alpha_output).lower()
    beta_str = str(beta_output).lower()
    conflict_signals = ["error", "failed", "conflict", "rejected"]
    if any(s in alpha_str or s in beta_str for s in conflict_signals):
        tension_detected = True
        proposal["tension_detected"] = True

    history_entry = "Ledger: Proposal ready for approval."
    if tension_detected:
        history_entry = "Ledger: Proposal ready (persona tension detected)."

    return {
        "ledger_proposal": proposal,
        "history": _append_history(state, history_entry)
    }

# --- Edge Logic ---

def should_interrupt(state: WardenState) -> Literal["continue", "interrupt"]:
    """
    HITL Interrupt Logic.
    Always interrupt for user approval on proposals unless auto-approve logic exists.
    """
    # For now, we simulate "check if user approved".
    # In LangGraph interrupt usage, we'd pause here.
    return "interrupt"

# --- Graph Assembly ---

def create_warden_graph(checkpointer: Any = None, enable_interrupt: bool = True) -> Any:
    """
    Constructs the compiled Warden StateGraph with Parallel Sovereignty.

    Args:
        checkpointer: Optional checkpointer for persistence (AsyncSqliteSaver or MemorySaver)
        enable_interrupt: If True, graph pauses at proposal_ledger for HITL
    """
    from langgraph.checkpoint.memory import MemorySaver

    workflow = StateGraph(WardenState)

    # Add Nodes
    workflow.add_node("planner", planner)
    workflow.add_node("worker_alpha", worker_alpha)
    workflow.add_node("worker_beta", worker_beta)
    workflow.add_node("synchronizer", synchronizer)
    workflow.add_node("proposal_ledger", proposal_ledger)

    # Entry
    workflow.set_entry_point("planner")

    # Flow
    # 1. Plan -> Fan Out
    workflow.add_edge("planner", "worker_alpha")
    workflow.add_edge("planner", "worker_beta")

    # 2. Fan In -> Synchronizer
    workflow.add_edge("worker_alpha", "synchronizer")
    workflow.add_edge("worker_beta", "synchronizer")

    # 3. Synchronizer -> Ledger
    workflow.add_edge("synchronizer", "proposal_ledger")

    # 4. Ledger -> End (or Interrupt Loop)
    workflow.add_edge("proposal_ledger", END)

    # Use provided checkpointer or fallback to MemorySaver
    memory = checkpointer if checkpointer else MemorySaver()

    # Compile with optional interrupt
    if enable_interrupt:
        app = workflow.compile(checkpointer=memory, interrupt_after=["proposal_ledger"])
    else:
        app = workflow.compile(checkpointer=memory)
    return app
