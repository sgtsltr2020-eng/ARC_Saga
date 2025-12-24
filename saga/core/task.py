"""
Task Primitive
==============

Defines the Task unit of work.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Task:
    """
    A unit of work to be executed by a subagent.
    
    Attributes:
        id: Unique task identifier
        description: What needs to be done
        weight: Simple (checklist) or complex (full vetting)
        assigned_agent: Which subagent is executing this
        status: Current state
        dependencies: Other task IDs that must complete first
        budget_allocation: Token/cost limit
        checklist: For simple tasks, minimum requirements
        vetting_criteria: For complex tasks, full quality checks
        self_check_result: Agent's own verification
        warden_verification: Warden's approval/rejection
        mimiry_measurement: Oracle's measurement against ideal (if consulted)
        trace_id: Links to decision trail
    """
    id: str
    description: str
    weight: Literal["simple", "complex"]
    assigned_agent: Optional[str] = None
    status: Literal["pending", "in_progress", "done", "blocked", "conflict"] = "pending"
    dependencies: List[str] = field(default_factory=list)
    budget_allocation: float = 0.0
    checklist: List[str] = field(default_factory=list)
    vetting_criteria: Dict[str, Any] = field(default_factory=dict)
    self_check_result: Optional[Dict[str, Any]] = None
    warden_verification: Optional[Literal["approved", "rejected"]] = None
    mimiry_measurement: Optional[Dict[str, Any]] = None  # NEW: Oracle measurement
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "weight": self.weight,
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "dependencies": self.dependencies,
            "budget_allocation": self.budget_allocation,
            "checklist": self.checklist,
            "vetting_criteria": self.vetting_criteria,
            "self_check_result": self.self_check_result,
            "warden_verification": self.warden_verification,
            "mimiry_measurement": self.mimiry_measurement,
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
