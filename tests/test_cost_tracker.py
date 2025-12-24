import pytest

from saga.llm.cost_tracker import CostTracker


def test_cost_tracker_initialization():
    tracker = CostTracker()
    assert tracker.get_total_cost() == 0.0
    assert len(tracker.task_costs) == 0

def test_record_task_cost():
    tracker = CostTracker()
    tracker.record_task_cost(
        task_id="task-1",
        cost=0.03,
        agent_name="CodingAgent"
    )

    assert tracker.get_total_cost() == 0.03
    assert len(tracker.task_costs) == 1
    assert tracker.get_task_cost("task-1") == 0.03

def test_budget_enforcement():
    tracker = CostTracker()
    tracker.budget_limit = 0.05  # Set low budget

    # Under budget
    within, remaining = tracker.check_budget()
    assert within is True
    assert remaining == 0.05

    tracker.record_task_cost("task-1", 0.03)
    within, remaining = tracker.check_budget()
    assert within is True
    assert remaining == pytest.approx(0.02)

    # Over budget
    tracker.record_task_cost("task-1", 0.03)
    # Total 0.06 > 0.05
    within, remaining = tracker.check_budget()
    assert within is False
    assert remaining < 0

def test_multiple_tasks():
    tracker = CostTracker()
    tracker.record_task_cost("task-1", 0.1)
    tracker.record_task_cost("task-2", 0.2)

    assert tracker.get_task_cost("task-1") == 0.1
    assert tracker.get_task_cost("task-2") == 0.2
    assert tracker.get_total_cost() == pytest.approx(0.3)

def test_reset():
    tracker = CostTracker()
    tracker.record_task_cost("task-1", 0.1)
    tracker.reset()
    assert tracker.get_total_cost() == 0.0
    assert len(tracker.task_costs) == 0
