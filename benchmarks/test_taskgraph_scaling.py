"""
TaskGraph Benchmark: DAG Scaling
"""


import networkx as nx
import pytest

from saga.core.task import Task
from saga.core.task_graph import TaskGraph


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("n_tasks", [10, 50, 100, 500])
def test_topological_sort_scaling(benchmark, n_tasks):
    """Benchmark topological_sort on N tasks."""

    def create_dag(n):
        g = TaskGraph()
        for i in range(n):
            task = Task(id=f"t{i}", description="Task", weight="simple")
            g.add_task(task)
            if i > 0:
                task.dependencies = [f"t{i-1}"]  # Chain deps
        return g.graph

    dag = create_dag(n_tasks)
    result = benchmark(nx.topological_sort, dag)
    # Handle generator vs list
    assert len(list(result)) == n_tasks
