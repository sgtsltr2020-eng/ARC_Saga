"""
Task Dependency Graph
=====================

Manages tasks with dependencies using NetworkX DAG.
"""

from typing import List, Optional

import networkx as nx  # type: ignore

from saga.core.task import Task


class TaskGraph:
    """
    Dependency graph for tasks.
    
    Enables:
    - Parallel execution of independent tasks
    - Topological ordering respecting dependencies
    - Cycle detection
    """
    
    def __init__(self) -> None:
        """Initialize empty DAG."""
        self.graph = nx.DiGraph()
    
    def add_task(self, task: Task) -> None:
        """Add task to graph."""
        self.graph.add_node(task.id, task=task)
        
        # Add dependency edges
        for dep_id in task.dependencies:
            self.graph.add_edge(dep_id, task.id)
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks with no pending dependencies.
        
        These can be executed in parallel.
        """
        ready: List[Task] = []
        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            
            # Check if all dependencies are done
            if all(
                self.graph.nodes[dep_id]['task'].status == 'done'
                for dep_id in self.graph.predecessors(node_id)
            ):
                if task.status == 'pending':
                    ready.append(task)
        
        return ready
    
    def is_cyclic(self) -> bool:
        """Check for circular dependencies."""
        return not nx.is_directed_acyclic_graph(self.graph)
        
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the graph."""
        return [self.graph.nodes[node_id]['task'] for node_id in self.graph.nodes()]
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID."""
        if task_id in self.graph:
            return self.graph.nodes[task_id]['task'] # type: ignore
        return None
