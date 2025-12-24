"""
Workflow Execution Patterns.

Implements IWorkflowStrategy for different execution modes.
Delegates the actual task execution to the provided executor function.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

from saga.orchestrator.errors import BudgetExceededError
from saga.orchestrator.judgement import VerdictParser, validate_artifacts
from saga.orchestrator.protocols import IWorkflowStrategy
from saga.orchestrator.roles import AgentRole
from saga.orchestrator.types import Result, Task

from ..error_instrumentation import log_with_context

# Type variables
T = TypeVar("T")
R = TypeVar("R")

# Callback type for executing a single task
TaskExecutor = Callable[[Task[Any]], Awaitable[Result[Any]]]


class SequentialStrategy(IWorkflowStrategy):
    """
    Executes tasks one after another in order.
    Aborts sequence if a task fails? No, the original implementation
    in core.py continued? Let's check.
    core.py _execute_sequential just appended results. It didn't stop.
    """

    async def execute(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        executor: TaskExecutor,
        correlation_id: str,
    ) -> list[Result[Any]]:
        results: list[Result[Any]] = []
        for task in tasks:
            result = await executor(task)
            results.append(result)
        return results


class ParallelStrategy(IWorkflowStrategy):
    """
    Executes all tasks concurrently.
    """

    async def execute(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        executor: TaskExecutor,
        correlation_id: str,
    ) -> list[Result[Any]]:
        if not tasks:
            return []
            
        coroutines = [executor(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=False)
        return list(results)


class DynamicStrategy(IWorkflowStrategy):
    """
    Executes tasks sequentially but stops early if a task fails.
    Useful for dependent steps where subsequent steps typically fail
    if the previous one failed.
    """

    async def execute(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        executor: TaskExecutor,
        correlation_id: str,
    ) -> list[Result[Any]]:
        results: list[Result[Any]] = []
        
        for task in tasks:
            result = await executor(task)
            results.append(result)

            if not result.success:
                # Stop on failure
                log_with_context(
                    "warning",
                    "dynamic_workflow_early_termination",
                    workflow_id=workflow_id,
                    failed_task_id=task.id,
                    completed_count=len(results),
                    remaining_count=len(tasks) - len(results),
                )
                break
                
        return results


class ArbitrationStrategy(IWorkflowStrategy):
    """
    Executes tasks using a Judge-Evaluated Arbitration pattern.
    
    Flow:
    1. Resolve Context Dependencies (Artifact Bus).
    2. Execute tasks in topological/layered order.
    3. Gate execution based on Token Budget (if manager provided).
    
    This strategy supports "Dual Quality Gate" where:
    - Independent Reviewers run in parallel.
    - A Judge synthesizes their outputs.
    """

    def __init__(
        self, 
        token_budget_manager: Any | None = None,
        strict_mode: bool = True
    ) -> None:
        """
        Initialize strategy.
        
        Args:
            token_budget_manager: Optional manager to enforce dynamic budget gates.
            strict_mode: If True, halts workflow on Judge REJECT/ESCALATE.
        """
        self._token_budget_manager = token_budget_manager
        self.strict_mode = strict_mode

    async def execute(
        self,
        workflow_id: str,
        tasks: list[Task[Any]],
        executor: TaskExecutor,
        correlation_id: str,
    ) -> list[Result[Any]]:
        # Map of Task ID -> Result
        results_map: dict[str, Result[Any]] = {}
        final_results_list: list[Result[Any]] = []

        # 1. Topological Sort (Simplified Layering for Phase 2)
        # We assume the tasks list is already partially ordered or we process
        # roughly in order, but we must strictly respect context_dependencies.
        # For a robust DAG, we would allow fully out-of-order execution,
        # but here we iterate and check dependencies.
        
        # In this implementation, we will iterate through the task list.
        # If a task has dependencies, we verify they are met.
        # This supports the "Worker -> [Reviewer A, Reviewer B] -> Judge" linear flow.
        
        # Group tasks by "layers" (tasks that can run in parallel)
        # A simple approach: Any task whose dependencies are satisfied and hasn't run yet.
        
        pending_tasks = list(tasks)
        completed_ids: set[str] = set()
        
        while pending_tasks:
            # Find all tasks ready to run
            ready_batch: list[Task[Any]] = []
            
            for task in pending_tasks:
                dependencies = task.metadata.get("context_dependencies", [])
                if all(dep_id in completed_ids for dep_id in dependencies):
                    ready_batch.append(task)
            
            if not ready_batch:
                # Cycle detected or dependency missing
                remaining_ids = [t.id for t in pending_tasks]
                log_with_context(
                    "error",
                    "arbitration_deadlock_detected",
                    workflow_id=workflow_id,
                    pending_tasks=remaining_ids,
                    completed_tasks=list(completed_ids)
                )
                raise ValueError(f"Arbitration Deadlock: Tasks {remaining_ids} have unmet dependencies.")

            # Budget Gate Check before batch execution
            if self._token_budget_manager:
                # Filter for AI tasks to estimate
                ai_tasks_in_batch = [
                    t for t in ready_batch 
                    if hasattr(t.input_data, "prompt") and hasattr(t.input_data, "model")
                    # Ideally check isinstance(t.input_data, AITaskInput) but need imports or loose check
                ]
                
                # Check for AITaskInput-like structure simply
                valid_ai_tasks = []
                for t in ai_tasks_in_batch:
                    # Duck typing check for AITaskInput fields
                    if all(hasattr(t.input_data, f) for f in ["prompt", "model", "provider"]):
                        valid_ai_tasks.append(t)

                if valid_ai_tasks:
                    # Estimate cost
                    allocations = await self._token_budget_manager.allocate_tokens(
                        valid_ai_tasks, memory_tier="standard" # Defaulting for Phase 2
                    )
                    batch_cost = sum(allocations.values())
                    
                    # Check against budget
                    budget = await self._token_budget_manager.check_budget()
                    if batch_cost > budget.remaining:
                        log_with_context(
                            "error",
                            "arbitration_budget_exceeded",
                            workflow_id=workflow_id,
                            batch_size=len(ready_batch),
                            estimated_cost=batch_cost,
                            remaining_budget=budget.remaining
                        )
                        raise BudgetExceededError(
                            workflow_id=workflow_id,
                            requested_tokens=batch_cost,
                            remaining_tokens=budget.remaining,
                            total_budget=budget.total
                        )

            # Execute Batch (Parallel)
            # We map over the batch to inject context first
            prepared_batch = []
            for task in ready_batch:
                # 2. Context Injection (Artifact Bus)
                dependencies = task.metadata.get("context_dependencies", [])
                if dependencies:
                    # Construct a context object/string from previous results
                    context_data = {}
                    for dep_id in dependencies:
                        res = results_map[dep_id]
                        if res.success and res.output_data:
                            # We assume output_data is accessible.
                            context_data[dep_id] = res.output_data
                    
                    # --- INPUT GUARD ---
                    # Ensure we aren't passing empty context to a Judge/Reviewer
                    try:
                        validate_artifacts(list(context_data.values()))
                    except ValueError as e:
                        # Fail fast if critical artifacts are missing
                         log_with_context("error", "arbitration_input_guard_failed", task_id=task.id, error=str(e))
                         raise
                    
                    # Augment the input_data.
                    if hasattr(task.input_data, "system_prompt"):
                        current_sys = getattr(task.input_data, "system_prompt") or ""
                        # Format context nicely
                        context_str = "\n\n=== UPSTREAM CONTEXT ===\n"
                        for dep_id, data in context_data.items():
                            context_str += f"Valid Source ({dep_id}):\n{data}\n"
                        context_str += "========================\n"
                        
                        # Create new input data with modification
                        from dataclasses import replace
                        new_input = replace(task.input_data, system_prompt=current_sys + context_str)
                        task = replace(task, input_data=new_input)
                
                prepared_batch.append(task)

            # Run the batch
            coroutines = [executor(t) for t in prepared_batch]
            batch_results = await asyncio.gather(*coroutines, return_exceptions=False)
            
            # Process results
            for task, result in zip(ready_batch, batch_results):
                results_map[task.id] = result
                completed_ids.add(task.id)
                # Keep original order for final list
                final_results_list.append(result)
                
                # --- JUDGE LOGIC & STRICT MODE ---
                # Check if this was a Judge task (via metadata role field)
                # We assume the caller tagged the task with `role="judge"`
                task_role = task.metadata.get("role")
                if task_role == AgentRole.JUDGE.value and result.success:
                    # Parse Verdict
                    # We assume output_data is the LLM string response
                    if isinstance(result.output_data, str):
                        verdict = VerdictParser.parse(result.output_data)
                        
                        # Store parsed object back in result (optional, or separate map?)
                        # For now, let's keep it clean. Maybe log it.
                        log_with_context(
                            "info", 
                            "judge_verdict_parsed", 
                            verdict_status=verdict.status,
                            rationale=verdict.rationale[:50]
                        )

                        if self.strict_mode and verdict.is_blocking():
                             log_with_context(
                                 "warning", 
                                 "arbitration_strict_halt", 
                                 reason="Judge issued Blocking Verdict",
                                 verdict=verdict
                             )
                             # Stop Workflow: Return incomplete list (safe?)
                             # Or better, mark remaining tasks as cancelled?
                             # For Phase 2: Break loop immediately.
                             # We still return what we have. Orchestrator handles "Incomplete".
                             pending_tasks.clear() # Clear pending to exit outer loop
                             break # Exit processing this batch (should be just the Judge usually)

            # Remove executed tasks from pending
            for t in ready_batch:
                if t in pending_tasks:
                    pending_tasks.remove(t)


        # Restore original order of results to match input tasks list
        # The Orchestrator expects results in the same order as tasks
        ordered_results = []
        for task in tasks:
            if task.id in results_map:
                ordered_results.append(results_map[task.id])
            else:
                # Should not happen if deadlock check passes
                pass
                
        return ordered_results
