"""
StaticAnalyzer - AST-based Code Structure Parser
=================================================

Provides "eyes" for USMA by parsing Python files into the Knowledge Graph.
Extracts functions, classes, imports, and call relationships.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 1 - Structural Vision
"""

import ast
import logging
from pathlib import Path
from typing import Any

from saga.core.memory.graph_engine import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RepoGraph,
)

logger = logging.getLogger(__name__)


# Directories to exclude from analysis
EXCLUDED_DIRS = {
    "venv", ".venv", "env", ".env",
    "__pycache__", ".git", ".saga",
    "node_modules", "htmlcov", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "build", "dist"
}


class StaticAnalyzer:
    """
    AST-based static analyzer for Python projects.

    Walks the project directory and extracts:
    - File structure (modules)
    - Class definitions
    - Function definitions
    - Import statements
    - Function call relationships

    Populates a RepoGraph with nodes and edges representing
    the structural relationships in the codebase.
    """

    def __init__(self, project_root: str | Path, graph: RepoGraph | None = None):
        """Initialize the static analyzer."""
        self.project_root = Path(project_root).resolve()
        self.graph = graph or RepoGraph(project_root)
        self._current_file: str | None = None
        self._current_class: str | None = None
        logger.info(f"StaticAnalyzer initialized for: {self.project_root}")

    def analyze_project(self) -> RepoGraph:
        """
        Analyze the entire project and populate the graph.

        Returns:
            RepoGraph with all discovered nodes and edges
        """
        logger.info(f"Starting project analysis: {self.project_root}")
        python_files = self._find_python_files()

        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

        logger.info(
            f"Analysis complete: {self.graph.node_count} nodes, "
            f"{self.graph.edge_count} edges"
        )
        return self.graph

    def analyze_file(self, file_path: str | Path) -> RepoGraph:
        """Analyze a single file and add to graph."""
        self._analyze_file(Path(file_path))
        return self.graph

    def _find_python_files(self) -> list[Path]:
        """Find all Python files in the project, excluding certain directories."""
        python_files: list[Path] = []

        for path in self.project_root.rglob("*.py"):
            # Check if any parent is in excluded dirs
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            python_files.append(path)

        logger.debug(f"Found {len(python_files)} Python files")
        return python_files

    def _analyze_file(self, file_path: Path) -> None:
        """Parse a single Python file and extract structure."""
        relative_path = file_path.relative_to(self.project_root)
        file_id = f"file:{relative_path}"
        self._current_file = file_id

        # Add file node
        self.graph.add_node(GraphNode(
            node_id=file_id,
            node_type=NodeType.FILE,
            name=file_path.name,
            file_path=str(relative_path),
            metadata={"absolute_path": str(file_path)}
        ))

        # Parse AST
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return

        # Extract structure
        self._extract_imports(tree, file_id)
        self._extract_classes(tree, file_id, str(relative_path))
        self._extract_functions(tree, file_id, str(relative_path))

    def _extract_imports(self, tree: ast.Module, file_id: str) -> None:
        """Extract import statements and create IMPORTS edges."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_id = f"module:{alias.name}"
                    self._ensure_module_node(module_id, alias.name)
                    self.graph.add_edge(GraphEdge(
                        source_id=file_id,
                        target_id=module_id,
                        edge_type=EdgeType.IMPORTS,
                        weight=1.0,
                        metadata={"line": node.lineno}
                    ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_id = f"module:{node.module}"
                    self._ensure_module_node(module_id, node.module)
                    self.graph.add_edge(GraphEdge(
                        source_id=file_id,
                        target_id=module_id,
                        edge_type=EdgeType.IMPORTS,
                        weight=1.0,
                        metadata={
                            "line": node.lineno,
                            "names": [a.name for a in node.names]
                        }
                    ))

    def _ensure_module_node(self, module_id: str, module_name: str) -> None:
        """Ensure a module node exists in the graph."""
        if not self.graph.has_node(module_id):
            self.graph.add_node(GraphNode(
                node_id=module_id,
                node_type=NodeType.MODULE,
                name=module_name
            ))

    def _extract_classes(
        self, tree: ast.Module, file_id: str, relative_path: str
    ) -> None:
        """Extract class definitions and inheritance relationships."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"class:{relative_path}:{node.name}"

                self.graph.add_node(GraphNode(
                    node_id=class_id,
                    node_type=NodeType.CLASS,
                    name=node.name,
                    file_path=relative_path,
                    line_number=node.lineno,
                    metadata={
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node) or ""
                    }
                ))

                # CONTAINS edge from file to class
                self.graph.add_edge(GraphEdge(
                    source_id=file_id,
                    target_id=class_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=1.0
                ))

                # INHERITS edges for base classes
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name:
                        # Create soft reference to base class
                        base_id = f"class:*:{base_name}"  # Wildcard for unresolved
                        if not self.graph.has_node(base_id):
                            self.graph.add_node(GraphNode(
                                node_id=base_id,
                                node_type=NodeType.CLASS,
                                name=base_name,
                                metadata={"unresolved": True}
                            ))
                        self.graph.add_edge(GraphEdge(
                            source_id=class_id,
                            target_id=base_id,
                            edge_type=EdgeType.INHERITS,
                            weight=1.0
                        ))

                # Extract methods within class
                self._current_class = class_id
                self._extract_methods(node, class_id, relative_path)
                self._current_class = None

    def _extract_methods(
        self, class_node: ast.ClassDef, class_id: str, relative_path: str
    ) -> None:
        """Extract method definitions from a class."""
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_id = f"func:{relative_path}:{class_node.name}.{node.name}"

                self.graph.add_node(GraphNode(
                    node_id=method_id,
                    node_type=NodeType.FUNCTION,
                    name=f"{class_node.name}.{node.name}",
                    file_path=relative_path,
                    line_number=node.lineno,
                    metadata={
                        "is_method": True,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or ""
                    }
                ))

                # CONTAINS edge from class to method
                self.graph.add_edge(GraphEdge(
                    source_id=class_id,
                    target_id=method_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=1.0
                ))

                # Extract function calls within method
                self._extract_calls(node, method_id)

    def _extract_functions(
        self, tree: ast.Module, file_id: str, relative_path: str
    ) -> None:
        """Extract top-level function definitions."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_id = f"func:{relative_path}:{node.name}"

                self.graph.add_node(GraphNode(
                    node_id=func_id,
                    node_type=NodeType.FUNCTION,
                    name=node.name,
                    file_path=relative_path,
                    line_number=node.lineno,
                    metadata={
                        "is_method": False,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or ""
                    }
                ))

                # CONTAINS edge from file to function
                self.graph.add_edge(GraphEdge(
                    source_id=file_id,
                    target_id=func_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=1.0
                ))

                # Extract function calls
                self._extract_calls(node, func_id)

    def _extract_calls(
        self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, caller_id: str
    ) -> None:
        """Extract function calls within a function body."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                callee_name = self._get_call_name(node)
                if callee_name:
                    # Create soft reference to called function
                    callee_id = f"func:*:{callee_name}"  # Wildcard for unresolved
                    if not self.graph.has_node(callee_id):
                        self.graph.add_node(GraphNode(
                            node_id=callee_id,
                            node_type=NodeType.FUNCTION,
                            name=callee_name,
                            metadata={"unresolved": True}
                        ))
                    self.graph.add_edge(GraphEdge(
                        source_id=caller_id,
                        target_id=callee_id,
                        edge_type=EdgeType.CALLS,
                        weight=0.9,  # Slightly lower for unresolved
                        metadata={"line": getattr(node, "lineno", None)}
                    ))

    # --- Helper Methods ---

    def _get_name(self, node: ast.expr) -> str | None:
        """Extract name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of a function call."""
        return self._get_name(node.func)

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_name(node) or ""
        elif isinstance(node, ast.Call):
            return self._get_call_name(node) or ""
        return ""

    def get_file_summary(self, file_path: str | Path) -> dict[str, Any]:
        """Get a summary of a file's structure from the graph."""
        relative_path = Path(file_path)
        if file_path and Path(file_path).is_absolute():
            relative_path = Path(file_path).relative_to(self.project_root)

        file_id = f"file:{relative_path}"

        if not self.graph.has_node(file_id):
            return {"error": f"File not indexed: {file_path}"}

        # Get contained entities
        classes = []
        functions = []
        imports = []

        for _, target, data in self.graph.get_edges_from(file_id):
            edge_type = data.get("edge_type")
            node = self.graph.get_node(target)

            if edge_type == EdgeType.CONTAINS.value:
                if node and node.node_type == NodeType.CLASS:
                    classes.append(node.name)
                elif node and node.node_type == NodeType.FUNCTION:
                    functions.append(node.name)
            elif edge_type == EdgeType.IMPORTS.value:
                if node:
                    imports.append(node.name)

        return {
            "file": str(relative_path),
            "classes": classes,
            "functions": functions,
            "imports": imports
        }
