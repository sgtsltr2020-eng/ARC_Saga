"""
AST-Based Code Analysis
=======================

Accurate detection using Python's ast module.
"""

import ast
from dataclasses import dataclass
from typing import List


@dataclass
class ASTViolation:
    """Code violation detected by AST analysis."""
    rule_name: str
    line_number: int
    description: str
    severity: str


class ASTCodeChecker:
    """AST-based code checker for Python."""
    
    def check_type_hints(self, code: str) -> List[ASTViolation]:
        """Check for missing type hints."""
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [ASTViolation("SyntaxError", 0, "Code has syntax errors", "CRITICAL")]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check parameters
                for arg in node.args.args:
                    if arg.arg == 'self' or arg.arg == 'cls':
                        continue
                    if arg.annotation is None:
                        violations.append(ASTViolation(
                            "MissingTypeHint",
                            node.lineno,
                            f"Function '{node.name}' parameter '{arg.arg}' lacks type hint",
                            "CRITICAL"
                        ))
                
                # Check return type
                if node.returns is None and node.name != "__init__":
                    violations.append(ASTViolation(
                        "MissingReturnType",
                        node.lineno,
                        f"Function '{node.name}' lacks return type annotation",
                        "CRITICAL"
                    ))
        
        return violations
    
    def check_bare_except(self, code: str) -> List[ASTViolation]:
        """Check for bare except clauses."""
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(ASTViolation(
                        "BareExcept",
                        node.lineno,
                        "Bare except clause detected (catches KeyboardInterrupt)",
                        "CRITICAL"
                    ))
        
        return violations
    
    def check_async_database_operations(self, code: str) -> List[ASTViolation]:
        """Check for sync database calls."""
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                # Check if function contains database operations
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute):
                        if child.attr in ['query', 'execute', 'commit']:
                            # Check if parent function is async
                            if not isinstance(node, ast.AsyncFunctionDef):
                                violations.append(ASTViolation(
                                    "SyncDatabaseOperation",
                                    node.lineno,
                                    f"Function '{node.name}' uses database operations but is not async",
                                    "CRITICAL"
                                ))
                                break
        
        return violations
