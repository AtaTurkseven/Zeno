"""CalculatorTool — evaluates a safe arithmetic expression."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from zeno.core.tools.base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Evaluate a safe arithmetic expression.

    Supported operations: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``,
    and the standard :mod:`math` constants ``pi`` and ``e``.

    Example
    -------
    .. code-block:: python

        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10 + 1")
        assert result.output == 1025.0
    """

    _SAFE_NAMES: dict[str, Any] = {
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    _ALLOWED_OPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "Evaluate a safe arithmetic expression.  "
                "Supports +, -, *, /, //, %, **, and math constants (pi, e, sqrt, …)."
            ),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate (e.g. '2 + 2').",
            }
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        expression: str = str(kwargs.get("expression", "")).strip()
        if not expression:
            return ToolResult(success=False, error="Missing 'expression' parameter.")

        try:
            tree = ast.parse(expression, mode="eval")
            self._validate(tree)
            result = self._eval_node(tree.body)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(success=False, error=str(exc))
        except SyntaxError:
            return ToolResult(
                success=False,
                error=f"Invalid expression syntax: {expression!r}",
            )

        return ToolResult(
            success=True,
            output=result,
            metadata={"expression": expression},
        )

    # ------------------------------------------------------------------
    # Internal safe-eval helpers
    # ------------------------------------------------------------------

    def _validate(self, tree: ast.AST) -> None:
        """Raise ValueError if *tree* contains unsafe nodes."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Function calls on objects are not allowed.")
                if node.func.id not in self._SAFE_NAMES:
                    raise ValueError(f"Function '{node.func.id}' is not allowed.")
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)):
                raise ValueError("Imports and attribute access are not allowed.")

    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate an AST *node*."""
        _binop = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        _unary = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self._SAFE_NAMES:
                raise ValueError(f"Name '{node.id}' is not allowed.")
            value = self._SAFE_NAMES[node.id]
            if callable(value):
                raise ValueError(
                    f"'{node.id}' is a function; call it with parentheses."
                )
            return float(value)
        if isinstance(node, ast.BinOp):
            op_fn = _binop.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _unary.get(type(node.op))
            if op_fn is None:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            return op_fn(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed.")
            func_name: str = node.func.id
            fn = self._SAFE_NAMES.get(func_name)
            if fn is None:
                raise ValueError(f"Unknown function: {func_name}")
            if not callable(fn):
                raise ValueError(f"'{func_name}' is not a callable function.")
            args = [self._eval_node(a) for a in node.args]
            return float(fn(*args))

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")
