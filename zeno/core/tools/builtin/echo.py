"""EchoTool — returns the input message unchanged.

Useful for testing the tool pipeline end-to-end and as a simple
demonstration that the tool registry is wired correctly.
"""

from __future__ import annotations

from typing import Any

from zeno.core.tools.base import BaseTool, ToolResult


class EchoTool(BaseTool):
    """Return the *message* argument unchanged.

    Example
    -------
    .. code-block:: python

        tool = EchoTool()
        result = tool.execute(message="hello")
        assert result.output == "hello"
    """

    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Return the input message unchanged.  Useful for testing.",
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "The message to echo back.",
            }
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        message = kwargs.get("message")
        if message is None:
            return ToolResult(success=False, error="Missing 'message' parameter.")
        return ToolResult(success=True, output=str(message))
