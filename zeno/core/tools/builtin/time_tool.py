"""TimeTool — returns the current date and/or time."""

from __future__ import annotations

import datetime
from typing import Any

from zeno.core.tools.base import BaseTool, ToolResult


class TimeTool(BaseTool):
    """Return the current local date, time, or datetime.

    Parameters
    ----------
    None — this tool requires no constructor arguments.

    Example
    -------
    .. code-block:: python

        tool = TimeTool()
        result = tool.execute(format="datetime")
        # result.output == "2025-01-15 10:30:00"
    """

    def __init__(self) -> None:
        super().__init__(
            name="time",
            description=(
                "Return the current local date and/or time.  "
                "Use format='date', 'time', or 'datetime' (default)."
            ),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "format": {
                "type": "string",
                "enum": ["date", "time", "datetime"],
                "default": "datetime",
                "description": "Which part to return: 'date', 'time', or 'datetime'.",
            }
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        fmt: str = str(kwargs.get("format", "datetime")).strip().lower()
        now = datetime.datetime.now()

        if fmt == "date":
            output = now.strftime("%Y-%m-%d")
        elif fmt == "time":
            output = now.strftime("%H:%M:%S")
        else:
            output = now.strftime("%Y-%m-%d %H:%M:%S")

        return ToolResult(
            success=True,
            output=output,
            metadata={"format": fmt, "iso": now.isoformat()},
        )
