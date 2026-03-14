"""ToolExecutor — runs tools looked up from the registry and handles errors."""

from __future__ import annotations

import logging
import time
from typing import Any

from zeno.core.tools.base import ToolResult
from zeno.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools by name, wrapping calls in error handling and timing.

    Parameters
    ----------
    registry:
        The :class:`~zeno.core.tools.registry.ToolRegistry` to look up tools from.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Look up *tool_name* and call its :meth:`execute` method.

        Parameters
        ----------
        tool_name:
            The registered name of the tool to run.
        **kwargs:
            Arguments forwarded verbatim to the tool's ``execute`` method.

        Returns
        -------
        ToolResult
            Always returns a :class:`ToolResult`; execution errors are
            captured and returned as ``success=False`` results.
        """
        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            msg = f"Tool '{tool_name}' is not registered."
            logger.warning(msg)
            return ToolResult(success=False, error=msg)

        logger.info("Executing tool '%s' with args: %s", tool_name, kwargs)
        start = time.monotonic()
        try:
            result = tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            msg = f"Tool '{tool_name}' raised an unexpected exception: {exc}"
            logger.exception(msg)
            return ToolResult(
                success=False, error=msg, metadata={"elapsed_s": elapsed}
            )

        elapsed = time.monotonic() - start
        result.metadata.setdefault("elapsed_s", elapsed)
        logger.info(
            "Tool '%s' finished in %.3fs | success=%s",
            tool_name,
            elapsed,
            result.success,
        )
        return result

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolExecutor(registry={self._registry!r})"
