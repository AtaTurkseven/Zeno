"""ToolRegistry — central store for all registered Zeno tools."""

from __future__ import annotations

import logging
from typing import Iterator

from zeno.core.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Maintains a name → tool mapping and provides discovery helpers.

    Tools are registered once and can then be looked up or listed at any time.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Add *tool* to the registry.

        Parameters
        ----------
        tool:
            A concrete :class:`~zeno.core.tools.base.BaseTool` instance.

        Raises
        ------
        ValueError
            When a tool with the same name has already been registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered.  "
                "Use unregister() first if you want to replace it."
            )
        self._tools[tool.name] = tool
        logger.debug("Tool registered: '%s'.", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by *name*.

        Parameters
        ----------
        name:
            The tool's unique identifier.

        Raises
        ------
        KeyError
            When no tool with *name* exists.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        del self._tools[name]
        logger.debug("Tool unregistered: '%s'.", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool:
        """Return the tool identified by *name*.

        Raises
        ------
        KeyError
            When the tool does not exist.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, str]]:
        """Return a summary list of all registered tools.

        Each entry is a dict with ``name``, ``description``, and
        ``parameters`` keys.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": str(t.parameters),
            }
            for t in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolRegistry(tools={list(self._tools)})"
