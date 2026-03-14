"""Abstract base class and result type for Zeno tools."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Encapsulates the outcome of running a tool.

    Attributes
    ----------
    success:
        ``True`` when the tool completed without error.
    output:
        Primary result value (string, dict, or any serialisable type).
    error:
        Human-readable error message when *success* is ``False``.
    metadata:
        Optional extra information (timing, raw response, …).
    """

    success: bool
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Contract that every Zeno tool must satisfy.

    A *tool* is a callable unit of work that the AI or the user can
    invoke by name.  Subclasses implement :meth:`execute` with their
    specific logic.

    Parameters
    ----------
    name:
        Unique identifier used to look up the tool in the registry.
    description:
        One-line human-readable summary shown to the AI and the user.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments.

        Parameters
        ----------
        **kwargs:
            Tool-specific parameters.

        Returns
        -------
        ToolResult
            Always returns a :class:`ToolResult`; never raises.
        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return a JSON-Schema-style description of accepted parameters.

        Example::

            {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            }
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
