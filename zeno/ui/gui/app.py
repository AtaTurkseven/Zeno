"""GUIApp — graphical user interface for Zeno.

.. todo::
    Implement a full GUI using Tkinter, PyQt6, or a web-based framework
    (e.g. Gradio or Streamlit) once the hardware control layer is stable.

The class below provides the interface contract so that ``main.py`` can
reference it without any GUI dependency being hard-required at runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zeno.core.ai.base import AIBase
    from zeno.core.memory.vector_store import VectorStore
    from zeno.core.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class GUIApp:
    """Placeholder for the Zeno graphical user interface.

    Parameters
    ----------
    ai_backend:
        Active AI backend.
    tool_executor:
        Tool executor instance.
    vector_store:
        Vector memory store.
    config:
        Optional UI configuration dict.
    """

    def __init__(
        self,
        ai_backend: AIBase,
        tool_executor: ToolExecutor,
        vector_store: VectorStore,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._ai = ai_backend
        self._executor = tool_executor
        self._memory = vector_store
        self._config = config or {}
        logger.debug("GUIApp initialised (stub).")

    def run(self) -> None:
        """Launch the GUI event loop.

        .. todo::
            Implement with a chosen GUI framework.
        """
        # TODO: Build and launch the GUI.
        logger.warning(
            "GUIApp.run() called but the GUI is not yet implemented.  "
            "Falling back to CLI mode."
        )
        from zeno.ui.cli.shell import CLIShell

        shell = CLIShell(
            ai_backend=self._ai,
            tool_executor=self._executor,
            vector_store=self._memory,
            config=self._config,
        )
        shell.run()
