"""CLIShell — interactive command-line interface for the Zeno assistant.

The shell provides a REPL where the user can:

- Type free-form messages that are forwarded to the active AI backend.
- Enter slash-commands (``/tool``, ``/memory``, ``/status``, …) to interact
  with subsystems directly.
- Exit with ``/quit`` or Ctrl-D.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zeno.core.ai.base import AIBase
    from zeno.core.memory.vector_store import VectorStore
    from zeno.core.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════╗
║          Zeno AI Assistant           ║
║   Type /help for available commands  ║
╚══════════════════════════════════════╝
"""

_HELP_TEXT = """
Available commands:
  /help              Show this help message
  /status            Show subsystem status
  /tool <name> [k=v] Run a registered tool
  /memory <query>    Search vector memory
  /quit              Exit the shell

Any other input is sent to the AI backend.
"""


class CLIShell:
    """Interactive REPL that connects the user to Zeno's subsystems.

    Parameters
    ----------
    ai_backend:
        The active AI backend (implements :class:`~zeno.core.ai.base.AIBase`).
    tool_executor:
        Executor used to run registered tools.
    vector_store:
        Vector memory store for episodic memory search.
    config:
        Optional UI config dict (currently unused; reserved for future options).
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
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the REPL loop.  Blocks until the user exits."""
        print(_BANNER, flush=True)
        self._running = True
        logger.info("CLIShell started.")

        while self._running:
            try:
                user_input = input("zeno> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting Zeno. Goodbye!", flush=True)
                self._running = False
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
            else:
                self._handle_ai_query(user_input)

        logger.info("CLIShell stopped.")

    def stop(self) -> None:
        """Signal the REPL loop to exit after the current iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> None:
        """Parse and dispatch a slash-command."""
        parts = raw.lstrip("/").split()
        if not parts:
            return
        cmd, *args = parts

        dispatch = {
            "help": self._cmd_help,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
            "status": self._cmd_status,
            "tool": self._cmd_tool,
            "memory": self._cmd_memory,
        }

        handler = dispatch.get(cmd.lower())
        if handler is None:
            print(f"Unknown command '/{cmd}'.  Type /help for a list.", flush=True)
        else:
            handler(args)

    def _handle_ai_query(self, text: str) -> None:
        """Forward *text* to the AI backend and print the response."""
        if not self._ai.is_available():
            print(
                "[AI backend unavailable — check your configuration.]", flush=True
            )
            return
        try:
            response = self._ai.generate(text)
            print(f"Zeno: {response}", flush=True)
        except RuntimeError as exc:
            print(f"[Error from AI backend: {exc}]", flush=True)
            logger.error("AI generation error: %s", exc)

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    def _cmd_help(self, _args: list[str]) -> None:
        print(_HELP_TEXT, flush=True)

    def _cmd_quit(self, _args: list[str]) -> None:
        print("Exiting Zeno. Goodbye!", flush=True)
        self._running = False

    def _cmd_status(self, _args: list[str]) -> None:
        ai_ok = self._ai.is_available()
        mem_size = self._memory.size
        print(
            f"  AI backend : {'✓ available' if ai_ok else '✗ unavailable'}\n"
            f"  Memory     : {mem_size} entries",
            flush=True,
        )

    def _cmd_tool(self, args: list[str]) -> None:
        """Run a registered tool.  Usage: ``/tool <name> [key=value ...]``"""
        if not args:
            print("Usage: /tool <name> [key=value ...]", flush=True)
            return
        tool_name, *kv_pairs = args
        kwargs: dict[str, Any] = {}
        for pair in kv_pairs:
            if "=" in pair:
                k, v = pair.split("=", 1)
                kwargs[k] = v
            else:
                print(f"Skipping malformed argument: {pair!r}", flush=True)

        result = self._executor.run(tool_name, **kwargs)
        if result.success:
            print(f"Tool '{tool_name}' result: {result.output}", flush=True)
        else:
            print(f"Tool '{tool_name}' error: {result.error}", flush=True)

    def _cmd_memory(self, args: list[str]) -> None:
        """Search vector memory.  Usage: ``/memory <query text>``"""
        if not args:
            print("Usage: /memory <query text>", flush=True)
            return
        # Without a real embedding model we search by simple text match.
        query = " ".join(args)
        import numpy as np  # local import to avoid hard dep at module level

        dim = self._memory.embedding_dim
        query_vec = np.zeros(dim, dtype=np.float32)
        results = self._memory.search(query_vec, top_k=5)
        if not results:
            print("No memory entries found.", flush=True)
            return
        for entry, score in results:
            print(f"  [{score:.3f}] {entry.text[:80]}", flush=True)
