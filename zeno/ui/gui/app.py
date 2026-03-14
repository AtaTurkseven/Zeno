"""GUIApp — graphical user interface for Zeno built with Tkinter.

The GUI provides a chat-style window where the user can:

- Type messages and receive AI responses.
- Run slash-commands (``/tool``, ``/status``, ``/memory``, ``/help``).
- View a status bar showing AI availability and memory size.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import tkinter
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

if TYPE_CHECKING:
    from zeno.core.ai.base import AIBase
    from zeno.core.memory.vector_store import VectorStore
    from zeno.core.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

_HELP_TEXT = (
    "Commands:\n"
    "  /help              Show this message\n"
    "  /status            Show subsystem status\n"
    "  /tool <name> [k=v] Run a registered tool\n"
    "  /memory <query>    Search vector memory\n"
    "  /quit              Exit the application\n\n"
    "Any other text is sent to the AI backend."
)


class GUIApp:
    """Graphical chat interface for Zeno built with Tkinter.

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
        ai_backend: "AIBase",
        tool_executor: "ToolExecutor",
        vector_store: "VectorStore",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._ai = ai_backend
        self._executor = tool_executor
        self._memory = vector_store
        self._config = config or {}
        logger.debug("GUIApp initialised.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the Tkinter window and start the event loop."""
        if not _TK_AVAILABLE:
            logger.error("Tkinter is not available.  Falling back to CLI mode.")
            self._fallback_cli()
            return

        from tkinter import scrolledtext

        root = tkinter.Tk()
        root.title("Zeno AI Assistant")
        root.geometry("800x600")
        root.minsize(500, 400)

        self._build_ui(root, tkinter, scrolledtext)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, root: Any, tk: Any, scrolledtext: Any) -> None:
        """Populate *root* with all widgets."""
        # ── Conversation display ──────────────────────────────────────
        self._chat = scrolledtext.ScrolledText(
            root,
            state="disabled",
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        self._chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # Tag colours for different speakers
        self._chat.tag_config("user", foreground="#9cdcfe")
        self._chat.tag_config("zeno", foreground="#4ec9b0")
        self._chat.tag_config("system", foreground="#ce9178")
        self._chat.tag_config("error", foreground="#f48771")

        # ── Status bar ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value=self._status_text())
        status_bar = tk.Label(
            root,
            textvariable=self._status_var,
            anchor="w",
            font=("Helvetica", 9),
            bg="#252526",
            fg="#858585",
            padx=6,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Input row ─────────────────────────────────────────────────
        input_frame = tk.Frame(root, bg="#252526")
        input_frame.pack(fill=tk.X, padx=8, pady=6)

        self._input_var = tk.StringVar()
        entry = tk.Entry(
            input_frame,
            textvariable=self._input_var,
            font=("Helvetica", 11),
            bg="#3c3c3c",
            fg="#d4d4d4",
            insertbackground="white",
            relief=tk.FLAT,
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))
        entry.bind("<Return>", lambda _e: self._on_send())
        entry.focus_set()

        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self._on_send,
            font=("Helvetica", 10, "bold"),
            bg="#0e639c",
            fg="white",
            activebackground="#1177bb",
            relief=tk.FLAT,
            padx=12,
            pady=6,
        )
        send_btn.pack(side=tk.RIGHT)

        # ── Welcome message ───────────────────────────────────────────
        self._append("Zeno AI Assistant — type /help for commands.\n", "system")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        """Handle the Send button / Return key."""
        text = self._input_var.get().strip()
        if not text:
            return
        self._input_var.set("")
        self._append(f"You: {text}\n", "user")

        if text.startswith("/"):
            self._handle_command(text)
        else:
            threading.Thread(
                target=self._handle_ai_query, args=(text,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Command handling (mirrors CLIShell logic)
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> None:
        parts = raw.lstrip("/").split()
        if not parts:
            return
        cmd, *args = parts

        handlers = {
            "help": lambda: self._append(_HELP_TEXT + "\n", "system"),
            "quit": self._do_quit,
            "exit": self._do_quit,
            "status": lambda: self._append(self._status_text() + "\n", "system"),
            "tool": lambda: self._cmd_tool(args),
            "memory": lambda: self._cmd_memory(args),
        }
        handler = handlers.get(cmd.lower())
        if handler is None:
            self._append(
                f"Unknown command '/{cmd}'.  Type /help for a list.\n", "error"
            )
        else:
            handler()

    def _do_quit(self) -> None:
        self._append("Exiting Zeno. Goodbye!\n", "system")
        # Schedule destroy so the message appears first
        root = self._chat.winfo_toplevel()
        root.after(200, root.destroy)

    def _handle_ai_query(self, text: str) -> None:
        if not self._ai.is_available():
            self._append(
                "[AI backend unavailable — check your configuration.]\n", "error"
            )
            return
        try:
            response = self._ai.generate(text)
            self._append(f"Zeno: {response}\n", "zeno")
        except RuntimeError as exc:
            self._append(f"[Error from AI backend: {exc}]\n", "error")
            logger.error("AI generation error: %s", exc)

    def _cmd_tool(self, args: list[str]) -> None:
        if not args:
            self._append("Usage: /tool <name> [key=value …]\n", "error")
            return
        tool_name, *kv_pairs = args
        kwargs: dict[str, Any] = {}
        for pair in kv_pairs:
            if "=" in pair:
                k, v = pair.split("=", 1)
                kwargs[k] = v

        result = self._executor.run(tool_name, **kwargs)
        if result.success:
            self._append(f"Tool '{tool_name}': {result.output}\n", "system")
        else:
            self._append(f"Tool '{tool_name}' error: {result.error}\n", "error")

    def _cmd_memory(self, args: list[str]) -> None:
        if not args:
            self._append("Usage: /memory <query text>\n", "error")
            return
        dim = self._memory.embedding_dim
        query_vec = np.zeros(dim, dtype=np.float32)
        results = self._memory.search(query_vec, top_k=5)
        if not results:
            self._append("No memory entries found.\n", "system")
            return
        for entry, score in results:
            self._append(f"  [{score:.3f}] {entry.text[:80]}\n", "system")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append(self, text: str, tag: str = "") -> None:
        """Append *text* to the chat display (thread-safe)."""
        def _do() -> None:
            self._chat.configure(state="normal")
            if tag:
                self._chat.insert("end", text, tag)
            else:
                self._chat.insert("end", text)
            self._chat.configure(state="disabled")
            self._chat.see("end")
            self._status_var.set(self._status_text())

        try:
            self._chat.after(0, _do)
        except tkinter.TclError:
            # Widget has been destroyed; discard the message silently.
            pass
        except Exception:  # noqa: BLE001
            logger.debug("GUIApp._append(): unexpected error scheduling update.")
            pass

    def _status_text(self) -> str:
        ai_ok = self._ai.is_available()
        mem_size = self._memory.size
        ai_label = "✓ AI available" if ai_ok else "✗ AI unavailable"
        return f"  {ai_label}   |   Memory: {mem_size} entries"

    def _fallback_cli(self) -> None:
        from zeno.ui.cli.shell import CLIShell

        shell = CLIShell(
            ai_backend=self._ai,
            tool_executor=self._executor,
            vector_store=self._memory,
            config=self._config,
        )
        shell.run()

