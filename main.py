"""Zeno — main entry point.

Bootstraps the system by:

1. Loading configuration from ``config/settings.yaml``.
2. Setting up logging.
3. Constructing the AI backend, memory store, tool registry, and executor.
4. Launching the selected UI (CLI or GUI).

Run with::

    python main.py [--config path/to/settings.yaml] [--ui cli|gui]
"""

from __future__ import annotations

import argparse
import logging
import sys

from zeno.config.manager import ConfigManager
from zeno.core.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _build_ai_backend(config: ConfigManager):
    """Instantiate and return the configured AI backend."""
    backend_name: str = config.get("ai.backend", "local")

    if backend_name == "local":
        from zeno.core.ai.local_llm import LocalLLM

        cfg = config.get_section("ai").get("local", {})
        return LocalLLM(cfg)

    if backend_name == "cloud":
        from zeno.core.ai.cloud_ai import CloudAI

        cfg = config.get_section("ai").get("cloud", {})
        return CloudAI(cfg)

    raise ValueError(
        f"Unknown AI backend '{backend_name}'.  "
        "Choose 'local' or 'cloud' in settings.yaml."
    )


def _build_memory(config: ConfigManager):
    """Instantiate and return the vector memory store."""
    from zeno.core.memory.vector_store import VectorStore

    mem_cfg = config.get_section("memory")
    return VectorStore(
        embedding_dim=int(mem_cfg.get("embedding_dim", 384)),
        max_entries=int(mem_cfg.get("max_entries", 10_000)),
    )


def _build_tool_system(config: ConfigManager):
    """Build and return ``(ToolRegistry, ToolExecutor)`` with built-in tools."""
    from zeno.core.tools.executor import ToolExecutor
    from zeno.core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    # TODO: Register domain-specific tools here.
    return registry, executor


def _build_ui(
    ui_mode: str,
    ai_backend,
    tool_executor,
    vector_store,
    config: ConfigManager,
):
    """Return the UI instance for the requested *ui_mode*."""
    ui_cfg = config.get_section("ui")

    if ui_mode == "cli":
        from zeno.ui.cli.shell import CLIShell

        return CLIShell(ai_backend, tool_executor, vector_store, ui_cfg)

    if ui_mode == "gui":
        from zeno.ui.gui.app import GUIApp

        return GUIApp(ai_backend, tool_executor, vector_store, ui_cfg)

    raise ValueError(
        f"Unknown UI mode '{ui_mode}'.  Choose 'cli' or 'gui' in settings.yaml."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="zeno",
        description="Zeno — modular physical AI assistant.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a YAML settings file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--ui",
        choices=["cli", "gui"],
        default=None,
        help="UI mode override (default: value from settings.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Bootstrap and run Zeno.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 on clean exit, non-zero on fatal error).
    """
    args = parse_args(argv)

    # --- Configuration -------------------------------------------------
    config = ConfigManager(config_path=args.config)

    # --- Logging -------------------------------------------------------
    setup_logging(
        level=config.get("system.log_level", "INFO"),
        log_file=config.get("system.log_file"),
    )
    logger.info("Zeno %s starting up.", config.get("system.version", "?"))

    # --- Subsystems ----------------------------------------------------
    try:
        ai_backend = _build_ai_backend(config)
        ai_backend.initialize()
        logger.info("AI backend: %s", ai_backend.__class__.__name__)

        vector_store = _build_memory(config)
        logger.info("Vector memory: %s", vector_store)

        _registry, tool_executor = _build_tool_system(config)
        logger.info("Tool registry ready.")

        # --- UI --------------------------------------------------------
        ui_mode: str = args.ui or config.get("ui.mode", "cli")
        ui = _build_ui(ui_mode, ai_backend, tool_executor, vector_store, config)
        logger.info("UI mode: %s", ui_mode)

        ui.run()

        ai_backend.shutdown()
        logger.info("Zeno shut down cleanly.")
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error during startup: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
