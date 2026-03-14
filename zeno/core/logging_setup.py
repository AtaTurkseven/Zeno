"""Logging initialisation helper for the Zeno system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger with a console handler and optional file handler.

    Parameters
    ----------
    level:
        Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
    log_file:
        Optional path to a log file.  The parent directory is created
        automatically if it does not already exist.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger(__name__).debug("Logging initialised at level %s", level)
