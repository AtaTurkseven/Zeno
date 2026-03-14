"""ConfigManager — loads and provides access to YAML-based configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


class ConfigManager:
    """Loads settings from a YAML file and exposes them as nested attribute access.

    Parameters
    ----------
    config_path:
        Path to the YAML settings file.  Defaults to ``config/settings.yaml``
        at the project root.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """(Re-)load configuration from disk."""
        if not self._path.exists():
            logger.warning("Config file not found at %s — using empty config.", self._path)
            self._data = {}
            return

        with self._path.open("r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh) or {}

        logger.debug("Configuration loaded from %s", self._path)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level config value by key.

        Parameters
        ----------
        key:
            Dot-separated path, e.g. ``"ai.backend"``.
        default:
            Value to return when the key is absent.
        """
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_section(self, section: str) -> dict[str, Any]:
        """Return an entire top-level section as a dict.

        Parameters
        ----------
        section:
            Top-level section name (e.g. ``"ai"``).
        """
        return dict(self._data.get(section, {}))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConfigManager(path={self._path})"
