"""Tests for ConfigManager."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zeno.config.manager import ConfigManager


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal YAML config and return its path."""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """\
            system:
              name: ZenoTest
              version: "0.0.1"
              log_level: DEBUG
            ai:
              backend: local
              local:
                model: llama3
                host: http://localhost:11434
                timeout: 10
            memory:
              embedding_dim: 8
              max_entries: 100
            """
        )
    )
    return cfg_file


def test_load_existing_config(tmp_config: Path) -> None:
    cm = ConfigManager(config_path=tmp_config)
    assert cm.get("system.name") == "ZenoTest"
    assert cm.get("system.version") == "0.0.1"


def test_get_nested_key(tmp_config: Path) -> None:
    cm = ConfigManager(config_path=tmp_config)
    assert cm.get("ai.backend") == "local"
    assert cm.get("ai.local.model") == "llama3"


def test_get_default_when_missing(tmp_config: Path) -> None:
    cm = ConfigManager(config_path=tmp_config)
    assert cm.get("does.not.exist", "fallback") == "fallback"


def test_get_section(tmp_config: Path) -> None:
    cm = ConfigManager(config_path=tmp_config)
    section = cm.get_section("memory")
    assert section["embedding_dim"] == 8
    assert section["max_entries"] == 100


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    cm = ConfigManager(config_path=tmp_path / "nonexistent.yaml")
    assert cm.get("anything") is None
    assert cm.get_section("anything") == {}


def test_reload(tmp_config: Path) -> None:
    cm = ConfigManager(config_path=tmp_config)
    assert cm.get("system.name") == "ZenoTest"
    # Overwrite the file and reload
    tmp_config.write_text("system:\n  name: ZenoUpdated\n")
    cm.load()
    assert cm.get("system.name") == "ZenoUpdated"
