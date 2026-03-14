"""Tests for the tools subsystem (BaseTool, ToolRegistry, ToolExecutor)."""

from __future__ import annotations

from typing import Any

import pytest

from zeno.core.tools.base import BaseTool, ToolResult
from zeno.core.tools.executor import ToolExecutor
from zeno.core.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Concrete tool implementations for testing
# ---------------------------------------------------------------------------

class _EchoTool(BaseTool):
    """Returns the input text as its output."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo input back.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {"text": {"type": "string"}}

    def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        return ToolResult(success=True, output=text)


class _FailingTool(BaseTool):
    """Always raises an exception."""

    def __init__(self) -> None:
        super().__init__(name="fail", description="Always fails.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("Intentional failure")


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_EchoTool())

    def test_contains(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        assert "echo" in registry
        assert "unknown" not in registry

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        registry.unregister("echo")
        assert "echo" not in registry

    def test_unregister_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    def test_len_and_iter(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        assert len(registry) == 1
        names = [t.name for t in registry]
        assert names == ["echo"]


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------

class TestToolExecutor:
    def _executor(self, *tools: BaseTool) -> ToolExecutor:
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        return ToolExecutor(registry)

    def test_run_success(self) -> None:
        executor = self._executor(_EchoTool())
        result = executor.run("echo", text="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_run_unregistered_tool(self) -> None:
        executor = self._executor()
        result = executor.run("missing")
        assert result.success is False
        assert "not registered" in result.error

    def test_run_tool_raises_exception(self) -> None:
        executor = self._executor(_FailingTool())
        result = executor.run("fail")
        assert result.success is False
        assert "Intentional failure" in result.error

    def test_run_adds_elapsed_metadata(self) -> None:
        executor = self._executor(_EchoTool())
        result = executor.run("echo", text="timing")
        assert "elapsed_s" in result.metadata
        assert result.metadata["elapsed_s"] >= 0.0
