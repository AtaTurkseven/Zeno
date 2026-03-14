"""Tests for built-in tools: CalculatorTool, TimeTool, EchoTool."""

from __future__ import annotations

import math
import re

import pytest

from zeno.core.tools.builtin import CalculatorTool, EchoTool, TimeTool


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def _tool(self) -> CalculatorTool:
        return CalculatorTool()

    def test_addition(self) -> None:
        result = self._tool().execute(expression="2 + 3")
        assert result.success is True
        assert result.output == 5.0

    def test_subtraction(self) -> None:
        result = self._tool().execute(expression="10 - 4")
        assert result.success is True
        assert result.output == 6.0

    def test_multiplication(self) -> None:
        result = self._tool().execute(expression="3 * 7")
        assert result.success is True
        assert result.output == 21.0

    def test_division(self) -> None:
        result = self._tool().execute(expression="10 / 4")
        assert result.success is True
        assert result.output == 2.5

    def test_floor_division(self) -> None:
        result = self._tool().execute(expression="10 // 3")
        assert result.success is True
        assert result.output == 3.0

    def test_modulo(self) -> None:
        result = self._tool().execute(expression="10 % 3")
        assert result.success is True
        assert result.output == 1.0

    def test_power(self) -> None:
        result = self._tool().execute(expression="2 ** 10")
        assert result.success is True
        assert result.output == 1024.0

    def test_negative_unary(self) -> None:
        result = self._tool().execute(expression="-5 + 10")
        assert result.success is True
        assert result.output == 5.0

    def test_pi_constant(self) -> None:
        result = self._tool().execute(expression="pi")
        assert result.success is True
        assert abs(result.output - math.pi) < 1e-9

    def test_sqrt_function(self) -> None:
        result = self._tool().execute(expression="sqrt(16)")
        assert result.success is True
        assert result.output == 4.0

    def test_nested_expression(self) -> None:
        result = self._tool().execute(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.output == 20.0

    def test_missing_expression(self) -> None:
        result = self._tool().execute()
        assert result.success is False
        assert "expression" in result.error.lower()

    def test_invalid_syntax(self) -> None:
        result = self._tool().execute(expression="2 +* 3")
        assert result.success is False

    def test_division_by_zero(self) -> None:
        result = self._tool().execute(expression="1 / 0")
        assert result.success is False

    def test_function_name_without_call_fails(self) -> None:
        result = self._tool().execute(expression="sqrt")
        assert result.success is False
        assert "function" in result.error.lower()

    def test_disallowed_import(self) -> None:
        result = self._tool().execute(expression="__import__('os')")
        assert result.success is False

    def test_disallowed_name(self) -> None:
        result = self._tool().execute(expression="__builtins__")
        assert result.success is False

    def test_metadata_contains_expression(self) -> None:
        result = self._tool().execute(expression="1 + 1")
        assert result.metadata.get("expression") == "1 + 1"

    def test_tool_name_and_description(self) -> None:
        tool = self._tool()
        assert tool.name == "calculator"
        assert tool.description

    def test_parameters_schema(self) -> None:
        params = self._tool().parameters
        assert "expression" in params


# ---------------------------------------------------------------------------
# TimeTool
# ---------------------------------------------------------------------------

class TestTimeTool:
    def _tool(self) -> TimeTool:
        return TimeTool()

    def test_datetime_default(self) -> None:
        result = self._tool().execute()
        assert result.success is True
        # Should match YYYY-MM-DD HH:MM:SS
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.output)

    def test_date_format(self) -> None:
        result = self._tool().execute(format="date")
        assert result.success is True
        assert re.match(r"\d{4}-\d{2}-\d{2}", result.output)
        assert len(result.output) == 10

    def test_time_format(self) -> None:
        result = self._tool().execute(format="time")
        assert result.success is True
        assert re.match(r"\d{2}:\d{2}:\d{2}", result.output)

    def test_datetime_format_explicit(self) -> None:
        result = self._tool().execute(format="datetime")
        assert result.success is True
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.output)

    def test_unknown_format_falls_back_to_datetime(self) -> None:
        result = self._tool().execute(format="unknown")
        assert result.success is True
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.output)

    def test_metadata_contains_iso(self) -> None:
        result = self._tool().execute()
        assert "iso" in result.metadata

    def test_tool_name(self) -> None:
        assert self._tool().name == "time"

    def test_parameters_schema(self) -> None:
        params = self._tool().parameters
        assert "format" in params


# ---------------------------------------------------------------------------
# EchoTool
# ---------------------------------------------------------------------------

class TestEchoTool:
    def _tool(self) -> EchoTool:
        return EchoTool()

    def test_echo_returns_message(self) -> None:
        result = self._tool().execute(message="hello world")
        assert result.success is True
        assert result.output == "hello world"

    def test_echo_empty_string(self) -> None:
        result = self._tool().execute(message="")
        assert result.success is True
        assert result.output == ""

    def test_echo_missing_message(self) -> None:
        result = self._tool().execute()
        assert result.success is False
        assert "message" in result.error.lower()

    def test_tool_name(self) -> None:
        assert self._tool().name == "echo"

    def test_parameters_schema(self) -> None:
        params = self._tool().parameters
        assert "message" in params
