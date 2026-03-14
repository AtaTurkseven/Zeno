"""Tests for device drivers (ESP32Controller, RoboticArm, SensorManager)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zeno.devices.sensors.sensor_manager import SensorManager


# ---------------------------------------------------------------------------
# SensorManager tests (no hardware dependency)
# ---------------------------------------------------------------------------

class TestSensorManager:
    def test_register_and_read(self) -> None:
        mgr = SensorManager()
        mgr.register("temp", lambda: 22.5)
        assert mgr.read("temp") == pytest.approx(22.5)

    def test_register_duplicate_raises(self) -> None:
        mgr = SensorManager()
        mgr.register("temp", lambda: 0)
        with pytest.raises(ValueError, match="already registered"):
            mgr.register("temp", lambda: 1)

    def test_unregister(self) -> None:
        mgr = SensorManager()
        mgr.register("temp", lambda: 0)
        mgr.unregister("temp")
        assert "temp" not in mgr.list_sensors()

    def test_unregister_missing_raises(self) -> None:
        mgr = SensorManager()
        with pytest.raises(KeyError):
            mgr.unregister("nonexistent")

    def test_read_missing_raises(self) -> None:
        mgr = SensorManager()
        with pytest.raises(KeyError):
            mgr.read("nonexistent")

    def test_read_all(self) -> None:
        mgr = SensorManager()
        mgr.register("a", lambda: 1)
        mgr.register("b", lambda: 2)
        readings = mgr.read_all()
        assert readings == {"a": 1, "b": 2}

    def test_error_returns_none(self) -> None:
        mgr = SensorManager()
        mgr.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = mgr.read("bad")
        assert result is None

    def test_last_reading_cached(self) -> None:
        mgr = SensorManager()
        calls = [42]
        mgr.register("s", lambda: calls[0])
        mgr.read("s")
        calls[0] = 99  # change what the sensor would return
        assert mgr.last_reading("s") == 42  # cached value

    def test_list_sensors(self) -> None:
        mgr = SensorManager()
        mgr.register("x", lambda: 0)
        mgr.register("y", lambda: 0)
        assert set(mgr.list_sensors()) == {"x", "y"}

    def test_len(self) -> None:
        mgr = SensorManager()
        mgr.register("a", lambda: 0)
        assert len(mgr) == 1


# ---------------------------------------------------------------------------
# RoboticArm — unit tests with mocked serial port
# ---------------------------------------------------------------------------

class TestRoboticArm:
    _cfg = {"port": "/dev/null", "baud_rate": 9600, "timeout": 1, "dof": 3}

    def _make_arm(self):
        from zeno.devices.robotic_arm.arm import RoboticArm

        arm = RoboticArm(self._cfg)
        # Patch the serial module so no real port is opened
        arm._serial = MagicMock()
        arm._serial.is_open = True
        arm._status = arm._status.__class__.CONNECTED  # DeviceStatus.CONNECTED
        return arm

    def test_dof(self) -> None:
        from zeno.devices.robotic_arm.arm import RoboticArm

        arm = RoboticArm(self._cfg)
        assert arm.dof == 3

    def test_move_joint_out_of_range_raises(self) -> None:
        arm = self._make_arm()
        with pytest.raises(IndexError):
            arm.move_joint(5, 90.0)

    def test_joint_angles_default_to_90(self) -> None:
        from zeno.devices.robotic_arm.arm import RoboticArm

        arm = RoboticArm(self._cfg)
        assert arm.joint_angles == [90.0, 90.0, 90.0]


# ---------------------------------------------------------------------------
# ESP32Controller — unit tests with mocked serial port
# ---------------------------------------------------------------------------

class TestESP32Controller:
    _cfg = {"port": "/dev/null", "baud_rate": 115200, "timeout": 1}

    def _make_ctrl(self):
        from zeno.devices.esp32.controller import ESP32Controller

        ctrl = ESP32Controller(self._cfg)
        ctrl._serial = MagicMock()
        ctrl._serial.is_open = True
        from zeno.core.hardware.base import DeviceStatus

        ctrl._status = DeviceStatus.CONNECTED
        return ctrl

    def test_send_command_returns_response(self) -> None:
        ctrl = self._make_ctrl()
        # side_effect ensures write() always reports all bytes written
        ctrl._serial.write = MagicMock(side_effect=lambda data: len(data))
        ctrl._serial.readline = MagicMock(return_value=b"OK\r\n")
        resp = ctrl.send_command("TEST")
        assert resp == "OK"

    def test_set_led_on(self) -> None:
        ctrl = self._make_ctrl()
        ctrl._serial.write = MagicMock(side_effect=lambda data: len(data))
        ctrl._serial.readline = MagicMock(return_value=b"OK\r\n")
        assert ctrl.set_led(True) is True

    def test_read_pin_returns_int(self) -> None:
        ctrl = self._make_ctrl()
        ctrl._serial.write = MagicMock(side_effect=lambda data: len(data))
        ctrl._serial.readline = MagicMock(return_value=b"2048\r\n")
        val = ctrl.read_pin(34)
        assert val == 2048

    def test_read_pin_bad_response_returns_none(self) -> None:
        ctrl = self._make_ctrl()
        ctrl._serial.write = MagicMock(side_effect=lambda data: len(data))
        ctrl._serial.readline = MagicMock(return_value=b"ERROR\r\n")
        assert ctrl.read_pin(34) is None
