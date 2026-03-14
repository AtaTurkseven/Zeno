"""ESP32Controller — high-level driver for an ESP32 microcontroller.

Commands are sent as newline-terminated ASCII strings over a serial UART link
and responses are read back as single lines.  The ESP32 firmware is expected
to accept human-readable commands such as ``"LED ON"`` or ``"READ PIN 34"``.
"""

from __future__ import annotations

import logging
from typing import Any

from zeno.core.hardware.serial_device import SerialDevice

logger = logging.getLogger(__name__)


class ESP32Controller(SerialDevice):
    """Communicates with an ESP32 over a USB-serial (UART) connection.

    Parameters
    ----------
    config:
        Dict with keys ``port``, ``baud_rate``, and ``timeout``.
        Typically sourced from ``devices.esp32`` in ``settings.yaml``.
    device_id:
        Optional label (default: ``"esp32-0"``).
    """

    def __init__(
        self, config: dict[str, Any], device_id: str = "esp32-0"
    ) -> None:
        super().__init__(device_id, config)

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def send_command(self, command: str) -> str:
        """Send a text *command* and return the single-line response.

        Parameters
        ----------
        command:
            ASCII command string (without line terminator).

        Returns
        -------
        str
            The response line from the firmware, or ``""`` on error.
        """
        if not self.is_connected:
            logger.error(
                "ESP32Controller '%s': send_command() called while disconnected.",
                self.device_id,
            )
            return ""

        self._logger.debug("ESP32 → %r", command)
        if not self.send_line(command):
            return ""

        response = self.receive_line()
        self._logger.debug("ESP32 ← %r", response)
        return response

    def set_led(self, state: bool) -> bool:
        """Turn the built-in LED on or off.

        Parameters
        ----------
        state:
            ``True`` to switch on, ``False`` to switch off.

        Returns
        -------
        bool
            ``True`` when the firmware acknowledged the command.
        """
        cmd = "LED ON" if state else "LED OFF"
        resp = self.send_command(cmd)
        success = resp.strip().upper() in {"OK", "ACK"}
        self._logger.info("set_led(%s) → %r (success=%s)", state, resp, success)
        return success

    def read_pin(self, pin: int) -> int | None:
        """Read the ADC value of a GPIO *pin*.

        Parameters
        ----------
        pin:
            GPIO pin number on the ESP32 (e.g. ``34`` for ADC1_CH6).

        Returns
        -------
        int or None
            ADC reading (0–4095), or ``None`` on error.
        """
        resp = self.send_command(f"READ PIN {pin}")
        try:
            return int(resp.strip())
        except ValueError:
            self._logger.warning(
                "ESP32 read_pin(%d) got unexpected response: %r", pin, resp
            )
            return None

    def reset(self) -> bool:
        """Send a software-reset command to the ESP32.

        Returns
        -------
        bool
            ``True`` when the firmware acknowledged the reset.
        """
        resp = self.send_command("RESET")
        return resp.strip().upper() in {"OK", "ACK", "RESET"}
