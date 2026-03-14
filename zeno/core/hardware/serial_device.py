"""SerialDevice — base class for devices connected over a serial port."""

from __future__ import annotations

import logging
from typing import Any

import serial  # pyserial

from zeno.core.hardware.base import BaseDevice, DeviceStatus

logger = logging.getLogger(__name__)


class SerialDevice(BaseDevice):
    """Implements :class:`BaseDevice` over a UART / USB-serial link.

    Parameters
    ----------
    device_id:
        Human-readable label (e.g. ``"esp32-0"``).
    config:
        Dict with keys:

        - ``port`` — serial port path (e.g. ``/dev/ttyUSB0``)
        - ``baud_rate`` — baud rate (default ``115200``)
        - ``timeout`` — read timeout in seconds (default ``1``)
    """

    def __init__(self, device_id: str, config: dict[str, Any]) -> None:
        super().__init__(device_id, config)
        self._port: str = config.get("port", "/dev/ttyUSB0")
        self._baud_rate: int = int(config.get("baud_rate", 115_200))
        self._timeout: float = float(config.get("timeout", 1.0))
        self._serial: serial.Serial | None = None

    # ------------------------------------------------------------------
    # BaseDevice implementation
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the serial port.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if an error occurs.
        """
        self._set_status(DeviceStatus.CONNECTING)
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=self._timeout,
            )
            self._set_status(DeviceStatus.CONNECTED)
            self._logger.info(
                "SerialDevice '%s' connected on %s @ %d baud.",
                self.device_id,
                self._port,
                self._baud_rate,
            )
            return True
        except serial.SerialException as exc:
            self._set_status(DeviceStatus.ERROR)
            self._logger.error(
                "SerialDevice '%s' failed to connect: %s", self.device_id, exc
            )
            return False

    def disconnect(self) -> None:
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None
        self._set_status(DeviceStatus.DISCONNECTED)
        self._logger.info("SerialDevice '%s' disconnected.", self.device_id)

    def send(self, data: bytes) -> bool:
        """Write *data* to the serial port.

        Returns
        -------
        bool
            ``True`` when all bytes were written successfully.
        """
        if not self.is_connected or self._serial is None:
            self._logger.warning(
                "SerialDevice '%s' send() called while not connected.", self.device_id
            )
            return False
        try:
            written = self._serial.write(data)
            return written == len(data)
        except serial.SerialException as exc:
            self._logger.error("SerialDevice '%s' send error: %s", self.device_id, exc)
            self._set_status(DeviceStatus.ERROR)
            return False

    def receive(self, num_bytes: int = 256) -> bytes:
        """Read up to *num_bytes* from the serial port.

        Returns
        -------
        bytes
            Received data (may be empty if nothing arrived within timeout).
        """
        if not self.is_connected or self._serial is None:
            self._logger.warning(
                "SerialDevice '%s' receive() called while not connected.", self.device_id
            )
            return b""
        try:
            return self._serial.read(num_bytes)
        except serial.SerialException as exc:
            self._logger.error(
                "SerialDevice '%s' receive error: %s", self.device_id, exc
            )
            self._set_status(DeviceStatus.ERROR)
            return b""

    def send_line(self, text: str, encoding: str = "utf-8") -> bool:
        """Send *text* as a UTF-8 line terminated with ``\\r\\n``.

        Parameters
        ----------
        text:
            Command or message string (without line terminator).
        encoding:
            Character encoding to use (default ``"utf-8"``).
        """
        return self.send((text + "\r\n").encode(encoding))

    def receive_line(self, encoding: str = "utf-8") -> str:
        """Read one ``\\n``-terminated line from the device.

        Returns
        -------
        str
            Decoded line (stripped of trailing whitespace).
        """
        if not self.is_connected or self._serial is None:
            return ""
        try:
            raw = self._serial.readline()
            return raw.decode(encoding, errors="replace").rstrip()
        except serial.SerialException as exc:
            self._logger.error(
                "SerialDevice '%s' readline error: %s", self.device_id, exc
            )
            self._set_status(DeviceStatus.ERROR)
            return ""
