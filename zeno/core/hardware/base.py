"""Abstract base class for all hardware devices managed by Zeno."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Life-cycle states of a managed hardware device."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


class BaseDevice(ABC):
    """Contract for every hardware device driver.

    Concrete device classes (ESP32, robotic arm, sensor, …) inherit from
    this class and implement the connect / disconnect / send / receive cycle.

    Parameters
    ----------
    device_id:
        A unique human-readable label (e.g. ``"esp32-0"``).
    config:
        Device-specific configuration dict.
    """

    def __init__(self, device_id: str, config: dict[str, Any]) -> None:
        self.device_id = device_id
        self.config = config
        self._status = DeviceStatus.DISCONNECTED
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> DeviceStatus:
        """Current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """``True`` when the device is in :attr:`DeviceStatus.CONNECTED` state."""
        return self._status == DeviceStatus.CONNECTED

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the physical device.

        Returns
        -------
        bool
            ``True`` on success.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection gracefully."""

    @abstractmethod
    def send(self, data: bytes) -> bool:
        """Write raw *data* to the device.

        Parameters
        ----------
        data:
            Raw bytes to transmit.

        Returns
        -------
        bool
            ``True`` when all bytes were sent.
        """

    @abstractmethod
    def receive(self, num_bytes: int = 256) -> bytes:
        """Read up to *num_bytes* from the device.

        Parameters
        ----------
        num_bytes:
            Maximum number of bytes to read.

        Returns
        -------
        bytes
            The received data (may be shorter than *num_bytes*).
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, status: DeviceStatus) -> None:
        """Update the device status and emit a debug log message."""
        self._status = status
        self._logger.debug("Device '%s' status → %s.", self.device_id, status.name)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"id={self.device_id!r}, status={self._status.name})"
        )
