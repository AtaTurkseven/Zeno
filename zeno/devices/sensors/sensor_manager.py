"""SensorManager — registers, polls, and aggregates values from multiple sensors.

Each sensor is represented by a callable (or a device object with a ``read()``
method) that returns a numeric or dict reading.  The manager polls all
registered sensors on demand and caches the most-recent values.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A sensor reader is any zero-argument callable that returns a reading.
SensorReader = Callable[[], Any]


class SensorManager:
    """Aggregates readings from named sensor callables.

    Usage example::

        mgr = SensorManager()
        mgr.register("temperature", lambda: 23.5)
        mgr.register("humidity", lambda: 60.0)
        readings = mgr.read_all()  # {"temperature": 23.5, "humidity": 60.0}
    """

    def __init__(self) -> None:
        self._sensors: dict[str, SensorReader] = {}
        self._last_readings: dict[str, Any] = {}
        self._last_read_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, reader: SensorReader) -> None:
        """Register a sensor under *name*.

        Parameters
        ----------
        name:
            Unique sensor identifier (e.g. ``"temperature"``).
        reader:
            A zero-argument callable that returns the current sensor value.

        Raises
        ------
        ValueError
            When a sensor with *name* is already registered.
        """
        if name in self._sensors:
            raise ValueError(
                f"Sensor '{name}' is already registered.  "
                "Call unregister() first to replace it."
            )
        self._sensors[name] = reader
        logger.debug("Sensor registered: '%s'.", name)

    def unregister(self, name: str) -> None:
        """Remove a sensor by *name*.

        Raises
        ------
        KeyError
            When no sensor with *name* is registered.
        """
        if name not in self._sensors:
            raise KeyError(f"Sensor '{name}' is not registered.")
        del self._sensors[name]
        self._last_readings.pop(name, None)
        self._last_read_time.pop(name, None)
        logger.debug("Sensor unregistered: '%s'.", name)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, name: str) -> Any:
        """Poll a single sensor by *name* and cache the result.

        Parameters
        ----------
        name:
            Registered sensor identifier.

        Returns
        -------
        Any
            The sensor reading, or ``None`` on error.

        Raises
        ------
        KeyError
            When *name* is not registered.
        """
        if name not in self._sensors:
            raise KeyError(f"Sensor '{name}' is not registered.")
        try:
            value = self._sensors[name]()
            self._last_readings[name] = value
            self._last_read_time[name] = time.monotonic()
            logger.debug("Sensor '%s' → %r", name, value)
            return value
        except Exception as exc:  # noqa: BLE001
            logger.error("Sensor '%s' read error: %s", name, exc)
            return None

    def read_all(self) -> dict[str, Any]:
        """Poll every registered sensor and return a name → value mapping.

        Returns
        -------
        dict
            Current reading for each sensor (``None`` on individual errors).
        """
        return {name: self.read(name) for name in self._sensors}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def last_reading(self, name: str) -> Any:
        """Return the cached reading from the previous :meth:`read` call.

        Returns ``None`` when the sensor has not been polled yet.
        """
        return self._last_readings.get(name)

    def list_sensors(self) -> list[str]:
        """Return a list of registered sensor names."""
        return list(self._sensors)

    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SensorManager(sensors={self.list_sensors()})"
