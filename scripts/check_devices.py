#!/usr/bin/env python3
"""check_devices.py — scan for connected serial devices and verify connectivity.

Usage::

    python scripts/check_devices.py [--port /dev/ttyUSB0] [--baud 115200]
"""

from __future__ import annotations

import argparse
import logging
import sys

# Ensure the project root is on sys.path when run directly.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zeno.config.manager import ConfigManager  # noqa: E402
from zeno.core.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _list_serial_ports() -> list[str]:
    """Return a list of available serial port names."""
    try:
        import serial.tools.list_ports as lp  # type: ignore[import]

        return [p.device for p in lp.comports()]
    except ImportError:
        logger.warning("pyserial list_ports not available.")
        return []


def main(argv: list[str] | None = None) -> int:
    """Entry point for the device check script."""
    parser = argparse.ArgumentParser(description="Scan for connected serial devices.")
    parser.add_argument("--port", default=None, help="Serial port to probe.")
    parser.add_argument("--baud", type=int, default=115_200, help="Baud rate.")
    args = parser.parse_args(argv)

    config = ConfigManager()
    setup_logging(level=config.get("system.log_level", "INFO"))

    available = _list_serial_ports()
    if available:
        print("Detected serial ports:")
        for port in available:
            print(f"  {port}")
    else:
        print("No serial ports detected.")

    if args.port:
        print(f"\nProbing {args.port} @ {args.baud} baud …")
        from zeno.core.hardware.serial_device import SerialDevice

        class _Probe(SerialDevice):
            """Minimal SerialDevice used only for connectivity probing."""

            def connect(self) -> bool:  # type: ignore[override]
                return super().connect()

        probe = _Probe("probe", {"port": args.port, "baud_rate": args.baud})
        connected = probe.connect()
        if connected:
            print("  Connection successful.")
            probe.disconnect()
        else:
            print("  Connection FAILED.")
        return 0 if connected else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
