"""RoboticArm — multi-DOF robotic arm driver over a serial link.

Commands follow a simple ASCII protocol understood by the arm's firmware::

    SERVO <joint_id> <angle>     — set joint angle (0–180 degrees)
    SERVO ALL HOME               — move all joints to home position
    GET POSITION                 — request current joint angles (CSV response)

This class manages the arm's joint state locally and synchronises it with
the physical hardware on every move command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from zeno.core.hardware.serial_device import SerialDevice

logger = logging.getLogger(__name__)

_MIN_ANGLE = 0.0
_MAX_ANGLE = 180.0


@dataclass
class Joint:
    """Represents a single joint / servo of the robotic arm.

    Attributes
    ----------
    joint_id:
        Zero-based joint index.
    angle:
        Current angle in degrees (0–180).
    min_angle:
        Minimum allowed angle in degrees.
    max_angle:
        Maximum allowed angle in degrees.
    """

    joint_id: int
    angle: float = 90.0
    min_angle: float = _MIN_ANGLE
    max_angle: float = _MAX_ANGLE

    def clamp(self, angle: float) -> float:
        """Return *angle* clamped to [min_angle, max_angle]."""
        return max(self.min_angle, min(self.max_angle, angle))


class RoboticArm(SerialDevice):
    """Controls a serial-connected robotic arm with up to *dof* joints.

    Parameters
    ----------
    config:
        Dict with keys ``port``, ``baud_rate``, ``timeout``, and ``dof``
        (degrees of freedom, default ``6``).
    device_id:
        Optional label (default: ``"robotic-arm-0"``).
    """

    def __init__(
        self, config: dict[str, Any], device_id: str = "robotic-arm-0"
    ) -> None:
        super().__init__(device_id, config)
        dof = int(config.get("dof", 6))
        self._joints: list[Joint] = [Joint(joint_id=i) for i in range(dof)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        """Number of degrees of freedom (joints)."""
        return len(self._joints)

    @property
    def joint_angles(self) -> list[float]:
        """Current angle of each joint in degrees."""
        return [j.angle for j in self._joints]

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def move_joint(self, joint_id: int, angle: float) -> bool:
        """Move joint *joint_id* to *angle* degrees.

        Parameters
        ----------
        joint_id:
            Zero-based index of the target joint.
        angle:
            Desired angle in degrees (clamped to [0, 180]).

        Returns
        -------
        bool
            ``True`` when the firmware acknowledged the command.

        Raises
        ------
        IndexError
            When *joint_id* is out of range.
        """
        if not 0 <= joint_id < self.dof:
            raise IndexError(
                f"joint_id {joint_id} out of range [0, {self.dof - 1}]."
            )
        joint = self._joints[joint_id]
        clamped = joint.clamp(angle)
        if clamped != angle:
            self._logger.warning(
                "Angle %.1f clamped to %.1f for joint %d.", angle, clamped, joint_id
            )

        resp = self._send_cmd(f"SERVO {joint_id} {clamped:.1f}")
        if resp.strip().upper() in {"OK", "ACK"}:
            joint.angle = clamped
            self._logger.info("Joint %d moved to %.1f°.", joint_id, clamped)
            return True
        self._logger.error(
            "move_joint(%d, %.1f) failed — firmware replied: %r", joint_id, clamped, resp
        )
        return False

    def home(self) -> bool:
        """Move all joints to the home position (90°).

        Returns
        -------
        bool
            ``True`` when the firmware acknowledged.
        """
        resp = self._send_cmd("SERVO ALL HOME")
        if resp.strip().upper() in {"OK", "ACK"}:
            for joint in self._joints:
                joint.angle = 90.0
            self._logger.info("Robotic arm homed.")
            return True
        self._logger.error("home() failed — firmware replied: %r", resp)
        return False

    def get_position(self) -> list[float] | None:
        """Query the arm's current joint angles from the firmware.

        Returns
        -------
        list of float or None
            Angle for each joint, or ``None`` when parsing fails.
        """
        resp = self._send_cmd("GET POSITION")
        try:
            angles = [float(a) for a in resp.split(",")]
            if len(angles) == self.dof:
                for j, a in zip(self._joints, angles):
                    j.angle = a
                return angles
            self._logger.warning(
                "get_position() expected %d values, got %d.", self.dof, len(angles)
            )
            return None
        except ValueError:
            self._logger.error("get_position() could not parse response: %r", resp)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_cmd(self, command: str) -> str:
        """Send *command* and return the response line."""
        if not self.is_connected:
            self._logger.error(
                "RoboticArm '%s': command sent while disconnected.", self.device_id
            )
            return ""
        self._logger.debug("Arm → %r", command)
        self.send_line(command)
        resp = self.receive_line()
        self._logger.debug("Arm ← %r", resp)
        return resp
