"""Camera — captures frames from a physical or virtual camera.

The implementation provides the full interface.  Attach an OpenCV capture
source by overriding :meth:`_open_capture` and :meth:`_read_frame`, or
use the class as-is with ``enabled=false`` in config during development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single captured image frame.

    Attributes
    ----------
    data:
        Raw pixel data as a NumPy array (H × W × C, uint8).
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    timestamp:
        Time the frame was captured (``time.monotonic()``).
    """

    data: np.ndarray
    width: int
    height: int
    timestamp: float


class Camera:
    """Manages frame capture from a camera device.

    Parameters
    ----------
    config:
        Dict with keys:

        - ``camera_index`` — integer device index (default ``0``)
        - ``resolution`` — ``[width, height]`` list (default ``[640, 480]``)
        - ``enabled`` — set to ``false`` to disable capture (default ``true``)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._index: int = int(config.get("camera_index", 0))
        resolution = config.get("resolution", [640, 480])
        self._width: int = int(resolution[0])
        self._height: int = int(resolution[1])
        self._enabled: bool = bool(config.get("enabled", True))
        self._capture: Any = None  # OpenCV VideoCapture when open
        logger.debug(
            "Camera configured (index=%d, res=%dx%d, enabled=%s).",
            self._index,
            self._width,
            self._height,
            self._enabled,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the camera capture device.

        Returns
        -------
        bool
            ``True`` on success or when camera is disabled (no-op).
        """
        if not self._enabled:
            logger.info("Camera is disabled — skipping open().")
            return True
        self._capture = self._open_capture(self._index)
        if self._capture is None:
            logger.error("Camera failed to open (index=%d).", self._index)
            return False
        logger.info("Camera opened (index=%d).", self._index)
        return True

    def close(self) -> None:
        """Release the camera device."""
        if self._capture is not None:
            self._release_capture(self._capture)
            self._capture = None
        logger.info("Camera closed.")

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture(self) -> Frame | None:
        """Capture and return a single frame.

        Returns
        -------
        Frame or None
            The captured frame, or ``None`` when the camera is closed or
            capture failed.
        """
        if not self._enabled:
            return None
        if self._capture is None:
            logger.warning("Camera.capture() called before open().")
            return None
        return self._read_frame(self._capture)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _open_capture(self, index: int) -> Any:
        """Open the underlying capture device using OpenCV.

        Parameters
        ----------
        index:
            Camera device index.

        Returns
        -------
        Any
            A ``cv2.VideoCapture`` handle, or ``None`` on failure (including
            when OpenCV is not installed).
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "opencv-python is not installed.  "
                "Install it with: pip install opencv-python-headless"
            )
            return None

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            logger.error("OpenCV could not open camera index=%d.", index)
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    def _read_frame(self, capture: Any) -> Frame | None:
        """Read one frame from an open ``cv2.VideoCapture``."""
        import time

        ret, img = capture.read()
        if not ret or img is None:
            logger.warning("Camera._read_frame() failed to grab frame.")
            return None
        height, width = img.shape[:2]
        return Frame(data=img, width=width, height=height, timestamp=time.monotonic())

    def _release_capture(self, capture: Any) -> None:
        """Release a ``cv2.VideoCapture`` handle."""
        capture.release()
