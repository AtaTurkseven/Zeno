"""Speaker — non-blocking WAV audio playback using sounddevice.

The :class:`Speaker` class plays WAV audio (provided as bytes) without
freezing the asyncio event loop.  Playback is dispatched to a
thread-pool executor; an internal lock ensures that only one clip plays
at a time.

Usage
-----
.. code-block:: python

    speaker = Speaker()
    await speaker.play(wav_bytes)
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Speaker:
    """Play WAV audio asynchronously without blocking the event loop.

    Parameters
    ----------
    device:
        Output device index or name passed to ``sounddevice.play``.
        ``None`` uses the system default.
    """

    def __init__(self, *, device: int | str | None = None) -> None:
        self._device = device
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def play(self, wav_bytes: bytes) -> None:
        """Play *wav_bytes* through the output device.

        Blocks (within the executor) until playback finishes so that
        consecutive calls are serialised and do not overlap.

        Parameters
        ----------
        wav_bytes:
            A complete WAV file as bytes.  Both 16-bit and 32-bit PCM are
            supported.  If the data is empty or unreadable, a warning is
            logged and the method returns immediately.
        """
        if not wav_bytes:
            return

        pcm, sample_rate = _decode_wav(wav_bytes)
        if pcm is None:
            logger.warning("Speaker.play(): failed to decode WAV data.")
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None, self._play_sync, pcm, sample_rate
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _play_sync(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Blocking playback — runs inside a thread-pool executor."""
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "sounddevice is not installed.  "
                "Install it with: pip install sounddevice"
            )
            return

        try:
            sd.play(pcm, samplerate=sample_rate, device=self._device)
            sd.wait()  # Block until playback is done (inside the executor thread).
        except Exception as exc:  # noqa: BLE001
            logger.error("Speaker playback error: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int] | tuple[None, int]:
    """Decode WAV bytes into a NumPy array and sample rate.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(pcm_array, sample_rate)`` on success.
    tuple[None, int]
        ``(None, 0)`` on failure.
    """
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels: int = wf.getnchannels()
            sample_width: int = wf.getsampwidth()  # bytes per sample
            sample_rate: int = wf.getframerate()
            raw: bytes = wf.readframes(wf.getnframes())

        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width)
        if dtype is None:
            logger.error("Speaker: unsupported sample width %d.", sample_width)
            return None, 0

        pcm = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels)

        return pcm, sample_rate
    except Exception as exc:  # noqa: BLE001
        logger.error("Speaker: WAV decode error: %s", exc)
        return None, 0
