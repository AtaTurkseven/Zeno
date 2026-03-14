"""MicStream — continuous, non-blocking microphone capture with simple VAD.

Architecture
------------
A ``sounddevice.InputStream`` callback pushes raw PCM chunks into a
thread-safe ``queue.Queue``.  An :meth:`MicStream.read_speech` coroutine
reads from that queue, applies energy-based voice activity detection (VAD),
and yields complete speech segments as ``bytes`` (16-bit PCM, mono).

Usage
-----
.. code-block:: python

    stream = MicStream(sample_rate=16000, channels=1)
    stream.start()
    async for audio in stream.read_speech():
        # audio is a WAV-encoded bytes object
        ...
    stream.stop()
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import wave
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)

_CHUNK_DURATION_S = 0.03  # 30 ms per chunk


class MicStream:
    """Captures audio from the default microphone without blocking the event loop.

    Parameters
    ----------
    sample_rate:
        Sample rate in Hz (default 16 000 — what whisper.cpp expects).
    channels:
        Number of channels (default 1 — mono).
    silence_threshold:
        RMS energy threshold below which audio is considered silence.
        Tune for your environment (default ``200`` for 16-bit PCM).
    speech_timeout_s:
        Seconds of trailing silence that marks the end of an utterance
        (default ``1.2``).
    max_duration_s:
        Hard cap on utterance length in seconds (default ``30``).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        silence_threshold: float = 200.0,
        speech_timeout_s: float = 1.2,
        max_duration_s: float = 30.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._threshold = silence_threshold
        self._speech_timeout_s = speech_timeout_s
        self._max_duration_s = max_duration_s

        self._chunk_frames = int(sample_rate * _CHUNK_DURATION_S)
        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream: object | None = None  # sounddevice.InputStream

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone input stream."""
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed.  "
                "Install it with: pip install sounddevice"
            ) from exc

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._chunk_frames,
            callback=self._sd_callback,
        )
        self._stream.start()  # type: ignore[union-attr]
        logger.info(
            "MicStream started (rate=%d Hz, channels=%d, chunk=%d frames).",
            self._sample_rate,
            self._channels,
            self._chunk_frames,
        )

    def stop(self) -> None:
        """Stop and close the microphone input stream."""
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
        logger.info("MicStream stopped.")

    # ------------------------------------------------------------------
    # Audio iteration
    # ------------------------------------------------------------------

    async def read_speech(self) -> AsyncIterator[bytes]:
        """Yield complete speech segments as WAV-encoded bytes.

        The generator runs indefinitely until cancelled.  Each yielded
        value is a self-contained WAV file (16-bit PCM, mono) suitable
        for passing directly to :class:`~zeno.audio.whisper_stt.WhisperSTT`.
        """
        loop = asyncio.get_running_loop()
        speech_frames: list[np.ndarray] = []
        silent_chunks = 0
        speaking = False

        silence_chunks_needed = int(
            self._speech_timeout_s / _CHUNK_DURATION_S
        )
        max_chunks = int(self._max_duration_s / _CHUNK_DURATION_S)

        while True:
            # Yield control while the queue is empty to avoid busy-wait.
            chunk = await loop.run_in_executor(None, self._dequeue_blocking)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            is_speech = rms >= self._threshold

            if is_speech:
                silent_chunks = 0
                if not speaking:
                    speaking = True
                    speech_frames = []
                    logger.debug("MicStream: speech started (rms=%.1f).", rms)
                speech_frames.append(chunk)
            elif speaking:
                speech_frames.append(chunk)
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed or len(speech_frames) >= max_chunks:
                    logger.debug(
                        "MicStream: speech ended (%d chunks).", len(speech_frames)
                    )
                    speaking = False
                    yield _encode_wav(
                        speech_frames, self._sample_rate, self._channels
                    )
                    speech_frames = []
                    silent_chunks = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: object,
    ) -> None:
        """sounddevice callback — called from a C thread, must not block."""
        if status:
            logger.warning("MicStream sounddevice status: %s", status)
        chunk = indata.copy()  # copy once; reuse for both put attempts
        try:
            self._raw_queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest chunk if the consumer is too slow.
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                pass
            self._raw_queue.put_nowait(chunk)

    def _dequeue_blocking(self) -> np.ndarray | None:
        """Block for up to 50 ms waiting for the next audio chunk."""
        try:
            return self._raw_queue.get(timeout=0.05)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_wav(
    frames: list[np.ndarray],
    sample_rate: int,
    channels: int,
) -> bytes:
    """Encode a list of int16 PCM chunks as an in-memory WAV file."""
    pcm = np.concatenate(frames, axis=0).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
