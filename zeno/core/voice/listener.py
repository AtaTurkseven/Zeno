"""VoiceListener — captures audio from a microphone and transcribes it.

The current implementation is a structured stub that defines the full interface.
Plug in a speech-recognition backend (e.g. Whisper, Vosk, Google STT) by
overriding :meth:`_transcribe_audio`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VoiceListener:
    """Listens for speech and returns the transcribed text.

    Parameters
    ----------
    config:
        Dict with keys:

        - ``input_device`` — audio device identifier (``"default"`` or index)
        - ``language`` — BCP-47 language tag (e.g. ``"en-US"``)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._device = config.get("input_device", "default")
        self._language = config.get("language", "en-US")
        self._enabled: bool = False
        logger.debug(
            "VoiceListener configured (device=%s, lang=%s).",
            self._device,
            self._language,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Activate the listener and open the audio input stream."""
        self._enabled = True
        logger.info("VoiceListener started (device=%s).", self._device)

    def stop(self) -> None:
        """Stop listening and release the audio input stream."""
        self._enabled = False
        logger.info("VoiceListener stopped.")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def listen(self) -> str | None:
        """Block until speech is detected and return the transcription.

        Returns
        -------
        str or None
            Transcribed text, or ``None`` when nothing was detected or the
            listener is not active.
        """
        if not self._enabled:
            logger.warning("VoiceListener.listen() called while listener is stopped.")
            return None

        # TODO: Integrate a real STT backend here (e.g. openai-whisper, Vosk).
        audio = self._capture_audio()
        if audio is None:
            return None
        return self._transcribe_audio(audio)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _capture_audio(self) -> bytes | None:
        """Capture a single utterance from the microphone.

        Returns
        -------
        bytes or None
            Raw PCM audio bytes, or ``None`` when no speech was detected.

        .. todo::
            Implement with ``pyaudio`` or ``sounddevice``.
        """
        # TODO: Implement audio capture.
        logger.debug("VoiceListener._capture_audio() — not yet implemented.")
        return None

    def _transcribe_audio(self, audio: bytes) -> str:
        """Convert raw PCM *audio* to text.

        Parameters
        ----------
        audio:
            Raw PCM audio bytes returned by :meth:`_capture_audio`.

        Returns
        -------
        str
            Transcribed text.

        .. todo::
            Implement with a chosen STT engine.
        """
        # TODO: Implement transcription.
        logger.debug("VoiceListener._transcribe_audio() — not yet implemented.")
        return ""
