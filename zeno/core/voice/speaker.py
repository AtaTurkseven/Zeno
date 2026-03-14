"""VoiceSpeaker — converts text to speech and plays it back.

The current implementation defines the full interface and provides a fallback
that logs the text.  Wire in a TTS backend (pyttsx3, gTTS, Coqui TTS, …)
by overriding :meth:`_synthesise`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VoiceSpeaker:
    """Converts text to speech and plays it through an audio output device.

    Parameters
    ----------
    config:
        Dict with keys:

        - ``output_device`` — audio device identifier (``"default"`` or index)
        - ``language`` — BCP-47 language tag (e.g. ``"en-US"``)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._device = config.get("output_device", "default")
        self._language = config.get("language", "en-US")
        logger.debug(
            "VoiceSpeaker configured (device=%s, lang=%s).",
            self._device,
            self._language,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Convert *text* to speech and play it back.

        Parameters
        ----------
        text:
            The sentence(s) to speak aloud.
        """
        if not text.strip():
            return

        logger.info("VoiceSpeaker speaking: %r", text)
        audio = self._synthesise(text)
        if audio:
            self._play(audio)
        else:
            # Fallback: just log the text so the system still "works"
            logger.debug("VoiceSpeaker fallback — text output only: %s", text)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _synthesise(self, text: str) -> bytes | None:
        """Convert *text* to raw PCM audio bytes.

        Returns
        -------
        bytes or None
            Audio data, or ``None`` to skip playback.

        .. todo::
            Implement with pyttsx3, gTTS, Coqui TTS, or similar.
        """
        # TODO: Implement TTS synthesis.
        logger.debug("VoiceSpeaker._synthesise() — not yet implemented.")
        return None

    def _play(self, audio: bytes) -> None:
        """Play raw PCM *audio* data through the output device.

        Parameters
        ----------
        audio:
            Audio bytes to play.

        .. todo::
            Implement with ``pyaudio`` or ``sounddevice``.
        """
        # TODO: Implement audio playback.
        logger.debug("VoiceSpeaker._play() — not yet implemented.")
