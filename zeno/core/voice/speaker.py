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
        """Convert *text* to speech using *pyttsx3* (offline TTS).

        Returns
        -------
        bytes or None
            ``None`` when *pyttsx3* handles playback internally, or when
            the library is not installed (fallback: log only).
        """
        try:
            import pyttsx3  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "pyttsx3 is not installed.  "
                "Install it with: pip install pyttsx3"
            )
            return None

        try:
            engine = pyttsx3.init()
            # Apply language hint as voice selection if possible
            voices = engine.getProperty("voices")
            lang_prefix = self._language.split("-")[0].lower()
            for voice in voices or []:
                vid: str = getattr(voice, "id", "") or ""
                if lang_prefix in vid.lower():
                    engine.setProperty("voice", vid)
                    break
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except RuntimeError as exc:
            logger.error("VoiceSpeaker pyttsx3 runtime error: %s", exc)
        except OSError as exc:
            logger.error("VoiceSpeaker audio device error: %s", exc)
        # pyttsx3 plays audio directly; no bytes to return.
        return None

    def _play(self, audio: bytes) -> None:
        """Play raw PCM *audio* data through the output device.

        Parameters
        ----------
        audio:
            Audio bytes to play.  When *pyttsx3* is used, playback is handled
            inside :meth:`_synthesise` and this method is never called.
        """
        logger.debug("VoiceSpeaker._play() — raw PCM playback not implemented.")
