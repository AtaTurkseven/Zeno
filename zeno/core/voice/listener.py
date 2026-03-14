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

    def _capture_audio(self) -> Any | None:
        """Capture a single utterance from the microphone using *SpeechRecognition*.

        Returns
        -------
        sr.AudioData or None
            A ``speech_recognition.AudioData`` object, or ``None`` when no
            speech was detected or when *SpeechRecognition* / *PyAudio* are
            not installed.
        """
        try:
            import speech_recognition as sr  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "SpeechRecognition is not installed.  "
                "Install it with: pip install SpeechRecognition pyaudio"
            )
            return None

        recognizer = sr.Recognizer()
        mic_index: int | None = None
        if isinstance(self._device, int):
            mic_index = self._device

        try:
            with sr.Microphone(device_index=mic_index) as source:
                logger.debug("VoiceListener adjusting for ambient noise…")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.debug("VoiceListener listening for speech…")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            return audio
        except sr.WaitTimeoutError:
            logger.debug("VoiceListener timed out waiting for speech.")
            return None
        except OSError as exc:
            logger.error("VoiceListener microphone error: %s", exc)
            return None

    def _transcribe_audio(self, audio: Any) -> str:
        """Transcribe *audio* using the Google Web Speech API.

        Parameters
        ----------
        audio:
            A ``speech_recognition.AudioData`` object returned by
            :meth:`_capture_audio`.

        Returns
        -------
        str
            Transcribed text, or an empty string when transcription fails.
        """
        try:
            import speech_recognition as sr  # type: ignore[import-untyped]
        except ImportError:
            return ""

        recognizer = sr.Recognizer()
        try:
            text: str = recognizer.recognize_google(audio, language=self._language)
            logger.info("VoiceListener transcribed: %r", text)
            return text
        except sr.UnknownValueError:
            logger.debug("VoiceListener could not understand audio.")
            return ""
        except sr.RequestError as exc:
            logger.error("VoiceListener STT request failed: %s", exc)
            return ""
