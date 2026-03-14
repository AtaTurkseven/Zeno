"""CallSystem — real-time voice conversation loop for Zeno.

The :class:`CallSystem` wires together:

- :class:`~zeno.audio.mic_stream.MicStream` — continuous microphone capture
- :class:`~zeno.audio.whisper_stt.WhisperSTT` — speech-to-text
- Ollama HTTP API — local language model
- :class:`~zeno.audio.kokoro_tts.KokoroTTS` — text-to-speech
- :class:`~zeno.audio.speaker.Speaker` — audio playback

The entire pipeline runs inside a single :func:`asyncio.run` event loop.

Quick-start
-----------
.. code-block:: python

    import asyncio
    from zeno.audio import CallSystem

    cs = CallSystem()
    asyncio.run(cs.start())

Configuration
-------------
All parameters have sensible defaults.  Pass keyword arguments to
override:

.. code-block:: python

    cs = CallSystem(
        whisper_host="http://localhost:9000",
        ollama_host="http://localhost:11434",
        ollama_model="llama3",
        tts_voice="af_heart",
        system_prompt="You are Zeno, a helpful physical AI assistant.",
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import requests

from zeno.audio.kokoro_tts import KokoroTTS
from zeno.audio.mic_stream import MicStream
from zeno.audio.speaker import Speaker
from zeno.audio.whisper_stt import WhisperSTT

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are Zeno, a modular physical AI assistant designed for real-world "
    "engineering projects.  Respond concisely in plain text."
)


class CallSystem:
    """Orchestrates a real-time voice conversation with a local LLM.

    Parameters
    ----------
    whisper_host:
        Base URL of the whisper.cpp HTTP server.
    whisper_language:
        BCP-47 language code forwarded to whisper.cpp (default ``"en"``).
    ollama_host:
        Base URL of the Ollama inference server.
    ollama_model:
        Ollama model to use for inference.
    tts_voice:
        Kokoro voice identifier.
    tts_command_template:
        Shell command template for Kokoro TTS.  See :class:`~zeno.audio.kokoro_tts.KokoroTTS`.
    system_prompt:
        System prompt prepended to every LLM conversation turn.
    sample_rate:
        Microphone sample rate in Hz (default 16 000).
    silence_threshold:
        RMS energy threshold for the VAD (default 200).
    speech_timeout_s:
        Seconds of trailing silence that marks end-of-utterance (default 1.2).
    ollama_timeout:
        HTTP timeout for Ollama requests in seconds (default 60).
    """

    def __init__(
        self,
        *,
        whisper_host: str = "http://localhost:9000",
        whisper_language: str = "en",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        tts_voice: str = "af_heart",
        tts_command_template: str | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        sample_rate: int = 16_000,
        silence_threshold: float = 200.0,
        speech_timeout_s: float = 1.2,
        ollama_timeout: int = 60,
    ) -> None:
        self._ollama_host = ollama_host.rstrip("/")
        self._ollama_model = ollama_model
        self._ollama_timeout = ollama_timeout
        self._system_prompt = system_prompt

        self._mic = MicStream(
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            speech_timeout_s=speech_timeout_s,
        )
        self._stt = WhisperSTT(
            host=whisper_host,
            language=whisper_language,
        )
        tts_kwargs: dict[str, object] = {"voice": tts_voice}
        if tts_command_template is not None:
            tts_kwargs["command_template"] = tts_command_template
        self._tts = KokoroTTS(**tts_kwargs)  # type: ignore[arg-type]
        self._speaker = Speaker()

        self._running = False
        self._conversation: list[dict[str, str]] = []  # rolling message history
        self._http_session = requests.Session()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the microphone and enter the voice-conversation loop.

        This coroutine runs until :meth:`stop` is called or a fatal error
        occurs.  Call it with ``asyncio.run(call_system.start())``.
        """
        self._running = True
        self._mic.start()
        logger.info("CallSystem started.  Listening…")

        try:
            async for audio_bytes in self._speech_stream():
                if not self._running:
                    break
                # ── STT ───────────────────────────────────────────────
                user_text = await self.listen(audio_bytes)
                if not user_text:
                    continue
                logger.info("User: %s", user_text)

                # ── LLM ───────────────────────────────────────────────
                reply = await self.ask_llm(user_text)
                if not reply:
                    continue
                logger.info("Zeno: %s", reply)

                # ── TTS + playback ────────────────────────────────────
                await self.speak(reply)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the microphone and shut down the pipeline."""
        if not self._running:
            return
        self._running = False
        self._mic.stop()
        self._stt.close()
        self._http_session.close()
        logger.info("CallSystem stopped.")

    async def listen(self, audio_bytes: bytes) -> str:
        """Transcribe *audio_bytes* and return the user's text.

        Parameters
        ----------
        audio_bytes:
            WAV-encoded audio segment produced by :class:`~zeno.audio.mic_stream.MicStream`.

        Returns
        -------
        str
            Transcribed text, or an empty string when nothing was detected.
        """
        text = await self._stt.transcribe(audio_bytes)
        return text.strip()

    async def ask_llm(self, user_text: str) -> str:
        """Send *user_text* to the Ollama LLM and return the reply.

        The conversation history is maintained across calls so the model
        has context from previous turns.

        Parameters
        ----------
        user_text:
            The user's transcribed utterance.

        Returns
        -------
        str
            The model's reply, or an empty string on error.
        """
        self._conversation.append({"role": "user", "content": user_text})

        # Build a single concatenated prompt with system preamble + history.
        prompt = self._build_prompt()

        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(None, self._ollama_generate, prompt)

        if reply:
            self._conversation.append({"role": "assistant", "content": reply})

        return reply

    async def speak(self, text: str) -> None:
        """Synthesise *text* via Kokoro TTS and play the audio.

        Parameters
        ----------
        text:
            The assistant's reply to speak aloud.
        """
        wav_bytes = await self._tts.synthesise(text)
        if wav_bytes:
            await self._speaker.play(wav_bytes)
        else:
            logger.warning("CallSystem.speak(): TTS produced no audio for: %r", text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _speech_stream(self) -> AsyncIterator[bytes]:
        """Delegate to :meth:`MicStream.read_speech`."""
        return self._mic.read_speech()

    def _build_prompt(self) -> str:
        """Assemble the full prompt string from history."""
        _role_label = {"user": "User", "assistant": "Assistant"}
        lines: list[str] = [f"System: {self._system_prompt}", ""]
        for msg in self._conversation:
            label = _role_label.get(msg["role"], msg["role"].capitalize())
            lines.append(f"{label}: {msg['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _ollama_generate(self, prompt: str) -> str:
        """Synchronous Ollama call — runs in a thread-pool executor."""
        url = f"{self._ollama_host}/api/generate"
        payload: dict[str, object] = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = self._http_session.post(
                url, json=payload, timeout=self._ollama_timeout
            )
            response.raise_for_status()
            data = response.json()
            reply: str = data.get("response", "").strip()
            logger.debug("Ollama response (%d chars).", len(reply))
            return reply
        except requests.ConnectionError:
            logger.error(
                "CallSystem: cannot reach Ollama at %s.  Is `ollama serve` running?",
                self._ollama_host,
            )
            return ""
        except requests.Timeout:
            logger.error("CallSystem: Ollama request timed out.")
            return ""
        except requests.RequestException as exc:
            logger.error("CallSystem: Ollama request failed: %s", exc)
            return ""
