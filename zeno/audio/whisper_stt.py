"""WhisperSTT — Speech-to-Text via the whisper.cpp HTTP server.

The whisper.cpp server exposes a simple REST endpoint that accepts audio
data and returns a JSON transcript.  Start the server with::

    ./server -m models/ggml-base.en.bin --port 9000

API contract
------------
``POST /inference``
    Form fields:

    - ``file``        — audio file (WAV, MP3, …)
    - ``temperature`` — sampling temperature (default ``0.0``)
    - ``language``    — BCP-47 language code (e.g. ``"en"``)

    Response JSON:

    .. code-block:: json

        {"text": "transcribed text here"}

Usage
-----
.. code-block:: python

    stt = WhisperSTT(host="http://localhost:9000")
    text = await stt.transcribe(wav_bytes)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Transcribes audio by calling a running whisper.cpp HTTP server.

    Parameters
    ----------
    host:
        Base URL of the whisper.cpp server (default ``"http://localhost:9000"``).
    language:
        BCP-47 language tag sent to the server (default ``"en"``).
    temperature:
        Whisper sampling temperature (default ``0.0`` — greedy decoding).
    timeout:
        HTTP request timeout in seconds (default ``30``).
    """

    _INFERENCE_PATH = "/inference"

    def __init__(
        self,
        *,
        host: str = "http://localhost:9000",
        language: str = "en",
        temperature: float = 0.0,
        timeout: int = 30,
    ) -> None:
        self._host = host.rstrip("/")
        self._language = language
        self._temperature = temperature
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: bytes, filename: str = "audio.wav") -> str:
        """Transcribe *audio* and return the transcript.

        The HTTP call is made in a thread-pool executor so it does not
        block the asyncio event loop.

        Parameters
        ----------
        audio:
            WAV-encoded PCM bytes (as produced by :func:`zeno.audio.mic_stream._encode_wav`).
        filename:
            Filename hint sent with the multipart upload
            (default ``"audio.wav"``).

        Returns
        -------
        str
            Transcribed text, stripped of leading/trailing whitespace.
            Returns an empty string when transcription fails.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio, filename
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transcribe_sync(self, audio: bytes, filename: str) -> str:
        """Synchronous transcription — runs in a thread-pool executor."""
        url = f"{self._host}{self._INFERENCE_PATH}"
        files: dict[str, Any] = {
            "file": (filename, audio, "audio/wav"),
        }
        data = {
            "temperature": str(self._temperature),
            "language": self._language,
        }
        try:
            response = self._session.post(
                url, files=files, data=data, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.ConnectionError:
            logger.error(
                "WhisperSTT: cannot reach whisper.cpp server at %s.  "
                "Is it running?",
                self._host,
            )
            return ""
        except requests.Timeout:
            logger.error("WhisperSTT: request to %s timed out.", url)
            return ""
        except requests.RequestException as exc:
            logger.error("WhisperSTT: HTTP error: %s", exc)
            return ""

        payload = response.json()
        text: str = payload.get("text", "").strip()
        logger.info("WhisperSTT transcribed: %r", text)
        return text
