"""KokoroTTS — Text-to-Speech via the Kokoro neural TTS engine.

Kokoro is invoked as an external subprocess without shell interpretation,
which prevents shell-injection vulnerabilities when user text is passed
as a command-line argument.

Default behaviour::

    python -m kokoro --text <text> --output <path> --voice <voice>

All arguments are passed as a list to :func:`subprocess.run` (``shell=False``),
so the text is never interpreted by a shell.

Usage
-----
.. code-block:: python

    tts = KokoroTTS(voice="af_heart")
    wav_bytes = await tts.synthesise("Hello, I am Zeno.")
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default base command.  ``{voice}`` and ``{output}`` are string-formatted
#: *after* the argument list is built — they are never shell-interpolated.
_DEFAULT_BASE_COMMAND = ["python", "-m", "kokoro"]


class KokoroTTS:
    """Generate speech from text using the Kokoro neural TTS engine.

    The subprocess is always invoked with ``shell=False``.  Text is passed
    as a discrete ``--text`` argument, so shell metacharacters in user speech
    cannot cause injection.

    Parameters
    ----------
    voice:
        Kokoro voice identifier (default ``"af_heart"``).
    base_command:
        The executable and its fixed arguments as a list.  Zeno appends
        ``["--text", text, "--output", output_path, "--voice", voice]``
        automatically.  Override to point to a different TTS binary.
    timeout:
        Maximum seconds to wait for the subprocess (default ``60``).
    sample_rate:
        Expected sample rate of the output WAV (informational; default 24 000).
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        base_command: list[str] | None = None,
        timeout: int = 60,
        sample_rate: int = 24_000,
    ) -> None:
        self._voice = voice
        self._base_command: list[str] = (
            base_command if base_command is not None else list(_DEFAULT_BASE_COMMAND)
        )
        self._timeout = timeout
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesise(self, text: str) -> bytes | None:
        """Convert *text* to speech and return WAV bytes.

        The subprocess is run in a thread-pool executor so it does not
        block the asyncio event loop.

        Parameters
        ----------
        text:
            The sentence(s) to synthesise.

        Returns
        -------
        bytes or None
            WAV-encoded audio, or ``None`` when synthesis fails.
        """
        if not text.strip():
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesise_sync, text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _synthesise_sync(self, text: str) -> bytes | None:
        """Synchronous TTS — runs in a thread-pool executor."""
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="zeno_tts_"
        ) as tmp:
            output_path = tmp.name

        try:
            args = self._build_args(text, output_path)
            logger.debug("KokoroTTS running: %s", args)
            result = subprocess.run(
                args,
                shell=False,  # Never use shell=True; text is a discrete argument.
                capture_output=True,
                timeout=self._timeout,
            )
            if result.returncode != 0:
                logger.error(
                    "KokoroTTS subprocess failed (exit %d): %s",
                    result.returncode,
                    result.stderr.decode(errors="replace"),
                )
                return None

            output = Path(output_path)
            if not output.exists() or output.stat().st_size == 0:
                logger.error(
                    "KokoroTTS: output file not created or empty: %s", output_path
                )
                return None

            wav_bytes = output.read_bytes()
            logger.info(
                "KokoroTTS synthesised %d bytes for %d-char text.",
                len(wav_bytes),
                len(text),
            )
            return wav_bytes

        except subprocess.TimeoutExpired:
            logger.error("KokoroTTS subprocess timed out after %ds.", self._timeout)
            return None
        except OSError as exc:
            logger.error("KokoroTTS OS error: %s", exc)
            return None
        finally:
            try:
                os.unlink(output_path)
            except OSError:
                pass

    def _build_args(self, text: str, output_path: str) -> list[str]:
        """Build the subprocess argument list.

        Text, output path, and voice are passed as discrete arguments —
        never interpolated into a shell string — so no quoting is needed
        and shell injection is impossible.
        """
        return [
            *self._base_command,
            "--text", text,
            "--output", output_path,
            "--voice", self._voice,
        ]

