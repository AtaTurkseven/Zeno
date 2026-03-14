"""Tests for the zeno/audio/ real-time voice call system."""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import wave
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(
    pcm: np.ndarray | None = None,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Create a minimal in-memory WAV file for testing."""
    if pcm is None:
        pcm = np.zeros(sample_rate, dtype=np.int16)  # 1 second of silence
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

# ===========================================================================
# MicStream tests
# ===========================================================================

class TestMicStream:
    """Test MicStream without a real microphone."""

    def _make_stream(self, **kw) -> "MicStream":
        from zeno.audio.mic_stream import MicStream
        return MicStream(**kw)

    # ── start / stop ────────────────────────────────────────────────────────

    def test_start_raises_when_sounddevice_missing(self) -> None:
        stream = self._make_stream()
        with patch.dict(sys.modules, {"sounddevice": None}):
            with pytest.raises(RuntimeError, match="sounddevice"):
                stream.start()

    def test_stop_without_start_is_safe(self) -> None:
        stream = self._make_stream()
        stream.stop()  # Should not raise

    def test_start_and_stop_with_mock_sd(self) -> None:
        stream = self._make_stream()
        mock_sd = MagicMock()
        mock_input = MagicMock()
        mock_sd.InputStream.return_value = mock_input
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            stream.start()
            stream.stop()
        mock_input.start.assert_called_once()
        mock_input.stop.assert_called_once()
        mock_input.close.assert_called_once()

    # ── _sd_callback ────────────────────────────────────────────────────────

    def test_sd_callback_enqueues_chunk(self) -> None:
        stream = self._make_stream()
        chunk = np.zeros((480, 1), dtype=np.int16)
        stream._sd_callback(chunk, 480, None, None)
        assert not stream._raw_queue.empty()

    def test_sd_callback_drops_oldest_on_full_queue(self) -> None:
        stream = self._make_stream()
        # Completely fill the queue
        for _ in range(stream._raw_queue.maxsize):
            stream._raw_queue.put_nowait(np.zeros(1, dtype=np.int16))
        # Callback should evict one and insert new
        new_chunk = np.ones(1, dtype=np.int16)
        stream._sd_callback(new_chunk, 1, None, None)
        assert stream._raw_queue.qsize() == stream._raw_queue.maxsize

    # ── _dequeue_blocking ───────────────────────────────────────────────────

    def test_dequeue_returns_none_on_empty(self) -> None:
        stream = self._make_stream()
        result = stream._dequeue_blocking()
        assert result is None

    def test_dequeue_returns_chunk(self) -> None:
        stream = self._make_stream()
        chunk = np.ones(5, dtype=np.int16)
        stream._raw_queue.put_nowait(chunk)
        result = stream._dequeue_blocking()
        assert result is not None
        np.testing.assert_array_equal(result, chunk)

    # ── read_speech ─────────────────────────────────────────────────────────

    def test_read_speech_yields_wav_bytes(self) -> None:
        """Simulate speech by injecting loud chunks followed by silence."""
        from zeno.audio.mic_stream import MicStream

        stream = MicStream(
            sample_rate=16_000,
            silence_threshold=100.0,
            speech_timeout_s=0.06,  # very short for test speed
        )

        # 5 loud chunks (speech)
        loud = np.full((stream._chunk_frames, 1), 5000, dtype=np.int16)
        # 4 silent chunks (end-of-utterance)
        silent = np.zeros((stream._chunk_frames, 1), dtype=np.int16)

        for _ in range(5):
            stream._raw_queue.put_nowait(loud)
        for _ in range(4):
            stream._raw_queue.put_nowait(silent)

        async def _collect_one() -> bytes:
            async for segment in stream.read_speech():
                return segment
            return b""

        result = asyncio.run(_collect_one())
        assert isinstance(result, bytes)
        assert len(result) > 44  # WAV header is 44 bytes; must have PCM data too

    # ── _encode_wav helper ──────────────────────────────────────────────────

    def test_encode_wav_produces_valid_wav(self) -> None:
        from zeno.audio.mic_stream import _encode_wav

        frames = [np.zeros((480, 1), dtype=np.int16) for _ in range(3)]
        wav = _encode_wav(frames, sample_rate=16_000, channels=1)
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 16_000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


# ===========================================================================
# WhisperSTT tests
# ===========================================================================

class TestWhisperSTT:
    def _stt(self, **kw) -> "WhisperSTT":
        from zeno.audio.whisper_stt import WhisperSTT
        return WhisperSTT(**kw)

    def test_transcribe_success(self) -> None:
        stt = self._stt()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"text": "  hello world  "}
        with patch.object(stt._session, "post", return_value=mock_resp):
            result = asyncio.run(stt.transcribe(b"fake_wav"))
        assert result == "hello world"

    def test_transcribe_returns_empty_on_connection_error(self) -> None:
        import requests
        stt = self._stt()
        with patch.object(
            stt._session, "post", side_effect=requests.ConnectionError("down")
        ):
            result = asyncio.run(stt.transcribe(b"fake_wav"))
        assert result == ""

    def test_transcribe_returns_empty_on_timeout(self) -> None:
        import requests
        stt = self._stt()
        with patch.object(
            stt._session, "post", side_effect=requests.Timeout("timeout")
        ):
            result = asyncio.run(stt.transcribe(b"fake_wav"))
        assert result == ""

    def test_transcribe_returns_empty_on_http_error(self) -> None:
        import requests
        stt = self._stt()
        with patch.object(
            stt._session, "post", side_effect=requests.HTTPError("500")
        ):
            result = asyncio.run(stt.transcribe(b"fake_wav"))
        assert result == ""

    def test_transcribe_missing_text_key(self) -> None:
        stt = self._stt()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}  # no "text" key
        with patch.object(stt._session, "post", return_value=mock_resp):
            result = asyncio.run(stt.transcribe(b"fake_wav"))
        assert result == ""

    def test_close_closes_session(self) -> None:
        stt = self._stt()
        with patch.object(stt._session, "close") as mock_close:
            stt.close()
        mock_close.assert_called_once()

    def test_default_host_and_language(self) -> None:
        from zeno.audio.whisper_stt import WhisperSTT
        stt = WhisperSTT()
        assert stt._host == "http://localhost:9000"
        assert stt._language == "en"

    def test_custom_host_strips_trailing_slash(self) -> None:
        from zeno.audio.whisper_stt import WhisperSTT
        stt = WhisperSTT(host="http://localhost:9000/")
        assert not stt._host.endswith("/")


# ===========================================================================
# KokoroTTS tests
# ===========================================================================

class TestKokoroTTS:
    def _tts(self, **kw) -> "KokoroTTS":
        from zeno.audio.kokoro_tts import KokoroTTS
        return KokoroTTS(**kw)

    def test_synthesise_returns_none_for_empty_text(self) -> None:
        tts = self._tts()
        result = asyncio.run(tts.synthesise("   "))
        assert result is None

    def test_synthesise_returns_wav_on_success(self, tmp_path) -> None:
        tts = self._tts()
        fake_wav = _make_wav()

        def _fake_run(args, **kw):
            # The output path is the value after --output in the arg list
            idx = args.index("--output")
            out = args[idx + 1]
            with open(out, "wb") as f:
                f.write(fake_wav)
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch("subprocess.run", side_effect=_fake_run):
            result = asyncio.run(tts.synthesise("hello"))
        assert result == fake_wav

    def test_synthesise_returns_none_on_nonzero_exit(self) -> None:
        tts = self._tts()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"error"
        with patch("subprocess.run", return_value=mock_result):
            result = asyncio.run(tts.synthesise("hello"))
        assert result is None

    def test_synthesise_returns_none_on_timeout(self) -> None:
        import subprocess
        tts = self._tts(timeout=1)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            result = asyncio.run(tts.synthesise("hello"))
        assert result is None

    def test_synthesise_returns_none_on_os_error(self) -> None:
        tts = self._tts()
        with patch("subprocess.run", side_effect=OSError("no such file")):
            result = asyncio.run(tts.synthesise("hello"))
        assert result is None

    def test_build_command_contains_voice(self) -> None:
        tts = self._tts(voice="af_sky")
        args = tts._build_args("hello", "/tmp/out.wav")
        assert "af_sky" in args
        assert "/tmp/out.wav" in args
        assert "hello" in args

    def test_build_args_text_is_separate_argument(self) -> None:
        """Text must be a discrete argument, not embedded in a shell string."""
        tts = self._tts(voice="af_heart")
        tricky_text = "hello; rm -rf /"
        args = tts._build_args(tricky_text, "/tmp/out.wav")
        # Text must appear as its own element — not shell-interpreted
        assert tricky_text in args
        # No shell-splitting should have been applied
        assert "--text" in args
        idx = args.index("--text")
        assert args[idx + 1] == tricky_text

    def test_default_voice(self) -> None:
        from zeno.audio.kokoro_tts import KokoroTTS
        tts = KokoroTTS()
        assert tts._voice == "af_heart"


# ===========================================================================
# Speaker tests
# ===========================================================================

class TestSpeaker:
    def _speaker(self) -> "Speaker":
        from zeno.audio.speaker import Speaker
        return Speaker()

    def test_play_empty_bytes_is_noop(self) -> None:
        speaker = self._speaker()
        asyncio.run(speaker.play(b""))  # Should not raise

    def test_play_invalid_wav_logs_warning(self) -> None:
        speaker = self._speaker()
        asyncio.run(speaker.play(b"not a wav file"))  # Should not raise

    def test_play_valid_wav_calls_sounddevice(self) -> None:
        speaker = self._speaker()
        wav = _make_wav()
        mock_sd = MagicMock()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            asyncio.run(speaker.play(wav))
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    def test_play_logs_warning_when_sounddevice_missing(self) -> None:
        speaker = self._speaker()
        wav = _make_wav()
        with patch.dict(sys.modules, {"sounddevice": None}):
            asyncio.run(speaker.play(wav))  # Should not raise

    def test_decode_wav_stereo(self) -> None:
        from zeno.audio.speaker import _decode_wav
        stereo = np.zeros((8000, 2), dtype=np.int16)
        wav = _make_wav(pcm=stereo.flatten(), channels=2)
        pcm, sr = _decode_wav(wav)
        assert pcm is not None
        assert sr == 16_000
        assert pcm.ndim == 2
        assert pcm.shape[1] == 2

    def test_decode_wav_8bit_uses_uint8(self) -> None:
        from zeno.audio.speaker import _decode_wav
        # 8-bit WAV PCM is unsigned (uint8)
        pcm_data = np.full(1000, 128, dtype=np.uint8)
        wav = _make_wav(pcm=pcm_data, sample_width=1)
        pcm, sr = _decode_wav(wav)
        assert pcm is not None
        assert pcm.dtype == np.uint8

    def test_decode_wav_invalid_returns_none(self) -> None:
        from zeno.audio.speaker import _decode_wav
        pcm, sr = _decode_wav(b"garbage")
        assert pcm is None
        assert sr == 0


# ===========================================================================
# CallSystem tests
# ===========================================================================

class TestCallSystem:
    def _cs(self, **kw) -> "CallSystem":
        from zeno.audio.call_system import CallSystem
        return CallSystem(**kw)

    # ── stop is idempotent ───────────────────────────────────────────────────

    def test_stop_without_start_is_safe(self) -> None:
        cs = self._cs()
        cs.stop()  # Should not raise

    def test_double_stop_is_safe(self) -> None:
        cs = self._cs()
        cs._running = True
        cs.stop()
        cs.stop()  # Second call should be a no-op

    # ── listen ───────────────────────────────────────────────────────────────

    def test_listen_returns_transcript(self) -> None:
        cs = self._cs()
        with patch.object(cs._stt, "transcribe", new_callable=AsyncMock, return_value="  hello  "):
            result = asyncio.run(cs.listen(b"fake_audio"))
        assert result == "hello"

    def test_listen_returns_empty_on_stt_failure(self) -> None:
        cs = self._cs()
        with patch.object(cs._stt, "transcribe", new_callable=AsyncMock, return_value=""):
            result = asyncio.run(cs.listen(b"fake_audio"))
        assert result == ""

    # ── ask_llm ──────────────────────────────────────────────────────────────

    def test_ask_llm_returns_reply(self) -> None:
        cs = self._cs()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "I am Zeno."}
        with patch.object(cs._http_session, "post", return_value=mock_resp):
            reply = asyncio.run(cs.ask_llm("who are you?"))
        assert reply == "I am Zeno."

    def test_ask_llm_appends_conversation(self) -> None:
        cs = self._cs()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "reply"}
        with patch.object(cs._http_session, "post", return_value=mock_resp):
            asyncio.run(cs.ask_llm("hi"))
        assert len(cs._conversation) == 2
        assert cs._conversation[0]["role"] == "user"
        assert cs._conversation[1]["role"] == "assistant"

    def test_ask_llm_returns_empty_on_connection_error(self) -> None:
        import requests
        cs = self._cs()
        with patch.object(
            cs._http_session, "post", side_effect=requests.ConnectionError("down")
        ):
            reply = asyncio.run(cs.ask_llm("hello"))
        assert reply == ""

    def test_ask_llm_returns_empty_on_timeout(self) -> None:
        import requests
        cs = self._cs()
        with patch.object(
            cs._http_session, "post", side_effect=requests.Timeout("timeout")
        ):
            reply = asyncio.run(cs.ask_llm("hello"))
        assert reply == ""

    # ── speak ────────────────────────────────────────────────────────────────

    def test_speak_calls_tts_and_speaker(self) -> None:
        cs = self._cs()
        fake_wav = _make_wav()
        with (
            patch.object(cs._tts, "synthesise", new_callable=AsyncMock, return_value=fake_wav),
            patch.object(cs._speaker, "play", new_callable=AsyncMock) as mock_play,
        ):
            asyncio.run(cs.speak("hello"))
        mock_play.assert_called_once_with(fake_wav)

    def test_speak_skips_player_when_tts_returns_none(self) -> None:
        cs = self._cs()
        with (
            patch.object(cs._tts, "synthesise", new_callable=AsyncMock, return_value=None),
            patch.object(cs._speaker, "play", new_callable=AsyncMock) as mock_play,
        ):
            asyncio.run(cs.speak("hello"))
        mock_play.assert_not_called()

    # ── _build_prompt ────────────────────────────────────────────────────────

    def test_build_prompt_includes_system_prompt(self) -> None:
        cs = self._cs(system_prompt="Be helpful.")
        cs._conversation = [{"role": "user", "content": "hi"}]
        prompt = cs._build_prompt()
        assert "Be helpful." in prompt
        assert "User: hi" in prompt
        assert prompt.endswith("Assistant:")

    def test_build_prompt_multi_turn(self) -> None:
        cs = self._cs()
        cs._conversation = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"},
        ]
        prompt = cs._build_prompt()
        assert "User: hello" in prompt
        assert "Assistant: hi" in prompt
        assert "User: how are you?" in prompt

    def test_build_prompt_role_labels_are_consistent(self) -> None:
        cs = self._cs()
        cs._conversation = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
        prompt = cs._build_prompt()
        # Must use the mapped labels, not raw role strings
        assert "User: x" in prompt
        assert "Assistant: y" in prompt

    # ── default configuration ────────────────────────────────────────────────

    def test_default_ollama_host(self) -> None:
        from zeno.audio.call_system import CallSystem
        cs = CallSystem()
        assert cs._ollama_host == "http://localhost:11434"

    def test_default_model(self) -> None:
        from zeno.audio.call_system import CallSystem
        cs = CallSystem()
        assert cs._ollama_model == "llama3"

    def test_custom_parameters(self) -> None:
        from zeno.audio.call_system import CallSystem
        cs = CallSystem(
            ollama_model="mistral",
            ollama_host="http://192.168.1.10:11434",
            system_prompt="Custom prompt.",
        )
        assert cs._ollama_model == "mistral"
        assert "192.168.1.10" in cs._ollama_host
        assert cs._system_prompt == "Custom prompt."
