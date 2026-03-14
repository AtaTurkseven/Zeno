"""Zeno audio subsystem — real-time voice call pipeline.

Modules
-------
mic_stream
    Continuous, non-blocking microphone capture with VAD.
whisper_stt
    Speech-to-Text via whisper.cpp HTTP server.
kokoro_tts
    Text-to-Speech via Kokoro subprocess.
speaker
    Non-blocking WAV audio playback.
call_system
    High-level :class:`CallSystem` that wires all components together in an
    asyncio event loop.
"""

from zeno.audio.call_system import CallSystem
from zeno.audio.kokoro_tts import KokoroTTS
from zeno.audio.mic_stream import MicStream
from zeno.audio.speaker import Speaker
from zeno.audio.whisper_stt import WhisperSTT

__all__ = ["CallSystem", "KokoroTTS", "MicStream", "Speaker", "WhisperSTT"]
