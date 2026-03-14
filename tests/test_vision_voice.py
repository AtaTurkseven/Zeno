"""Tests for camera, voice listener, and voice speaker modules."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zeno.core.vision.camera import Camera, Frame
from zeno.core.voice.listener import VoiceListener
from zeno.core.voice.speaker import VoiceSpeaker


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

class TestCamera:
    def _camera(self, **overrides) -> Camera:
        cfg = {"camera_index": 0, "resolution": [640, 480], "enabled": True}
        cfg.update(overrides)
        return Camera(cfg)

    def test_disabled_camera_open_returns_true(self) -> None:
        cam = self._camera(enabled=False)
        assert cam.open() is True

    def test_disabled_camera_capture_returns_none(self) -> None:
        cam = self._camera(enabled=False)
        assert cam.capture() is None

    def test_capture_before_open_returns_none(self) -> None:
        cam = self._camera()
        # Do not call open(); _capture is still None
        assert cam.capture() is None

    def test_close_without_open_is_safe(self) -> None:
        cam = self._camera()
        cam.close()  # Should not raise

    def test_open_fails_when_opencv_missing(self) -> None:
        cam = self._camera()
        # Simulate cv2 not installed
        with patch.dict(sys.modules, {"cv2": None}):
            result = cam.open()
        assert result is False

    def test_open_fails_when_capture_not_opened(self) -> None:
        cam = self._camera()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cam.open()
        assert result is False

    def test_open_success_with_mock_opencv(self) -> None:
        cam = self._camera()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cam.open()
        assert result is True

    def test_close_releases_capture(self) -> None:
        cam = self._camera()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cam.open()
        cam.close()
        mock_cap.release.assert_called_once()

    def test_capture_returns_frame_with_mock_opencv(self) -> None:
        cam = self._camera()
        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_img)
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cam.open()
            frame = cam.capture()
        assert isinstance(frame, Frame)
        assert frame.width == 640
        assert frame.height == 480

    def test_capture_returns_none_on_read_failure(self) -> None:
        cam = self._camera()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cam.open()
            frame = cam.capture()
        assert frame is None


# ---------------------------------------------------------------------------
# VoiceListener tests
# ---------------------------------------------------------------------------

class TestVoiceListener:
    def _listener(self, **overrides) -> VoiceListener:
        cfg = {"input_device": "default", "language": "en-US"}
        cfg.update(overrides)
        return VoiceListener(cfg)

    def test_listen_while_stopped_returns_none(self) -> None:
        listener = self._listener()
        # Listener is not started
        assert listener.listen() is None

    def test_start_and_stop(self) -> None:
        listener = self._listener()
        listener.start()
        assert listener._enabled is True
        listener.stop()
        assert listener._enabled is False

    def test_capture_audio_returns_none_when_sr_missing(self) -> None:
        listener = self._listener()
        listener.start()
        with patch.dict(sys.modules, {"speech_recognition": None}):
            result = listener._capture_audio()
        assert result is None

    def test_transcribe_audio_returns_empty_when_sr_missing(self) -> None:
        listener = self._listener()
        with patch.dict(sys.modules, {"speech_recognition": None}):
            result = listener._transcribe_audio(object())  # any non-None value
        assert result == ""

    def test_listen_returns_none_when_capture_returns_none(self) -> None:
        listener = self._listener()
        listener.start()
        with patch.object(listener, "_capture_audio", return_value=None):
            result = listener.listen()
        assert result is None

    def test_listen_returns_transcription(self) -> None:
        listener = self._listener()
        listener.start()
        fake_audio = object()  # acts as an sr.AudioData placeholder
        with (
            patch.object(listener, "_capture_audio", return_value=fake_audio),
            patch.object(listener, "_transcribe_audio", return_value="hello"),
        ):
            result = listener.listen()
        assert result == "hello"


# ---------------------------------------------------------------------------
# VoiceSpeaker tests
# ---------------------------------------------------------------------------

class TestVoiceSpeaker:
    def _speaker(self, **overrides) -> VoiceSpeaker:
        cfg = {"output_device": "default", "language": "en-US"}
        cfg.update(overrides)
        return VoiceSpeaker(cfg)

    def test_speak_empty_string_is_noop(self) -> None:
        speaker = self._speaker()
        # Should not raise even with no TTS engine
        speaker.speak("   ")

    def test_speak_calls_synthesise(self) -> None:
        speaker = self._speaker()
        with patch.object(speaker, "_synthesise", return_value=None) as mock_synth:
            speaker.speak("hello")
        mock_synth.assert_called_once_with("hello")

    def test_synthesise_returns_none_when_pyttsx3_missing(self) -> None:
        speaker = self._speaker()
        with patch.dict(sys.modules, {"pyttsx3": None}):
            result = speaker._synthesise("test text")
        assert result is None

    def test_synthesise_with_mock_pyttsx3(self) -> None:
        speaker = self._speaker()
        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
            result = speaker._synthesise("hello world")
        mock_engine.say.assert_called_once_with("hello world")
        mock_engine.runAndWait.assert_called_once()
        assert result is None  # pyttsx3 plays directly; no bytes returned
