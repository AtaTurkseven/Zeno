"""Tests for the AI layer (base + backends)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from zeno.core.ai.base import AIBase
from zeno.core.ai.cloud_ai import CloudAI
from zeno.core.ai.local_llm import LocalLLM


# ---------------------------------------------------------------------------
# Concrete stub for abstract base
# ---------------------------------------------------------------------------

class _DummyAI(AIBase):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"echo:{prompt}"

    def is_available(self) -> bool:
        return True


class TestAIBase:
    def test_generate_and_available(self) -> None:
        ai = _DummyAI({})
        assert ai.generate("hello") == "echo:hello"
        assert ai.is_available() is True

    def test_initialize_shutdown(self) -> None:
        ai = _DummyAI({})
        ai.initialize()
        ai.shutdown()


# ---------------------------------------------------------------------------
# LocalLLM tests
# ---------------------------------------------------------------------------

class TestLocalLLM:
    _cfg = {"model": "llama3", "host": "http://localhost:11434", "timeout": 5}

    def test_is_available_when_server_up(self) -> None:
        llm = LocalLLM(self._cfg)
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch.object(llm._session, "get", return_value=mock_resp):
            assert llm.is_available() is True

    def test_is_available_when_server_down(self) -> None:
        import requests

        llm = LocalLLM(self._cfg)
        with patch.object(
            llm._session, "get", side_effect=requests.ConnectionError("down")
        ):
            assert llm.is_available() is False

    def test_generate_success(self) -> None:
        llm = LocalLLM(self._cfg)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "Hello from LLM"}
        with patch.object(llm._session, "post", return_value=mock_resp):
            result = llm.generate("hi")
        assert result == "Hello from LLM"

    def test_generate_raises_on_request_error(self) -> None:
        import requests

        llm = LocalLLM(self._cfg)
        with patch.object(
            llm._session,
            "post",
            side_effect=requests.ConnectionError("unreachable"),
        ):
            with pytest.raises(RuntimeError, match="LocalLLM request failed"):
                llm.generate("hi")

    def test_shutdown_closes_session(self) -> None:
        llm = LocalLLM(self._cfg)
        with patch.object(llm._session, "close") as mock_close:
            llm.shutdown()
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# CloudAI tests
# ---------------------------------------------------------------------------

class TestCloudAI:
    _cfg = {
        "provider": "openai",
        "api_key_env": "ZENO_TEST_API_KEY",
        "model": "gpt-4o",
        "timeout": 5,
    }

    def test_is_available_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZENO_TEST_API_KEY", "sk-test")
        ai = CloudAI(self._cfg)
        assert ai.is_available() is True

    def test_is_available_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZENO_TEST_API_KEY", raising=False)
        ai = CloudAI(self._cfg)
        assert ai.is_available() is False

    def test_generate_raises_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZENO_TEST_API_KEY", raising=False)
        ai = CloudAI(self._cfg)
        with pytest.raises(RuntimeError, match="API key not set"):
            ai.generate("hello")

    def test_generate_openai_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZENO_TEST_API_KEY", "sk-test")
        ai = CloudAI(self._cfg)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "OpenAI reply"}}]
        }
        with patch.object(ai._session, "post", return_value=mock_resp):
            result = ai.generate("hello")
        assert result == "OpenAI reply"

    def test_unsupported_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZENO_TEST_API_KEY", "sk-test")
        cfg = {**self._cfg, "provider": "unknown_provider"}
        ai = CloudAI(cfg)
        with pytest.raises(RuntimeError, match="Unsupported cloud provider"):
            ai.generate("hello")
