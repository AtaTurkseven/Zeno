"""Local LLM backend that communicates with an Ollama-compatible server."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from zeno.core.ai.base import AIBase

logger = logging.getLogger(__name__)


class LocalLLM(AIBase):
    """Sends prompts to a locally-running Ollama inference server.

    The Ollama REST API is assumed to be running at ``config['host']``.
    Start it with ``ollama serve`` and pull a model with ``ollama pull <model>``.

    Parameters
    ----------
    config:
        Dict with keys ``model``, ``host``, and ``timeout`` (seconds).
    """

    _GENERATE_PATH = "/api/generate"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model: str = config.get("model", "llama3")
        self._host: str = config.get("host", "http://localhost:11434").rstrip("/")
        self._timeout: int = int(config.get("timeout", 60))
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # AIBase implementation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* to Ollama and return the full response text.

        Parameters
        ----------
        prompt:
            The input text for the model.
        **kwargs:
            Optional overrides: ``model``, ``temperature``, ``stream``.

        Returns
        -------
        str
            The concatenated response from the model.

        Raises
        ------
        RuntimeError
            When the Ollama server returns an error or is unreachable.
        """
        url = f"{self._host}{self._GENERATE_PATH}"
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "prompt": prompt,
            "stream": kwargs.get("stream", False),
        }
        if "temperature" in kwargs:
            payload.setdefault("options", {})["temperature"] = kwargs["temperature"]

        self._logger.debug("LocalLLM request to %s | model=%s", url, payload["model"])

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"LocalLLM request failed: {exc}") from exc

        data = response.json()
        text: str = data.get("response", "")
        self._logger.debug("LocalLLM response (%d chars).", len(text))
        return text

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server responds to a health check."""
        try:
            resp = self._session.get(f"{self._host}/api/tags", timeout=3)
            return resp.ok
        except requests.RequestException:
            return False

    def shutdown(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
        super().shutdown()
