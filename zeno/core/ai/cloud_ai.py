"""Cloud AI backend that communicates with OpenAI-compatible REST APIs."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from zeno.core.ai.base import AIBase

logger = logging.getLogger(__name__)


class CloudAI(AIBase):
    """Sends prompts to a cloud AI provider (OpenAI, Anthropic, …).

    The API key is read from the environment variable specified in
    ``config['api_key_env']`` (default: ``OPENAI_API_KEY``).

    Parameters
    ----------
    config:
        Dict with keys ``provider``, ``api_key_env``, ``model``, and ``timeout``.
    """

    # Endpoint templates keyed by provider name
    _ENDPOINTS: dict[str, str] = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._provider: str = config.get("provider", "openai")
        self._model: str = config.get("model", "gpt-4o")
        self._timeout: int = int(config.get("timeout", 30))
        self._api_key_env: str = config.get("api_key_env", "OPENAI_API_KEY")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # AIBase implementation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* to the cloud provider and return the response.

        Parameters
        ----------
        prompt:
            The user message to send.
        **kwargs:
            Optional overrides: ``model``, ``temperature``, ``max_tokens``.

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        RuntimeError
            When the API key is missing, or the request fails.
        """
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not set.  Export the environment variable '{self._api_key_env}'."
            )

        endpoint = self._ENDPOINTS.get(self._provider)
        if not endpoint:
            raise RuntimeError(f"Unsupported cloud provider: '{self._provider}'")

        if self._provider == "openai":
            return self._call_openai(endpoint, api_key, prompt, **kwargs)
        if self._provider == "anthropic":
            return self._call_anthropic(endpoint, api_key, prompt, **kwargs)

        raise RuntimeError(f"Provider '{self._provider}' handler not implemented.")

    def is_available(self) -> bool:
        """Return ``True`` when the API key environment variable is set."""
        return bool(os.environ.get(self._api_key_env))

    def shutdown(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
        super().shutdown()

    # ------------------------------------------------------------------
    # Provider-specific helpers
    # ------------------------------------------------------------------

    def _call_openai(
        self, endpoint: str, api_key: str, prompt: str, **kwargs: Any
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        self._logger.debug("CloudAI OpenAI request | model=%s", payload["model"])
        try:
            resp = self._session.post(
                endpoint, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"CloudAI request failed: {exc}") from exc

        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]
        self._logger.debug("CloudAI response (%d chars).", len(text))
        return text

    def _call_anthropic(
        self, endpoint: str, api_key: str, prompt: str, **kwargs: Any
    ) -> str:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
        }

        self._logger.debug("CloudAI Anthropic request | model=%s", payload["model"])
        try:
            resp = self._session.post(
                endpoint, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"CloudAI (Anthropic) request failed: {exc}") from exc

        data = resp.json()
        text: str = data["content"][0]["text"]
        self._logger.debug("CloudAI response (%d chars).", len(text))
        return text
