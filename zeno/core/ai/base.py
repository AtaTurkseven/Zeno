"""Abstract base class for all AI backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class AIBase(ABC):
    """Contract that every AI backend must satisfy.

    All concrete backends (local LLM, cloud API, …) inherit from this class
    and implement :meth:`generate` and :meth:`is_available`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        config:
            Backend-specific configuration dict (e.g. the ``ai.local``
            section from ``settings.yaml``).
        """
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response for *prompt*.

        Parameters
        ----------
        prompt:
            The user / system prompt to send to the model.
        **kwargs:
            Backend-specific overrides (temperature, max_tokens, …).

        Returns
        -------
        str
            The model's response text.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the backend is reachable and ready."""

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Perform any one-time setup (model loading, connection, …)."""
        self._logger.debug("%s initialised.", self.__class__.__name__)

    def shutdown(self) -> None:
        """Release resources held by the backend."""
        self._logger.debug("%s shut down.", self.__class__.__name__)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(config={self.config})"
