"""One key in, one director out. See config.py for how the key is chosen."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .base_director import DirectorError
from .claude_code_provider import ClaudeCodeDirector
from .config import BACKENDS, InfinityConfig
from .exchange import ExchangeDirector
from .gemini_antigravity_provider import GeminiAntigravityDirector
from .ollama_provider import OllamaDirector
from .pi_provider import PiDirector
from .scripted_provider import ScriptedDirector

__all__ = ["REGISTRY", "create_director", "available_backends"]

REGISTRY = {
    "claude_code": ClaudeCodeDirector,
    "gemini_antigravity": GeminiAntigravityDirector,
    "ollama": OllamaDirector,
    "pi": PiDirector,
    "scripted": ScriptedDirector,
}


def available_backends() -> tuple:
    return BACKENDS


def create_director(config: InfinityConfig, session_root: str, session: str = "",
                    should_cancel: Optional[Callable[[], bool]] = None,
                    runner: Optional[Callable[..., Any]] = None,
                    on_status: Optional[Callable[[str], None]] = None,
                    overrides: Optional[dict] = None) -> ExchangeDirector:
    """The active director for one session.

    `overrides` are settings the graph supplies for this run -- the scripted
    director's script from the Loop Controller's widget -- laid over the
    backend's section of the config.
    """
    cls = REGISTRY.get(config.backend)
    if cls is None:  # pragma: no cover - InfinityConfig.resolve only yields known names
        raise DirectorError("unknown director backend %r; one of %s"
                            % (config.backend, ", ".join(BACKENDS)))
    settings = config.backend_settings()
    settings.update(overrides or {})
    return cls(settings, session_root, should_cancel=should_cancel, runner=runner,
               on_status=on_status, session=session)
