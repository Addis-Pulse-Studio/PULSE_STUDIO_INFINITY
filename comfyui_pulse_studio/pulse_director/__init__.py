"""Pulse Infinity's pluggable director. Stdlib only.

  base_director -- BaseDirector (the spec's inspect_shot / direct_next_beat, plus
                   step), Capabilities, EvidencePack, TurnRequest, the errors
  exchange      -- ExchangeDirector, run_turn / validate_answer, the transports
  mailbox       -- the Director Desk: pulse-desk/1 over files
  cli_transport -- the one place a process is started
  prompts       -- the director brief and how a turn is put
  config        -- the director_backend toggle and every default
  director_factory -- one key in, one director out
  claude_code_provider, gemini_antigravity_provider, ollama_provider,
  pi_provider, scripted_provider -- the backends
"""

from .base_director import (
    KIND_OBSERVE,
    KIND_OPEN,
    KIND_PLAN,
    KIND_TURN,
    BaseDirector,
    Capabilities,
    DirectorCancelled,
    DirectorError,
    DirectorRejected,
    DirectorTimeout,
    EvidencePack,
    TurnRequest,
    default_constraints,
)
from .config import (
    BACKENDS,
    DEFAULT_BACKEND,
    FROM_CONFIG,
    InfinityConfig,
    find_config_file,
    parse_config_text,
)
from .director_factory import REGISTRY, available_backends, create_director
from .exchange import ExchangeDirector, run_turn, validate_answer
from .mailbox import PROTOCOL, Mailbox
from .prompts import DIRECTOR_BRIEF, OBSERVER_BRIEF, convert_path, render_prompt
from .scripted_provider import parse_script

__all__ = [
    "BACKENDS",
    "BaseDirector",
    "Capabilities",
    "DEFAULT_BACKEND",
    "DIRECTOR_BRIEF",
    "DirectorCancelled",
    "DirectorError",
    "DirectorRejected",
    "DirectorTimeout",
    "EvidencePack",
    "ExchangeDirector",
    "FROM_CONFIG",
    "InfinityConfig",
    "KIND_OBSERVE",
    "KIND_OPEN",
    "KIND_PLAN",
    "KIND_TURN",
    "Mailbox",
    "OBSERVER_BRIEF",
    "PROTOCOL",
    "REGISTRY",
    "TurnRequest",
    "available_backends",
    "convert_path",
    "create_director",
    "default_constraints",
    "find_config_file",
    "parse_config_text",
    "parse_script",
    "render_prompt",
    "run_turn",
    "validate_answer",
]
