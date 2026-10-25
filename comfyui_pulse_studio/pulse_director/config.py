"""Pulse Infinity's configuration: one key picks the director, the rest has defaults.

THE TOGGLE
----------
    director_backend: claude_code      # or gemini_antigravity | ollama | pi | scripted

One active director at a time, chosen from four places, first match wins:

  1. the Loop Controller's `director_backend` widget, unless it says
     `from_config` (its default) -- a per-graph override;
  2. the environment variable PULSE_DIRECTOR_BACKEND -- a per-launch override;
  3. `director_backend` in pulse_infinity.yaml (or .json);
  4. `claude_code`, the built-in default.

The report always names the winner and where it came from, so "why is it waiting
for Claude" has a one-line answer.

WHERE THE FILE LIVES
--------------------
`$PULSE_INFINITY_CONFIG` if set, else `pulse_infinity.yaml` (or `.json`) in
ComfyUI's user directory -- outside the pack, so an update never touches it. The
pack ships `pulse_infinity.example.yaml` to copy from.

The file is parsed by the node layer, with ComfyUI's own PyYAML. This module only
ever receives the parsed dict (or parses JSON itself): the core imports no YAML
library, because CI and the desk tool run it with nothing installed.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "FROM_CONFIG",
    "ENV_BACKEND",
    "ENV_CONFIG",
    "CONFIG_NAMES",
    "DEFAULTS",
    "InfinityConfig",
    "parse_config_text",
    "find_config_file",
]

BACKENDS = ("claude_code", "gemini_antigravity", "ollama", "pi", "scripted")
DEFAULT_BACKEND = "claude_code"
FROM_CONFIG = "from_config"
ENV_BACKEND = "PULSE_DIRECTOR_BACKEND"
ENV_CONFIG = "PULSE_INFINITY_CONFIG"
CONFIG_NAMES = ("pulse_infinity.yaml", "pulse_infinity.yml", "pulse_infinity.json")

#: Every setting with its default. A config file overrides any subset of it.
DEFAULTS: dict = {
    "director_backend": DEFAULT_BACKEND,
    # Folder under ComfyUI/output that holds one folder per session.
    "sessions_dir": "pulse_infinity",
    "loop": {
        "max_attempts": 3,            # per turn, before the beat is refused
        "on_invalid": "fail",         # or "accept": render the last answer anyway, warned
        "blacklist_size": 8,
        "blacklist_ttl": 4,           # beats a forbidden action stays forbidden
        "scene_budget": 8,            # beats before a scene must cut
        "min_beats_before_end": 3,    # end_of_story is refused before this
        "reel_every": 1,              # rebuild reel.mp4 every N committed beats
    },
    "backends": {
        "claude_code": {},
        "gemini_antigravity": {},
        "ollama": {},
        "pi": {},
        "scripted": {},
    },
}


def _merge(base: dict, override: Mapping) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_config_text(text: str, name: str = "",
                      yaml_loader: Optional[Callable[[str], Any]] = None
                      ) -> tuple[dict, list[str]]:
    """A config file's text, as a dict. Returns (data, problems).

    JSON is read here. YAML needs `yaml_loader` -- the node layer passes
    `yaml.safe_load` -- because the core imports no YAML library.
    """
    text = text or ""
    if not text.strip():
        return {}, []
    is_json = name.lower().endswith(".json") or text.lstrip().startswith("{")
    try:
        if is_json:
            data = json.loads(text)
        elif yaml_loader is None:
            return {}, ["%s is YAML, and no YAML reader was available to read it"
                        % (name or "the config")]
        else:
            data = yaml_loader(text)
    except Exception as exc:  # any parser's error is the user's to see
        return {}, ["%s could not be read: %s" % (name or "the config", exc)]
    if data is None:
        return {}, []
    if not isinstance(data, dict):
        return {}, ["%s must hold a mapping at the top level" % (name or "the config")]
    return data, []


def find_config_file(user_directory: str = "",
                     environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The config file to read, or None: $PULSE_INFINITY_CONFIG, else the user dir."""
    environ = os.environ if environ is None else environ
    explicit = (environ.get(ENV_CONFIG) or "").strip()
    if explicit:
        return explicit
    for name in CONFIG_NAMES:
        path = os.path.join(user_directory or "", name)
        if user_directory and os.path.isfile(path):
            return path
    return None


class InfinityConfig:
    """The merged settings, and which director won and why."""

    __slots__ = ("data", "backend", "backend_source", "path", "problems")

    def __init__(self, data: dict, backend: str, backend_source: str, path: str = "",
                 problems: Optional[list[str]] = None) -> None:
        self.data = data
        self.backend = backend
        self.backend_source = backend_source
        self.path = path
        self.problems = list(problems or [])

    @classmethod
    def resolve(cls, file_data: Optional[Mapping] = None, widget: str = FROM_CONFIG,
                environ: Optional[Mapping[str, str]] = None, path: str = "",
                file_problems: Optional[list[str]] = None) -> "InfinityConfig":
        """Merge the file over the defaults and pick the director. Never raises;
        an unusable choice is a problem, and the next source down is used."""
        environ = os.environ if environ is None else environ
        problems = list(file_problems or [])
        data = _merge(DEFAULTS, file_data or {})

        candidates = []
        if widget and widget != FROM_CONFIG:
            candidates.append((widget, "the Loop Controller's director_backend widget"))
        if (environ.get(ENV_BACKEND) or "").strip():
            candidates.append((environ[ENV_BACKEND].strip(), "$%s" % ENV_BACKEND))
        if file_data and file_data.get("director_backend"):
            candidates.append((str(file_data["director_backend"]).strip(),
                               "director_backend in %s" % (path or "the config file")))
        candidates.append((DEFAULT_BACKEND, "the built-in default"))

        for name, source in candidates:
            if name in BACKENDS:
                return cls(data, name, source, path, problems)
            problems.append("%s names director backend %r, which is not one of %s; using "
                            "the next choice" % (source, name, ", ".join(BACKENDS)))
        return cls(data, DEFAULT_BACKEND, "the built-in default", path, problems)  # pragma: no cover

    def loop(self, key: str) -> Any:
        return self.data["loop"].get(key, DEFAULTS["loop"].get(key))

    def backend_settings(self, name: Optional[str] = None) -> dict:
        return dict((self.data.get("backends") or {}).get(name or self.backend) or {})

    def describe(self) -> str:
        return "%s (from %s)" % (self.backend, self.backend_source)
