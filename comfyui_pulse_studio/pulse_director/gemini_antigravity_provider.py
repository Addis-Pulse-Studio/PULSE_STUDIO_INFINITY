"""Backend B: Gemini, directing from an Antigravity workspace.

Like Claude Code, an agent in a folder, reached through the Director Desk. The
workspace kit installs Antigravity rules and a `/pulse-direct` workflow
(`.agent/rules/`, `.agent/workflows/`) with the same brief, and the agent submits
through the same `tools/pulse_desk.py`.

What it adds is the long multimodal context: each turn points it at the whole
rendered clip of the beat -- video *and* its native sound -- before the stills,
so it can check what was actually said and how, not only what was meant.
Whether Antigravity's agent can open an mp4 from disk is a property of its
tools; the request also lists the contact sheet and last frame, so it never has
less to go on than Claude Code.

Antigravity has no headless CLI of its own. The `cli` transport runs the Gemini
CLI (`gemini -p`) instead, for unattended films on the same model family.

A cloud agent: the clip and frames it reads leave this machine through it.
"""

from __future__ import annotations

from .base_director import Capabilities
from .exchange import ExchangeDirector

__all__ = ["GeminiAntigravityDirector"]


class GeminiAntigravityDirector(ExchangeDirector):
    name = "gemini_antigravity"
    label = "Gemini (Antigravity)"
    capabilities = Capabilities(sees_frames=True, sees_video=True, hears_audio=True,
                                cloud=True)
    default_transport = "mailbox"
    default_settings = {
        "transport": "mailbox",
        "timeout_seconds": 900,
        "poll_seconds": 0.5,
        "cli": ["gemini", "-p", "{instruction}"],
        "path_style": "native",
    }
