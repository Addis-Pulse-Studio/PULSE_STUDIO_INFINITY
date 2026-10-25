"""Backend A, the default: Claude Code, directing from its workspace.

Claude Code is an agent in a folder, so it is reached the way an agent works:
through files. With the `mailbox` transport (the default) ComfyUI posts each turn
on the Director Desk, and a Claude Code session running the `pulse-director` skill
(installed by tools/setup_director_workspace.py) waits at the desk, reads the
request and the evidence frames from disk -- its Read tool views images -- and
submits the turn through `tools/pulse_desk.py`, which runs this pack's own
validator first. Under `/loop` it directs indefinitely.

It sees stills, not video and not sound: each turn points it at a contact sheet
of the beat (first, quarter, half, three quarters, last frame) and the last
frame. Speech it cannot hear is taken from the plan's intent and kept marked as
such in memory.

The `cli` transport runs `claude -p` headless for unattended films. When ComfyUI
runs on Windows and Claude Code under WSL, the command goes through `wsl.exe`
and `path_style: wsl` respells every path for it -- see pulse_infinity.example.yaml.

Claude Code is a cloud agent: the frames it reads leave this machine through it.
The pack itself sends nothing, and the node report says which backend is active.
"""

from __future__ import annotations

from .base_director import Capabilities
from .exchange import ExchangeDirector

__all__ = ["ClaudeCodeDirector"]


class ClaudeCodeDirector(ExchangeDirector):
    name = "claude_code"
    label = "Claude Code"
    capabilities = Capabilities(sees_frames=True, sees_video=False, hears_audio=False,
                                cloud=True)
    default_transport = "mailbox"
    default_settings = {
        "transport": "mailbox",
        "timeout_seconds": 900,
        "poll_seconds": 0.5,
        # Headless alternative. `--add-dir` lets it Read the session's frames.
        "cli": ["claude", "-p", "--output-format", "json", "--allowedTools", "Read",
                "--add-dir", "{session_root}"],
        "path_style": "native",
    }
