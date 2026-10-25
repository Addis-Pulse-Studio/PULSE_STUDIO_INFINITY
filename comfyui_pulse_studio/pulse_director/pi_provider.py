"""Backend D: the pi coding agent, for light and offline direction.

pi is a terminal agent that runs against whatever model it is configured with,
local ones through Ollama included -- so it is the lightweight way to direct a
test film with no cloud at all. Two ways in, like Claude Code:

  * `cli` (the default): `pi -p` print mode, the prompt on stdin, the JSON on
    stdout;
  * `mailbox`: pi running in the director workspace, using the desk exactly as
    Claude Code does.

Whether it sees the frames depends on the model behind it: `vision: false` for
a text-only one, whose observations are then the plan's intent.
"""

from __future__ import annotations

from .base_director import Capabilities
from .exchange import ExchangeDirector

__all__ = ["PiDirector"]


class PiDirector(ExchangeDirector):
    name = "pi"
    label = "pi agent"
    capabilities = Capabilities(sees_frames=True, sees_video=False, hears_audio=False,
                                cloud=False)
    default_transport = "cli"
    default_settings = {
        "transport": "cli",
        "vision": True,
        "timeout_seconds": 600,
        "cli": ["pi", "-p", "{instruction}"],
        "path_style": "native",
        "free_vram_before_call": True,
    }

    def __init__(self, settings=None, *args, **kwargs) -> None:
        super().__init__(settings, *args, **kwargs)
        if not self.settings.get("vision", True):
            self.capabilities = Capabilities(sees_frames=False, cloud=False)
