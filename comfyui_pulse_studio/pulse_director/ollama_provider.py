"""Backend C: a local model through Ollama's own CLI. Offline, nothing leaves.

`ollama run <model> --format json`, with the prompt on stdin. The pack opens no
socket: the CLI talks to the Ollama server, the pack talks to the CLI. For a
vision model (Gemma 4, a Qwen-VL, any multimodal tag) the prompt names the
contact sheet and the last frame on lines of their own, which is how Ollama's CLI
attaches images; set `vision: false` for a text-only model, and its
observations are taken from the plan's intent, marked as such.

The GPU is shared. H3 holds about 21 GB of a 32 GB card while it is loaded, and
a 27B model wants most of the rest, so `free_vram_before_call` (on by default
here) lets the node layer unload ComfyUI's models before the call, and
`--keepalive {keep_alive}` (0s by default) makes Ollama unload its own model the
moment it has answered, so H3 has the card back for the next beat.

The default command was checked against Ollama 0.35: `--nowordwrap`, because
the CLI word-wraps with terminal escape codes even into a pipe, and
`--hidethinking`, because a thinking model's reasoning is printed as plain text
ahead of the answer. Both are also cleaned up on the way in, for older builds.
Use a model tag that exists (`ollama list`): a bare `gemma4` asks Ollama for
`gemma4:latest`, which it would try to download.

The default, qwen3.8:27b, was chosen by running both candidates on the same
turn -- a beat whose stills show nothing of the plan. It answered in one attempt
and reported the empty frames as empty; gemma4:12b also answered in one attempt
but narrated the plan instead of the pictures. gemma4:12b is the lighter choice
when VRAM is tight.
"""

from __future__ import annotations

from .base_director import Capabilities
from .exchange import ExchangeDirector

__all__ = ["OllamaDirector"]


class OllamaDirector(ExchangeDirector):
    name = "ollama"
    label = "Ollama (local)"
    capabilities = Capabilities(sees_frames=True, sees_video=False, hears_audio=False,
                                cloud=False)
    default_transport = "cli"
    default_settings = {
        "transport": "cli",
        "model": "qwen3.8:27b",
        "vision": True,
        "timeout_seconds": 600,
        "keep_alive": "0s",
        "cli": ["ollama", "run", "{model}", "--format", "json", "--nowordwrap",
                "--hidethinking", "--keepalive", "{keep_alive}"],
        "path_style": "native",
        "free_vram_before_call": True,
    }

    def __init__(self, settings=None, *args, **kwargs) -> None:
        super().__init__(settings, *args, **kwargs)
        if not self.settings.get("vision", True):
            # Per instance: a text-only tag must not change what the class says.
            self.capabilities = Capabilities(sees_frames=False, cloud=False)
