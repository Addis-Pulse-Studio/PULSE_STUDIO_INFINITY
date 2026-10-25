"""Backend C: a local model through Ollama's own CLI. Offline, nothing leaves.

`ollama run <model> --format json`, with the prompt on stdin. The pack opens no
socket: the CLI talks to the Ollama server, the pack talks to the CLI. For a
vision model (Gemma 4, a Qwen-VL, any multimodal tag) the prompt names the
contact sheet and the last frame on lines of their own, which is how Ollama's CLI
attaches images; set `vision: false` for a text-only model, and its
observations are taken from the plan's intent, marked as such.

The GPU is shared. H3 holds about 21 GB of a 32 GB card while it is loaded, and
a 27B model wants most of the rest, so `free_vram_before_call` (on by default
here) lets the node layer unload ComfyUI's models before the call. Ollama keeps
its own model resident afterwards unless told not to: add
`["--keepalive", "0s"]`-style flags to `cli` if your Ollama version supports
them, or set OLLAMA_KEEP_ALIVE for the server.

Thinking models' <think> blocks are stripped before the JSON is read.
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
        "model": "gemma4",
        "vision": True,
        "timeout_seconds": 600,
        "cli": ["ollama", "run", "{model}", "--format", "json"],
        "path_style": "native",
        "free_vram_before_call": True,
    }

    def __init__(self, settings=None, *args, **kwargs) -> None:
        super().__init__(settings, *args, **kwargs)
        if not self.settings.get("vision", True):
            # Per instance: a text-only tag must not change what the class says.
            self.capabilities = Capabilities(sees_frames=False, cloud=False)
