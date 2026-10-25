"""Where a session's memory lives on disk, and how it survives a crash.

`scene_memory.json` sits at the root of the session folder -- the "active
project directory" -- so a person can open it and see what the film remembers.
Every commit writes it durably (durable.write_json_durably: temp file, fsync,
atomic replace), and also writes a snapshot of the memory as it stood *before*
that beat, `memory_snapshots/beat_NNNN.json`. The snapshot is what makes undo
cheap: a rerolled or rejected beat restores the memory it was planned from,
rather than unpicking its changes.

A memory file that will not parse is never overwritten. It is moved aside to
`scene_memory.corrupt.json` -- it may be the only record of a long film -- and the
newest snapshot is loaded in its place, said plainly in the notes. The same
discipline as segcache.Manifest.load.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from ..durable import write_json_durably
from .scene_memory import SceneMemory

__all__ = [
    "MEMORY_FILE",
    "SNAPSHOT_DIR",
    "save_memory",
    "load_memory",
    "snapshot_memory",
    "restore_snapshot",
    "snapshot_beats",
]

MEMORY_FILE = "scene_memory.json"
SNAPSHOT_DIR = "memory_snapshots"
_SNAPSHOT_RE = re.compile(r"^beat_(\d{4,})\.json$")


def _snapshot_path(directory: str, beat: int) -> str:
    return os.path.join(directory, SNAPSHOT_DIR, "beat_%04d.json" % int(beat))


def save_memory(directory: str, memory: SceneMemory) -> str:
    """Write `scene_memory.json` durably. Returns its path."""
    return write_json_durably(os.path.join(directory, MEMORY_FILE), memory.to_dict(),
                              prefix=".scene_memory-")


def snapshot_memory(directory: str, memory: SceneMemory, beat: int) -> str:
    """Keep the memory as it stood before beat `beat` was committed."""
    return write_json_durably(_snapshot_path(directory, beat), memory.to_dict(),
                              prefix=".snapshot-")


def snapshot_beats(directory: str) -> list[int]:
    """The beats a snapshot exists for, ascending."""
    try:
        names = os.listdir(os.path.join(directory, SNAPSHOT_DIR))
    except OSError:
        return []
    return sorted(int(m.group(1)) for m in map(_SNAPSHOT_RE.match, names) if m)


def _read(path: str) -> SceneMemory:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "current_beat" not in data:
        raise ValueError("not a scene memory document")
    return SceneMemory.from_dict(data)


def restore_snapshot(directory: str, beat: int) -> Optional[SceneMemory]:
    """The memory as it stood before beat `beat`, or None if no snapshot is kept."""
    path = _snapshot_path(directory, beat)
    if not os.path.exists(path):
        return None
    try:
        return _read(path)
    except (OSError, ValueError):
        return None


def load_memory(directory: str, story_goal: str = "") -> tuple[SceneMemory, list[str]]:
    """The session's memory, or a fresh one. Returns (memory, notes).

    Never raises for a bad file: a corrupt memory is moved aside and the newest
    snapshot that reads is loaded instead, with a note saying exactly that.
    """
    path = os.path.join(directory, MEMORY_FILE)
    if not os.path.exists(path):
        return SceneMemory(story_goal), []
    try:
        return _read(path), []
    except (OSError, ValueError) as exc:
        spoiled = os.path.join(directory, "scene_memory.corrupt.json")
        try:
            os.replace(path, spoiled)
        except OSError:  # pragma: no cover - unwritable directory
            pass
        for beat in reversed(snapshot_beats(directory)):
            restored = restore_snapshot(directory, beat)
            if restored is not None:
                return restored, [
                    "scene_memory.json could not be read (%s); it was kept as "
                    "scene_memory.corrupt.json and the memory from before beat %d was "
                    "restored" % (exc, beat)]
        return SceneMemory(story_goal), [
            "scene_memory.json could not be read (%s) and no snapshot exists; it was "
            "kept as scene_memory.corrupt.json and the film starts with a fresh memory"
            % (exc,)]
