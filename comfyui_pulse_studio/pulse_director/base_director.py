"""The pluggable director: one contract, whichever model decides the next beat.

Pulse Infinity is model-agnostic in both directions. Nothing here knows which
diffusion weights render a beat, and nothing in the loop knows which model
directs it: Claude Code reading a file mailbox, Gemini in Antigravity, a local
model behind its CLI, or a script. Every one of them is a `BaseDirector`, asked
the same question in the same shape and held to the same answer
(pulse_memory.schema).

THE SPEC'S TWO METHODS, AND THE ONE THE LOOP USES
-------------------------------------------------
`inspect_shot(clip_path, last_frame, current_memory)` and
`direct_next_beat(memory, story_goal)` are the abstract interface the mission
asks for, and every backend implements them. They are also two round trips.
The loop asks both questions at once -- `step(request, memory)`, one `TurnRequest`
answered with one `TurnResponse` (observation + plan) -- because for a director
that is a person-paced agent session, one exchange per beat instead of two is
the difference between a film and a wait.

`last_frame` is a PNG path or an `EvidencePack`, never a tensor. The core may not
import torch (tests/test_compiler_hardening.py), and every director needs files
anyway: an agent reads images from disk, a CLI is handed paths. The node layer
writes the frames first.

PATHS
-----
Every path in a request is relative to the session folder. ComfyUI runs on
Windows; Claude Code may run under WSL in the same moment, looking at the same
files as `/mnt/d/...`. A relative path is the one spelling both resolve.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Callable, Optional

from ..pulse_memory.scene_memory import SceneMemory
from ..pulse_memory.schema import (
    MAX_BEAT_SECONDS,
    MIN_BEAT_SECONDS,
    DirectedShotPlan,
    ObservationResult,
    TurnResponse,
)

__all__ = [
    "KIND_OPEN",
    "KIND_TURN",
    "KIND_OBSERVE",
    "KIND_PLAN",
    "KINDS",
    "DirectorError",
    "DirectorTimeout",
    "DirectorCancelled",
    "DirectorRejected",
    "Capabilities",
    "EvidencePack",
    "TurnRequest",
    "BaseDirector",
]

#: The opening turn: plan beat 1; nothing has been rendered to observe.
KIND_OPEN = "open"
#: Observe the beat just rendered, and plan the next -- one exchange.
KIND_TURN = "turn"
#: Observe only (the spec's inspect_shot on its own).
KIND_OBSERVE = "observe"
#: Plan only (the spec's direct_next_beat on its own, or a replan).
KIND_PLAN = "plan"
KINDS = (KIND_OPEN, KIND_TURN, KIND_OBSERVE, KIND_PLAN)


class DirectorError(RuntimeError):
    """The director could not be reached, or did not answer usefully."""


class DirectorTimeout(DirectorError):
    """Nobody answered in time -- for a mailbox, usually nobody is at the desk."""


class DirectorCancelled(DirectorError):
    """ComfyUI's interrupt was pressed while the director was thinking."""


class DirectorRejected(DirectorError):
    """Every attempt broke the contract. `problems` says how, most recent last."""

    def __init__(self, message: str, problems: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.problems = list(problems or [])


class Capabilities:
    """What a backend can take in, and whether what it takes in leaves the machine."""

    __slots__ = ("sees_frames", "sees_video", "hears_audio", "cloud")

    def __init__(self, sees_frames: bool = True, sees_video: bool = False,
                 hears_audio: bool = False, cloud: bool = False) -> None:
        self.sees_frames = bool(sees_frames)
        self.sees_video = bool(sees_video)
        self.hears_audio = bool(hears_audio)
        self.cloud = bool(cloud)

    def to_dict(self) -> dict:
        return {"sees_frames": self.sees_frames, "sees_video": self.sees_video,
                "hears_audio": self.hears_audio, "cloud": self.cloud}

    @classmethod
    def from_dict(cls, data: Any) -> "Capabilities":
        data = data if isinstance(data, dict) else {}
        return cls(data.get("sees_frames", True), data.get("sees_video", False),
                   data.get("hears_audio", False), data.get("cloud", False))

    def describe(self) -> str:
        senses = [s for s, on in (("frames", self.sees_frames), ("video", self.sees_video),
                                  ("audio", self.hears_audio)) if on] or ["text only"]
        return " + ".join(senses)


class EvidencePack:
    """What a rendered beat left on disk for a director to look at.

    Paths are relative to `root`, the session folder. `frames` are evenly spaced
    stills (first, quarter, half, three quarters, last); `contact_sheet` is all of
    them in one image -- one file a director can read to see the motion.
    """

    __slots__ = ("root", "beat", "clip", "audio", "last_frame", "frames", "contact_sheet")

    def __init__(self, root: str, beat: int, clip: str = "", audio: str = "",
                 last_frame: str = "", frames: Optional[list[str]] = None,
                 contact_sheet: str = "") -> None:
        self.root = root
        self.beat = int(beat)
        self.clip = clip
        self.audio = audio
        self.last_frame = last_frame
        self.frames = list(frames or [])
        self.contact_sheet = contact_sheet

    def absolute(self, relative: str) -> str:
        return os.path.join(self.root, *relative.replace("\\", "/").split("/")) if relative else ""

    def look_at(self, capabilities: Capabilities, all_frames: bool = False) -> list[str]:
        """The files this backend should be pointed at, most telling first."""
        files = []
        if capabilities.sees_video and self.clip:
            files.append(self.clip)
        if capabilities.hears_audio and self.audio and not capabilities.sees_video:
            files.append(self.audio)
        if capabilities.sees_frames:
            files += [f for f in (self.contact_sheet, self.last_frame) if f]
            if all_frames:
                files += [f for f in self.frames if f not in files]
        return files

    def to_dict(self) -> dict:
        return {"beat": self.beat, "clip": self.clip, "audio": self.audio,
                "last_frame": self.last_frame, "frames": list(self.frames),
                "contact_sheet": self.contact_sheet}

    @classmethod
    def from_dict(cls, root: str, data: Any) -> "EvidencePack":
        data = data if isinstance(data, dict) else {}
        return cls(root, int(data.get("beat") or 0), str(data.get("clip") or ""),
                   str(data.get("audio") or ""), str(data.get("last_frame") or ""),
                   [str(f) for f in (data.get("frames") or []) if f],
                   str(data.get("contact_sheet") or ""))


class TurnRequest:
    """One question to a director, in full. What the mailbox writes, what a CLI is fed.

    `beat` is the beat being *planned*; on a turn the beat being observed is
    `beat - 1`. `memory` is the bounded view (SceneMemory.view_for_director), and
    `rejection` carries the previous attempt's problems, so a retry is told
    exactly what to fix.
    """

    __slots__ = ("kind", "session", "beat", "attempt", "story_goal", "memory",
                 "memory_text", "last_plan", "evidence", "constraints", "rejection",
                 "cast", "pacing", "guidance")

    def __init__(self, kind: str, session: str, beat: int, attempt: int = 1,
                 story_goal: str = "", memory: Optional[dict] = None, memory_text: str = "",
                 last_plan: Optional[dict] = None, evidence: Optional[EvidencePack] = None,
                 constraints: Optional[dict] = None, rejection: Optional[list[str]] = None,
                 cast: Optional[list[dict]] = None, pacing: Optional[dict] = None,
                 guidance: str = "") -> None:
        if kind not in KINDS:
            raise ValueError("turn kind must be one of %r, got %r" % (KINDS, kind))
        self.kind = kind
        self.session = session
        self.beat = int(beat)
        self.attempt = int(attempt)
        self.story_goal = story_goal
        self.memory = dict(memory or {})
        self.memory_text = memory_text
        self.last_plan = last_plan
        self.evidence = evidence
        self.constraints = dict(constraints or {})
        self.rejection = list(rejection or [])
        self.cast = list(cast or [])
        self.pacing = dict(pacing or {})
        self.guidance = guidance

    @property
    def turn_id(self) -> str:
        return "b%04d-%s-a%d" % (self.beat, self.kind, self.attempt)

    @property
    def observes(self) -> bool:
        return self.kind in (KIND_TURN, KIND_OBSERVE)

    @property
    def plans(self) -> bool:
        return self.kind in (KIND_OPEN, KIND_TURN, KIND_PLAN)

    @property
    def schema_kind(self) -> str:
        """Which pulse_memory.schema.json_schema answers this request."""
        return {KIND_OPEN: "open", KIND_TURN: "turn", KIND_OBSERVE: "observation",
                KIND_PLAN: "plan"}[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "turn_id": self.turn_id, "session": self.session,
                "beat": self.beat, "observe_beat": self.beat - 1 if self.observes else None,
                "attempt": self.attempt, "story_goal": self.story_goal,
                "memory": self.memory, "memory_text": self.memory_text,
                "last_plan": self.last_plan,
                "evidence": self.evidence.to_dict() if self.evidence is not None else None,
                "constraints": self.constraints, "rejection": list(self.rejection),
                "cast": list(self.cast), "pacing": dict(self.pacing),
                "guidance": self.guidance}

    @classmethod
    def from_dict(cls, root: str, data: dict) -> "TurnRequest":
        evidence = data.get("evidence")
        return cls(data["kind"], data.get("session", ""), int(data["beat"]),
                   int(data.get("attempt") or 1), data.get("story_goal", ""),
                   data.get("memory"), data.get("memory_text", ""), data.get("last_plan"),
                   EvidencePack.from_dict(root, evidence) if evidence else None,
                   data.get("constraints"), data.get("rejection"), data.get("cast"),
                   data.get("pacing"), data.get("guidance", ""))


def default_constraints(memory: SceneMemory, scene_budget: int = 8,
                        may_end: bool = False) -> dict:
    """The rules a plan is checked against, stated up front so a director can obey
    them the first time instead of learning them from a rejection."""
    from ..constants import FPS, MAX_DIALOGUE_WPM

    return {
        "duration_seconds": [round(MIN_BEAT_SECONDS, 3), round(MAX_BEAT_SECONDS, 3)],
        "fps": FPS,
        "frame_grid": "17k+5 frames; durations snap up to it",
        "max_dialogue_wpm": MAX_DIALOGUE_WPM,
        "cast": memory.cast_names,
        "forbidden_actions": memory.forbidden_actions,
        "scene_beats_left": max(0, scene_budget - memory.scene_length),
        "must_change_scene": bool(memory.scene and memory.scene_length >= scene_budget),
        "stalled": memory.stagnant,
        "may_end_story": bool(may_end),
    }


class BaseDirector(abc.ABC):
    """Whatever decides the next beat. Subclasses set `name` and `capabilities`."""

    name = "base"
    label = "Director"
    capabilities = Capabilities()

    def __init__(self, settings: Optional[dict] = None, session_root: str = "",
                 should_cancel: Optional[Callable[[], bool]] = None) -> None:
        self.settings = dict(settings or {})
        self.session_root = session_root
        self.should_cancel = should_cancel or (lambda: False)

    # ── the spec's interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def inspect_shot(self, clip_path: str, last_frame: "str | EvidencePack",
                     current_memory: SceneMemory) -> ObservationResult:
        """What the beat just rendered actually shows, as a delta on memory."""

    @abc.abstractmethod
    def direct_next_beat(self, memory: SceneMemory, story_goal: str) -> DirectedShotPlan:
        """The next beat, as a DirectedShotPlan."""

    # ── what the loop calls ─────────────────────────────────────────────────

    def step(self, request: TurnRequest, memory: SceneMemory) -> TurnResponse:
        """Both answers for one beat. The default asks the two questions in turn;
        exchange-based backends answer both in one round trip instead."""
        observation = None
        if request.observes and request.evidence is not None:
            observation = self.inspect_shot(request.evidence.absolute(request.evidence.clip),
                                            request.evidence, memory)
        plan = self.direct_next_beat(memory, request.story_goal)
        return TurnResponse(plan, observation)

    def describe(self) -> str:
        """One line for the report: who is directing, how, and where data goes."""
        where = ("cloud agent -- the evidence frames are read by it and may leave this "
                 "machine through it" if self.capabilities.cloud else "runs on this machine")
        return "%s (%s; sees %s; %s)" % (self.name, self.transport_name, self.capabilities.describe(),
                                         where)

    @property
    def transport_name(self) -> str:
        return str(self.settings.get("transport") or "inline")

    def close(self) -> None:
        """Release anything held between turns. Most backends hold nothing."""
