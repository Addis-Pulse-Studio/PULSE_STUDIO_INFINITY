"""The contracts every director backend honours: plan, observation, turn.

Pulse Infinity's director is pluggable -- Claude Code over a file mailbox, Gemini
in Antigravity, a local model through its CLI, a script -- and the only thing
the loop may assume about any of them is that what comes back fits these
shapes. So the shapes are enforced here, once, by code every backend and every
tool shares (the node layer, the session, and `tools/pulse_desk.py`, which hands
an agent the same verdict before it submits).

THE DIRECTOR OUTPUT CONTRACT (`DirectedShotPlan`)
-------------------------------------------------
Exactly the fields the mission spec names -- `shot_index`, `visual_prompt`,
`negative_prompt`, `camera_motion`, `duration_seconds`, `dialogue`,
`continuity_anchors` -- and `to_dict()` writes them back exactly as given, so a
plan read and written again is byte-for-byte the plan the director sent. The
optional additions (`beat`, `scene`, `scene_change`, `new_characters`,
`characters_present`, `end_of_story`) are written only when set.

What the validator enforces, and why:

  * the duration is snapped *up* to H3's 17k+5 frame grid, inside the trained
    range (124-362 frames, 5.17-15.08 s). `duration_seconds` keeps what the
    director wrote; `frames` and `rendered_seconds` say what will be rendered,
    and the snap is reported rather than hidden;
  * no hand-typed `<Picture 3>` / `<Video 1>` / `<Audio 2>`. Ordinals are the
    compiler's to assign (assets.LITERAL_TAG_RE is the same rule the bin uses);
    a director names a cast member as `@Name` and the compiler resolves it;
  * every `@Name` is a cast member, when the cast is known -- an unresolved
    mention reaches the model as the literal text "@Bob";
  * the line fits the beat: words per minute at most constants.MAX_DIALOGUE_WPM,
    the rate H3's speech starts to garble past.

Problems are returned, never raised -- the repo-wide convention -- so a director
that got three things wrong hears about all three in one retry.

UNTRUSTED INPUT
---------------
Everything parsed here was written by a model, sometimes through an agent's
file write. `unwrap_json` accepts the shapes models actually emit -- a bare
object, a ```json fence, prose around an object, a tool_use block, a CLI's
result envelope, a <think> preamble -- and nothing in a plan is ever used as a
path. Lengths are capped, so a runaway generation cannot become a 2 MB prompt.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from ..assets import LITERAL_TAG_RE, REF_TOKEN_RE
from ..constants import FPS, MAX_DIALOGUE_WPM, MAX_WINDOW_FRAMES, MIN_TRAINED_FRAMES
from ..frames import align_frame_count, frames_to_seconds

__all__ = [
    "PLAN_REQUIRED",
    "PLAN_OPTIONAL",
    "MIN_BEAT_SECONDS",
    "MAX_BEAT_SECONDS",
    "PROVENANCE_OBSERVED",
    "PROVENANCE_INTENDED",
    "Dialogue",
    "ContinuityAnchors",
    "NewCharacter",
    "DirectedShotPlan",
    "CharacterDelta",
    "EnvironmentDelta",
    "ObservationResult",
    "TurnResponse",
    "unwrap_json",
    "mentions",
    "json_schema",
]

#: The fields the mission spec's Director Output Contract names, in its order.
PLAN_REQUIRED = ("shot_index", "visual_prompt", "negative_prompt", "camera_motion",
                 "duration_seconds", "dialogue", "continuity_anchors")
#: Additions a director may send. Written back only when set.
PLAN_OPTIONAL = ("beat", "scene", "scene_change", "new_characters",
                 "characters_present", "end_of_story")

MIN_BEAT_SECONDS = frames_to_seconds(MIN_TRAINED_FRAMES)   # 5.1667
MAX_BEAT_SECONDS = frames_to_seconds(MAX_WINDOW_FRAMES)    # 15.0833

#: Who vouches for an observation: a director that looked at the frames, or one
#: that could not (a script, a text-only model), whose "observation" is only
#: what the plan intended. Memory keeps the difference.
PROVENANCE_OBSERVED = "observed"
PROVENANCE_INTENDED = "intended"

#: Caps on model-written text. Generous for real direction; small enough that a
#: generation stuck in a loop cannot grow the next prompt without bound.
MAX_TEXT = 2000
MAX_SHORT = 300
MAX_ITEMS = 12

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


# ── small coercions, each reporting what it could not use ───────────────────

def _text(data: dict, field: str, problems: list[str], where: str,
          required: bool = False, limit: int = MAX_TEXT) -> str:
    value = data.get(field)
    if value is None:
        if required:
            problems.append("%s: `%s` is missing" % (where, field))
        return ""
    if not isinstance(value, str):
        problems.append("%s: `%s` must be text, got %s" % (where, field, type(value).__name__))
        return ""
    value = value.strip()
    if required and not value:
        problems.append("%s: `%s` is empty" % (where, field))
    if len(value) > limit:
        problems.append("%s: `%s` is %d characters; keep it under %d"
                        % (where, field, len(value), limit))
        value = value[:limit]
    return value


def _texts(data: dict, field: str, problems: list[str], where: str,
           limit: int = MAX_ITEMS) -> list[str]:
    value = data.get(field)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        problems.append("%s: `%s` must be a list of text" % (where, field))
        return []
    out = []
    for item in value:
        if not isinstance(item, str):
            problems.append("%s: every item of `%s` must be text" % (where, field))
            continue
        item = item.strip()
        if item:
            out.append(item[:MAX_SHORT])
    if len(out) > limit:
        problems.append("%s: `%s` has %d items; at most %d" % (where, field, len(out), limit))
        out = out[:limit]
    return out


def _flag(data: dict, field: str, problems: list[str], where: str) -> Optional[bool]:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, bool):
        problems.append("%s: `%s` must be true or false" % (where, field))
        return None
    return value


def _mapping(value: Any, field: str, problems: list[str], where: str) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, dict):
        problems.append("%s: `%s` must be an object" % (where, field))
        return None
    return value


def word_count(text: str) -> int:
    """Spoken words in a line, in any script. Bracketed delivery tags do not count."""
    return len(_WORD_RE.findall(re.sub(r"\[[^\]]*\]", " ", text or "")))


def mentions(text: str) -> list[str]:
    """The cast names a line cites as @Name or @[Name], in order, de-duplicated."""
    found = []
    for match in REF_TOKEN_RE.finditer(text or ""):
        name = (match.group(2) or match.group(3) or "").strip()
        if name and name not in found:
            found.append(name)
    return found


# ── the plan ────────────────────────────────────────────────────────────────

class Dialogue:
    """One spoken line: who says it, what, and how. Empty text means a silent beat."""

    __slots__ = ("speaker", "text", "emotion_tone")

    def __init__(self, speaker: str = "", text: str = "", emotion_tone: str = "") -> None:
        self.speaker = speaker
        self.text = text
        self.emotion_tone = emotion_tone

    @property
    def spoken(self) -> bool:
        return bool(self.text.strip())

    @property
    def words(self) -> int:
        return word_count(self.text)

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "text": self.text, "emotion_tone": self.emotion_tone}

    @classmethod
    def from_dict(cls, data: Any, problems: list[str], where: str = "dialogue"
                  ) -> Optional["Dialogue"]:
        if data is None:
            return None
        data = _mapping(data, "dialogue", problems, where)
        if data is None:
            return None
        line = cls(_text(data, "speaker", problems, where, limit=MAX_SHORT),
                   _text(data, "text", problems, where, limit=MAX_SHORT * 2),
                   _text(data, "emotion_tone", problems, where, limit=MAX_SHORT))
        if line.spoken and not line.speaker:
            problems.append("%s: a line is spoken but `speaker` is empty" % where)
        return line


class ContinuityAnchors:
    """What must still be true on screen: things held, the costume, the place."""

    __slots__ = ("held_items", "costume", "environment")

    def __init__(self, held_items: Optional[list[str]] = None, costume: str = "",
                 environment: str = "") -> None:
        self.held_items = list(held_items or [])
        self.costume = costume
        self.environment = environment

    def to_dict(self) -> dict:
        return {"held_items": list(self.held_items), "costume": self.costume,
                "environment": self.environment}

    def sentence(self) -> str:
        """The anchors as one line of direction for the shot text."""
        bits = []
        if self.held_items:
            bits.append("holding " + ", ".join(self.held_items))
        if self.costume:
            bits.append("wearing " + self.costume)
        if self.environment:
            bits.append("in " + self.environment)
        return ("Continuity: " + "; ".join(bits) + ".") if bits else ""

    @classmethod
    def from_dict(cls, data: Any, problems: list[str], where: str = "continuity_anchors"
                  ) -> "ContinuityAnchors":
        data = _mapping(data, "continuity_anchors", problems, where) or {}
        return cls(_texts(data, "held_items", problems, where),
                   _text(data, "costume", problems, where, limit=MAX_SHORT),
                   _text(data, "environment", problems, where, limit=MAX_SHORT))


class NewCharacter:
    """A character the director introduces, described concretely enough to pin."""

    __slots__ = ("name", "appearance")

    def __init__(self, name: str, appearance: str = "") -> None:
        self.name = name
        self.appearance = appearance

    def to_dict(self) -> dict:
        return {"name": self.name, "appearance": self.appearance}

    @classmethod
    def list_from(cls, value: Any, problems: list[str], where: str) -> list["NewCharacter"]:
        if value is None:
            return []
        if not isinstance(value, list):
            problems.append("%s: `new_characters` must be a list" % where)
            return []
        out = []
        for item in value[:MAX_ITEMS]:
            if not isinstance(item, dict):
                problems.append("%s: each new character is an object with name and "
                                "appearance" % where)
                continue
            name = _text(item, "name", problems, where + ".new_characters", True, MAX_SHORT)
            if name:
                out.append(cls(name, _text(item, "appearance", problems,
                                           where + ".new_characters", limit=MAX_SHORT)))
        return out


class DirectedShotPlan:
    """The next beat, as the director decided it. The mission's output contract.

    `duration_seconds` is what the director asked for; `frames` and
    `rendered_seconds` are what H3 will actually render, after the snap to the
    frame grid. `notes` holds the non-fatal adjustments, for the report.
    """

    __slots__ = ("shot_index", "visual_prompt", "negative_prompt", "camera_motion",
                 "duration_seconds", "dialogue", "continuity_anchors", "beat", "scene",
                 "scene_change", "new_characters", "characters_present", "end_of_story",
                 "frames", "notes")

    def __init__(self, shot_index: int, visual_prompt: str, negative_prompt: str = "",
                 camera_motion: str = "", duration_seconds: float = MIN_BEAT_SECONDS,
                 dialogue: Optional[Dialogue] = None,
                 continuity_anchors: Optional[ContinuityAnchors] = None,
                 beat: str = "", scene: str = "", scene_change: Optional[bool] = None,
                 new_characters: Optional[list[NewCharacter]] = None,
                 characters_present: Optional[list[str]] = None,
                 end_of_story: Optional[bool] = None, fps: int = FPS) -> None:
        self.shot_index = int(shot_index)
        self.visual_prompt = visual_prompt
        self.negative_prompt = negative_prompt
        self.camera_motion = camera_motion
        self.duration_seconds = duration_seconds
        self.dialogue = dialogue
        self.continuity_anchors = continuity_anchors or ContinuityAnchors()
        self.beat = beat
        self.scene = scene
        self.scene_change = scene_change
        self.new_characters = list(new_characters or [])
        self.characters_present = list(characters_present or [])
        self.end_of_story = end_of_story
        self.notes: list[str] = []
        self.frames = _beat_frames(duration_seconds, fps, self.notes)

    @property
    def rendered_seconds(self) -> float:
        return frames_to_seconds(self.frames)

    @property
    def cuts(self) -> bool:
        """True when this beat opens on a cut rather than continuing the last one."""
        return bool(self.scene_change)

    @property
    def spoken(self) -> bool:
        return bool(self.dialogue is not None and self.dialogue.spoken)

    def to_dict(self) -> dict:
        out = {
            "shot_index": self.shot_index,
            "visual_prompt": self.visual_prompt,
            "negative_prompt": self.negative_prompt,
            "camera_motion": self.camera_motion,
            "duration_seconds": self.duration_seconds,
            "dialogue": self.dialogue.to_dict() if self.dialogue is not None else None,
            "continuity_anchors": self.continuity_anchors.to_dict(),
        }
        if self.beat:
            out["beat"] = self.beat
        if self.scene:
            out["scene"] = self.scene
        if self.scene_change is not None:
            out["scene_change"] = self.scene_change
        if self.new_characters:
            out["new_characters"] = [c.to_dict() for c in self.new_characters]
        if self.characters_present:
            out["characters_present"] = list(self.characters_present)
        if self.end_of_story is not None:
            out["end_of_story"] = self.end_of_story
        return out

    @classmethod
    def from_dict(cls, data: Any, cast_names: Optional[Iterable[str]] = None,
                  fps: int = FPS, where: str = "plan"
                  ) -> tuple[Optional["DirectedShotPlan"], list[str]]:
        """Parse and check one plan. Returns (plan, problems); plan is None when
        the problems leave nothing to render.

        `cast_names` is who may be cited as @Name and may speak: the cast wired
        into the loop plus everyone memory has met. None skips that check -- a
        tool validating a plan with no session to hand.
        """
        problems: list[str] = []
        data = _mapping(data, where, problems, where)
        if data is None:
            return None, problems or ["%s: expected a JSON object" % where]

        shot_index = data.get("shot_index")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int) or shot_index < 1:
            problems.append("%s: `shot_index` must be a whole number from 1, got %r"
                            % (where, shot_index))
            shot_index = 1
        visual = _text(data, "visual_prompt", problems, where, required=True)
        negative = _text(data, "negative_prompt", problems, where)
        camera = _text(data, "camera_motion", problems, where, limit=MAX_SHORT)

        duration = data.get("duration_seconds", MIN_BEAT_SECONDS)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) \
                or duration != duration or duration <= 0:
            problems.append("%s: `duration_seconds` must be a positive number of seconds, "
                            "got %r" % (where, duration))
            duration = MIN_BEAT_SECONDS

        dialogue = Dialogue.from_dict(data.get("dialogue"), problems, where + ".dialogue")
        anchors = ContinuityAnchors.from_dict(data.get("continuity_anchors"), problems,
                                              where + ".continuity_anchors")
        plan = cls(shot_index, visual, negative, camera, duration, dialogue, anchors,
                   beat=_text(data, "beat", problems, where, limit=MAX_SHORT),
                   scene=_text(data, "scene", problems, where, limit=MAX_SHORT),
                   scene_change=_flag(data, "scene_change", problems, where),
                   new_characters=NewCharacter.list_from(data.get("new_characters"),
                                                         problems, where),
                   characters_present=_texts(data, "characters_present", problems, where),
                   end_of_story=_flag(data, "end_of_story", problems, where), fps=fps)

        unknown = sorted(set(data) - set(PLAN_REQUIRED) - set(PLAN_OPTIONAL))
        if unknown:
            plan.notes.append("%s: ignored unknown field(s) %s" % (where, ", ".join(unknown)))
        problems.extend(plan.check(cast_names))
        if not visual:
            return None, problems
        return plan, problems

    def check(self, cast_names: Optional[Iterable[str]] = None) -> list[str]:
        """The content rules, on a plan already parsed. Empty means renderable."""
        problems = []
        texts = [("visual_prompt", self.visual_prompt), ("camera_motion", self.camera_motion),
                 ("continuity_anchors", " ".join(self.continuity_anchors.held_items + [
                     self.continuity_anchors.costume, self.continuity_anchors.environment]))]
        if self.dialogue is not None:
            texts.append(("dialogue", self.dialogue.text))
        for field, text in texts:
            for match in LITERAL_TAG_RE.finditer(text or ""):
                problems.append(
                    "`%s` contains a hand-typed %s. Ordinals are assigned by the "
                    "compiler -- name a cast member as @Name and it resolves the tag"
                    % (field, match.group(0)))

        if cast_names is not None:
            known = {n.casefold(): n for n in cast_names}
            known.update({c.name.casefold(): c.name for c in self.new_characters})
            for name in mentions(self.visual_prompt):
                if name.casefold() not in known:
                    problems.append(
                        "`visual_prompt` cites @%s, who is not in the cast (%s). Use a "
                        "cast name, or introduce them in `new_characters`"
                        % (name, ", ".join(sorted(known.values())) or "nobody yet"))
            if self.spoken and self.dialogue.speaker.lstrip("@").casefold() not in known:
                problems.append(
                    "`dialogue.speaker` %r is not in the cast (%s)"
                    % (self.dialogue.speaker, ", ".join(sorted(known.values())) or "nobody yet"))

        if self.spoken:
            seconds = self.rendered_seconds
            rate = self.dialogue.words / (seconds / 60.0)
            if rate > MAX_DIALOGUE_WPM:
                fits = int(MAX_DIALOGUE_WPM * seconds / 60.0)
                problems.append(
                    "the line is %d words in a %.2f s beat (%.0f words a minute); at most "
                    "%d fit -- shorten it or lengthen the beat (up to %.2f s)"
                    % (self.dialogue.words, seconds, rate, fits, MAX_BEAT_SECONDS))
        return problems


def _beat_frames(seconds: Any, fps: int, notes: list[str]) -> int:
    """Seconds asked for -> frames H3 renders: on the 17k+5 grid, in the trained range."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        seconds = MIN_BEAT_SECONDS
    # To the nearest whole frame first, then up to the grid. A director writing
    # the documented "5.17 s" means the 124-frame floor (124.08 frames), and
    # rounding straight up from 124.08 would hand it 141.
    frames = align_frame_count(max(1, int(round(seconds * fps))))
    if frames < MIN_TRAINED_FRAMES:
        notes.append("a %.2f s beat is below H3's trained 124-frame floor; it renders "
                     "as %.2f s" % (seconds, MIN_BEAT_SECONDS))
        frames = MIN_TRAINED_FRAMES
    elif frames > MAX_WINDOW_FRAMES:
        notes.append("a %.2f s beat is over one window's %.2f s ceiling; it renders as "
                     "%.2f s" % (seconds, MAX_BEAT_SECONDS, MAX_BEAT_SECONDS))
        frames = MAX_WINDOW_FRAMES
    elif abs(frames_to_seconds(frames, fps) - seconds) > 1e-6:
        notes.append("%.2f s snapped to the frame grid: %d frames (%.2f s)"
                     % (seconds, frames, frames_to_seconds(frames, fps)))
    return frames


# ── the observation ─────────────────────────────────────────────────────────

class CharacterDelta:
    """What one beat changed about one character. Every field is a change, or empty."""

    __slots__ = ("posture", "appearance_change", "inventory_add", "inventory_remove",
                 "injuries", "emotional_state", "present")

    FIELDS = ("posture", "appearance_change", "inventory_add", "inventory_remove",
              "injuries", "emotional_state", "present")

    def __init__(self, posture: str = "", appearance_change: str = "",
                 inventory_add: Optional[list[str]] = None,
                 inventory_remove: Optional[list[str]] = None,
                 injuries: Optional[list[str]] = None, emotional_state: str = "",
                 present: Optional[bool] = None) -> None:
        self.posture = posture
        self.appearance_change = appearance_change
        self.inventory_add = list(inventory_add or [])
        self.inventory_remove = list(inventory_remove or [])
        self.injuries = list(injuries or [])
        self.emotional_state = emotional_state
        self.present = present

    @property
    def changes_anything(self) -> bool:
        return bool(self.appearance_change or self.inventory_add or self.inventory_remove
                    or self.injuries)

    def to_dict(self) -> dict:
        out = {}
        for field in self.FIELDS:
            value = getattr(self, field)
            if value not in ("", [], None):
                out[field] = value
        return out

    @classmethod
    def from_dict(cls, data: Any, problems: list[str], where: str) -> "CharacterDelta":
        data = _mapping(data, where, problems, where) or {}
        return cls(_text(data, "posture", problems, where, limit=MAX_SHORT),
                   _text(data, "appearance_change", problems, where, limit=MAX_SHORT),
                   _texts(data, "inventory_add", problems, where),
                   _texts(data, "inventory_remove", problems, where),
                   _texts(data, "injuries", problems, where),
                   _text(data, "emotional_state", problems, where, limit=MAX_SHORT),
                   _flag(data, "present", problems, where))


class EnvironmentDelta:
    """What one beat changed about the place. Empty fields are unchanged."""

    __slots__ = ("setting", "time_of_day", "lighting", "camera_orientation", "landmarks")

    FIELDS = ("setting", "time_of_day", "lighting", "camera_orientation", "landmarks")

    def __init__(self, setting: str = "", time_of_day: str = "", lighting: str = "",
                 camera_orientation: str = "", landmarks: Optional[list[str]] = None) -> None:
        self.setting = setting
        self.time_of_day = time_of_day
        self.lighting = lighting
        self.camera_orientation = camera_orientation
        self.landmarks = list(landmarks or [])

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.FIELDS if getattr(self, f) not in ("", [])}

    @classmethod
    def from_dict(cls, data: Any, problems: list[str], where: str) -> "EnvironmentDelta":
        data = _mapping(data, "environment", problems, where) or {}
        return cls(*(_text(data, f, problems, where, limit=MAX_SHORT)
                     for f in cls.FIELDS[:-1]),
                   landmarks=_texts(data, "landmarks", problems, where))


class ObservationResult:
    """What a beat actually showed, as a delta on the scene memory.

    Deltas, not a full restatement -- the memory already holds everything that
    has not changed, and a director asked to restate it all drifts the wording
    of things that did not move. `completed_actions` are the story events the
    beat finished (they become the anti-loop blacklist); `forbid` is anything
    else the director saw repeating; `stationary` says nobody really moved.
    """

    __slots__ = ("shot_index", "summary", "completed_actions", "characters", "environment",
                 "new_characters", "forbid", "stationary", "continuity_issues",
                 "provenance", "notes")

    def __init__(self, shot_index: int, summary: str = "",
                 completed_actions: Optional[list[str]] = None,
                 characters: Optional[dict[str, CharacterDelta]] = None,
                 environment: Optional[EnvironmentDelta] = None,
                 new_characters: Optional[list[NewCharacter]] = None,
                 forbid: Optional[list[str]] = None, stationary: bool = False,
                 continuity_issues: Optional[list[str]] = None,
                 provenance: str = PROVENANCE_OBSERVED) -> None:
        self.shot_index = int(shot_index)
        self.summary = summary
        self.completed_actions = list(completed_actions or [])
        self.characters = dict(characters or {})
        self.environment = environment or EnvironmentDelta()
        self.new_characters = list(new_characters or [])
        self.forbid = list(forbid or [])
        self.stationary = bool(stationary)
        self.continuity_issues = list(continuity_issues or [])
        self.provenance = provenance
        self.notes: list[str] = []

    @property
    def made_progress(self) -> bool:
        """Did anything happen? No completed action and no state change is a stall."""
        return bool(self.completed_actions or self.new_characters
                    or any(d.changes_anything for d in self.characters.values()))

    def to_dict(self) -> dict:
        out = {"shot_index": self.shot_index, "summary": self.summary,
               "completed_actions": list(self.completed_actions),
               "characters": {n: d.to_dict() for n, d in self.characters.items()},
               "environment": self.environment.to_dict(),
               "stationary": self.stationary}
        if self.new_characters:
            out["new_characters"] = [c.to_dict() for c in self.new_characters]
        if self.forbid:
            out["forbid"] = list(self.forbid)
        if self.continuity_issues:
            out["continuity_issues"] = list(self.continuity_issues)
        if self.provenance != PROVENANCE_OBSERVED:
            out["provenance"] = self.provenance
        return out

    @classmethod
    def from_dict(cls, data: Any, where: str = "observation"
                  ) -> tuple[Optional["ObservationResult"], list[str]]:
        problems: list[str] = []
        data = _mapping(data, where, problems, where)
        if data is None:
            return None, problems or ["%s: expected a JSON object" % where]
        shot_index = data.get("shot_index", 1)
        if isinstance(shot_index, bool) or not isinstance(shot_index, int) or shot_index < 1:
            problems.append("%s: `shot_index` must be a whole number from 1" % where)
            shot_index = 1
        characters = {}
        raw = _mapping(data.get("characters"), "characters", problems, where) or {}
        for name, delta in list(raw.items())[:MAX_ITEMS]:
            if isinstance(name, str) and name.strip():
                characters[name.strip()[:MAX_SHORT]] = CharacterDelta.from_dict(
                    delta, problems, "%s.characters.%s" % (where, name))
        provenance = data.get("provenance", PROVENANCE_OBSERVED)
        if provenance not in (PROVENANCE_OBSERVED, PROVENANCE_INTENDED):
            problems.append("%s: `provenance` must be observed or intended" % where)
            provenance = PROVENANCE_OBSERVED
        stationary = _flag(data, "stationary", problems, where)
        obs = cls(shot_index, _text(data, "summary", problems, where),
                  completed_actions=_texts(data, "completed_actions", problems, where, 3),
                  characters=characters,
                  environment=EnvironmentDelta.from_dict(data.get("environment"), problems,
                                                         where + ".environment"),
                  new_characters=NewCharacter.list_from(data.get("new_characters"),
                                                        problems, where),
                  forbid=_texts(data, "forbid", problems, where),
                  stationary=bool(stationary),
                  continuity_issues=_texts(data, "continuity_issues", problems, where),
                  provenance=provenance)
        return obs, problems

    @classmethod
    def from_intent(cls, plan: DirectedShotPlan) -> "ObservationResult":
        """The observation of a director that cannot see: the plan, taken at its word.

        Flagged `intended`, so memory never mistakes it for something watched --
        a script's beats and a text-only model's guesses are requests, not facts.
        """
        actions = [plan.beat] if plan.beat else []
        obs = cls(plan.shot_index, summary=plan.beat or plan.visual_prompt[:MAX_SHORT],
                  completed_actions=actions, new_characters=plan.new_characters,
                  provenance=PROVENANCE_INTENDED)
        if plan.continuity_anchors.environment:
            obs.environment = EnvironmentDelta(setting=plan.continuity_anchors.environment)
        return obs


class TurnResponse:
    """One director turn: what the last beat showed, and the next beat.

    The opening turn has nothing to observe, so `observation` is optional there
    and required on every turn after it.
    """

    __slots__ = ("observation", "plan")

    def __init__(self, plan: DirectedShotPlan,
                 observation: Optional[ObservationResult] = None) -> None:
        self.plan = plan
        self.observation = observation

    def to_dict(self) -> dict:
        out = {"plan": self.plan.to_dict()}
        if self.observation is not None:
            out["observation"] = self.observation.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Any, cast_names: Optional[Iterable[str]] = None,
                  needs_observation: bool = True, fps: int = FPS
                  ) -> tuple[Optional["TurnResponse"], list[str]]:
        problems: list[str] = []
        if not isinstance(data, dict):
            return None, ["the response must be a JSON object with `plan`%s"
                          % (" and `observation`" if needs_observation else "")]
        if "plan" not in data and "visual_prompt" in data and not needs_observation:
            data = {"plan": data}          # an opening turn may send the plan bare
        observation = None
        if data.get("observation") is not None:
            observation, more = ObservationResult.from_dict(data["observation"])
            problems.extend(more)
        elif needs_observation:
            problems.append("`observation` is missing -- say what the last beat showed "
                            "before planning the next")
        names = list(cast_names) if cast_names is not None else None
        if names is not None and observation is not None:
            names += [c.name for c in observation.new_characters]
        plan, more = DirectedShotPlan.from_dict(data.get("plan"), cast_names=names,
                                                fps=fps)
        problems.extend(more)
        if plan is None:
            return None, problems
        return cls(plan, observation), problems


# ── reading what a model actually wrote ─────────────────────────────────────

def unwrap_json(raw: Any) -> tuple[Optional[Any], list[str]]:
    """The JSON object inside whatever a director emitted. Returns (data, problems).

    Accepts, in order: an already-parsed object; a tool_use block or a message's
    content list holding one (Claude tool-use formatting); a CLI result envelope
    (`{"result": "..."}`, `{"response": "..."}`); and text -- with any <think>
    block removed, then a ```json fence, else the first balanced {...} in it.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return unwrap_json(block.get("input"))
        texts = [b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"]
        return unwrap_json("\n".join(texts)) if texts else (None, ["no JSON object found"])
    if isinstance(raw, dict):
        if raw.get("type") == "tool_use" and isinstance(raw.get("input"), dict):
            return raw["input"], []
        if isinstance(raw.get("content"), list) and "plan" not in raw:
            return unwrap_json(raw["content"])
        for envelope in ("result", "response"):
            if isinstance(raw.get(envelope), str) and not _looks_like_contract(raw):
                return unwrap_json(raw[envelope])
        return raw, []
    if not isinstance(raw, str):
        return None, ["expected JSON text, got %s" % type(raw).__name__]

    text = _THINK_RE.sub("", raw).strip()
    if not text:
        return None, ["the response was empty"]
    try:
        return unwrap_json(json.loads(text))
    except ValueError:
        pass
    for fenced in _FENCE_RE.findall(text):
        try:
            return unwrap_json(json.loads(fenced.strip()))
        except ValueError:
            continue
    candidate = _first_object(text)
    if candidate is not None:
        try:
            return unwrap_json(json.loads(candidate))
        except ValueError as exc:
            return None, ["the response holds something like JSON that does not parse: %s"
                          % exc]
    return None, ["no JSON object found in the response"]


def _looks_like_contract(data: dict) -> bool:
    return any(k in data for k in ("plan", "observation", "visual_prompt", "shot_index"))


def _first_object(text: str) -> Optional[str]:
    """The first balanced {...} in text, minding strings and escapes."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


# ── JSON Schema, for agents, tool definitions and docs ──────────────────────

def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


_DIALOGUE = {
    "type": ["object", "null"],
    "description": "The spoken line, or null for a silent beat.",
    "properties": {"speaker": _string("A cast name."),
                   "text": _string("The words, short enough for the beat."),
                   "emotion_tone": _string("Delivery, e.g. 'whispering, anxious'.")},
    "required": ["speaker", "text", "emotion_tone"],
    "additionalProperties": False,
}
_ANCHORS = {
    "type": "object",
    "properties": {"held_items": _strings("What is in hand, e.g. 'brass key in left hand'."),
                   "costume": _string("The costume as it is now."),
                   "environment": _string("Where, and in what light.")},
    "required": ["held_items", "costume", "environment"],
    "additionalProperties": False,
}
_NEW_CHARACTERS = {"type": "array", "items": {
    "type": "object", "properties": {"name": _string("Name, as it will be cited."),
                                     "appearance": _string("Concrete, pinnable look.")},
    "required": ["name", "appearance"], "additionalProperties": False}}

_PLAN_SCHEMA = {
    "type": "object",
    "description": "DirectedShotPlan: the next beat.",
    "properties": {
        "shot_index": {"type": "integer", "minimum": 1},
        "visual_prompt": _string("What the camera records, in temporal order. Cite cast "
                                 "as @Name; never type <Picture N> tags."),
        "negative_prompt": _string("What must not happen in this shot."),
        "camera_motion": _string("e.g. 'slow dolly forward, low angle'."),
        "duration_seconds": {"type": "number", "minimum": MIN_BEAT_SECONDS,
                             "maximum": MAX_BEAT_SECONDS,
                             "description": "Snapped up to H3's 17k+5 frame grid."},
        "dialogue": _DIALOGUE,
        "continuity_anchors": _ANCHORS,
        "beat": _string("One line: the story step this shot completes."),
        "scene": _string("Short name of the scene."),
        "scene_change": {"type": "boolean",
                         "description": "true opens on a cut instead of continuing."},
        "new_characters": _NEW_CHARACTERS,
        "characters_present": _strings("Cast names on screen."),
        "end_of_story": {"type": "boolean", "description": "true when the goal is met."},
    },
    "required": list(PLAN_REQUIRED),
    "additionalProperties": False,
}

_CHARACTER_DELTA = {
    "type": "object",
    "properties": {"posture": _string("Where and how they stand now."),
                   "appearance_change": _string("A visible change: torn, wet, injured."),
                   "inventory_add": _strings("Picked up."),
                   "inventory_remove": _strings("Put down, lost, used up."),
                   "injuries": _strings("New injuries."),
                   "emotional_state": _string("As it reads on screen."),
                   "present": {"type": "boolean"}},
    "additionalProperties": False,
}

_OBSERVATION_SCHEMA = {
    "type": "object",
    "description": "ObservationResult: what the last beat actually showed, as changes.",
    "properties": {
        "shot_index": {"type": "integer", "minimum": 1},
        "summary": _string("One or two sentences, what is on screen, not what was meant."),
        "completed_actions": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                              "description": "Story events this beat finished."},
        "characters": {"type": "object", "additionalProperties": _CHARACTER_DELTA},
        "environment": {"type": "object", "properties": {
            f: _string(f.replace("_", " ")) for f in EnvironmentDelta.FIELDS[:-1]},
            "additionalProperties": True},
        "new_characters": _NEW_CHARACTERS,
        "forbid": _strings("Anything seen repeating that must not happen again."),
        "stationary": {"type": "boolean", "description": "Nobody really moved."},
        "continuity_issues": _strings("Drift seen: a costume, a prop, a face."),
    },
    "required": ["shot_index", "summary", "completed_actions"],
    "additionalProperties": False,
}

_TURN_SCHEMA = {
    "type": "object",
    "description": "One director turn: the observation of the last beat and the next plan.",
    "properties": {"observation": _OBSERVATION_SCHEMA, "plan": _PLAN_SCHEMA},
    "required": ["observation", "plan"],
    "additionalProperties": False,
}


def json_schema(kind: str) -> dict:
    """The JSON Schema for 'plan', 'observation', 'turn' or 'open' (a turn with no
    observation). A deep copy, so a caller may annotate it freely."""
    table = {"plan": _PLAN_SCHEMA, "observation": _OBSERVATION_SCHEMA, "turn": _TURN_SCHEMA}
    if kind == "open":
        schema = json.loads(json.dumps(_TURN_SCHEMA))
        schema["required"] = ["plan"]
        schema["description"] = "The opening turn: the first plan, nothing to observe yet."
        return schema
    if kind not in table:
        raise ValueError("schema kind must be plan, observation, turn or open, got %r" % kind)
    return json.loads(json.dumps(table[kind]))
