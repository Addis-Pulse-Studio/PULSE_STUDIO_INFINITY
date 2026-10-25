"""The scene-graph memory a generated film keeps between beats.

H3 has no memory: every beat is rendered from its own prompt and one carried
frame. Everything the film has to remember -- who is wearing what, what is in
whose hand, where the light is coming from, what already happened and must not
happen again -- lives here, and is handed to the director every turn.

THE CONTRACT, AND WHAT IS ADDED TO IT
-------------------------------------
`to_dict()` writes the mission's Scene Memory State Contract with its keys and
shapes unchanged: `story_goal`, `current_beat`, `characters` (each with
`appearance`, `inventory`, `last_posture`), `environment` (`setting`,
`time_of_day`, `lighting`), `action_history`, `forbidden_actions`.

The ledger the spec asks for adds fields where they belong, written only when
they hold something, so the spec's own example reads back exactly as it was:

  * per character -- `pinned_appearance` (the identity line, restated verbatim
    in every prompt and never overwritten by an observation), `costume_tags`,
    `injuries`, `emotional_state`, first and last beat seen, `cast` (bound to a
    wired PulseCharacter) and `provenance`;
  * environment -- `camera_orientation` and `landmarks`, the spatial anchors.

The loop's own bookkeeping -- the blacklist's expiry beats, the scene tracker,
the beat intents, the chapters, the stagnation count -- lives under one
namespaced key, `pulse`, so it can never collide with a field a director sends.

TWO KINDS OF TRUTH
------------------
An observation from a director that looked at the frames is `observed`; one
from a script or a text-only model is only what the plan `intended`. Both update
the memory -- a story has to go on -- but the difference is kept, and shown to
the next director, so an intention is never mistaken for something filmed.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable, Optional

from .antiloop import ActionBlacklist, content_words, forbidden_phrase, jaccard
from .schema import (
    PROVENANCE_INTENDED,
    PROVENANCE_OBSERVED,
    DirectedShotPlan,
    ObservationResult,
)

__all__ = [
    "MEMORY_SCHEMA",
    "CharacterState",
    "Environment",
    "SceneMemory",
]

MEMORY_SCHEMA = 1

#: Fuzzy name matching: "alice", "Alice." and "Alice Liddell" fold into "Alice".
NAME_CUTOFF = 0.82
#: When the history passes this, its oldest entries fold into a chapter.
COMPACT_TRIGGER = 22
COMPACT_COUNT = 10
#: How many recent events the director sees in full.
VIEW_EVENTS = 8


def _clean(text: Any) -> str:
    return str(text or "").strip()


class CharacterState:
    """One entity in the ledger, as it is *now*."""

    __slots__ = ("name", "appearance", "inventory", "last_posture", "pinned_appearance",
                 "costume_tags", "injuries", "emotional_state", "first_seen_beat",
                 "last_seen_beat", "cast", "provenance")

    SPEC_FIELDS = ("appearance", "inventory", "last_posture")

    def __init__(self, name: str, appearance: str = "", inventory: Optional[list[str]] = None,
                 last_posture: str = "", pinned_appearance: str = "",
                 costume_tags: Optional[list[str]] = None,
                 injuries: Optional[list[str]] = None, emotional_state: str = "",
                 first_seen_beat: Optional[int] = None, last_seen_beat: Optional[int] = None,
                 cast: bool = False, provenance: str = "") -> None:
        self.name = name
        self.appearance = appearance
        self.inventory = list(inventory or [])
        self.last_posture = last_posture
        self.pinned_appearance = pinned_appearance
        self.costume_tags = list(costume_tags or [])
        self.injuries = list(injuries or [])
        self.emotional_state = emotional_state
        self.first_seen_beat = first_seen_beat
        self.last_seen_beat = last_seen_beat
        self.cast = bool(cast)
        self.provenance = provenance

    @property
    def identity(self) -> str:
        """The description every prompt restates: the pinned line, else the current one."""
        return self.pinned_appearance or self.appearance

    def to_dict(self) -> dict:
        out = {"appearance": self.appearance, "inventory": list(self.inventory),
               "last_posture": self.last_posture}
        extras = (("pinned_appearance", self.pinned_appearance),
                  ("costume_tags", list(self.costume_tags)),
                  ("injuries", list(self.injuries)),
                  ("emotional_state", self.emotional_state),
                  ("first_seen_beat", self.first_seen_beat),
                  ("last_seen_beat", self.last_seen_beat),
                  ("cast", self.cast or None),
                  ("provenance", self.provenance))
        for key, value in extras:
            if value not in ("", [], None):
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, name: str, data: Any) -> "CharacterState":
        data = data if isinstance(data, dict) else {}

        def _list(key: str) -> list[str]:
            value = data.get(key)
            return [_clean(v) for v in value if _clean(v)] if isinstance(value, list) else []

        def _beat(key: str) -> Optional[int]:
            value = data.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        return cls(name, _clean(data.get("appearance")), _list("inventory"),
                   _clean(data.get("last_posture")), _clean(data.get("pinned_appearance")),
                   _list("costume_tags"), _list("injuries"), _clean(data.get("emotional_state")),
                   _beat("first_seen_beat"), _beat("last_seen_beat"),
                   bool(data.get("cast")), _clean(data.get("provenance")))


class Environment:
    """Where the film is: the spec's three fields, plus the spatial anchors."""

    __slots__ = ("setting", "time_of_day", "lighting", "camera_orientation", "landmarks")

    def __init__(self, setting: str = "", time_of_day: str = "", lighting: str = "",
                 camera_orientation: str = "", landmarks: Optional[list[str]] = None) -> None:
        self.setting = setting
        self.time_of_day = time_of_day
        self.lighting = lighting
        self.camera_orientation = camera_orientation
        self.landmarks = list(landmarks or [])

    def to_dict(self) -> dict:
        out = {"setting": self.setting, "time_of_day": self.time_of_day,
               "lighting": self.lighting}
        if self.camera_orientation:
            out["camera_orientation"] = self.camera_orientation
        if self.landmarks:
            out["landmarks"] = list(self.landmarks)
        return out

    def words(self) -> list[str]:
        return [self.setting, self.time_of_day, self.lighting] + list(self.landmarks)

    @classmethod
    def from_dict(cls, data: Any) -> "Environment":
        data = data if isinstance(data, dict) else {}
        landmarks = data.get("landmarks")
        return cls(_clean(data.get("setting")), _clean(data.get("time_of_day")),
                   _clean(data.get("lighting")), _clean(data.get("camera_orientation")),
                   [_clean(v) for v in landmarks if _clean(v)]
                   if isinstance(landmarks, list) else [])


class SceneMemory:
    """Everything a generated film remembers between beats.

    `current_beat` is the number of the last beat committed -- 0 before the
    first. The plan being written is always for `current_beat + 1`.
    """

    __slots__ = ("story_goal", "current_beat", "characters", "environment",
                 "action_history", "blacklist", "scene", "scene_started_beat",
                 "scene_length", "beats", "chapters", "stagnant_streak",
                 "continuity_issues", "last_prompt", "revision")

    def __init__(self, story_goal: str = "", blacklist_size: int = 8,
                 blacklist_ttl: int = 4) -> None:
        self.story_goal = story_goal
        self.current_beat = 0
        self.characters: dict[str, CharacterState] = {}
        self.environment = Environment()
        self.action_history: list[str] = []
        self.blacklist = ActionBlacklist(blacklist_size, blacklist_ttl)
        self.scene = ""
        self.scene_started_beat = 0
        self.scene_length = 0
        #: [{"beat": n, "intent": "...", "scene": "...", "provenance": "..."}]
        self.beats: list[dict] = []
        #: [{"beats": [first, last], "summary": "..."}] -- folded history.
        self.chapters: list[dict] = []
        self.stagnant_streak = 0
        self.continuity_issues: list[str] = []
        self.last_prompt = ""
        self.revision = 0

    # ── the ledger ──────────────────────────────────────────────────────────

    def resolve_name(self, name: str) -> Optional[str]:
        """The ledger's name for `name`, or None if this is someone new.

        Exact (ignoring case and a leading @) first, then a name that starts with
        a known one ("Alice Liddell" is Alice), then difflib at NAME_CUTOFF.
        """
        wanted = _clean(name).lstrip("@").rstrip(".").casefold()
        if not wanted:
            return None
        by_fold = {known.casefold(): known for known in self.characters}
        if wanted in by_fold:
            return by_fold[wanted]
        for fold, known in by_fold.items():
            if wanted.startswith(fold + " ") or fold.startswith(wanted + " "):
                return known
        close = difflib.get_close_matches(wanted, list(by_fold), n=1, cutoff=NAME_CUTOFF)
        return by_fold[close[0]] if close else None

    def character(self, name: str, create: bool = True) -> Optional[CharacterState]:
        known = self.resolve_name(name)
        if known is not None:
            return self.characters[known]
        if not create or not _clean(name):
            return None
        state = CharacterState(_clean(name).lstrip("@"))
        self.characters[state.name] = state
        return state

    def seed_cast(self, cast: Iterable[tuple[str, str]]) -> list[str]:
        """Bind the wired PulseCharacters: (name, description) pairs.

        A cast member's description is their pinned identity -- what the PulseCharacter
        node says they look like is what the film must keep them looking like.
        Returns the names newly added.
        """
        added = []
        for name, description in cast:
            if not _clean(name):
                continue
            existed = self.resolve_name(name) is not None
            state = self.character(name)
            state.cast = True
            if description and not state.pinned_appearance:
                state.pinned_appearance = _clean(description)
            if description and not state.appearance:
                state.appearance = _clean(description)
            if not existed:
                added.append(state.name)
        return added

    @property
    def cast_names(self) -> list[str]:
        return list(self.characters)

    @property
    def forbidden_actions(self) -> list[str]:
        return self.blacklist.active()

    @property
    def stagnant(self) -> bool:
        return self.stagnant_streak >= 2

    def static_words(self) -> list[str]:
        """Text restated in every prompt on purpose -- removed before repeat checks."""
        words = self.environment.words()
        for state in self.characters.values():
            words += [state.name, state.identity, state.appearance] + state.costume_tags
        return [w for w in words if w]

    # ── a beat happens ──────────────────────────────────────────────────────

    def commit(self, plan: DirectedShotPlan, observation: Optional[ObservationResult] = None,
               max_scene_beats: int = 8) -> list[str]:
        """Record beat `plan.shot_index` as done. Returns notes for the report.

        The order matters: the plan's intent first (scene tracking, new
        characters), then what was actually seen on top of it, then the
        blacklist and the stagnation count, then compaction.
        """
        notes = []
        beat = int(plan.shot_index)
        observation = observation or ObservationResult.from_intent(plan)

        # Scene tracking -- a change of scene name or an explicit cut starts one.
        scene = plan.scene or self.scene
        if plan.cuts or (plan.scene and plan.scene != self.scene) or not self.scene:
            self.scene, self.scene_started_beat, self.scene_length = scene, beat, 1
        else:
            self.scene_length += 1

        for new in list(plan.new_characters) + list(observation.new_characters):
            state = self.character(new.name)
            if new.appearance and not state.pinned_appearance:
                state.pinned_appearance = new.appearance
            if new.appearance and not state.appearance:
                state.appearance = new.appearance

        provenance = observation.provenance
        for name in plan.characters_present:
            self._seen(self.character(name), beat, provenance)
        for name, delta in observation.characters.items():
            state = self.character(name)
            if delta.present is False:
                continue
            self._seen(state, beat, provenance)
            if delta.posture:
                state.last_posture = delta.posture
            if delta.appearance_change:
                # The current look moves; the pinned identity does not.
                state.appearance = delta.appearance_change
            for item in delta.inventory_add:
                if not any(item.casefold() == held.casefold() for held in state.inventory):
                    state.inventory.append(item)
            for item in delta.inventory_remove:
                state.inventory = [held for held in state.inventory
                                   if jaccard(content_words(held), content_words(item)) < 0.5]
            for injury in delta.injuries:
                if injury not in state.injuries:
                    state.injuries.append(injury)
            if delta.emotional_state:
                state.emotional_state = delta.emotional_state

        env = observation.environment
        for field in ("setting", "time_of_day", "lighting", "camera_orientation"):
            if getattr(env, field):
                setattr(self.environment, field, getattr(env, field))
        if env.landmarks:
            self.environment.landmarks = list(dict.fromkeys(
                self.environment.landmarks + env.landmarks))[-6:]
        if not self.environment.setting and plan.continuity_anchors.environment:
            self.environment.setting = plan.continuity_anchors.environment

        # Held items the plan pinned are, at the least, what the plan intended.
        for item in plan.continuity_anchors.held_items:
            holder = self._holder_of(item)
            if holder is not None and not any(item.casefold() == h.casefold()
                                              for h in holder.inventory):
                holder.inventory.append(item)

        # The blacklist: what finished must not be restaged; what was seen
        # repeating must not repeat; a beat where nobody moved must not recur.
        self.blacklist.expire(beat)
        for action in observation.completed_actions:
            self.action_history.append(action)
            self.blacklist.add(forbidden_phrase(action), beat, provenance)
        for item in observation.forbid:
            self.blacklist.add(item, beat, "observed")
        if observation.stationary:
            where = self.environment.setting or self.scene or "the same spot"
            self.blacklist.add("standing still at %s" % where, beat, "observed")

        self.stagnant_streak = 0 if observation.made_progress else self.stagnant_streak + 1
        if self.stagnant:
            notes.append("beats %d-%d changed nothing; the next beat must"
                         % (beat - self.stagnant_streak + 1, beat)
                         + " complete a visible change")
        self.continuity_issues = list(observation.continuity_issues)

        self.beats.append({"beat": beat, "intent": plan.beat or plan.visual_prompt[:200],
                           "scene": self.scene, "provenance": provenance})
        self.last_prompt = plan.visual_prompt
        self.current_beat = max(self.current_beat, beat)
        notes.extend(self._compact())
        self.revision += 1
        return notes

    def _seen(self, state: Optional[CharacterState], beat: int, provenance: str) -> None:
        if state is None:
            return
        if state.first_seen_beat is None:
            state.first_seen_beat = beat
        state.last_seen_beat = beat
        if provenance == PROVENANCE_OBSERVED or not state.provenance:
            state.provenance = provenance

    def _holder_of(self, item: str) -> Optional[CharacterState]:
        """Who an anchor like "brass key in Alice's left hand" belongs to, if anyone."""
        folded = item.casefold()
        for state in self.characters.values():
            if state.name.casefold() in folded:
                return state
        present = [s for s in self.characters.values()
                   if s.last_seen_beat == self.current_beat + 1 or len(self.characters) == 1]
        return present[0] if len(present) == 1 else None

    def _compact(self) -> list[str]:
        """Fold the oldest history into a chapter once it passes COMPACT_TRIGGER.

        Deterministic on purpose -- the summary is the folded events themselves,
        joined -- so compaction can never invent a fact. What it buys is a director
        prompt that stops growing with the film.
        """
        if len(self.action_history) <= COMPACT_TRIGGER:
            return []
        folded, self.action_history = (self.action_history[:COMPACT_COUNT],
                                       self.action_history[COMPACT_COUNT:])
        covered = [b["beat"] for b in self.beats[:COMPACT_COUNT]] or [self.current_beat]
        self.chapters.append({"beats": [min(covered), max(covered)],
                              "summary": "; ".join(folded)[:600]})
        return ["memory folded %d events into chapter %d" % (len(folded), len(self.chapters))]

    # ── what the director is shown ──────────────────────────────────────────

    def view_for_director(self, events: int = VIEW_EVENTS) -> dict:
        """A bounded view: never grows with the film's length."""
        return {
            "story_goal": self.story_goal,
            "beats_done": self.current_beat,
            "next_beat": self.current_beat + 1,
            "scene": {"name": self.scene, "beats_so_far": self.scene_length},
            "chapters": [dict(c) for c in self.chapters[-6:]],
            "cast": {name: {"identity": s.identity, "now": s.appearance,
                            "holding": list(s.inventory), "posture": s.last_posture,
                            "injuries": list(s.injuries), "mood": s.emotional_state,
                            "seen": s.provenance or "not yet on screen"}
                     for name, s in self.characters.items()},
            "environment": self.environment.to_dict(),
            "recent_events": list(self.action_history[-events:]),
            "recent_intentions": [{"beat": b["beat"], "intent": b["intent"],
                                   "verified": b["provenance"] == PROVENANCE_OBSERVED}
                                  for b in self.beats[-events:]],
            "forbidden_actions": self.forbidden_actions,
            "stagnant": self.stagnant,
            "continuity_issues": list(self.continuity_issues),
        }

    def view_text(self, events: int = VIEW_EVENTS) -> str:
        """The same view as prose, for a director that reads a prompt, not JSON."""
        view = self.view_for_director(events)
        lines = ["Story goal: %s" % (self.story_goal or "(open)"),
                 "Beats done: %d. Scene %r, %d beat(s) in."
                 % (self.current_beat, self.scene or "-", self.scene_length)]
        for chapter in view["chapters"]:
            lines.append("Earlier (beats %d-%d): %s" % (chapter["beats"][0],
                                                         chapter["beats"][1], chapter["summary"]))
        if view["cast"]:
            lines.append("Cast as they are now:")
            for name, s in view["cast"].items():
                bits = [s["identity"] or "no description yet"]
                if s["now"] and s["now"] != s["identity"]:
                    bits.append("now: " + s["now"])
                if s["holding"]:
                    bits.append("holding " + ", ".join(s["holding"]))
                if s["posture"]:
                    bits.append(s["posture"])
                if s["injuries"]:
                    bits.append("injured: " + ", ".join(s["injuries"]))
                if s["mood"]:
                    bits.append(s["mood"])
                lines.append("  - @%s: %s" % (name, "; ".join(bits)))
        env = self.environment
        place = ", ".join(x for x in (env.setting, env.time_of_day, env.lighting) if x)
        if place:
            lines.append("Place: " + place
                         + (" | camera: " + env.camera_orientation if env.camera_orientation else "")
                         + (" | landmarks: " + ", ".join(env.landmarks) if env.landmarks else ""))
        if view["recent_events"]:
            lines.append("Just happened: " + "; ".join(view["recent_events"]))
        if view["recent_intentions"]:
            lines.append("Earlier beat intentions (not all verified on screen; do not repeat them):")
            for item in view["recent_intentions"]:
                lines.append("  %d. %s%s" % (item["beat"], item["intent"],
                                             "" if item["verified"] else " (unverified)"))
        if view["forbidden_actions"]:
            lines.append("Forbidden now: " + "; ".join(view["forbidden_actions"]))
        if view["stagnant"]:
            lines.append("STALLED: recent beats changed nothing. The next beat must complete "
                         "a visible change of situation, or cut to a new scene.")
        if view["continuity_issues"]:
            lines.append("Drift seen last beat, correct it: " + "; ".join(view["continuity_issues"]))
        return "\n".join(lines)

    # ── serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "story_goal": self.story_goal,
            "current_beat": self.current_beat,
            "characters": {name: s.to_dict() for name, s in self.characters.items()},
            "environment": self.environment.to_dict(),
            "action_history": list(self.action_history),
            "forbidden_actions": self.forbidden_actions,
            "pulse": {
                "memory_schema": MEMORY_SCHEMA,
                "revision": self.revision,
                "blacklist": self.blacklist.to_list(),
                "blacklist_size": self.blacklist.size,
                "blacklist_ttl": self.blacklist.ttl,
                "scene": {"name": self.scene, "started_beat": self.scene_started_beat,
                          "length": self.scene_length},
                "beats": [dict(b) for b in self.beats[-64:]],
                "chapters": [dict(c) for c in self.chapters],
                "stagnant_streak": self.stagnant_streak,
                "continuity_issues": list(self.continuity_issues),
                "last_prompt": self.last_prompt,
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SceneMemory":
        """Read a memory -- one this module wrote, or a bare spec-contract document.

        A document with no `pulse` block (the spec's example, or one a person
        wrote by hand) is taken at its word: its forbidden actions enter the
        blacklist as if added on its current beat.
        """
        data = data if isinstance(data, dict) else {}
        pulse = data.get("pulse") if isinstance(data.get("pulse"), dict) else {}
        memory = cls(_clean(data.get("story_goal")),
                     blacklist_size=int(pulse.get("blacklist_size") or 8),
                     blacklist_ttl=int(pulse.get("blacklist_ttl") or 4))
        beat = data.get("current_beat")
        memory.current_beat = beat if isinstance(beat, int) and not isinstance(beat, bool) else 0
        characters = data.get("characters") if isinstance(data.get("characters"), dict) else {}
        for name, state in characters.items():
            if _clean(name):
                memory.characters[_clean(name)] = CharacterState.from_dict(_clean(name), state)
        memory.environment = Environment.from_dict(data.get("environment"))
        history = data.get("action_history")
        memory.action_history = ([_clean(a) for a in history if _clean(a)]
                                 if isinstance(history, list) else [])

        if pulse.get("blacklist"):
            memory.blacklist = ActionBlacklist.from_list(
                pulse["blacklist"], memory.blacklist.size, memory.blacklist.ttl)
        else:
            forbidden = data.get("forbidden_actions")
            # Oldest first into the ring, so `active()` (newest first) reads the
            # list back in the order it was written.
            for phrase in reversed(forbidden if isinstance(forbidden, list) else []):
                if _clean(phrase):
                    memory.blacklist.add(_clean(phrase), memory.current_beat, "loaded")
        scene = pulse.get("scene") if isinstance(pulse.get("scene"), dict) else {}
        memory.scene = _clean(scene.get("name"))
        memory.scene_started_beat = int(scene.get("started_beat") or 0)
        memory.scene_length = int(scene.get("length") or 0)
        memory.beats = [b for b in (pulse.get("beats") or []) if isinstance(b, dict)]
        memory.chapters = [c for c in (pulse.get("chapters") or []) if isinstance(c, dict)]
        memory.stagnant_streak = int(pulse.get("stagnant_streak") or 0)
        memory.continuity_issues = [_clean(i) for i in (pulse.get("continuity_issues") or [])
                                    if _clean(i)]
        memory.last_prompt = _clean(pulse.get("last_prompt"))
        memory.revision = int(pulse.get("revision") or 0)
        return memory

    def copy(self) -> "SceneMemory":
        return SceneMemory.from_dict(self.to_dict())


# Re-exported for callers that only import the memory module.
__all__ += ["PROVENANCE_OBSERVED", "PROVENANCE_INTENDED"]
