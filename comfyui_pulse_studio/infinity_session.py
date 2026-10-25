"""Pulse Infinity's loop, as state on disk: one beat per ComfyUI queue run.

The four Infinity nodes are thin; this is where the loop actually lives. A
session is a folder, `ComfyUI/output/<sessions_dir>/<name>/`, and everything a
beat needs to know about the beats before it is in that folder:

    session.json            who, where in the story, what is planned next
    scene_memory.json       what the film remembers (pulse_memory)
    memory_snapshots/       the memory before each beat, for undo
    control.json            pause / resume / stop / reroll / replan requests
    beats/beat_NNNN/        plan.json, and one render folder per attempt (aK/)
    desk/                   the Director Desk (pulse_director.mailbox)
    journal/                every director exchange, accepted ones replayable
    schemas/                the JSON Schemas a director answers to
    reel/                   the committed beats, in order

ONE BEAT PER QUEUE RUN
----------------------
The Loop Controller reads the session and hands the planned beat downstream;
Pulse Infinity Render renders it into `beats/beat_NNNN/aK`; the Observer shows
the result to the director, commits what it says to memory, and stores the next
plan. Queue the graph N times and N beats render, each run starting from the
state the one before left. A crash loses at most the beat in flight, and the
segment cache and the journal make re-running it cheap.

IDEMPOTENT, BECAUSE COMFYUI RE-RUNS NODES
----------------------------------------
ComfyUI's RAM-pressure cache may evict a node's output and execute it again in
the same run. So nothing here does anything twice:

  * a director exchange is journaled under a key naming the session's undo
    epoch, the beat, the kind of turn, and -- for a turn that observes -- the
    render it observed. Asked again, an accepted answer is replayed, not
    re-asked;
  * a beat is committed once per render: committing the same render again
    returns the stored outcome.

UNDO
----
A reroll renders the same plan again (a new attempt, so a new shot id, so new
seeds); a replan asks the director for a new plan. Both restore the memory
snapshot from before the beat and advance the session's epoch, so no answer
from before the undo can be replayed into the world after it.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Callable, Optional

from .durable import write_json_durably
from .pulse_director.base_director import (
    KIND_OBSERVE,
    KIND_OPEN,
    KIND_PLAN,
    KIND_TURN,
    DirectorError,
    EvidencePack,
    TurnRequest,
    default_constraints,
)
from .pulse_director.config import InfinityConfig
from .pulse_director.exchange import ExchangeDirector, validate_answer
from .pulse_director.mailbox import utc_now
from .pulse_director.scripted_provider import parse_script
from .pulse_memory import store
from .pulse_memory.scene_memory import SceneMemory
from .pulse_memory.schema import DirectedShotPlan, ObservationResult, TurnResponse, json_schema
from .pulse_timeline import text_shot_id
from .segcache import carry_in_descriptor

__all__ = [
    "SESSION_FILE",
    "CONTROL_FILE",
    "SESSION_SCHEMA",
    "STATUS_ACTIVE",
    "STATUS_PAUSED",
    "STATUS_COMPLETE",
    "CONTROL_ACTIONS",
    "SessionError",
    "BeatOutcome",
    "InfinitySession",
    "safe_session_name",
    "peek_revision",
]

SESSION_FILE = "session.json"
CONTROL_FILE = "control.json"
SESSION_SCHEMA = 1

STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_COMPLETE = "complete"

CONTROL_ACTIONS = ("pause", "resume", "stop", "reroll", "replan")

_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {"com%d" % i for i in range(1, 10)} | {
    "lpt%d" % i for i in range(1, 10)}


class SessionError(RuntimeError):
    """The session cannot do what was asked -- said in words a person can act on."""


def safe_session_name(name: str) -> str:
    """A session name that is a safe folder name on Windows and Linux alike."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")[:48]
    if not cleaned:
        cleaned = "session"
    if cleaned.casefold() in _WINDOWS_RESERVED:
        cleaned += "_"
    return cleaned


def peek_revision(root: str) -> str:
    """`<uid>:<revision>` for IS_CHANGED, or "new" -- a cheap read, no side effects.

    The uid is part of it so a session reset (a fresh uid, revision 0 again) can
    never collide with a value ComfyUI's cache already holds.
    """
    try:
        with open(os.path.join(root, SESSION_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return "%s:%s" % (data.get("uid"), data.get("revision"))
    except (OSError, ValueError, AttributeError):
        return "new"


class BeatOutcome:
    """What a committed beat changed, for the Observer's report."""

    __slots__ = ("beat", "observation", "next_plan", "complete", "notes", "replayed")

    def __init__(self, beat: int, observation: Optional[ObservationResult] = None,
                 next_plan: Optional[DirectedShotPlan] = None, complete: bool = False,
                 notes: Optional[list[str]] = None, replayed: bool = False) -> None:
        self.beat = beat
        self.observation = observation
        self.next_plan = next_plan
        self.complete = complete
        self.notes = list(notes or [])
        self.replayed = replayed

    def to_dict(self) -> dict:
        return {"beat": self.beat,
                "observation": self.observation.to_dict() if self.observation else None,
                "next_plan": self.next_plan.to_dict() if self.next_plan else None,
                "complete": self.complete, "notes": list(self.notes)}


class InfinitySession:
    """One film in progress. See the module docstring."""

    def __init__(self, output_dir: str, name: str, config: InfinityConfig) -> None:
        self.output_dir = output_dir
        self.name = safe_session_name(name)
        self.config = config
        self.sessions_dir = str(config.data.get("sessions_dir") or "pulse_infinity").strip("/\\")
        self.root = os.path.join(output_dir, self.sessions_dir, self.name)
        self.state = self._load_state()
        self.memory, self.memory_notes = store.load_memory(self.root, self.state["story_goal"])

    # ── state on disk ───────────────────────────────────────────────────────

    def _fresh_state(self) -> dict:
        return {"schema": SESSION_SCHEMA, "name": self.name, "uid": uuid.uuid4().hex[:12],
                "created_utc": utc_now(), "status": STATUS_ACTIVE, "story_goal": "",
                "world": "", "beat": 0, "pending_beat": 1, "attempt": 1, "epoch": 0,
                "revision": 0, "final_beat": None, "pending_plan": None,
                "replan_note": "", "beats": {}, "reel": []}

    def _load_state(self) -> dict:
        path = os.path.join(self.root, SESSION_FILE)
        if not os.path.exists(path):
            return self._fresh_state()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or "uid" not in data:
                raise ValueError("not a session document")
        except (OSError, ValueError) as exc:
            raise SessionError(
                "session.json in %s cannot be read (%s). Move it aside to start the "
                "session afresh, or restore it from the journal." % (self.root, exc))
        state = self._fresh_state()
        state.update(data)
        return state

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        write_json_durably(os.path.join(self.root, SESSION_FILE), self.state, prefix=".session-")

    def _bump(self) -> None:
        self.state["revision"] = int(self.state.get("revision") or 0) + 1

    @property
    def uid(self) -> str:
        return self.state["uid"]

    @property
    def revision_token(self) -> str:
        return "%s:%s" % (self.state["uid"], self.state["revision"])

    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def pending_beat(self) -> int:
        return int(self.state["pending_beat"])

    @property
    def attempt(self) -> int:
        return int(self.state["attempt"])

    def relative(self, path: str) -> str:
        return os.path.relpath(path, self.root).replace("\\", "/")

    def beat_dir(self, beat: int) -> str:
        return os.path.join(self.root, "beats", "beat_%04d" % int(beat))

    def run_location(self, beat: Optional[int] = None, attempt: Optional[int] = None
                     ) -> tuple[str, str]:
        """(run_dir, run_id) for Pulse Render: relative to ComfyUI's output folder."""
        beat = self.pending_beat if beat is None else beat
        attempt = self.attempt if attempt is None else attempt
        return ("%s/%s/beats/beat_%04d" % (self.sessions_dir, self.name, beat), "a%d" % attempt)

    def shot_id(self, beat: Optional[int] = None, attempt: Optional[int] = None) -> str:
        """Deterministic per session, beat, attempt and epoch: a resume hits the
        segment cache; a reroll is a new id, so new seeds; an undo can never land
        on a shot from before it."""
        beat = self.pending_beat if beat is None else beat
        attempt = self.attempt if attempt is None else attempt
        return text_shot_id("pulse-infinity:" + self.uid,
                            "beat %d attempt %d epoch %d" % (beat, attempt, self.state["epoch"]))

    # ── the story's settings, as the Loop Controller gives them ─────────────

    def configure(self, story_goal: str = "", world: str = "",
                  cast: Optional[list[tuple[str, str]]] = None) -> list[str]:
        """Take the widget values. Non-destructive: a goal edited mid-film moves the
        goal, it does not restart the film."""
        notes = []
        if story_goal and story_goal != self.state["story_goal"]:
            if self.state["story_goal"]:
                notes.append("story goal changed mid-film; the next beat aims at the new one")
            self.state["story_goal"] = story_goal
            self.memory.story_goal = story_goal
        if world and world != self.state["world"]:
            self.state["world"] = world
        if cast:
            added = self.memory.seed_cast(cast)
            if added:
                notes.append("cast bound: %s" % ", ".join(added))
        return notes

    def cast_list(self) -> list[dict]:
        return [{"name": n, "description": s.identity} for n, s in self.memory.characters.items()]

    # ── control ─────────────────────────────────────────────────────────────

    def take_control(self) -> list[str]:
        """Apply control.json, if present, and remove it. Returns what was done."""
        path = os.path.join(self.root, CONTROL_FILE)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = {}
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover
            pass
        action = (data or {}).get("action")
        if action not in CONTROL_ACTIONS:
            return ["control.json asked for %r, which is not one of %s; ignored"
                    % (action, ", ".join(CONTROL_ACTIONS))]
        return [self.control(action, note=str((data or {}).get("note") or ""))]

    def control(self, action: str, note: str = "") -> str:
        """pause / resume / stop / reroll / replan. Returns what happened, in words."""
        if action == "pause":
            self.state["status"] = STATUS_PAUSED
            message = "paused after beat %d" % self.state["beat"]
        elif action == "resume":
            if self.state["status"] == STATUS_COMPLETE:
                return "the film is complete; there is nothing to resume"
            self.state["status"] = STATUS_ACTIVE
            message = "resumed at beat %d" % self.pending_beat
        elif action == "stop":
            self.state["status"] = STATUS_COMPLETE
            message = "stopped after beat %d" % self.state["beat"]
        elif action in ("reroll", "replan"):
            message = self._undo_last(action, note)
        else:
            raise SessionError("unknown control action %r" % (action,))
        self._bump()
        self.save()
        return message

    def _undo_last(self, action: str, note: str) -> str:
        beat = int(self.state["beat"])
        if beat < 1:
            if action == "replan":
                self.state["pending_plan"] = None
                self.state["epoch"] += 1
                self.state["replan_note"] = note
                return "beat 1 will be planned afresh"
            return "nothing has been rendered yet to reroll"
        restored = store.restore_snapshot(self.root, beat)
        if restored is not None:
            self.memory = restored
            store.save_memory(self.root, self.memory)
        record = self.state["beats"].pop(str(beat), {}) or {}
        self.state["reel"] = [r for r in self.state["reel"] if int(r.get("beat", 0)) != beat]
        self.state["beat"] = beat - 1
        self.state["pending_beat"] = beat
        self.state["attempt"] = int(record.get("attempt") or 1) + 1
        self.state["epoch"] += 1
        self.state["status"] = STATUS_ACTIVE
        self.state["final_beat"] = None
        if action == "reroll":
            self.state["pending_plan"] = record.get("plan") or self.state.get("pending_plan")
            return "beat %d will render again from the same plan (attempt %d)" % (
                beat, self.state["attempt"])
        self.state["pending_plan"] = None
        self.state["replan_note"] = note or "The previous take of this beat was rejected."
        return "beat %d will be planned again" % beat

    # ── the director's turns ────────────────────────────────────────────────

    def _journal_key(self, kind: str, beat: int, observed_key: str = "") -> str:
        key = "b%04d-%s-e%d" % (beat, kind, self.state["epoch"])
        return key + ("-k%s" % observed_key[:12] if observed_key else "")

    def _journal_path(self, key: str) -> str:
        return os.path.join(self.root, "journal", key + ".accepted.json")

    def _journaler(self, key: str) -> Callable[..., None]:
        def record(request: TurnRequest, raw: Any, turn: Optional[TurnResponse],
                   problems: list[str]) -> None:
            entry = {"utc": utc_now(), "turn_id": request.turn_id, "kind": request.kind,
                     "beat": request.beat, "attempt": request.attempt,
                     "accepted": turn is not None and not problems, "problems": problems,
                     "raw": raw if isinstance(raw, (dict, list)) else str(raw)[:20000]}
            write_json_durably(os.path.join(self.root, "journal", "%s.json" % request.turn_id),
                               entry, prefix=".journal-")
            if entry["accepted"]:
                write_json_durably(self._journal_path(key), {"raw": entry["raw"],
                                                             "turn_id": request.turn_id},
                                   prefix=".journal-")
        return record

    def _replayed(self, key: str, request: TurnRequest,
                  last_plan: Optional[DirectedShotPlan]) -> Optional[TurnResponse]:
        path = self._journal_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)["raw"]
        except (OSError, ValueError, KeyError):
            return None
        turn, problems, _ = validate_answer(raw, request, self.memory, last_plan,
                                            int(self.config.loop("scene_budget")))
        return turn if turn is not None and not problems else None

    def _write_schemas(self) -> None:
        folder = os.path.join(self.root, "schemas")
        if os.path.isdir(folder):
            return
        for kind in ("plan", "observation", "turn", "open"):
            write_json_durably(os.path.join(folder, "%s.schema.json" % kind), json_schema(kind),
                               prefix=".schema-")

    def _request(self, kind: str, beat: int, director: ExchangeDirector,
                 evidence: Optional[EvidencePack] = None,
                 last_plan: Optional[dict] = None) -> TurnRequest:
        max_beats = int(self.state.get("max_beats") or 0)
        may_end = (beat >= int(self.config.loop("min_beats_before_end"))
                   or (max_beats and beat >= max_beats)
                   or not director.can_continue(beat + 1))
        seconds = sum(float(r.get("seconds") or 0) for r in self.state["reel"])
        guidance = self.state.get("replan_note") or ""
        return TurnRequest(
            kind, self.name, beat, story_goal=self.state["story_goal"],
            memory=self.memory.view_for_director(), memory_text=self.memory.view_text(),
            last_plan=last_plan, evidence=evidence,
            constraints=default_constraints(self.memory, int(self.config.loop("scene_budget")),
                                            may_end=bool(may_end)),
            cast=self.cast_list(), guidance=(self.state["world"] + ("\n" + guidance if guidance
                                                                     else "")).strip(),
            pacing={"beat": beat, "max_beats": max_beats or "open-ended",
                    "seconds_so_far": round(seconds, 2)},
            epoch=int(self.state["epoch"]))

    def _ask(self, director: ExchangeDirector, request: TurnRequest, key: str,
             last_plan: Optional[DirectedShotPlan] = None) -> TurnResponse:
        # On disk before anything is asked: a director looking for its first turn
        # finds the session by its folder, and a fresh session has nothing else
        # written yet.
        self.save()
        self._write_schemas()
        replay = self._replayed(key, request, last_plan)
        if replay is not None:
            return replay
        return director.step(request, self.memory, last_plan=last_plan,
                             max_attempts=int(self.config.loop("max_attempts")),
                             scene_budget=int(self.config.loop("scene_budget")),
                             on_attempt=self._journaler(key),
                             accept_invalid=self.config.loop("on_invalid") == "accept")

    def plan_for_pending(self, director: ExchangeDirector, opening: str = "") -> DirectedShotPlan:
        """The plan for the beat about to render -- stored, or asked for, once.

        Beat 1 comes from the `opening` text when the user wrote one, in the
        scripted format (pulse_director.scripted_provider); otherwise the director
        opens the film. A replanned beat asks again with the user's note.
        """
        if self.state.get("pending_plan"):
            plan, problems = DirectedShotPlan.from_dict(self.state["pending_plan"])
            if plan is not None and not problems:
                return plan
        beat = self.pending_beat
        if beat == 1 and opening.strip() and self.state["epoch"] == 0:
            plans, problems = parse_script(opening)
            if problems or not plans:
                raise SessionError("the opening beat cannot be read: %s"
                                   % ("; ".join(problems) or "it is empty"))
            data = dict(plans[0], shot_index=1)
            plan, problems = DirectedShotPlan.from_dict(data, cast_names=self.memory.cast_names)
            if plan is None or problems:
                raise SessionError("the opening beat breaks the contract: %s"
                                   % "; ".join(problems))
        else:
            kind = KIND_OPEN if beat == 1 else KIND_PLAN
            request = self._request(kind, beat, director)
            plan = self._ask(director, request, self._journal_key(kind, beat)).plan
        self._store_plan(plan)
        return plan

    def _store_plan(self, plan: DirectedShotPlan) -> None:
        self.state["pending_plan"] = plan.to_dict()
        self.state["replan_note"] = ""
        write_json_durably(os.path.join(self.beat_dir(plan.shot_index), "plan.json"),
                           plan.to_dict(), prefix=".plan-")
        if plan.end_of_story:
            self.state["final_beat"] = plan.shot_index
        self.save()

    # ── what the render left ────────────────────────────────────────────────

    def record_render(self, beat: int, attempt: int, record: dict) -> str:
        """Pulse Infinity Render's account of one attempt: keys, files, canvas."""
        path = os.path.join(self.beat_dir(beat), "a%d" % int(attempt), "beat.json")
        return write_json_durably(path, dict(record, beat=int(beat), attempt=int(attempt)),
                                  prefix=".beat-")

    def render_record(self, beat: int, attempt: int) -> Optional[dict]:
        path = os.path.join(self.beat_dir(beat), "a%d" % int(attempt), "beat.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def continue_from(self) -> Optional[dict]:
        """The `continue_from` for the pending beat: the last committed beat's last
        segment, or None when there is none or the plan cuts."""
        committed = int(self.state["beat"])
        if committed < 1:
            return None
        record = self.state["beats"].get(str(committed)) or {}
        if not record.get("last_key"):
            return None
        run_dir, run_id = self.run_location(committed, int(record.get("attempt") or 1))
        return carry_in_descriptor(run_dir, run_id, record["last_key"])

    # ── commit ──────────────────────────────────────────────────────────────

    def commit_rendered(self, director: ExchangeDirector, evidence: EvidencePack,
                        render: dict, beat: Optional[int] = None,
                        attempt: Optional[int] = None) -> BeatOutcome:
        """The beat just rendered: show it to the director, commit, plan the next.

        `render` is the record Pulse Infinity Render wrote (record_render), and
        `beat` / `attempt` say which render it is -- taken from the render, never
        from the session's pending beat, which has already moved on once this beat
        is committed. Doing this twice for one render returns the first outcome
        unchanged; a render the session has moved past is refused.
        """
        beat = int(beat if beat is not None else render.get("beat") or self.pending_beat)
        attempt = int(attempt if attempt is not None else render.get("attempt") or self.attempt)
        key = str(render.get("last_key") or "")
        done = self.state["beats"].get(str(beat))
        if done and done.get("last_key") == key and int(done.get("attempt") or 0) == attempt:
            return BeatOutcome(beat, next_plan=self._pending_plan_object(),
                               complete=self.status == STATUS_COMPLETE,
                               notes=["beat %d was already committed; nothing repeated" % beat],
                               replayed=True)
        if int(self.state["beat"]) >= beat:
            raise SessionError("beat %d is already committed from another render; reroll or "
                               "replan it to render it again" % beat)
        if beat != self.pending_beat or attempt != self.attempt:
            raise SessionError(
                "this render is beat %d attempt %d, but the session is waiting for beat %d "
                "attempt %d -- it was undone or overtaken; queue again to render the "
                "current beat" % (beat, attempt, self.pending_beat, self.attempt))

        last_plan = self._pending_plan_object()
        if last_plan is None:
            raise SessionError("no plan is stored for beat %d -- was the session reset "
                               "between the Loop Controller and the Observer?" % beat)

        max_beats = int(self.state.get("max_beats") or 0)
        ends_here = bool((max_beats and beat >= max_beats)
                         or self.state.get("final_beat") == beat
                         or not director.can_continue(beat + 1))
        kind = KIND_OBSERVE if ends_here else KIND_TURN
        request = self._request(kind, beat + 1, director, evidence=evidence,
                                last_plan=last_plan.to_dict())
        turn = self._ask(director, request, self._journal_key(kind, beat + 1, key), last_plan)

        notes = list(getattr(director, "last_notes", []) or [])
        if self.memory.current_beat >= beat:
            # A crash after the memory was saved and before session.json was: the
            # memory already holds this beat. Back to the snapshot taken before it,
            # so the commit below is the only one.
            before = store.restore_snapshot(self.root, beat)
            if before is not None:
                self.memory = before
        store.snapshot_memory(self.root, self.memory, beat)
        notes.extend(self.memory.commit(last_plan, turn.observation,
                                        max_scene_beats=int(self.config.loop("scene_budget"))))
        store.save_memory(self.root, self.memory)

        self.state["beats"][str(beat)] = {
            "attempt": attempt, "last_key": key, "plan": last_plan.to_dict(),
            "committed_utc": utc_now(), "seconds": render.get("seconds"),
            "summary": (turn.observation.summary if turn.observation else "")}
        self.state["reel"].append({"beat": beat, "attempt": attempt,
                                   "segments": list(render.get("segments") or []),
                                   "seconds": render.get("seconds"),
                                   "width": render.get("width"), "height": render.get("height"),
                                   "fps": render.get("fps")})
        self.state["beat"] = beat
        complete = ends_here
        if turn.plan is not None and not ends_here:
            self.state["pending_beat"] = beat + 1
            self.state["attempt"] = 1
            self._store_plan(turn.plan)
        else:
            self.state["pending_plan"] = None
            self.state["status"] = STATUS_COMPLETE
            complete = True
            notes.append("the film is complete at beat %d" % beat)
        self._bump()
        self.save()
        return BeatOutcome(beat, turn.observation, None if complete else turn.plan,
                           complete, notes)

    def _pending_plan_object(self) -> Optional[DirectedShotPlan]:
        if not self.state.get("pending_plan"):
            return None
        plan, _ = DirectedShotPlan.from_dict(self.state["pending_plan"])
        return plan

    # ── setup ───────────────────────────────────────────────────────────────

    def set_limits(self, max_beats: int = 0) -> None:
        self.state["max_beats"] = max(0, int(max_beats or 0))

    def describe(self) -> str:
        where = "%s/%s" % (self.sessions_dir, self.name)
        if self.status == STATUS_COMPLETE:
            return "session %s: complete after beat %d" % (where, int(self.state["beat"]))
        return ("session %s: beat %d committed, beat %d next (attempt %d), %s"
                % (where, int(self.state["beat"]), self.pending_beat, self.attempt,
                   self.status))


# Kept for callers that only import the session module.
__all__ += ["DirectorError"]
