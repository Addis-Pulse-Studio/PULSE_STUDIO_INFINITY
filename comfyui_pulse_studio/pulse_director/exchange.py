"""Directors reached by sending a request and reading JSON back -- every backend.

A backend is a transport plus a few facts about itself (what it can see,
whether it is a cloud agent, what its CLI is called). The transport is one of
three:

  * `mailbox` -- the Director Desk (mailbox.py): Claude Code, Antigravity, pi;
  * `cli`     -- the backend's own command line (cli_transport.py): Ollama, pi,
                 and `claude -p` / `gemini -p` for unattended runs;
  * `inline`  -- the scripted director answers itself, in process.

Whatever comes back goes through `run_turn`, the one loop that validates an
answer and asks again with the reasons when it falls short -- so a script, a
27B local model and a frontier agent are held to exactly the same contract, and
none of them can put an unvalidated plan in front of the renderer.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from ..durable import write_text_durably
from ..pulse_memory.antiloop import check_plan
from ..pulse_memory.scene_memory import SceneMemory
from ..pulse_memory.schema import (
    DirectedShotPlan,
    ObservationResult,
    TurnResponse,
    json_schema,
    unwrap_json,
)
from .base_director import (
    KIND_OBSERVE,
    KIND_OPEN,
    KIND_PLAN,
    KIND_TURN,
    BaseDirector,
    DirectorError,
    DirectorRejected,
    EvidencePack,
    TurnRequest,
    default_constraints,
)
from .cli_transport import CliRunner, build_argv
from .mailbox import Mailbox
from .prompts import convert_path, render_prompt

__all__ = [
    "ExchangeDirector",
    "MailboxTransport",
    "CliTransport",
    "InlineTransport",
    "validate_answer",
    "run_turn",
]


# ── checking one answer ─────────────────────────────────────────────────────

def validate_answer(raw: Any, request: TurnRequest, memory: SceneMemory,
                    last_plan: Optional[DirectedShotPlan] = None,
                    scene_budget: int = 8) -> tuple[Optional[TurnResponse], list[str], list[str]]:
    """Parse and judge one answer. Returns (turn, problems, notes).

    On a turn, the plan is judged against the memory *as it will be* once the
    observation is committed: the beat just observed may have completed the
    very action the plan restages, and that must count against it.
    """
    notes: list[str] = []
    data, problems = unwrap_json(raw)
    if data is None:
        if isinstance(raw, str) and raw.strip():
            problems = problems + ["the answer was text, not JSON -- answer with the JSON "
                                   "object only"]
        return None, problems, notes

    if request.kind == KIND_OBSERVE:
        payload = data.get("observation", data) if isinstance(data, dict) else data
        observation, problems = ObservationResult.from_dict(payload)
        if observation is not None and observation.shot_index != request.beat - 1:
            notes.append("observation shot_index %d corrected to %d"
                         % (observation.shot_index, request.beat - 1))
            observation.shot_index = request.beat - 1
        return (TurnResponse(None, observation) if observation is not None else None,
                problems, notes)

    cast = memory.cast_names + [c.get("name") for c in request.cast if c.get("name")]
    turn, problems = TurnResponse.from_dict(data, cast_names=cast,
                                            needs_observation=request.kind == KIND_TURN)
    if turn is None:
        return None, problems, notes
    plan = turn.plan
    if plan.shot_index != request.beat:
        notes.append("plan shot_index %d corrected to %d, the beat being planned"
                     % (plan.shot_index, request.beat))
        plan.shot_index = request.beat
    if turn.observation is not None and turn.observation.shot_index != request.beat - 1:
        turn.observation.shot_index = request.beat - 1
    notes.extend(plan.notes)

    judged = memory
    if request.kind == KIND_TURN and last_plan is not None and turn.observation is not None:
        judged = memory.copy()
        judged.commit(last_plan, turn.observation, max_scene_beats=scene_budget)
    problems = problems + check_plan(plan, judged, max_scene_beats=scene_budget)
    if plan.end_of_story and not request.constraints.get("may_end_story", True):
        problems.append("end_of_story is not allowed yet -- the constraints say how many "
                        "beats the film must run first")
    return turn, problems, notes


def run_turn(director: "ExchangeDirector", request: TurnRequest, memory: SceneMemory,
             last_plan: Optional[DirectedShotPlan] = None, max_attempts: int = 3,
             scene_budget: int = 8, on_attempt: Optional[Callable[..., None]] = None
             ) -> tuple[TurnResponse, list[str]]:
    """Ask, judge, and ask again with the reasons -- up to `max_attempts` times.

    Returns (turn, notes). Raises DirectorRejected when every attempt broke the
    contract, carrying the last attempt's problems. `on_attempt(request, raw,
    turn, problems)` sees every attempt, accepted or not -- the session journals
    them through it.
    """
    problems: list[str] = []
    first = request.attempt
    for attempt in range(first, first + max(1, int(max_attempts))):
        request.attempt = attempt
        request.rejection = list(problems)
        raw = director.exchange(request)
        turn, problems, notes = validate_answer(raw, request, memory, last_plan, scene_budget)
        if on_attempt is not None:
            on_attempt(request, raw, turn, problems)
        director.settle(request, accepted=not problems and turn is not None, problems=problems)
        if turn is not None and not problems:
            return turn, notes
    raise DirectorRejected(
        "%s broke the contract on %d attempt(s) at %s; the last answer's problems: %s"
        % (director.name, max(1, int(max_attempts)), request.turn_id, "; ".join(problems)),
        problems)


# ── transports ──────────────────────────────────────────────────────────────

class MailboxTransport:
    """Post the request on the desk; wait for the agent's answer."""

    name = "mailbox"

    def __init__(self, mailbox: Mailbox) -> None:
        self.mailbox = mailbox

    def round_trip(self, request: TurnRequest, director: BaseDirector) -> Any:
        caps = director.capabilities
        evidence = request.evidence
        extra = {
            "backend": director.name,
            "capabilities": caps.to_dict(),
            "look_at": (evidence.look_at(caps, all_frames=not caps.sees_video)
                        if evidence is not None and request.observes else []),
            "answer_schema": json_schema(request.schema_kind),
        }
        self.mailbox.post(request, extra)
        return self.mailbox.wait(request.turn_id)

    def settle(self, request: TurnRequest, accepted: bool, problems: list[str]) -> None:
        if accepted:
            self.mailbox.accept(request.turn_id, {"accepted": True})
        else:
            self.mailbox.reject(request.turn_id, problems)


class CliTransport:
    """Feed the rendered prompt to the backend's CLI on stdin; read stdout."""

    name = "cli"

    def __init__(self, template: list[str], session_root: str, timeout: float = 600.0,
                 path_style: str = "native", model: str = "",
                 runner: Optional[Callable[..., Any]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 instruction: str = "") -> None:
        self.template = list(template)
        self.session_root = session_root
        self.timeout = float(timeout)
        self.path_style = path_style
        self.model = model
        self.runner = runner or CliRunner()
        self.should_cancel = should_cancel
        self.instruction = instruction or ("Answer the director request on standard input "
                                           "with the JSON object only.")

    def round_trip(self, request: TurnRequest, director: BaseDirector) -> Any:
        caps = director.capabilities
        prompt = render_prompt(request, caps, path_style=self.path_style)
        prompt_path = os.path.join(self.session_root, "desk", "cli", request.turn_id + ".prompt.txt")
        write_text_durably(prompt_path, prompt, prefix=".prompt-")
        images = []
        if request.observes and request.evidence is not None and caps.sees_frames:
            images = [convert_path(os.path.abspath(request.evidence.absolute(f)), self.path_style)
                      for f in request.evidence.look_at(caps) if f.lower().endswith(".png")]
        argv = build_argv(self.template, {
            "model": self.model,
            "session_root": convert_path(os.path.abspath(self.session_root), self.path_style),
            "prompt_file": convert_path(os.path.abspath(prompt_path), self.path_style),
            "images": images,
            "instruction": self.instruction,
            "turn_id": request.turn_id,
        })
        result = self.runner(argv, prompt, self.timeout, should_cancel=self.should_cancel,
                             cwd=self.session_root)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-600:]
            raise DirectorError("%s exited with code %d: %s" % (argv[0], result.returncode,
                                                                 tail or "(no output)"))
        return result.stdout

    def settle(self, request: TurnRequest, accepted: bool, problems: list[str]) -> None:
        """A CLI keeps no desk; the session journal is the record."""


class InlineTransport:
    """The director answers itself (the scripted backend)."""

    name = "inline"

    def round_trip(self, request: TurnRequest, director: BaseDirector) -> Any:
        return director.answer(request)

    def settle(self, request: TurnRequest, accepted: bool, problems: list[str]) -> None:
        pass


# ── the base every backend builds on ────────────────────────────────────────

class ExchangeDirector(BaseDirector):
    """A director whose whole interface is one round trip. See the module docstring."""

    default_transport = "mailbox"
    #: Merged under the user's settings for this backend (config.py).
    default_settings: dict = {}

    def __init__(self, settings: Optional[dict] = None, session_root: str = "",
                 should_cancel: Optional[Callable[[], bool]] = None,
                 runner: Optional[Callable[..., Any]] = None,
                 on_status: Optional[Callable[[str], None]] = None,
                 session: str = "") -> None:
        merged = dict(self.default_settings)
        merged.update(settings or {})
        merged.setdefault("transport", self.default_transport)
        super().__init__(merged, session_root, should_cancel)
        self.session = session or os.path.basename(os.path.normpath(session_root or "")) or ""
        self.transport = self._make_transport(runner, on_status)

    def _make_transport(self, runner, on_status):
        kind = self.settings["transport"]
        if kind == "mailbox":
            return MailboxTransport(Mailbox(
                self.session_root, poll_seconds=float(self.settings.get("poll_seconds", 0.5)),
                timeout_seconds=float(self.settings.get("timeout_seconds", 900)),
                should_cancel=self.should_cancel, on_status=on_status))
        if kind == "cli":
            template = self.settings.get("cli")
            if not template:
                raise DirectorError("%s is set to the cli transport but has no `cli` command "
                                    "in pulse_infinity.yaml" % self.name)
            return CliTransport(template, self.session_root,
                                timeout=float(self.settings.get("timeout_seconds", 600)),
                                path_style=str(self.settings.get("path_style", "native")),
                                model=str(self.settings.get("model", "")), runner=runner,
                                should_cancel=self.should_cancel,
                                instruction=str(self.settings.get("instruction", "")))
        if kind == "inline":
            return InlineTransport()
        raise DirectorError("%s: transport must be mailbox, cli or inline, got %r"
                            % (self.name, kind))

    @property
    def transport_name(self) -> str:
        return self.transport.name

    # ── the one primitive ───────────────────────────────────────────────────

    def exchange(self, request: TurnRequest) -> Any:
        return self.transport.round_trip(request, self)

    def settle(self, request: TurnRequest, accepted: bool, problems: list[str]) -> None:
        self.transport.settle(request, accepted, problems)

    def answer(self, request: TurnRequest) -> Any:  # pragma: no cover - inline only
        raise DirectorError("%s does not answer in process" % self.name)

    def can_continue(self, beat: int) -> bool:
        """Whether this director has a beat `beat` to give. Only a script runs out."""
        return True

    # ── the loop's call ─────────────────────────────────────────────────────

    def step(self, request: TurnRequest, memory: SceneMemory,
             last_plan: Optional[DirectedShotPlan] = None, max_attempts: int = 3,
             scene_budget: int = 8, on_attempt: Optional[Callable[..., None]] = None
             ) -> TurnResponse:
        turn, _ = run_turn(self, request, memory, last_plan, max_attempts, scene_budget,
                           on_attempt)
        return turn

    # ── the spec's interface, on the same round trip ────────────────────────

    def _request(self, kind: str, memory: SceneMemory, story_goal: str, beat: int,
                 evidence: Optional[EvidencePack] = None) -> TurnRequest:
        return TurnRequest(
            kind, self.session, beat, story_goal=story_goal or memory.story_goal,
            memory=memory.view_for_director(), memory_text=memory.view_text(),
            evidence=evidence, constraints=default_constraints(memory, may_end=True),
            cast=[{"name": n, "description": s.identity} for n, s in memory.characters.items()])

    def inspect_shot(self, clip_path: str, last_frame: "str | EvidencePack",
                     current_memory: SceneMemory) -> ObservationResult:
        observed = current_memory.current_beat + 1
        if isinstance(last_frame, EvidencePack):
            evidence = last_frame
        else:
            evidence = EvidencePack(self.session_root, observed,
                                    clip=self._relative(clip_path),
                                    last_frame=self._relative(last_frame))
        request = self._request(KIND_OBSERVE, current_memory, current_memory.story_goal,
                                observed + 1, evidence)
        turn, _ = run_turn(self, request, current_memory)
        return turn.observation

    def direct_next_beat(self, memory: SceneMemory, story_goal: str) -> DirectedShotPlan:
        kind = KIND_OPEN if memory.current_beat == 0 else KIND_PLAN
        request = self._request(kind, memory, story_goal, memory.current_beat + 1)
        turn, _ = run_turn(self, request, memory)
        return turn.plan

    def _relative(self, path: str) -> str:
        if not path:
            return ""
        if not self.session_root:
            return path
        return os.path.relpath(os.path.abspath(path), os.path.abspath(self.session_root)
                               ).replace("\\", "/")


# The turn kinds a caller may need alongside the base class.
__all__ += ["KIND_OPEN", "KIND_TURN", "KIND_OBSERVE", "KIND_PLAN"]
