"""The scripted director: offline, deterministic, and held to the same contract.

For wiring a graph, smoke-testing a GPU box, and every headless test, a director
that needs no model at all. It answers each turn from a script -- the Loop
Controller's `script` widget, or `backends.scripted.script` in the config -- and
its "observation" is the plan it last gave, taken at its word and marked
`intended` in memory. Its answers go through the same validator as any model's,
so a script that restages a beat is refused like one.

SCRIPT FORMAT
-------------
A JSON list of plan objects, or the plain format below: beats separated by a
line of `---`, one `key: value` per line, every key optional except `visual`.

    beat: Alice leaves the clock behind
    scene: Clock garden
    visual: @Alice turns away from the stopped clock onto the garden path.
    camera: slow dolly forward, low angle
    say: Alice | whispering, anxious | The time has already stopped...
    hold: brass key in left hand
    wear: torn blue apron dress
    where: overgrown clock garden, twilight
    seconds: 5.17
    cut: no
    avoid: standing still, looking down at hands
    ---
    visual: ...

A line with no key continues the previous value, so a long `visual` may wrap.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..pulse_memory.schema import MIN_BEAT_SECONDS, DirectedShotPlan, ObservationResult
from .base_director import Capabilities, DirectorError, TurnRequest
from .exchange import ExchangeDirector

__all__ = ["ScriptedDirector", "parse_script"]

_KEY_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.*)$")
_KEYS = {"beat", "scene", "visual", "camera", "say", "hold", "wear", "where", "seconds",
         "cut", "avoid", "end"}


def _yes(value: str) -> bool:
    return value.strip().casefold() in ("yes", "true", "y", "1", "cut")


def parse_script(text: Any) -> tuple[list[dict], list[str]]:
    """A script, as plan dicts (shot_index left for the director to fill)."""
    if isinstance(text, list):
        return [p for p in text if isinstance(p, dict)], []
    text = str(text or "").strip()
    if not text:
        return [], []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except ValueError as exc:
            return [], ["the script looks like JSON but does not parse: %s" % exc]
        return [p for p in data if isinstance(p, dict)], []

    plans, problems = [], []
    for number, block in enumerate(re.split(r"^\s*---+\s*$", text, flags=re.MULTILINE), 1):
        fields: dict[str, str] = {}
        key = None
        for line in block.splitlines():
            if not line.strip():
                continue
            match = _KEY_RE.match(line)
            if match and match.group(1).casefold() in _KEYS:
                key = match.group(1).casefold()
                fields[key] = match.group(2).strip()
            elif key:
                fields[key] = (fields[key] + " " + line.strip()).strip()
            else:
                key = "visual"
                fields[key] = line.strip()
        if not fields:
            continue
        if not fields.get("visual"):
            problems.append("script beat %d has no `visual:` line" % number)
            continue
        dialogue = None
        if fields.get("say"):
            speaker, _, rest = fields["say"].partition("|")
            tone, _, line_text = rest.partition("|")
            if not line_text:
                tone, line_text = "", tone
            dialogue = {"speaker": speaker.strip().lstrip("@"), "text": line_text.strip(),
                        "emotion_tone": tone.strip()}
        try:
            seconds = float(fields.get("seconds") or MIN_BEAT_SECONDS)
        except ValueError:
            problems.append("script beat %d: seconds %r is not a number"
                            % (number, fields["seconds"]))
            seconds = MIN_BEAT_SECONDS
        plan = {
            "shot_index": 1,
            "visual_prompt": fields["visual"],
            "negative_prompt": fields.get("avoid", ""),
            "camera_motion": fields.get("camera", ""),
            "duration_seconds": seconds,
            "dialogue": dialogue,
            "continuity_anchors": {
                "held_items": [h.strip() for h in fields.get("hold", "").split(";") if h.strip()],
                "costume": fields.get("wear", ""),
                "environment": fields.get("where", "")},
        }
        if fields.get("beat"):
            plan["beat"] = fields["beat"]
        if fields.get("scene"):
            plan["scene"] = fields["scene"]
        if fields.get("cut"):
            plan["scene_change"] = _yes(fields["cut"])
        if fields.get("end"):
            plan["end_of_story"] = _yes(fields["end"])
        plans.append(plan)
    return plans, problems


class ScriptedDirector(ExchangeDirector):
    name = "scripted"
    label = "Scripted (offline)"
    capabilities = Capabilities(sees_frames=False, sees_video=False, hears_audio=False,
                                cloud=False)
    default_transport = "inline"
    default_settings = {"transport": "inline", "script": [], "loop_script": False}

    def __init__(self, settings: Optional[dict] = None, *args, **kwargs) -> None:
        super().__init__(settings, *args, **kwargs)
        self.beats, problems = parse_script(self.settings.get("script"))
        if problems:
            raise DirectorError("the script cannot be used: " + "; ".join(problems))

    def can_continue(self, beat: int) -> bool:
        return bool(self.beats) and (beat <= len(self.beats)
                                     or bool(self.settings.get("loop_script")))

    def _scripted(self, beat: int) -> dict:
        if not self.beats:
            raise DirectorError("the scripted director has no script -- write one in the "
                                "Loop Controller's `script` widget")
        index = (beat - 1) % len(self.beats) if self.settings.get("loop_script") \
            else min(beat, len(self.beats)) - 1
        plan = json.loads(json.dumps(self.beats[index]))
        plan["shot_index"] = beat
        if not self.settings.get("loop_script") and beat >= len(self.beats):
            plan.setdefault("end_of_story", True)
        return plan

    def answer(self, request: TurnRequest) -> Any:
        """The turn, from the script. The observation of the last beat is its plan."""
        out = {}
        if request.observes:
            last = None
            if request.last_plan:
                last, _ = DirectedShotPlan.from_dict(request.last_plan)
            if last is None:
                last, _ = DirectedShotPlan.from_dict(self._scripted(max(1, request.beat - 1)))
            out["observation"] = ObservationResult.from_intent(last).to_dict()
        if request.plans:
            out["plan"] = self._scripted(request.beat)
        return out if request.kind != "observe" else out["observation"]
