"""What a director is told: the brief, and how one turn is put to it.

One brief serves every backend. A CLI backend gets it prepended to each prompt;
an agent at the desk (Claude Code, Antigravity) gets it once, as the skill or
rules file the workspace kit installs -- generated from this module, so the two
can never say different things.

The brief is written for MiniMax H3 as Pulse Studio drives it: one beat is one
window of 5.17-15.08 s, the compiler writes H3's structured prompt from the
director's prose, cast members are cited as @Name and resolved to their
reference pictures, and the renderer carries the previous beat's last frame in
by itself. The craft rules are the ones that keep a self-directed film moving.
They draw on what HyperFlow Infinity learned about self-play direction; the
words are our own.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..pulse_memory.schema import json_schema
from .base_director import (
    KIND_OBSERVE,
    KIND_OPEN,
    KIND_PLAN,
    KIND_TURN,
    Capabilities,
    TurnRequest,
)

__all__ = ["DIRECTOR_BRIEF", "OBSERVER_BRIEF", "render_prompt", "convert_path"]

DIRECTOR_BRIEF = """\
You are the director of a long-form generated film. A video-and-audio model,
MiniMax H3, renders it one short beat at a time (5.17 to 15.08 seconds each,
with native sound). After every beat you see what was actually rendered, and you
decide the next one. There is no fixed script: the story goal is where the film
is heading, the scene memory is what has happened, and the evidence is what is
really on screen now.

Answer with JSON only -- no prose around it -- matching the schema you are given.

How to write a beat
- One visible change per beat. Each beat must finish a concrete step of the
  story: something is reached, gained, lost, opened, revealed, decided. A new
  reaction, a new camera angle or a new mood alone is not a step.
- Never restage what already happened. The forbidden actions list is binding;
  so are the recent events. "She opens the gate" and "the gate swings open as
  she pushes" are the same beat.
- Continue from what was rendered, not from what you intended. If the evidence
  shows the action stopped short, start from where it stopped.
- Write only what a camera records, in temporal order: framing and movement,
  light, the setting, each character's look, and the physical action.
- Cite cast members as @Name (exactly as listed). Never type <Picture N>,
  <Video N> or <Audio N> tags -- the compiler assigns those.
- Restate each present character's identity line word for word every beat, and
  keep `continuity_anchors` true: what is in whose hand, the costume as it is
  now, the place and its light. The renderer has no memory of its own.
- The previous beat's last frame is carried into the next beat automatically,
  unless you set "scene_change": true to cut to a new place or time. Prefer
  continuous scenes of a few beats; cut when a scene's purpose is done.
- Keep dialogue short enough to speak inside the beat (at most about 200 words a
  minute), name the speaker, and give a delivery in `emotion_tone`. End a beat on
  a breath, not mid-line: the next beat starts on a new shot.
- Describe the soundscape as continuing; do not invent music or sound tags.
- `negative_prompt` lists what must not happen in this shot. It is enforced by
  checking your plan, not by the renderer, so make the visual_prompt itself
  avoid those things.
- No sexual content, no graphic gore, no on-screen text, subtitles or logos.
- Set "end_of_story": true only when the story goal has been reached on screen
  and the constraints say ending is allowed.
"""

OBSERVER_BRIEF = """\
Report what the evidence actually shows, as changes to the scene memory. Look at
the stills before you read the plan: a planned person, place or object that is not
visible is missing -- say so in `summary`, list it in `continuity_issues`, and do
not report a planned action as done unless you can see it done.
- `summary`: one or two sentences of what is on screen -- not what was meant.
- `completed_actions`: at most three story events this beat finished, each a
  short past-tense phrase ("pushed the heavy gate open"). Glances, smiles,
  steps and camera moves are not events. A beat that only develops a mood has
  none.
- `characters`: per character, only what changed -- posture now, things picked
  up (`inventory_add`) or put down (`inventory_remove`), a visible change to
  their look (`appearance_change`), new `injuries`, `emotional_state`.
- `environment`: only what changed -- setting, time_of_day, lighting,
  camera_orientation, landmarks.
- `stationary`: true when nobody really moved.
- `forbid`: anything you saw repeating that must not happen again.
- `continuity_issues`: drift you can see -- a costume, a prop, a face that does
  not match the memory. The next beat will be asked to correct it.
"""


def convert_path(path: str, style: str = "native") -> str:
    """A path spelled for the system that will open it.

    "wsl" turns D:\\x\\y into /mnt/d/x/y (a Linux agent reading a Windows
    ComfyUI's output); "windows" does the reverse; "native" leaves it alone.
    """
    if not path or style == "native":
        return path
    if style == "wsl" and len(path) > 2 and path[1] == ":":
        return "/mnt/%s/%s" % (path[0].lower(), path[2:].replace("\\", "/").lstrip("/"))
    if style == "windows" and path.startswith("/mnt/") and len(path) > 6 and path[6:7] in ("/", ""):
        return "%s:\\%s" % (path[5].upper(), path[7:].replace("/", "\\"))
    return path


def _ask_line(kind: str) -> str:
    return {
        KIND_OPEN: "Write the opening beat (shot_index %(beat)d). Nothing has been "
                   "rendered yet. Answer {\"plan\": {...}}.",
        KIND_TURN: "First observe beat %(observed)d from the evidence, then write beat "
                   "%(beat)d. Answer {\"observation\": {...}, \"plan\": {...}}.",
        KIND_OBSERVE: "Observe beat %(observed)d from the evidence. Answer the "
                      "observation object.",
        KIND_PLAN: "Write beat %(beat)d. Answer the plan object.",
    }[kind]


def render_prompt(request: TurnRequest, capabilities: Capabilities,
                  path_style: str = "native", include_brief: bool = True,
                  schema: Optional[dict] = None) -> str:
    """The whole prompt for a backend fed by text: brief, situation, evidence, ask.

    Evidence paths are absolute, spelled for the reader (`path_style`), each on
    its own line -- Ollama's CLI attaches an image when a prompt line names one,
    and an agent can open each in turn.
    """
    parts = []
    if include_brief:
        parts.append(DIRECTOR_BRIEF.strip())
        if request.observes:
            parts.append(OBSERVER_BRIEF.strip())
    parts.append("Story goal: %s" % (request.story_goal or "(open-ended)"))
    if request.cast:
        parts.append("Cast (cite as @Name; identity lines are binding):\n" + "\n".join(
            "  - @%s: %s" % (c.get("name"), c.get("description") or "")
            for c in request.cast))
    if request.memory_text:
        parts.append("Scene memory:\n" + request.memory_text)
    if request.last_plan:
        parts.append("The plan the last beat was rendered from (an intention, not a fact):\n"
                     + json.dumps(request.last_plan, indent=1, ensure_ascii=False))
    if request.pacing:
        parts.append("Pacing: " + ", ".join("%s=%s" % kv for kv in sorted(request.pacing.items())))
    if request.constraints:
        parts.append("Constraints:\n" + json.dumps(request.constraints, indent=1,
                                                   ensure_ascii=False))
    if request.guidance:
        parts.append("Creative direction from the user: " + request.guidance)

    evidence = request.evidence
    if request.observes and evidence is not None:
        files = evidence.look_at(capabilities, all_frames=not capabilities.sees_video)
        if files:
            what = ("the full rendered clip of beat %d, with its sound" % evidence.beat
                    if capabilities.sees_video else
                    "stills of beat %d: a contact sheet of evenly spaced frames, then its "
                    "last frame" % evidence.beat)
            parts.append("Evidence -- %s:\n%s" % (what, "\n".join(
                convert_path(os.path.abspath(evidence.absolute(f)), path_style)
                for f in files)))
        else:
            parts.append("You cannot see this beat. Report the observation from the plan "
                         "it was rendered from, and keep it to what that plan made certain.")

    if request.rejection:
        parts.append("Your previous answer was rejected. Fix every one of these and answer "
                     "again with the complete JSON:\n" + "\n".join(
                         "  - " + p for p in request.rejection))
    parts.append(_ask_line(request.kind) % {"beat": request.beat,
                                            "observed": request.beat - 1})
    shape = schema if schema is not None else json_schema(request.schema_kind)
    parts.append("JSON Schema of the answer:\n" + json.dumps(shape, ensure_ascii=False))
    return "\n\n".join(parts) + "\n"
