#!/usr/bin/env python3
"""The Director Desk, from the director's side: wait, look, answer.

    python3 tools/pulse_desk.py next --wait 540        the next turn ComfyUI asks
    python3 tools/pulse_desk.py submit <turn> answer.json
    python3 tools/pulse_desk.py status | sessions | memory | schema turn
    python3 tools/pulse_desk.py control pause|resume|stop|reroll|replan [--note ...]

For Claude Code, Antigravity, pi or a person: Pulse Studio Infinity asks each
turn's question through files in the session folder (pulse_director.mailbox,
protocol pulse-desk/1), and this is the tool an agent uses to answer. `submit`
runs the pack's own validator first -- the same rules ComfyUI will apply -- so an
agent sees every problem with its answer before it costs a round trip.

Stdlib only, and it imports nothing but the pack's pure core, so it runs under
any Python 3.10+, in WSL or on Windows, with nothing installed. Paths in a
request are relative to the session folder; a Windows path given to this tool
under WSL (D:\\...) is read as /mnt/d/... automatically.

It never writes inside the pack, and inside a session it only ever writes the
answer to a turn, the heartbeat, and control.json.
"""

import argparse
import json
import os
import sys
import time

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

from comfyui_pulse_studio.pulse_director import Mailbox, TurnRequest  # noqa: E402
from comfyui_pulse_studio.pulse_director.exchange import validate_answer  # noqa: E402
from comfyui_pulse_studio.pulse_director.prompts import convert_path  # noqa: E402
from comfyui_pulse_studio.pulse_memory.schema import (  # noqa: E402
    DirectedShotPlan,
    json_schema,
    unwrap_json,
)
from comfyui_pulse_studio.pulse_memory.scene_memory import SceneMemory  # noqa: E402

WORKSPACE_FILE = ".pulse_desk.json"
CONTROL_ACTIONS = ("pause", "resume", "stop", "reroll", "replan")


def _on_wsl():
    if os.name == "nt":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _local(path):
    """A path spelled for this machine: D:\\x under WSL becomes /mnt/d/x."""
    if not path:
        return path
    if _on_wsl() and len(path) > 2 and path[1] == ":":
        return convert_path(path, "wsl")
    if os.name == "nt" and path.startswith("/mnt/"):
        return convert_path(path, "windows")
    return path


def _workspace_settings():
    """The director workspace's .pulse_desk.json, from here or any parent folder."""
    here = os.getcwd()
    while True:
        candidate = os.path.join(here, WORKSPACE_FILE)
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, ValueError):
                return {}
        parent = os.path.dirname(here)
        if parent == here:
            return {}
        here = parent


def sessions_root(args):
    root = (args.root or os.environ.get("PULSE_INFINITY_SESSIONS")
            or _workspace_settings().get("sessions_root"))
    if not root:
        sys.exit("pulse_desk: no sessions folder. Pass --root <ComfyUI/output/pulse_infinity>, "
                 "set PULSE_INFINITY_SESSIONS, or run tools/setup_director_workspace.py.")
    return _local(root)


def list_sessions(root):
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [n for n in names if os.path.isfile(os.path.join(root, n, "session.json"))]


def _pending(root, session=None):
    """[(session, message)] waiting for an answer, oldest first."""
    found = []
    for name in ([session] if session else list_sessions(root)):
        box = Mailbox(os.path.join(root, name))
        for message in box.pending():
            found.append((name, message))
    found.sort(key=lambda pair: pair[1].get("posted_utc", ""))
    return found


def _read_memory(session_root):
    """The session's memory, read only -- this tool never repairs another's file."""
    try:
        with open(os.path.join(session_root, "scene_memory.json"), "r", encoding="utf-8") as fh:
            return SceneMemory.from_dict(json.load(fh))
    except (OSError, ValueError):
        return SceneMemory()


def _brief(session_root, message):
    """The turn, as an agent should read it: what to do, what to look at, the rules."""
    request = TurnRequest.from_dict(session_root, message)
    caps = message.get("capabilities") or {}
    lines = ["TURN %s  (session %s, %s)" % (message["turn_id"], message.get("session"),
                                            message.get("backend", "director")),
             "Kind: %s -- %s" % (request.kind, {
                 "open": "write the opening beat",
                 "turn": "observe beat %d, then plan beat %d" % (request.beat - 1, request.beat),
                 "observe": "observe beat %d" % (request.beat - 1),
                 "plan": "plan beat %d" % request.beat}[request.kind]),
             "Story goal: %s" % (request.story_goal or "(open-ended)")]
    if request.guidance:
        lines.append("User direction: %s" % request.guidance)
    if request.cast:
        lines.append("Cast (cite as @Name):")
        lines += ["  @%s: %s" % (c.get("name"), c.get("description")) for c in request.cast]
    look = message.get("look_at") or []
    if look:
        lines.append("Look at, in this order (%s):" % ", ".join(
            k for k in ("sees_video", "hears_audio", "sees_frames") if caps.get(k)))
        lines += ["  " + os.path.abspath(os.path.join(session_root, *p.split("/")))
                  for p in look]
    if request.memory_text:
        lines += ["", "Scene memory:", request.memory_text]
    if request.last_plan:
        lines += ["", "The plan the last beat was rendered from (intent, not fact):",
                  json.dumps(request.last_plan, indent=1, ensure_ascii=False)]
    if request.constraints:
        lines += ["", "Constraints:", json.dumps(request.constraints, indent=1,
                                                 ensure_ascii=False)]
    if request.rejection:
        lines += ["", "Your previous answer was rejected -- fix all of these:"]
        lines += ["  - " + p for p in request.rejection]
    answer = '{"observation": {...}, "plan": {...}}' if request.observes and request.plans \
        else ('{"plan": {...}}' if request.plans else '{"observation": {...}}')
    lines += ["", "Answer with %s (schema: pulse_desk.py schema %s), then:" % (
        answer, request.schema_kind),
              "  %s %s submit %s <answer.json>" % ("python" if os.name == "nt" else "python3",
                                                     os.path.abspath(__file__),
                                                     message["turn_id"])]
    return "\n".join(lines)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_sessions(args):
    root = sessions_root(args)
    for name in list_sessions(root):
        with open(os.path.join(root, name, "session.json"), "r", encoding="utf-8") as fh:
            state = json.load(fh)
        print("%-24s %-9s beat %s committed, beat %s next" % (
            name, state.get("status"), state.get("beat"), state.get("pending_beat")))
    return 0


def cmd_status(args):
    root = sessions_root(args)
    waiting = _pending(root, args.session)
    for name in ([args.session] if args.session else list_sessions(root)):
        box = Mailbox(os.path.join(root, name))
        age = box.heartbeat_age()
        print("%s: %d turn(s) waiting; a director was last at the desk %s" % (
            name, sum(1 for s, _ in waiting if s == name),
            "never" if age is None else "%.0f s ago" % age))
    return 0


def cmd_next(args):
    root = sessions_root(args)
    deadline = time.monotonic() + max(0.0, float(args.wait or 0))
    last_beat = 0.0
    while True:
        waiting = _pending(root, args.session)
        now = time.monotonic()
        if now - last_beat > 10:
            for name in ([args.session] if args.session else list_sessions(root)):
                Mailbox(os.path.join(root, name)).beat_heart(args.who)
            last_beat = now
        if waiting:
            name, message = waiting[0]
            session_root = os.path.join(root, name)
            if args.json:
                message = dict(message, session_root=session_root,
                               look_at_absolute=[os.path.abspath(os.path.join(
                                   session_root, *p.split("/"))) for p in
                                   message.get("look_at") or []])
                print(json.dumps(message, indent=1, ensure_ascii=False))
            else:
                print(_brief(session_root, message))
            return 0
        if now >= deadline:
            print("no turn waiting%s" % (" after %.0f s" % args.wait if args.wait else ""))
            return 3
        time.sleep(1.0)


def _find(root, turn_id, session=None):
    for name in ([session] if session else list_sessions(root)):
        box = Mailbox(os.path.join(root, name))
        path = box.inbox_path(turn_id)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return name, box, json.load(fh)
    sys.exit("pulse_desk: no turn %s is waiting on the desk" % turn_id)


def cmd_show(args):
    root = sessions_root(args)
    name, _, message = _find(root, args.turn, args.session)
    print(_brief(os.path.join(root, name), message))
    return 0


def cmd_submit(args):
    root = sessions_root(args)
    name, box, message = _find(root, args.turn, args.session)
    session_root = os.path.join(root, name)
    if args.answer == "-":
        text = sys.stdin.read()
    else:
        with open(args.answer, "r", encoding="utf-8") as fh:
            text = fh.read()
    data, problems = unwrap_json(text)
    if data is None:
        print("REJECTED -- the answer is not JSON:\n  " + "\n  ".join(problems))
        return 1
    request = TurnRequest.from_dict(session_root, message)
    memory = _read_memory(session_root)
    last_plan = None
    if request.last_plan:
        last_plan, _ = DirectedShotPlan.from_dict(request.last_plan)
    budget = int((request.constraints or {}).get("scene_beats_left", 8)) + memory.scene_length
    turn, problems, notes = validate_answer(data, request, memory, last_plan, budget)
    if (turn is None or problems) and not args.force:
        print("REJECTED -- fix every one of these and submit again:")
        print("\n".join("  - " + p for p in problems or ["the answer has no usable plan"]))
        return 1
    box.submit(args.turn, data)
    print("submitted %s%s" % (args.turn, "".join("\n  note: " + n for n in notes)))
    return 0


def cmd_schema(args):
    print(json.dumps(json_schema(args.kind), indent=1, ensure_ascii=False))
    return 0


def cmd_memory(args):
    root = sessions_root(args)
    name = args.session or (list_sessions(root) or [None])[-1]
    if not name:
        sys.exit("pulse_desk: no sessions yet")
    memory = _read_memory(os.path.join(root, name))
    print(json.dumps(memory.to_dict(), indent=1, ensure_ascii=False) if args.json
          else memory.view_text())
    return 0


def cmd_control(args):
    """control.json is read by the Loop node at the start of the next queue run."""
    root = sessions_root(args)
    name = args.session or (list_sessions(root) or [None])[-1]
    if not name:
        sys.exit("pulse_desk: no sessions yet")
    path = os.path.join(root, name, "control.json")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"action": args.action, "note": args.note or ""}, fh)
    os.replace(path + ".tmp", path)
    print("%s: %s requested; it applies when the graph is next queued" % (name, args.action))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pulse_desk", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", help="the sessions folder (ComfyUI/output/pulse_infinity)")
    parser.add_argument("--session", help="one session by name (default: whichever asks)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sessions").set_defaults(fn=cmd_sessions)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    nxt = sub.add_parser("next")
    nxt.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a turn")
    nxt.add_argument("--json", action="store_true")
    nxt.add_argument("--who", default="director", help="named in the heartbeat")
    nxt.set_defaults(fn=cmd_next)
    show = sub.add_parser("show")
    show.add_argument("turn")
    show.set_defaults(fn=cmd_show)
    submit = sub.add_parser("submit")
    submit.add_argument("turn")
    submit.add_argument("answer", help="a JSON file, or - for stdin")
    submit.add_argument("--force", action="store_true",
                        help="submit even when the validator objects (ComfyUI will still check)")
    submit.set_defaults(fn=cmd_submit)
    schema = sub.add_parser("schema")
    schema.add_argument("kind", choices=("plan", "observation", "turn", "open"))
    schema.set_defaults(fn=cmd_schema)
    memory = sub.add_parser("memory")
    memory.add_argument("--json", action="store_true")
    memory.set_defaults(fn=cmd_memory)
    control = sub.add_parser("control")
    control.add_argument("action", choices=CONTROL_ACTIONS)
    control.add_argument("--note", default="")
    control.set_defaults(fn=cmd_control)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
