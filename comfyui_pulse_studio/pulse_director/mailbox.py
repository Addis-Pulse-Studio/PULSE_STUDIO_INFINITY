"""The Director Desk: a filesystem mailbox between ComfyUI and an agent.

Claude Code and Antigravity are agents in a workspace, not servers. They read
files and write files. So that is the whole protocol -- `pulse-desk/1`:

    <session>/desk/inbox/<turn>.json      ComfyUI asks           (written once)
    <session>/desk/outbox/<turn>.json     the agent answers      (written once)
    <session>/desk/done/                  accepted pairs, kept as the record
    <session>/desk/rejected/              answers that broke the contract, + why
    <session>/desk/heartbeat.json         "a director is at the desk"

`<turn>` is `b0004-turn-a1`: the beat being planned, the kind of turn, the
attempt. A rejected answer is never retried under the same name -- the next
attempt is `a2`, with the problems in its request -- so neither side ever
replaces a file the other may have open. On Windows that replace fails outright.

WHY POLLING
-----------
ComfyUI writes from Windows; the agent may be reading from WSL, where inotify
does not see writes made by Windows processes. Polling every half second costs
nothing next to a director that thinks for twenty.

UNTRUSTED
---------
An answer is whatever an agent wrote. It is size-capped, parsed by
pulse_memory.schema.unwrap_json, and validated by the loop; nothing in it is
ever followed as a path.

THE PACK SENDS NOTHING
----------------------
These are local files. Nothing leaves the machine unless the person running
the session points an agent at this folder -- and a cloud agent then reads the
evidence frames, which the node report says plainly (BaseDirector.describe).

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..durable import write_json_durably
from .base_director import DirectorCancelled, DirectorTimeout, TurnRequest

__all__ = [
    "PROTOCOL",
    "DESK_DIR",
    "MAX_RESPONSE_BYTES",
    "Mailbox",
    "utc_now",
]

PROTOCOL = "pulse-desk/1"
DESK_DIR = "desk"
#: An answer larger than this is not a plan, it is a runaway generation.
MAX_RESPONSE_BYTES = 256 * 1024
#: A heartbeat older than this means nobody is at the desk.
HEARTBEAT_GRACE_SECONDS = 120.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Mailbox:
    """One session's desk. Paths are the session folder's; nothing else is touched."""

    __slots__ = ("root", "poll_seconds", "timeout_seconds", "should_cancel", "sleep",
                 "clock", "on_status", "status_every")

    def __init__(self, root: str, poll_seconds: float = 0.5, timeout_seconds: float = 900.0,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 on_status: Optional[Callable[[str], None]] = None,
                 status_every: float = 30.0) -> None:
        self.root = root
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.timeout_seconds = float(timeout_seconds or 0)
        self.should_cancel = should_cancel or (lambda: False)
        self.sleep = sleep
        self.clock = clock
        self.on_status = on_status
        self.status_every = float(status_every)

    # ── layout ──────────────────────────────────────────────────────────────

    def _dir(self, name: str) -> str:
        return os.path.join(self.root, DESK_DIR, name)

    def inbox_path(self, turn_id: str) -> str:
        return os.path.join(self._dir("inbox"), turn_id + ".json")

    def outbox_path(self, turn_id: str) -> str:
        return os.path.join(self._dir("outbox"), turn_id + ".json")

    def relative(self, path: str) -> str:
        return os.path.relpath(path, self.root).replace("\\", "/")

    # ── ComfyUI's side ──────────────────────────────────────────────────────

    def post(self, request: TurnRequest, extra: Optional[dict] = None) -> str:
        """Put a request on the desk. Idempotent: a request already there stays.

        A node re-run (ComfyUI's cache can evict and re-execute any node) posts the
        same turn again; the agent may already be answering the copy on disk, so
        it is left exactly as it is.
        """
        path = self.inbox_path(request.turn_id)
        if os.path.exists(path):
            return path
        message = dict(request.to_dict())
        message.update(extra or {})
        message.update({"protocol": PROTOCOL, "posted_utc": utc_now(),
                        "reply_to": self.relative(self.outbox_path(request.turn_id))})
        return write_json_durably(path, message, prefix=".post-")

    def wait(self, turn_id: str) -> Any:
        """Block until the answer to `turn_id` is on the desk, and return it parsed.

        Returns the raw JSON (validation is the loop's), or the raw text when the
        agent wrote something that is not JSON -- that becomes the rejection.
        Raises DirectorCancelled on ComfyUI's interrupt and DirectorTimeout when
        `timeout_seconds` passes (0 waits for ever).
        """
        started = self.clock()
        last_status = started
        path = self.outbox_path(turn_id)
        while True:
            if self.should_cancel():
                raise DirectorCancelled("interrupted while waiting for the director")
            if os.path.exists(path):
                answer = self._read_answer(path)
                if answer is not None:
                    return answer
            now = self.clock()
            waited = now - started
            if self.timeout_seconds and waited >= self.timeout_seconds:
                raise DirectorTimeout(self._nobody_home(turn_id, waited))
            if self.on_status and now - last_status >= self.status_every:
                last_status = now
                self.on_status(self.status_line(turn_id, waited))
            self.sleep(self.poll_seconds)

    def _read_answer(self, path: str) -> Any:
        """The answer file, or None while it is still being written."""
        try:
            size = os.path.getsize(path)
            if size > MAX_RESPONSE_BYTES:
                return ("the answer file is %d bytes; an answer is at most %d"
                        % (size, MAX_RESPONSE_BYTES))
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return None
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except ValueError:
            # Either a write caught half-way (an agent not using the desk tool) or
            # prose. Give a half-written file one more poll before calling it prose.
            self.sleep(self.poll_seconds)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    again = handle.read()
            except OSError:
                return None
            if again != text:
                return None
            return text

    def accept(self, turn_id: str, verdict: Optional[dict] = None) -> None:
        """File an accepted pair under done/, with the loop's verdict beside it."""
        self._file(turn_id, "done", verdict)

    def reject(self, turn_id: str, problems: list[str]) -> None:
        """File a rejected answer under rejected/, with every problem it had."""
        self._file(turn_id, "rejected", {"problems": list(problems), "utc": utc_now()})

    def _file(self, turn_id: str, folder: str, note: Optional[dict]) -> None:
        target = self._dir(folder)
        os.makedirs(target, exist_ok=True)
        for source, suffix in ((self.inbox_path(turn_id), ".request.json"),
                               (self.outbox_path(turn_id), ".response.json")):
            if os.path.exists(source):
                try:
                    os.replace(source, os.path.join(target, turn_id + suffix))
                except OSError:  # pragma: no cover - the agent still has it open
                    pass
        if note is not None:
            write_json_durably(os.path.join(target, turn_id + ".verdict.json"), note,
                               prefix=".verdict-")

    # ── the agent's side (tools/pulse_desk.py) ──────────────────────────────

    def pending(self) -> list[dict]:
        """Requests with no answer yet, oldest first."""
        inbox = self._dir("inbox")
        try:
            names = sorted(n for n in os.listdir(inbox) if n.endswith(".json"))
        except OSError:
            return []
        out = []
        for name in names:
            turn_id = name[:-5]
            if os.path.exists(self.outbox_path(turn_id)):
                continue
            try:
                with open(os.path.join(inbox, name), "r", encoding="utf-8") as handle:
                    message = json.load(handle)
            except (OSError, ValueError):
                continue
            message.setdefault("turn_id", turn_id)
            out.append(message)
        return out

    def submit(self, turn_id: str, answer: Any) -> str:
        """Write the answer to `turn_id`, once. Refuses a second answer."""
        path = self.outbox_path(turn_id)
        if os.path.exists(path):
            raise FileExistsError("turn %s already has an answer on the desk" % turn_id)
        if not os.path.exists(self.inbox_path(turn_id)):
            raise FileNotFoundError("no request %s is waiting on the desk" % turn_id)
        return write_json_durably(path, answer, prefix=".answer-")

    def beat_heart(self, who: str = "director") -> None:
        write_json_durably(os.path.join(self.root, DESK_DIR, "heartbeat.json"),
                           {"who": who, "utc": utc_now(), "epoch": time.time()},
                           prefix=".heartbeat-")

    def heartbeat_age(self) -> Optional[float]:
        """Seconds since a director last said it was at the desk, or None if never."""
        try:
            with open(os.path.join(self.root, DESK_DIR, "heartbeat.json"), "r",
                      encoding="utf-8") as handle:
                return max(0.0, time.time() - float(json.load(handle)["epoch"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ── telling a person what is going on ───────────────────────────────────

    def status_line(self, turn_id: str, waited: float) -> str:
        age = self.heartbeat_age()
        if age is None or age > HEARTBEAT_GRACE_SECONDS:
            return ("waiting %.0f s for the director on %s -- nobody has checked the desk "
                    "%s. Start it: /pulse-director in Claude Code, or /pulse-direct in "
                    "Antigravity, in the director workspace."
                    % (waited, turn_id, "yet" if age is None else "for %.0f s" % age))
        return "waiting %.0f s for the director on %s (at the desk %.0f s ago)" % (
            waited, turn_id, age)

    def _nobody_home(self, turn_id: str, waited: float) -> str:
        return ("no answer to %s after %.0f s. %s The request is still on the desk at %s; "
                "queue again to keep waiting for it."
                % (turn_id, waited, self.status_line(turn_id, waited).split(" -- ", 1)[-1]
                   if " -- " in self.status_line(turn_id, waited) else "",
                   self.relative(self.inbox_path(turn_id))))
