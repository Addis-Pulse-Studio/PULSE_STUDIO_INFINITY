"""DaVinci Resolve Studio's Super Scale, run from a ComfyUI node: the decisions.

Pulse Resolve Super Scale (../resolve_nodes.py) hands a finished film to Resolve
Studio and takes back the 2K or 4K film Resolve renders. Resolve's scripting
module never loads into ComfyUI:

  * ComfyUI runs Python 3.11; the module is driven from Resolve's own Python
    (3.14 in Resolve 21.1.1), where tools/resolve_superscale.py is proven live.
  * The module links Windows' socket library (fusionscript.dll imports
    WS2_32). In ComfyUI's process that would put a socket client in the pack.
  * A crash in native code there would take the ComfyUI server down with it.

So the node runs the helper under ResolvePython.exe as a child process, and every
decision on the way is a pure function here, tested without Resolve or ComfyUI:

    where Resolve and its Python are          find_install
    which film goes to Resolve                native_film, already_delivered
    the canvas it is delivered on             delivery_canvas
    where the result is written               output_dir, output_name
    the helper's command line                 helper_argv
    reading what the helper prints            parse_progress, parse_result,
                                              never_answered
    whether Resolve is running                resolve_listed
    an unfinished film, removed               discard_unfinished

The helper imports TARGETS, MODES and CODECS from here, so the node's combos and
the helper's choices are one list. Stdlib only: Resolve's Python imports this.
"""

import ntpath
import os
import re
import time

from .concat import upscale_size

__all__ = [
    "CODECS", "MODES", "TARGETS",
    "OPEN_PROJECT_MODES", "OPEN_PROJECT_STOP", "OPEN_PROJECT_SWITCH",
    "WHEN_DONE_LEAVE", "WHEN_DONE_MODES", "WHEN_DONE_QUIT",
    "NATIVE_FILM", "RESOLVE_DIR_ENV", "TASKLIST_ARGV", "ResolveBridgeError",
    "already_delivered", "delivery_canvas", "discard_unfinished", "explain", "find_install",
    "helper_argv", "native_film", "never_answered", "output_dir", "output_name",
    "parse_progress", "parse_result", "resolve_listed",
]

#: Delivery sizes Super Scale renders to, on Pulse's own `<target>_exact` canvas.
TARGETS = ("1440p", "2160p")
#: Super Scale's multipliers. `auto` is 2x Enhanced with Resolve's own resize
#: covering the rest, measured the best of them on a Pulse film (README).
MODES = ("auto", "2x", "2x_enhanced", "3x", "4x")
#: Render codecs: Resolve's own encoder of each, before a vendor's.
CODECS = ("h265", "h264")

#: What the node does when one of the user's own projects is open in Resolve.
OPEN_PROJECT_STOP = "stop"
OPEN_PROJECT_SWITCH = "switch_and_restore"
OPEN_PROJECT_MODES = (OPEN_PROJECT_STOP, OPEN_PROJECT_SWITCH)

#: What happens to Resolve after the render.
WHEN_DONE_QUIT = "quit_if_started_here"
WHEN_DONE_LEAVE = "leave_running"
WHEN_DONE_MODES = (WHEN_DONE_QUIT, WHEN_DONE_LEAVE)

#: Names Resolve's folder when it is not where Resolve installs itself.
RESOLVE_DIR_ENV = "PULSE_RESOLVE_DIR"
RESOLVE_IMAGE = "Resolve.exe"
#: Windows' process list, filtered to Resolve: one quoted CSV row per process.
TASKLIST_ARGV = ("tasklist", "/FI", "IMAGENAME eq %s" % RESOLVE_IMAGE, "/FO", "CSV", "/NH")

#: The film a Pulse run folder always holds (render.py), and its deliveries.
NATIVE_FILM = "assembled.mp4"
_DELIVERY_RE = re.compile(r"^assembled_.+\.mp4$", re.IGNORECASE)
#: Under ComfyUI's output folder, for a film that does not already live there.
FALLBACK_SUBDIR = "pulse_resolve"
#: How the helper begins the one line it writes to stderr when it fails.
ERROR_PREFIX = "resolve_superscale: "
#: How the helper says Resolve's scripting never answered (see never_answered).
NEVER_ANSWERED = "DaVinci Resolve did not answer within"
_PROGRESS_RE = re.compile(r"^\s*(\d{1,3})(?:\.\d+)?\s*%\s*$")
#: The helper speaks in command-line flags; the node's user sees widgets.
_IN_NODE_WORDS = (
    ("pass --switch-project", "set open_project to switch_and_restore"),
    ("the .bat starts it", "the node starts it when it is not"),
)


class ResolveBridgeError(RuntimeError):
    """Something the node cannot do; the message says what, and what to change."""


def find_install(env, exists):
    """(Resolve.exe, ResolvePython.exe) on this machine.

    `env` is the environment (os.environ) and `exists` a file test
    (os.path.isfile). The folder is $PULSE_RESOLVE_DIR when it is set, and
    otherwise where Resolve installs itself under %ProgramFiles%.
    """
    folder = (env.get(RESOLVE_DIR_ENV) or "").strip().strip('"')
    named = bool(folder)
    if not named:
        root = env.get("ProgramFiles") or env.get("PROGRAMFILES") or "C:\\Program Files"
        folder = ntpath.join(root, "Blackmagic Design", "DaVinci Resolve")
    resolve_exe = ntpath.join(folder, RESOLVE_IMAGE)
    python_exe = ntpath.join(folder, "ResolvePython", "ResolvePython.exe")
    if not exists(resolve_exe):
        if named:
            raise ResolveBridgeError("%s is %s, but there is no %s in it."
                                     % (RESOLVE_DIR_ENV, folder, RESOLVE_IMAGE))
        raise ResolveBridgeError(
            "DaVinci Resolve Studio is not installed at %s. Super Scale is a Studio "
            "feature. If Resolve is in another folder, set %s to that folder and "
            "restart ComfyUI." % (folder, RESOLVE_DIR_ENV))
    if not exists(python_exe):
        raise ResolveBridgeError(
            "Resolve's own Python is not at %s. The node runs Super Scale through it, "
            "so it needs a DaVinci Resolve that ships one (21.1.1 does)." % python_exe)
    return resolve_exe, python_exe


def delivery_canvas(width, height, target):
    """The canvas a film is delivered on at `target`: Pulse's own `<target>_exact`.

    3840x2160 or 2560x1440, turned for portrait and square for square -- the
    canvas Pulse Render's `_exact` and `_rtx` modes deliver too.
    """
    if target not in TARGETS:
        raise ValueError("target must be one of %s, got %r" % (", ".join(TARGETS), target))
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("a film of %sx%s has no size to deliver" % (width, height))
    return upscale_size(width, height, "%s_exact" % target)[2:]


def already_delivered(width, height, canvas_w, canvas_h):
    """True when the film already fills the canvas both ways: nothing to upscale."""
    return width >= canvas_w and height >= canvas_h


def native_film(path, exists):
    """(film, swapped): Pulse's native film in place of one of its deliveries.

    Pulse Render hands on `assembled_<mode>.mp4` when its own `upscale` is set: a
    film Lanczos or RTX has already enlarged. Super Scale is meant to start from
    what H3 rendered, so when `assembled.mp4` sits beside it, that is the film
    sent, as the .bat does with a run folder. Anything else is sent as it is.
    """
    folder, name = os.path.split(path)
    if _DELIVERY_RE.match(name):
        native = os.path.join(folder, NATIVE_FILM)
        if exists(native):
            return native, True
    return path, False


def output_dir(film, output_root):
    """The folder the Super Scaled film is written to.

    Beside the film when it lives under ComfyUI's output folder: a Pulse run
    folder, next to assembled.mp4 and its other deliveries, where the .bat
    writes too. A film from anywhere else -- ComfyUI's input or temp folder,
    another drive -- goes to output/pulse_resolve, so nothing is ever written
    into a folder of sources.
    """
    folder = os.path.dirname(os.path.abspath(film))
    root = os.path.abspath(output_root)
    return folder if _within(folder, root) else os.path.join(root, FALLBACK_SUBDIR)


def output_name(film, target):
    """'assembled.mp4', '2160p' -> 'assembled_2160p_resolve.mp4'."""
    stem = os.path.splitext(os.path.basename(film))[0]
    return "%s_%s_resolve.mp4" % (stem, target)


def discard_unfinished(path, since, wait=0.0, poll=0.5):
    """Remove the film a render that did not finish left behind.

    Only a file written since `since` (when the render began) is touched, never
    an older film of the same name. Resolve can hold a stopped render's file open
    until it quits, so the removal is retried for `wait` seconds. Returns False
    while the unfinished film is still there, True otherwise.
    """
    deadline = time.monotonic() + wait
    while True:
        try:
            if os.path.getmtime(path) < since:
                return True
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:  # still open in Resolve
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)


def _within(path, root):
    path, root = os.path.normcase(path), os.path.normcase(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def helper_argv(python_exe, helper, film, out_dir, stop_file, target="2160p", mode="auto",
                sharpness=0.5, noise_reduction=0.5, codec="h265", switch_project=False,
                quit_after=False, wait=180):
    """The command that runs tools/resolve_superscale.py under Resolve's Python.

    `--quit` belongs only to a Resolve the node started itself; a Resolve that was
    already open is the user's, and stays open.
    """
    for value, allowed, label in ((target, TARGETS, "target"), (mode, MODES, "mode"),
                                  (codec, CODECS, "codec")):
        if value not in allowed:
            raise ResolveBridgeError("%s must be one of %s, got %r"
                                     % (label, ", ".join(allowed), value))
    for value, label in ((sharpness, "sharpness"), (noise_reduction, "noise_reduction")):
        if not 0.0 <= float(value) <= 1.0:
            raise ResolveBridgeError("%s must be between 0 and 1, got %r" % (label, value))
    argv = [python_exe, helper, film, "--target", target, "--mode", mode,
            "--sharpness", "%g" % float(sharpness), "--nr", "%g" % float(noise_reduction),
            "--codec", codec, "--out", out_dir, "--wait", str(int(wait)),
            "--stop-file", stop_file]
    if switch_project:
        argv.append("--switch-project")
    if quit_after:
        argv.append("--quit")
    return argv


def parse_progress(line):
    """The render's percentage from one of the helper's `  45%` lines, else None."""
    match = _PROGRESS_RE.match(line or "")
    if not match:
        return None
    value = int(match.group(1))
    return value if value <= 100 else None


def parse_result(stdout, stderr, returncode, modified, since):
    """The film the helper wrote, or ResolveBridgeError in the helper's own words.

    On success the helper's last line is the file it wrote. `modified(path)` is
    that file's modification time, None when it is missing, and `since` is when
    the helper started. A file older than that was not written by this render --
    Resolve wrote its film under another name, or wrote nothing -- and handing the
    old film on as the new one would be the worst way to fail.
    """
    if returncode == 0:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        path = lines[-1] if lines else ""
        stamp = modified(path) if path else None
        if stamp is None:
            raise ResolveBridgeError(
                "Resolve's render finished, but the helper names no file it wrote%s."
                % (" (its last line was %r)" % path if path else ""))
        if stamp < since:
            raise ResolveBridgeError(
                "Resolve reported the render complete, but %s is older than this run: "
                "Resolve wrote its film under another name, or not at all. Look in %s."
                % (path, os.path.dirname(path)))
        return path
    said = [line.strip()[len(ERROR_PREFIX):] for line in (stderr or "").splitlines()
            if line.strip().startswith(ERROR_PREFIX)]
    if said:
        raise ResolveBridgeError(explain(said[-1]))
    last = [line.strip() for line in (stderr or "").splitlines() if line.strip()] or \
        [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    raise ResolveBridgeError("Resolve's Python stopped with exit code %s%s"
                             % (returncode, ": " + last[-1] if last else ""))


def never_answered(stderr):
    """Whether the helper gave up waiting for Resolve's scripting to answer.

    On a cold start Resolve's own script server can fail to come up ("Failed to
    connect to script server" in Resolve's log), and then nothing answers for
    that whole session. A Resolve the node started is restarted once for it.
    """
    return any(line.strip().startswith(ERROR_PREFIX + NEVER_ANSWERED)
               for line in (stderr or "").splitlines())


def explain(message):
    """The helper's message with its command-line flags named as the node's widgets."""
    for words, in_node in _IN_NODE_WORDS:
        message = message.replace(words, in_node)
    return message


def resolve_listed(tasklist_csv):
    """Whether `tasklist /FO CSV /NH` lists Resolve.

    A running Resolve is a row that opens with its quoted image name. The line
    tasklist prints when nothing matches has no quotes in it, in whatever
    language Windows speaks.
    """
    wanted = '"%s"' % RESOLVE_IMAGE.lower()
    return any(line.strip().lower().startswith(wanted)
               for line in (tasklist_csv or "").splitlines())
