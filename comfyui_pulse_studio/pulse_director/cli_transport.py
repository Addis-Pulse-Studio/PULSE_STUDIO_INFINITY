"""Running a program: a director's own CLI, and DaVinci Resolve.

Pulse Infinity's directors run `ollama run`, `pi -p`, `claude -p`, `gemini -p`.
Pulse Resolve Super Scale starts `Resolve.exe -nogui` and runs the Super Scale
helper under Resolve's own Python.

This is the one module in the pack allowed to start a process
(tests/test_infinity_directors.py enforces it). It is how the pack reaches a
local model, or Resolve, while the pack itself stays socket-free: the CLI talks
to its model, Resolve's Python talks to Resolve, and the pack talks to them over
stdin and stdout.

THE RULES IT KEEPS
------------------
  * No shell, ever. The command is an argv list from the user's config, with a
    few `{placeholders}` filled in by whole-argument substitution. The prompt --
    model-visible text that can contain anything -- goes on stdin, never into an
    argument, so no quoting rule can be broken by it.
  * Executables are resolved with `shutil.which`, which finds the `.cmd` shim an
    npm-installed CLI is on Windows. A shim is a batch file and needs cmd.exe to
    run; it gets `cmd.exe /d /c` and arguments that are only flags and paths the
    pack built itself.
  * UTF-8 both ways, whatever the console code page says -- Windows defaults to
    cp1252, which cannot carry an Amharic line.
  * Cancel and timeout kill the whole process tree (`taskkill /T /F` on Windows,
    the process group elsewhere), so an interrupted render does not leave a model
    running in the background holding VRAM.
  * On Windows no console window flashes up per call (CREATE_NO_WINDOW).
  * A program meant to outlive the call (Resolve) gets none of our handles. A
    child holding the write end of a pipe keeps its reader waiting until the
    child exits, which for Resolve means hours. It also gets a process group of
    its own, so ComfyUI's Ctrl+C does not reach it.
  * A long job with progress to show (Resolve's render) is read a line at a
    time. A cancel first asks it to stop by itself, through the caller's
    `on_cancel`, and kills its tree only after a grace period: a killed render
    leaves its temporary project behind in Resolve.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping, Optional, Sequence

from .base_director import DirectorCancelled, DirectorError, DirectorTimeout

__all__ = ["CliResult", "CliRunner", "build_argv", "run_cli", "run_lines", "spawn_detached"]

_WINDOWS = os.name == "nt"
#: Terminal control sequences. Some CLIs (Ollama's) emit them into a pipe too.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class CliResult:
    __slots__ = ("argv", "stdout", "stderr", "returncode", "seconds")

    def __init__(self, argv: list[str], stdout: str, stderr: str, returncode: int,
                 seconds: float) -> None:
        self.argv = argv
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.seconds = seconds


def build_argv(template: Sequence[str], values: Mapping[str, object]) -> list[str]:
    """Fill an argv template. `{name}` is replaced inside an argument; an argument
    that is exactly `{name}` and whose value is a list expands to one argument per
    item (`{images}` -> every image path). Unknown names are an error, not a
    silently empty flag."""
    if not template or not all(isinstance(a, str) for a in template):
        raise DirectorError("a CLI command must be a list of strings, got %r" % (template,))
    argv = []
    for arg in template:
        key = arg[1:-1] if arg.startswith("{") and arg.endswith("}") else None
        if key is not None and isinstance(values.get(key), (list, tuple)):
            argv.extend(str(v) for v in values[key])
            continue
        try:
            argv.append(arg.format_map(_Strict(values)))
        except KeyError as exc:
            raise DirectorError("the CLI command uses {%s}, which is not one of %s"
                                % (exc.args[0], ", ".join(sorted(values))))
    return [a for a in argv if a != ""]


class _Strict(dict):
    def __init__(self, values: Mapping[str, object]) -> None:
        super().__init__({k: ("" if v is None else v) for k, v in values.items()
                          if not isinstance(v, (list, tuple))})

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _resolve(argv: list[str]) -> list[str]:
    found = shutil.which(argv[0])
    if not found:
        raise DirectorError(
            "the director CLI %r is not on PATH for the ComfyUI process. Install it, or "
            "set the full path as the first entry of the command in pulse_infinity.yaml"
            % argv[0])
    if _WINDOWS and found.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/d", "/c", found] + argv[1:]
    return [found] + argv[1:]


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ValueError):  # pragma: no cover - already gone
        pass
    try:
        proc.kill()
    except OSError:  # pragma: no cover
        pass


def run_cli(argv: list[str], stdin_text: str, timeout: float = 600.0,
            should_cancel: Optional[Callable[[], bool]] = None, cwd: Optional[str] = None,
            env: Optional[Mapping[str, str]] = None, poll: float = 0.25) -> CliResult:
    """Run one director call to completion, or kill it trying."""
    should_cancel = should_cancel or (lambda: False)
    command = _resolve(list(argv))
    kwargs = {}
    if _WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    merged = dict(os.environ)
    merged.update(env or {})
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd or None, env=merged,
                                text=True, encoding="utf-8", errors="replace", **kwargs)
    except OSError as exc:
        raise DirectorError("could not start %s: %s" % (command[0], exc))
    stdout = stderr = ""
    first = True
    try:
        while True:
            if should_cancel():
                raise DirectorCancelled("interrupted while %s was thinking" % argv[0])
            if timeout and time.monotonic() - started > timeout:
                raise DirectorTimeout("%s did not answer within %.0f s" % (argv[0], timeout))
            try:
                out, err = proc.communicate(input=stdin_text if first else None, timeout=poll)
                stdout, stderr = out or "", err or ""
                break
            except subprocess.TimeoutExpired:
                first = False
                continue
    except BaseException:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:  # pragma: no cover - a child that will not die
            pass
        raise
    return CliResult(command, _ANSI_RE.sub("", stdout), stderr, proc.returncode,
                     time.monotonic() - started)


def _existing(argv: Sequence[str]) -> list[str]:
    """`argv` resolved like run_cli's, but a full path that is not there is named
    as such rather than as a director CLI missing from PATH."""
    if os.path.isabs(argv[0]) and not os.path.isfile(argv[0]):
        raise DirectorError("%s does not exist" % argv[0])
    return _resolve(list(argv))


def spawn_detached(argv: Sequence[str], cwd: Optional[str] = None) -> subprocess.Popen:
    """Start a program meant to outlive the call -- Resolve, headless -- and return
    its handle without waiting for it."""
    command = _existing(argv)
    kwargs = {}
    if _WINDOWS:
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, cwd=cwd or None, close_fds=True,
                                **kwargs)
    except OSError as exc:
        raise DirectorError("could not start %s: %s" % (command[0], exc))


def run_lines(argv: Sequence[str], on_line: Callable[[str], None], timeout: float = 0.0,
              should_cancel: Optional[Callable[[], bool]] = None,
              on_cancel: Optional[Callable[[], None]] = None, grace: float = 0.0,
              cwd: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
              poll: float = 0.25) -> CliResult:
    """Run a long job to completion, handing `on_line` each line it prints as it
    comes.

    The first time `should_cancel` answers True, `on_cancel` is called and the job
    gets `grace` seconds to finish by itself before its tree is killed; either
    way DirectorCancelled is raised. A timeout kills at once. `on_line` runs on a
    reader thread, and an exception from it never stops the job.
    """
    should_cancel = should_cancel or (lambda: False)
    command = _existing(argv)
    name = os.path.basename(argv[0])
    kwargs = {}
    if _WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    merged = dict(os.environ)
    merged.update(env or {})
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd or None, env=merged,
                                text=True, encoding="utf-8", errors="replace", **kwargs)
    except OSError as exc:
        raise DirectorError("could not start %s: %s" % (command[0], exc))
    lines: list[str] = []
    errors: list[str] = []

    def read_stdout():
        for raw in proc.stdout:
            line = _ANSI_RE.sub("", raw.rstrip("\r\n"))
            lines.append(line)
            try:
                on_line(line)
            except Exception:  # a progress display must never stop the job
                pass

    def read_stderr():
        errors.append(proc.stderr.read())

    readers = [(threading.Thread(target=read_stdout, daemon=True), proc.stdout),
               (threading.Thread(target=read_stderr, daemon=True), proc.stderr)]
    for reader, _ in readers:
        reader.start()

    def finish():
        for reader, stream in readers:
            reader.join(timeout=5)
            # A pipe a grandchild still holds open keeps its reader blocked;
            # closing it under that reader could hang here instead.
            if not reader.is_alive():
                stream.close()

    cancelled_at = None
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if cancelled_at is None and should_cancel():
                cancelled_at = now
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:  # the grace period still applies, then the kill
                        pass
            if cancelled_at is not None and now - cancelled_at >= grace:
                raise DirectorCancelled("interrupted while %s was running" % name)
            if timeout and now - started > timeout:
                raise DirectorTimeout("%s did not finish within %.0f s" % (name, timeout))
            time.sleep(poll)
    except BaseException:
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - a child that will not die
            pass
        finish()
        raise
    finish()
    if cancelled_at is not None:
        raise DirectorCancelled("%s stopped after it was interrupted" % name)
    return CliResult(command, "\n".join(lines), "".join(errors), proc.returncode,
                     time.monotonic() - started)


class CliRunner:
    """The call a CLI transport makes, as an object a test can replace."""

    def __call__(self, argv: list[str], stdin_text: str, timeout: float,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 cwd: Optional[str] = None) -> CliResult:
        return run_cli(argv, stdin_text, timeout=timeout, should_cancel=should_cancel, cwd=cwd)

