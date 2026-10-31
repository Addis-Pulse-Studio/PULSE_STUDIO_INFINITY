"""Pulse Resolve Super Scale: DaVinci Resolve Studio's Super Scale as a ComfyUI node.

Wire Pulse Render's `video` (or any VIDEO) in, and the film comes back at 2K or 4K,
upscaled by Resolve Studio's Super Scale in the same queue run as the render. 2x
Enhanced is the default, measured the best of the delivery routes on a Pulse film
(README, "Delivering at 2K / 4K").

ComfyUI never loads Resolve's scripting module; resolve_bridge.py says why. The
node runs the pack's helper, tools/resolve_superscale.py, under Resolve's own
Python as a child process. It starts Resolve headless when it is not running, and
quits it again afterwards. Before Resolve renders, the node hands the GPU back:
ComfyUI-Memory-Reclaim's All Models + Hard purge when that pack is installed,
ComfyUI's own unload otherwise. Every decision lives in
comfyui_pulse_studio/resolve_bridge.py and is tested headless. The processes start
in pulse_director/cli_transport.py, the one module allowed to start them.

Same slot contract as nodes.py: `schema_version` at widget 0, widgets append-only,
`"hidden": HIDDEN_INPUTS`. It is an output node, because the file it writes is
the point even when nothing is wired after it.
"""

import logging
import os
import time
import uuid

import folder_paths
from comfy_execution.graph import ExecutionBlocker

from . import media
from .comfyui_pulse_studio import resolve_bridge as bridge
from .comfyui_pulse_studio.pulse_director import DirectorCancelled, DirectorError, DirectorTimeout
from .comfyui_pulse_studio.pulse_director.cli_transport import run_cli, run_lines, spawn_detached
from .nodes import HIDDEN_INPUTS, _warn_on_node, schema_widget

log = logging.getLogger(__name__)

CATEGORY = "AddisPulse/H3"
PACK_DIR = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(PACK_DIR, "tools", "resolve_superscale.py")
#: Seconds the helper waits for Resolve to answer. A warm headless start answered
#: in 14 s; a cold one took 46 s just to start its script server.
CONNECT_WAIT = 120
#: After ComfyUI's Cancel: time for the helper to stop the render, delete its
#: project, quit Resolve and remove the unfinished film by itself, before its
#: process tree is killed.
CANCEL_GRACE = 40.0
#: Time Resolve gets to leave after it was asked to quit.
QUIT_WAIT = 60.0
#: Time a stopped render's file gets to be let go of, once the run is over.
UNFINISHED_WAIT = 10.0
#: The longest one render may take before the node gives up on it.
RENDER_TIMEOUT = 3 * 3600.0
#: ComfyUI-Memory-Reclaim's inline node, by the id saved workflows know it by,
#: and the values of its widgets for "everything, handed back to the driver".
MEMORY_RECLAIM_NODE = "MemoryReclaimUnload"
MEMORY_RECLAIM_ARGS = {"mode": "Always", "unload_target": "All Models",
                       "cache_purge": "Hard", "gc_collect": True}


# ── ComfyUI plumbing ────────────────────────────────────────────────────────

def _interrupted():
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:  # pragma: no cover - outside ComfyUI
        return False


def _is_interrupt(exc):
    try:
        import comfy.model_management as mm
        return isinstance(exc, mm.InterruptProcessingException)
    except Exception:  # pragma: no cover
        return False


def _raise_interrupted():
    """Tell ComfyUI the run was cancelled, rather than failed, the way its own nodes
    do: clear its interrupt flag as the exception goes up. Left set, the flag would
    stop the next prompt at its first progress update."""
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()  # clears the flag, and raises
    raise mm.InterruptProcessingException()  # the progress hook cleared it already


def _progress_bar(unique_id):
    try:
        import comfy.utils
    except Exception:  # pragma: no cover
        return None
    try:
        return comfy.utils.ProgressBar(100, node_id=unique_id)
    except TypeError:  # an older core, without node_id
        return comfy.utils.ProgressBar(100)


def _memory_reclaim():
    """ComfyUI-Memory-Reclaim's inline node class, when that pack is loaded."""
    try:
        import nodes as comfy_nodes  # ComfyUI's registry, not this pack's nodes.py
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(MEMORY_RECLAIM_NODE)
    except Exception:
        return None
    return cls if cls is not None and callable(getattr(cls, "run", None)) else None


def _free_gpu(video):
    """Hand ComfyUI's VRAM back before Resolve renders on the same GPU.

    Resolve is another process, so only memory returned to the driver helps it.
    ComfyUI-Memory-Reclaim's Hard purge returns what ComfyUI's own unload leaves
    in PyTorch's cache, and measures it. Returns a line for the report.
    """
    reclaim = _memory_reclaim()
    if reclaim is not None:
        try:
            summary = reclaim().run(video, **MEMORY_RECLAIM_ARGS)["result"][-1]
            return ("freed the GPU with ComfyUI-Memory-Reclaim (All Models, Hard):\n      "
                    + str(summary).replace("\n", "\n      "))
        except Exception as exc:
            log.warning("[PulseStudio] ComfyUI-Memory-Reclaim failed (%s); using ComfyUI's "
                        "own unload instead", exc)
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:  # pragma: no cover
        return "could not unload ComfyUI's models: %s" % exc
    return ("unloaded ComfyUI's models with ComfyUI's own unload (ComfyUI-Memory-Reclaim, "
            "when installed, also hands PyTorch's cache back to the driver)")


# ── files and processes ─────────────────────────────────────────────────────

def _source_file(video):
    """(path, note): a file Resolve can import for this VIDEO.

    A VIDEO loaded from a file, neither trimmed nor cropped, is that file.
    Anything else -- frames, a trimmed or cropped clip, an in-memory file -- is
    written to ComfyUI's temp folder first, with the VIDEO's own save_to.
    """
    try:
        source = video.get_stream_source()
    except Exception:
        source = None
    if isinstance(source, str) and os.path.isfile(source) and _whole(video, source):
        return source, None
    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    path = os.path.join(temp, "pulse_resolve_%s.mp4" % uuid.uuid4().hex[:12])
    video.save_to(path)
    return path, "the video was not a whole file on disk, so it was written to %s first" % path


def _whole(video, path):
    """Whether the VIDEO is all of `path`: no trim, and no crop."""
    trim = getattr(video, "get_active_trim_window", None)
    if trim is not None:
        try:
            start, duration = trim()
        except Exception:
            return False
        if start or duration:
            return False
    # A crop has no public reader, but a cropped clip reports its cropped size.
    try:
        return (tuple(video.get_dimensions())
                == tuple(media.video_from_file(path).get_dimensions()))
    except Exception:
        return False


def _modified(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _resolve_running():
    try:
        result = run_cli(list(bridge.TASKLIST_ARGV), "", timeout=30)
    except DirectorError as exc:
        raise RuntimeError("could not ask Windows whether DaVinci Resolve is running: %s"
                           % exc) from None
    return bridge.resolve_listed(result.stdout)


def _start_resolve(resolve_exe):
    try:
        process = spawn_detached([resolve_exe, "-nogui"], cwd=os.path.dirname(resolve_exe))
    except DirectorError as exc:
        raise RuntimeError("could not start DaVinci Resolve: %s" % exc) from None
    log.info("[PulseStudio] started DaVinci Resolve headless (pid %s)", process.pid)
    return process


def _end_resolve(process, wait=30.0):
    """End a headless Resolve the node started whose scripting never answered: it
    has no project of anyone's open, and nothing will answer in it again."""
    if process.poll() is None:
        process.kill()
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if not _resolve_running():
                return
        except RuntimeError:
            return
        time.sleep(1)


def _see_resolve_out(process, wait=QUIT_WAIT):
    """Wait for a Resolve the node started, and asked to quit, to be gone.

    Its GPU memory belongs to the next render. If it is still there after `wait`
    seconds -- the helper stopped before it could ask -- the headless Resolve the
    node started is ended. Returns a warning for the report, or None.
    """
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if not _resolve_running():
                return None
        except RuntimeError:
            break
        time.sleep(2)
    if process is not None and process.poll() is None:
        process.kill()
        return ("Resolve did not quit within %d s, so the node ended the headless Resolve "
                "it had started." % wait)
    return ("Resolve is still running %d s after it was asked to quit. It holds GPU "
            "memory until you close it." % wait)


# ── a session with Resolve, and the Super Scale run on it ───────────────────

def _resolve_session(make_argv, when_done, notes, unique_id=None, on_line=None,
                     cleanup=None):
    """Run one helper under Resolve's own Python, against a Resolve that is running
    or started here, and see it out again. Returns (result, began).

    `make_argv(quit_after, stop_file)` builds the helper's command. `on_line(line)`
    hears every line that is not a progress percentage. `cleanup(result)` runs in
    the `finally` when the helper did not succeed -- after a Resolve the node
    started has quit and let go of its files. ComfyUI's Cancel reaches the helper
    through the stop file, and a cold start whose scripting never answers is
    retried once.
    """
    start_resolve = not _resolve_running()
    quit_after = start_resolve and when_done == bridge.WHEN_DONE_QUIT
    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    stop_file = os.path.join(temp, "pulse_resolve_stop_%s" % uuid.uuid4().hex[:12])
    try:
        argv = make_argv(quit_after, stop_file)
    except bridge.ResolveBridgeError as exc:
        raise RuntimeError(str(exc)) from None
    resolve_exe = bridge.find_install(os.environ, os.path.isfile)[0]

    bar = _progress_bar(unique_id)
    state = {"latest": None, "shown": None, "interrupted": False}

    def heard(line):
        # The reader thread. ComfyUI's progress hook is never called from here:
        # on a cancel it raises and clears the interrupt flag, and the cancel
        # would be lost.
        percent = bridge.parse_progress(line)
        if percent is not None:
            state["latest"] = percent
        elif line.strip():
            log.info("[PulseStudio] Resolve: %s", line.strip())
            if on_line is not None:
                on_line(line.strip())

    def should_cancel():
        # The executor's thread, polled while the helper runs.
        if state["interrupted"] or _interrupted():
            return True
        latest = state["latest"]
        if bar is not None and latest is not None and latest != state["shown"]:
            state["shown"] = latest
            try:
                bar.update_absolute(latest, 100)
            except BaseException as exc:  # ComfyUI's interrupt is a BaseException
                if _is_interrupt(exc):
                    state["interrupted"] = True
                    return True
                if not isinstance(exc, Exception):
                    raise
                log.debug("[PulseStudio] progress update failed: %s", exc)
        return False

    def on_cancel():
        with open(stop_file, "w", encoding="utf-8") as handle:
            handle.write("cancelled from ComfyUI\n")

    # Resolve starts only now, with everything else checked, and from here on
    # the `finally` sees a Resolve it started out again. A cold start can leave
    # Resolve's scripting dead for the session, so a Resolve the node started
    # gets one restart for that.
    process = leftover = None
    began = time.time()
    cancelled = False
    result = None
    try:
        for attempt in (1, 2):
            if start_resolve:
                process = _start_resolve(resolve_exe)
                notes.append("started Resolve headless" if attempt == 1 else
                             "restarted Resolve: its scripting never answered the "
                             "first time")
            result = run_lines(argv, heard, timeout=RENDER_TIMEOUT,
                               should_cancel=should_cancel, on_cancel=on_cancel,
                               grace=CANCEL_GRACE, cwd=PACK_DIR)
            if not (start_resolve and attempt == 1 and result.returncode != 0
                    and bridge.never_answered(result.stderr)):
                break
            log.warning("[PulseStudio] Resolve's scripting never answered; restarting it")
            _end_resolve(process)
    except DirectorCancelled:
        cancelled = True
    except DirectorTimeout:
        raise RuntimeError("Resolve's helper did not finish within %d hours, so it was "
                           "stopped." % (RENDER_TIMEOUT // 3600)) from None
    except DirectorError as exc:
        raise RuntimeError("Resolve's Python could not be run: %s" % exc) from None
    finally:
        _remove(stop_file)
        if quit_after and process is not None:
            leftover = _see_resolve_out(process)
            if leftover:
                notes.append(leftover)
                _warn_on_node(unique_id, [leftover])
        if cleanup is not None and (result is None or result.returncode != 0):
            cleanup(result)
    if cancelled:
        _raise_interrupted()
    if quit_after and not leftover:
        notes.append("quit Resolve afterwards")
    return result, began


def _run_resolve(video, targets, mode, sharpness, noise_reduction, codec, open_project,
                 when_done, free_vram, unique_id=None, title="Pulse Resolve Super Scale"):
    """Super Scale `video` to every one of `targets` in one Resolve session.

    Returns ({target: file written}, report).
    """
    started = time.time()
    if os.name != "nt":
        raise RuntimeError("%s needs Windows: it drives DaVinci Resolve Studio through "
                           "Resolve's own Windows Python." % title)
    try:
        python_exe = bridge.find_install(os.environ, os.path.isfile)[1]
    except bridge.ResolveBridgeError as exc:
        raise RuntimeError(str(exc)) from None
    if not os.path.isfile(HELPER):
        raise RuntimeError("The Super Scale helper is missing from the pack: %s" % HELPER)

    notes = []
    film, note = _source_file(video)
    if note:
        notes.append(note)
    film, swapped = bridge.native_film(film, os.path.isfile)
    if swapped:
        notes.append("sent the native %s, not the copy Pulse Render had already "
                     "upscaled" % bridge.NATIVE_FILM)
    width, height = media.video_from_file(film).get_dimensions()
    for target in targets:
        canvas_w, canvas_h = bridge.delivery_canvas(width, height, target)
        if bridge.already_delivered(width, height, canvas_w, canvas_h):
            raise RuntimeError(
                "%s is already %dx%d, which fills the %s canvas (%dx%d), so Super Scale "
                "has nothing to add. Choose a larger size, or use the film as it is."
                % (os.path.basename(film), width, height, target, canvas_w, canvas_h))
    out_dir = bridge.output_dir(film, folder_paths.get_output_directory())
    os.makedirs(out_dir, exist_ok=True)
    if free_vram:
        notes.append(_free_gpu(video))

    def make_argv(quit_after, stop_file):
        return bridge.helper_argv(
            python_exe, HELPER, film, out_dir, stop_file, targets=targets, mode=mode,
            sharpness=sharpness, noise_reduction=noise_reduction, codec=codec,
            switch_project=open_project == bridge.OPEN_PROJECT_SWITCH,
            quit_after=quit_after, wait=CONNECT_WAIT)

    def cleanup(result):
        # A stopped or failed render can leave truncated films that Resolve held
        # open until it quit. Removed here, once it has let go.
        for target in targets:
            unfinished = os.path.join(out_dir, bridge.output_name(film, target))
            if not bridge.discard_unfinished(unfinished, began, wait=UNFINISHED_WAIT):
                held = ("an unfinished %s is left in %s: Resolve still holds it"
                        % (os.path.basename(unfinished), out_dir))
                notes.append(held)
                _warn_on_node(unique_id, [held])

    said = []
    began = time.time()  # the cleanup's floor; the session's own is passed on below
    result, began = _resolve_session(make_argv, when_done, notes, unique_id,
                                     on_line=said.append, cleanup=cleanup)
    try:
        paths = bridge.parse_result(result.stdout, result.stderr, result.returncode,
                                    _modified, since=began)
    except bridge.ResolveBridgeError as exc:
        raise RuntimeError(str(exc)) from None

    written = set(paths.values())
    lines = ["%s -> %s" % (title, ", ".join("%s %s" % (t, paths[t]) for t in targets))]
    lines += ["  %s" % line for line in said
              if line not in written and not line.startswith(bridge.RESULT_PREFIX)]
    lines += ["  %s" % note for note in notes]
    lines.append("  %.0f s in all" % (time.time() - started))
    report = "\n".join(lines)
    log.info("[PulseStudio] %s", report)
    return paths, report


# ── the widgets both nodes share ────────────────────────────────────────────

_MODE = (list(bridge.MODES), {"default": "auto", "tooltip":
    "Super Scale's multiplier. auto: 2x Enhanced, with Resolve's own resize covering "
    "what is left (up to 1.5x), measured the best on a Pulse film; a smaller film gets "
    "3x or 4x. Only 2x Enhanced uses sharpness and noise_reduction."})
_SHARPNESS = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
    "2x Enhanced only. 0.5 measured best; 1.0 turns eyes and skin crunchy. Raise it in "
    "small steps, if at all."})
_NOISE_REDUCTION = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "2x Enhanced only. 0.5 measured best."})
_CODEC = (list(bridge.CODECS), {"default": "h265", "tooltip":
    "Resolve's own encoder. Through its API Resolve takes no quality setting, so the "
    "encoder sets the bitrate: at 4K, H.265 wrote 34 Mbit/s and H.264 58 (a larger "
    "file)."})
_OPEN_PROJECT = (list(bridge.OPEN_PROJECT_MODES), {"default": bridge.OPEN_PROJECT_STOP,
                                                   "tooltip":
    "Resolve holds one project at a time, and the render happens in a temporary project "
    "of its own. stop: if one of your projects is open, stop and say so. "
    "switch_and_restore: switch away from it for the render and load it again "
    "afterwards."})
_WHEN_DONE = (list(bridge.WHEN_DONE_MODES), {"default": bridge.WHEN_DONE_QUIT, "tooltip":
    "quit_if_started_here: a Resolve the node started headless is quit afterwards, and "
    "its GPU memory comes back; a Resolve you had open is left as it is. leave_running: "
    "keep the headless Resolve up. The next run saves about 30 s, but Resolve holds GPU "
    "memory the next H3 render wants."})
_FREE_VRAM = ("BOOLEAN", {"default": True, "tooltip":
    "Unload ComfyUI's models before Resolve renders, since they share the GPU. With "
    "ComfyUI-Memory-Reclaim installed this is its All Models + Hard purge, which also "
    "hands PyTorch's cache back to the driver; otherwise ComfyUI's own unload. The "
    "models reload on the next run. Off when a Memory Reclaim gate already did it."})


# ── Pulse Resolve Super Scale ───────────────────────────────────────────────

class PulseResolveSuperScale:
    """A finished film at 1080p, 2K or 4K, through DaVinci Resolve Studio's Super Scale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip":
                    "The film to upscale. Wire Pulse Render's `video`: when Pulse's own "
                    "upscale has already enlarged it, the native assembled.mp4 beside it "
                    "is sent instead. Any other VIDEO works too."}),
                "schema_version": schema_widget(),
                "target": (list(bridge.TARGETS), {"default": "2160p", "tooltip":
                    "2160p: 3840x2160. 1440p: 2560x1440. 1080p: 1920x1080. Turned for "
                    "portrait, square for square, and cropped to fill: the canvas Pulse "
                    "Render's _exact and _rtx modes deliver."}),
                "mode": _MODE,
                "sharpness": _SHARPNESS,
                "noise_reduction": _NOISE_REDUCTION,
                "codec": _CODEC,
                "open_project": _OPEN_PROJECT,
                "when_done": _WHEN_DONE,
                "free_vram": _FREE_VRAM,
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "path", "report")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Upscales a finished film to 1080p, 2K or 4K with DaVinci Resolve "
                   "Studio's Super Scale. Resolve starts headless if it is not running and "
                   "quits afterwards; the film is written beside the one it came from.")

    def execute(self, video, schema_version, target, mode, sharpness, noise_reduction,
                codec, open_project, when_done, free_vram, unique_id=None):
        paths, report = _run_resolve(video, [target], mode, sharpness, noise_reduction,
                                     codec, open_project, when_done, free_vram, unique_id)
        path = paths[target]
        return (media.video_from_file(path), path, report)


# ── Pulse Resolve Deliver ───────────────────────────────────────────────────

class PulseResolveDeliver:
    """1080p, 2K and 4K switched on and off, from one Resolve session."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip":
                    "The film to deliver. Wire Pulse Render's `video` (through a Memory "
                    "Reclaim gate, ideally): the native assembled.mp4 is what Resolve "
                    "gets. Any other VIDEO works too."}),
                "schema_version": schema_widget(),
                "deliver_1080p": ("BOOLEAN", {"default": False, "tooltip":
                    "1920x1080 (1080x1920 vertical): Super Scale, then Resolve's own "
                    "resize down to the canvas."}),
                "deliver_2k": ("BOOLEAN", {"default": False, "tooltip":
                    "2560x1440 (1440x2560 vertical)."}),
                "deliver_4k": ("BOOLEAN", {"default": True, "tooltip":
                    "3840x2160 (2160x3840 vertical). Every size switched on comes from "
                    "one Resolve session: one start, one render job each."}),
                "mode": _MODE,
                "sharpness": _SHARPNESS,
                "noise_reduction": _NOISE_REDUCTION,
                "codec": _CODEC,
                "open_project": _OPEN_PROJECT,
                "when_done": _WHEN_DONE,
                "free_vram": _FREE_VRAM,
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("VIDEO", "VIDEO", "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video_1080p", "video_2k", "video_4k", "paths", "report")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Delivers a finished film at 1080p, 2K and/or 4K -- each switched on or "
                   "off -- through DaVinci Resolve Studio's Super Scale, from one Resolve "
                   "session. A size switched off sends nothing, so its Save node is skipped.")

    def execute(self, video, schema_version, deliver_1080p, deliver_2k, deliver_4k, mode,
                sharpness, noise_reduction, codec, open_project, when_done, free_vram,
                unique_id=None):
        switched = {"1080p": deliver_1080p, "2K": deliver_2k, "4K": deliver_4k}
        targets = [target for label, target in bridge.TIERS if switched[label]]
        if not targets:
            raise RuntimeError("Pulse Resolve Deliver has every size switched off. Switch on "
                               "deliver_1080p, deliver_2k or deliver_4k.")
        paths, report = _run_resolve(video, targets, mode, sharpness, noise_reduction,
                                     codec, open_project, when_done, free_vram, unique_id,
                                     title="Pulse Resolve Deliver")
        # A size switched off returns a silent blocker: its Save node simply does not
        # run, which is what ComfyUI's ExecutionBlocker(None) is for.
        videos = tuple(media.video_from_file(paths[target]) if target in paths
                       else ExecutionBlocker(None) for _, target in bridge.TIERS)
        listing = "\n".join("%s: %s" % (label, paths[target]) for label, target in bridge.TIERS
                            if target in paths)
        return videos + (listing, report)


NODE_CLASS_MAPPINGS = {"PulseResolveSuperScale": PulseResolveSuperScale,
                       "PulseResolveDeliver": PulseResolveDeliver}
NODE_DISPLAY_NAME_MAPPINGS = {"PulseResolveSuperScale": "Pulse Resolve Super Scale",
                              "PulseResolveDeliver": "Pulse Resolve Deliver"}
