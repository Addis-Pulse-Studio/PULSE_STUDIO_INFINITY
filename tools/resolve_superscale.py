#!/usr/bin/env python3
"""Upscale a finished Pulse film to 2K or 4K with DaVinci Resolve Studio's Super Scale.

    tools\\resolve_superscale.bat <film.mp4 | run folder> [--target 1440p]   (or drag and drop)
    ResolvePython.exe tools\\resolve_superscale.py <film.mp4 | run folder> [options]

Pulse Render's `upscale` delivers inside the graph: `_exact` with Lanczos, `_rtx`
with NVIDIA's RTX Video Super Resolution. This is the other route: Resolve
Studio's Super Scale, after the render, for the highest-quality pass and the A/B
against RTX. Given a run folder it takes the native `assembled.mp4`, never an
already-upscaled copy.

In a throwaway project, which it deletes afterwards:

  1. a timeline at the delivery canvas -- 3840x2160 or 2560x1440, turned for
     portrait, square for square: the same canvas Pulse's own `_exact` and `_rtx`
     modes deliver -- with input sizing `scaleToCrop` (cover, then centre-crop),
     at the film's frame rate;
  2. the film, with Super Scale set on its clip: by default 2x Enhanced, the
     mode with sharpness and noise reduction, with Resolve's own resize covering
     the rest of the way to the canvas -- measured the best of Resolve's modes on
     a Pulse film (see ENHANCED_REACH);
  3. one render: MP4, H.265 with Resolve's own encoder, audio kept, written
     beside the film as `<name>_<target>_resolve.mp4`.

It talks only to the Resolve on this machine, through Resolve's own scripting
module, and starts nothing: `resolve_superscale.bat` launches Resolve headless
(`-nogui`) when it is not running, and so does the Pulse Resolve Super Scale node,
and this waits for it. One-time setup: Resolve *Studio* (Super Scale is a Studio
feature), and Preferences > System > General > External scripting using: Local.

ComfyUI never imports this file. It runs under Resolve's own Python
(ResolvePython.exe), from the .bat, or as a child process of the Pulse Resolve
Super Scale node -- which reads its `NN%` lines for its progress bar and asks it
to stop, on ComfyUI's Cancel, with `--stop-file`. The decisions -- canvas,
multiplier, codec, file names -- are pure functions, tested in
tests/test_delivery.py without Resolve.
"""

import argparse
import os
import sys
import time

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

# One list each, shared with the node's combos.
from comfyui_pulse_studio.resolve_bridge import (  # noqa: E402
    CODECS,
    MODES,
    NATIVE_FILM,
    TARGETS,
    delivery_canvas,
    discard_unfinished,
    output_name,
)

#: The project Resolve opens by itself -- nobody's work.
SCRATCH_PROJECT = "Untitled Project"
#: Seconds a cancelled render is given to stop before its project is deleted.
STOP_WAIT = 15
#: Resolve can hold a stopped render's file open until it quits: seconds to keep
#: trying to remove it while Resolve runs, and once it has been asked to quit.
UNFINISHED_WAIT = 2
QUIT_RELEASE_WAIT = 15


class SuperScaleError(RuntimeError):
    """Something Resolve refused or could not do; the message says what."""


# ── decisions (pure) ────────────────────────────────────────────────────────

def canvas(width, height, target):
    """The delivery canvas for a film: Pulse's own `<target>_exact` canvas."""
    try:
        return delivery_canvas(width, height, target)
    except ValueError as exc:
        raise SuperScaleError(str(exc)) from None


def stop_requested(stop_file):
    """Whether whoever started this has asked it to stop: the node, on ComfyUI's
    Cancel, by creating the stop file."""
    return bool(stop_file) and os.path.exists(stop_file)


def unfinished(error, path, since):
    """`error`, once the film its render left unfinished is removed. While Resolve
    still holds that file open, the error carries it, for main() to remove once
    Resolve has quit."""
    if not discard_unfinished(path, since, wait=UNFINISHED_WAIT):
        error.unfinished = (path, since)
    return error


#: How much of the way to the canvas Resolve's own resize may cover after 2x
#: Enhanced, for `auto` to choose it. Measured on a Pulse film, 1344x768 to 3840x2160
#: (1.43x of resize after 2x), Resolve 21.1.1, the same frame each way:
#:
#:     mode                  fine detail on a face     scaled back down to the source
#:     2x Enhanced 0.5/0.5          15.8                      45.2 dB
#:     4x                           10.2                      44.7 dB
#:     3x                            9.9                      42.8 dB
#:     2x Enhanced 1.0/0.2          92.6 (oversharpened)      35.1 dB
#:     RTX VSR (2160p_rtx)          15.3                      38.6 dB
#:     Lanczos (2160p_exact)        10.0                      43.6 dB
#:
#: (Luma; detail is the Laplacian's variance.) 2x Enhanced at its default strengths
#: kept as much detail as RTX VSR and stayed the closest to the source. A bigger
#: remaining resize than this is unmeasured, and the plain multipliers take over.
ENHANCED_REACH = 1.5


def superscale_for(width, height, canvas_w, canvas_h, mode="auto"):
    """(multiplier, enhanced, note) for Super Scale on this film.

    `auto` is 2x Enhanced when Resolve's own resize has at most ENHANCED_REACH
    left to cover after it -- every Pulse canvas to 1440p and to 4K. Past that
    it is the smallest of 3x and 4x that covers the canvas, and `note` says when
    even 4x falls short and Resolve's own resize makes up the rest.
    """
    if mode not in MODES:
        raise SuperScaleError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
    if mode != "auto":
        return int(mode[0]), mode == "2x_enhanced", ""
    if max(canvas_w / (2.0 * width), canvas_h / (2.0 * height)) <= ENHANCED_REACH + 1e-9:
        return 2, True, ""
    for multiplier in (3, 4):
        if width * multiplier >= canvas_w and height * multiplier >= canvas_h:
            return multiplier, False, ""
    return 4, False, ("4x Super Scale makes %dx%d of %dx%d, short of %dx%d; Resolve "
                      "scales the rest" % (width * 4, height * 4, width, height,
                                           canvas_w, canvas_h))


def describe(multiplier, enhanced):
    return "2x Enhanced" if enhanced else "%dx" % multiplier


def parse_resolution(text):
    """'1344x768' (Resolve's "Resolution" clip property) -> (1344, 768)."""
    try:
        width, height = (int(v) for v in str(text).lower().replace(" ", "").split("x"))
    except ValueError:
        raise SuperScaleError("Resolve reports the film's resolution as %r" % (text,)) from None
    return width, height


def timeline_rate(text):
    """Resolve's "FPS" clip property ('24.000') as a timelineFrameRate value ('24')."""
    try:
        rate = round(float(text), 3)
    except (TypeError, ValueError):
        return None
    return ("%.3f" % rate).rstrip("0").rstrip(".") if rate > 0 else None


def pick_codec(codecs, want="h265"):
    """The render codec to ask for, from GetRenderCodecs('mp4') -- {description: name}.

    Resolve's own build of the codec, else a vendor's (NVIDIA, Intel), else -- for
    H.265 -- H.264, which every Resolve can write. Own first because neither takes a
    quality setting through the API (Resolve 21.1.1 refuses every VideoQuality value
    for every MP4 codec), so the encoder alone sets the bitrate: at 4K its own H.265
    wrote 34 Mbit/s and its NVIDIA one 22, in about the same time.
    """
    if want not in CODECS:
        raise SuperScaleError("codec must be one of %s, got %r" % (", ".join(CODECS), want))

    def family(codec):
        keys = ("H265", "HEVC") if codec == "h265" else ("H264", "AVC")
        return [(d, n) for d, n in (codecs or {}).items()
                if any(k in (d + " " + n).upper().replace(".", "") for k in keys)]

    vendors = ("NVIDIA", "NVENC", "INTEL", "AMD", "APPLE")
    for codec in ((want, "h264") if want == "h265" else (want,)):
        found = family(codec)
        own = [n for d, n in found if not any(v in (d + n).upper() for v in vendors)]
        if own or found:
            return (own or [n for _, n in found])[0]
    raise SuperScaleError("Resolve lists no H.264 or H.265 encoder for MP4: %r" % (codecs,))


def choose_source(path, names=None):
    """The film to upscale: `path` itself, or a run folder's native assembled.mp4."""
    if names is None:
        if not os.path.isdir(path):
            return path
        names = os.listdir(path)
    if NATIVE_FILM in names:
        return os.path.join(path, NATIVE_FILM)
    raise SuperScaleError("%s holds no %s -- point this at a Pulse run folder or at a "
                          "film" % (path, NATIVE_FILM))


def windows_path(path):
    """/mnt/d/x -> D:\\x, so a path typed in WSL reaches Resolve as Windows sees it."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 2 and parts[0] == "" and parts[1] == "mnt" and len(parts[2]) == 1:
        return "%s:\\%s" % (parts[2].upper(), "\\".join(parts[3:]))
    return path


# ── Resolve ─────────────────────────────────────────────────────────────────

def connect(timeout, stop_file=None):
    """The running Resolve, waited for: a headless one takes a while to answer."""
    try:
        import DaVinciResolveScript as dvr
    except ImportError as exc:
        raise SuperScaleError(
            "DaVinciResolveScript would not load (%s). Run this with Resolve's own "
            "Python -- ResolvePython.exe, which resolve_superscale.bat uses." % (exc,)) from None
    deadline = time.monotonic() + timeout
    while True:
        resolve = dvr.scriptapp("Resolve")
        if resolve is not None and resolve.GetProjectManager() is not None:
            return resolve
        if stop_requested(stop_file):
            raise SuperScaleError("stopped before Resolve answered: the stop file appeared")
        if time.monotonic() >= deadline:
            raise SuperScaleError(
                "DaVinci Resolve did not answer within %d s. Is it running -- the .bat "
                "starts it -- and is Preferences > System > General > External "
                "scripting using set to Local?" % timeout)
        time.sleep(3)


def superscale(resolve, src, target="2160p", mode="auto", sharpness=0.5, noise_reduction=0.5,
               codec="h265", out_dir=None, keep_project=False, switch_project=False,
               stop_file=None, say=print):
    """Render `src` through Super Scale at `target`. Returns the file written.

    Resolve holds one project at a time, so making the temporary one closes
    whatever is open. A project of yours is not closed unless `switch_project`
    says so, and then it is loaded again afterwards.

    When `stop_file` appears the render is stopped and its partial file removed,
    and the temporary project is deleted as it always is.
    """
    if stop_requested(stop_file):
        raise SuperScaleError("stopped before the render began: the stop file appeared")
    manager = resolve.GetProjectManager()
    current = manager.GetCurrentProject()
    previous = (current.GetName() if current is not None else "") or ""
    yours = previous not in ("", SCRATCH_PROJECT)
    if yours and not switch_project:
        raise SuperScaleError(
            "Resolve has your project %r open, and this renders in a temporary project of "
            "its own. Save and close it first, or pass --switch-project to switch away from "
            "it and load it again afterwards." % previous)
    name = "PulseSuperScale_%s" % time.strftime("%Y%m%d_%H%M%S")
    project = manager.CreateProject(name)
    if project is None:
        raise SuperScaleError("Resolve would not create the project %r" % name)
    job = None
    try:
        pool = project.GetMediaPool()
        # A plain list of paths. Resolve 21.1.1 answers None to the [{"FilePath": ...}]
        # form its README recommends in place of this one, for a movie file.
        items = pool.ImportMedia([src]) or pool.ImportMedia([{"FilePath": src}])
        if not items:
            raise SuperScaleError("Resolve could not import %s" % src)
        clip = items[0]
        width, height = parse_resolution(clip.GetClipProperty("Resolution"))
        canvas_w, canvas_h = canvas(width, height, target)
        multiplier, enhanced, note = superscale_for(width, height, canvas_w, canvas_h, mode)
        say("%s: %dx%d -> Super Scale %s -> %dx%d"
            % (os.path.basename(src), width, height, describe(multiplier, enhanced),
               canvas_w, canvas_h))
        if note:
            say("note: " + note)

        settings = {"timelineResolutionWidth": str(canvas_w),
                    "timelineResolutionHeight": str(canvas_h),
                    "timelineInputResMismatchBehavior": "scaleToCrop"}
        rate = timeline_rate(clip.GetClipProperty("FPS"))
        if rate:
            settings["timelineFrameRate"] = rate
        if not project.SetSettings(settings):
            raise SuperScaleError("Resolve refused the timeline settings %r" % (settings,))

        if enhanced:
            applied = clip.SetClipProperty("Super Scale", 2, float(sharpness),
                                           float(noise_reduction))
        else:
            applied = clip.SetClipProperty("Super Scale", multiplier)
        if not applied:
            raise SuperScaleError("Resolve refused Super Scale %s on the clip. Super Scale "
                                  "is a DaVinci Resolve Studio feature."
                                  % describe(multiplier, enhanced))

        timeline = pool.CreateTimelineFromClips("Pulse Super Scale", [{"mediaPoolItem": clip}])
        if timeline is None:
            raise SuperScaleError("Resolve could not make a timeline from %s" % src)
        project.SetCurrentTimeline(timeline)

        encoder = pick_codec(project.GetRenderCodecs("mp4"), codec)
        out_dir = out_dir or os.path.dirname(src)
        out_file = output_name(src, target)
        project.SetCurrentRenderMode(1)  # one file for the whole timeline
        if not project.SetCurrentRenderFormatAndCodec("mp4", encoder):
            raise SuperScaleError("Resolve refused MP4 / %s" % encoder)
        if not project.SetRenderSettings({
                "SelectAllFrames": True, "TargetDir": out_dir,
                "CustomName": os.path.splitext(out_file)[0], "ExportVideo": True,
                "ExportAudio": True, "FormatWidth": canvas_w, "FormatHeight": canvas_h}):
            raise SuperScaleError("Resolve refused the render settings")
        if not project.SetRenderSettings({"VideoQuality": "Best"}):
            say("note: Resolve sets %s's bitrate itself (it takes no quality setting "
                "from a script)" % encoder)

        out_path = os.path.join(out_dir, out_file)
        began = time.time()
        job = project.AddRenderJob()
        if not job or not project.StartRendering([job]):
            raise SuperScaleError("Resolve would not start the render")
        say("rendering %s with %s ..." % (out_file, encoder))
        shown = None
        while project.IsRenderingInProgress():
            if stop_requested(stop_file):
                project.StopRendering()
                for _ in range(STOP_WAIT):
                    if not project.IsRenderingInProgress():
                        break
                    time.sleep(1)
                raise unfinished(SuperScaleError("stopped: the render was cancelled part-way"),
                                 out_path, began)
            done = (project.GetRenderJobStatus(job) or {}).get("CompletionPercentage")
            if done is not None and done != shown:
                say("  %s%%" % done)
                shown = done
            time.sleep(2)
        status = project.GetRenderJobStatus(job) or {}
        if status.get("JobStatus") != "Complete":
            raise unfinished(SuperScaleError("Resolve's render ended %s %s"
                                             % (status.get("JobStatus"), status.get("Error", ""))),
                             out_path, began)
        return out_path
    finally:
        if keep_project:
            manager.SaveProject()
            say("kept the Resolve project %r" % name)
        else:
            if job:
                project.DeleteRenderJob(job)
            manager.CloseProject(project)
            manager.DeleteProject(name)
        if yours:
            manager.LoadProject(previous)


def _line_by_line():
    """UTF-8, flushed a line at a time.

    Through a pipe Python buffers its output in blocks, so the node reading this
    would see the render's progress only once it was over; and a console code
    page that cannot spell a path in the film's folder would stop the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):  # no stream (pythonw), or not a text one
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Upscale a finished Pulse film to 2K/4K with DaVinci Resolve Studio's "
                    "Super Scale.")
    parser.add_argument("film", help="a film, or a Pulse run folder (its assembled.mp4)")
    parser.add_argument("--target", choices=TARGETS, default="2160p")
    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="Super Scale multiplier; auto: 2x Enhanced, with Resolve's "
                             "resize covering the rest (up to 1.5x), else 3x or 4x")
    parser.add_argument("--sharpness", type=float, default=0.5, help="2x Enhanced, 0-1")
    parser.add_argument("--nr", type=float, default=0.5,
                        help="2x Enhanced noise reduction, 0-1")
    parser.add_argument("--codec", choices=CODECS, default="h265")
    parser.add_argument("--out", help="folder to write to (default: beside the film)")
    parser.add_argument("--keep-project", action="store_true",
                        help="keep the Resolve project instead of deleting it")
    parser.add_argument("--switch-project", action="store_true",
                        help="if one of your projects is open in Resolve, switch away from "
                             "it for the render and load it again afterwards")
    parser.add_argument("--quit", action="store_true",
                        help="quit Resolve afterwards (the .bat passes this when it "
                             "started Resolve itself)")
    parser.add_argument("--wait", type=int, default=120,
                        help="seconds to wait for Resolve to answer")
    parser.add_argument("--stop-file",
                        help="stop the render, and clean up, as soon as this file exists "
                             "(the Pulse Resolve Super Scale node creates it on ComfyUI's "
                             "Cancel)")
    args = parser.parse_args(argv)

    for value, label in ((args.sharpness, "--sharpness"), (args.nr, "--nr")):
        if not 0.0 <= value <= 1.0:
            parser.error("%s must be between 0 and 1" % label)
    _line_by_line()
    try:
        src = os.path.abspath(choose_source(windows_path(args.film)))
        if not os.path.isfile(src):
            raise SuperScaleError("no such film: %s" % src)
        resolve = connect(args.wait, stop_file=args.stop_file)
        print("%s %s" % (resolve.GetProductName(), resolve.GetVersionString()))
        try:
            out = superscale(resolve, src, target=args.target, mode=args.mode,
                             sharpness=args.sharpness, noise_reduction=args.nr,
                             codec=args.codec,
                             out_dir=args.out and os.path.abspath(windows_path(args.out)),
                             keep_project=args.keep_project,
                             switch_project=args.switch_project,
                             stop_file=args.stop_file)
        finally:
            if args.quit:
                resolve.Quit()
    except SuperScaleError as exc:
        left = getattr(exc, "unfinished", None)
        if left and not discard_unfinished(*left, wait=QUIT_RELEASE_WAIT if args.quit else 0):
            print("note: Resolve still holds the unfinished %s, so it is left there" % left[0],
                  file=sys.stderr)
        print("resolve_superscale: %s" % exc, file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
