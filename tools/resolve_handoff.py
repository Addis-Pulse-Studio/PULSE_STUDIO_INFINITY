#!/usr/bin/env python3
"""Hand a Pulse film's stems to DaVinci Resolve's Fairlight, as a project ready to mix.

    ResolvePython.exe tools\\resolve_handoff.py --picture <film.mp4> \\
        --stem "Dialogue=<dialogue.wav>" --stem "H3 sound=<h3_sound.wav>" ... --name <project>

Pulse Audio Export writes the stems and runs this, when its `handoff` is
resolve_fairlight. It lays them out in a project of their own:

  1. a project named `--name` (with a number added if one of yours is called
     that already), at the picture's frame rate and size;
  2. a timeline with the picture on V1 -- video only -- and every stem on an audio
     track of its own, named after it, all starting at the timeline's first frame;
  3. saved, and kept: it is yours now. Only a project this left half-built is
     deleted.

It prints a `PULSE_RESULT {"project": ...}` line for the node. Like
resolve_superscale.py it talks only to the Resolve on this machine, through
Resolve's own scripting module, and starts nothing: the node starts Resolve when
it is not running. It runs under Resolve's own Python, never inside ComfyUI.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import resolve_superscale as rs  # noqa: E402  -- also puts the pack on sys.path
from comfyui_pulse_studio.resolve_bridge import RESULT_PREFIX  # noqa: E402

#: AppendToTimeline's mediaType: the picture without its own sound, a stem as sound.
VIDEO_ONLY = 1
AUDIO_ONLY = 2


def parse_stem(text):
    """'Dialogue=D:\\run\\stems\\dialogue.wav' -> ('Dialogue', 'D:\\run\\stems\\dialogue.wav')."""
    label, sep, path = (text or "").partition("=")
    if not sep or not label.strip() or not path.strip():
        raise rs.SuperScaleError("a stem is LABEL=FILE, got %r" % (text,))
    return label.strip(), path.strip()


def project_name(name, existing):
    """`name`, or `name 2`, `name 3` ... -- never the name of a project you have."""
    if name not in existing:
        return name
    number = 2
    while "%s %d" % (name, number) in existing:
        number += 1
    return "%s %d" % (name, number)


def _by_file(items):
    """{basename: media pool item}: ImportMedia's order is not promised."""
    found = {}
    for item in items or []:
        path = item.GetClipProperty("File Path") or ""
        found[os.path.basename(path.replace("\\", "/")).lower()] = item
    return found


def handoff(resolve, picture, stems, name, switch_project=False, stop_file=None, say=print):
    """Build and save the hand-off project. Returns its name."""
    if rs.stop_requested(stop_file):
        raise rs.SuperScaleError("stopped before the hand-off began: the stop file appeared")
    manager = resolve.GetProjectManager()
    current = manager.GetCurrentProject()
    previous = (current.GetName() if current is not None else "") or ""
    yours = previous not in ("", rs.SCRATCH_PROJECT)
    if yours and not switch_project:
        raise rs.SuperScaleError(
            "Resolve has your project %r open, and the hand-off builds a project of its "
            "own. Save and close it first, or pass --switch-project to switch away from it "
            "and load it again afterwards." % previous)
    manager.GotoRootFolder()
    name = project_name(name, set(manager.GetProjectListInCurrentFolder() or []))
    project = manager.CreateProject(name)
    if project is None:
        raise rs.SuperScaleError("Resolve would not create the project %r" % name)
    done = False
    try:
        pool = project.GetMediaPool()
        files = [picture] + [path for _, path in stems]
        items = _by_file(pool.ImportMedia(files))
        missing = [f for f in files if os.path.basename(f).lower() not in items]
        if missing:
            raise rs.SuperScaleError("Resolve could not import %s" % ", ".join(missing))
        pic = items[os.path.basename(picture).lower()]
        width, height = rs.parse_resolution(pic.GetClipProperty("Resolution"))
        settings = {"timelineResolutionWidth": str(width),
                    "timelineResolutionHeight": str(height)}
        rate = rs.timeline_rate(pic.GetClipProperty("FPS"))
        if rate:
            settings["timelineFrameRate"] = rate
        if not project.SetSettings(settings):
            raise rs.SuperScaleError("Resolve refused the timeline settings %r" % (settings,))

        timeline = pool.CreateEmptyTimeline("Pulse stems")
        if timeline is None:
            raise rs.SuperScaleError("Resolve would not make an empty timeline")
        project.SetCurrentTimeline(timeline)
        while (timeline.GetTrackCount("audio") or 0) < len(stems):
            if not timeline.AddTrack("audio", "stereo"):
                raise rs.SuperScaleError("Resolve would not add audio track %d"
                                         % ((timeline.GetTrackCount("audio") or 0) + 1))
        start = timeline.GetStartFrame()
        if not pool.AppendToTimeline([{"mediaPoolItem": pic, "mediaType": VIDEO_ONLY,
                                       "trackIndex": 1, "recordFrame": start}]):
            raise rs.SuperScaleError("Resolve would not put the picture on V1")
        for index, (label, path) in enumerate(stems, 1):
            item = items[os.path.basename(path).lower()]
            if not pool.AppendToTimeline([{"mediaPoolItem": item, "mediaType": AUDIO_ONLY,
                                           "trackIndex": index, "recordFrame": start}]):
                raise rs.SuperScaleError("Resolve would not put %s on A%d" % (label, index))
            timeline.SetTrackName("audio", index, label)
        if not manager.SaveProject():
            raise rs.SuperScaleError("Resolve would not save the project %r" % name)
        say("project %r: the picture on V1, %d stem(s) on A1-A%d (%s)"
            % (name, len(stems), len(stems), ", ".join(label for label, _ in stems)))
        done = True
        return name
    finally:
        if not done:
            manager.CloseProject(project)
            manager.DeleteProject(name)
        if yours:
            manager.LoadProject(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hand a Pulse film's stems to DaVinci Resolve's Fairlight as a project.")
    parser.add_argument("--picture", required=True, help="the film the stems belong to")
    parser.add_argument("--stem", action="append", required=True,
                        help="LABEL=FILE, once per stem, in track order")
    parser.add_argument("--name", required=True, help="the project's name")
    parser.add_argument("--switch-project", action="store_true",
                        help="if one of your projects is open in Resolve, switch away from "
                             "it and load it again afterwards")
    parser.add_argument("--quit", action="store_true", help="quit Resolve afterwards")
    parser.add_argument("--wait", type=int, default=120,
                        help="seconds to wait for Resolve to answer")
    parser.add_argument("--stop-file", help="stop as soon as this file exists")
    args = parser.parse_args(argv)

    rs._line_by_line()
    try:
        stems = [parse_stem(text) for text in args.stem]
        picture = os.path.abspath(rs.windows_path(args.picture))
        for path in [picture] + [p for _, p in stems]:
            if not os.path.isfile(path):
                raise rs.SuperScaleError("no such file: %s" % path)
        stems = [(label, os.path.abspath(rs.windows_path(path))) for label, path in stems]
        resolve = rs.connect(args.wait, stop_file=args.stop_file)
        print("%s %s" % (resolve.GetProductName(), resolve.GetVersionString()))
        try:
            name = handoff(resolve, picture, stems, args.name,
                           switch_project=args.switch_project, stop_file=args.stop_file)
        finally:
            if args.quit:
                resolve.Quit()
    except rs.SuperScaleError as exc:
        print("resolve_superscale: %s" % exc, file=sys.stderr)
        return 1
    print(RESULT_PREFIX + json.dumps({"project": name}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
