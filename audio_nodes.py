"""Pulse Audio: the film's sound as separate tracks on the film clock.

    Pulse Audio Timeline   the timeline's dialogue, placed, trimmed and gated to its
                           shots, as one film-length track -- everyone's, or one
                           character's -- with a table of who is heard when
    Pulse Audio Export     the stems of a rendered film, as WAV files and a
                           stems.json, beside it; optionally handed to DaVinci
                           Resolve's Fairlight as a project ready to mix

Two stems need no separation model, because Pulse placed them itself: the
dialogue (render.reference_audio_track, exactly as Pulse Render muxes it) and H3's
own sound (assembled_sound.flac, which Pulse Render keeps beside the film). Music,
effects, Foley and ambience arrive on the export's sockets. Every decision lives
in comfyui_pulse_studio/pulse_audio.py and is tested headless; the hand-off runs
tools/resolve_handoff.py under Resolve's own Python, through the same Resolve
session the Super Scale nodes use.

Same slot contract as nodes.py: `schema_version` at widget 0, widgets
append-only, `"hidden": HIDDEN_INPUTS`.
"""

import json
import logging
import math
import os
import time

import folder_paths

from . import media, render
from .comfyui_pulse_studio import lipsync
from .comfyui_pulse_studio import pulse_audio as pa
from .comfyui_pulse_studio import resolve_bridge as bridge
from .comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO
from .comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, FPS
from .nodes import HIDDEN_INPUTS, _unwrap_timeline, schema_widget
from .resolve_nodes import CONNECT_WAIT, _resolve_session, _source_file

log = logging.getLogger(__name__)

CATEGORY = "AddisPulse/H3"
PACK_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOFF = os.path.join(PACK_DIR, "tools", "resolve_handoff.py")
#: The labels the stems carry in Resolve's track headers.
TRACK_LABELS = {"dialogue": "Dialogue", "h3_sound": "H3 sound", "music": "Music",
                "sfx": "SFX", "foley": "Foley", "ambience": "Ambience", "mix": "Mix"}


# ── the timeline's sound ────────────────────────────────────────────────────

def _timeline_parts(timeline):
    """(document, side, windows, fps) of a Pulse Slate `timeline`, or ValueError."""
    document, side = _unwrap_timeline(timeline)
    if document is None or side is None or side.timeline is None or side.plan is None:
        raise ValueError("Connect a Pulse Slate node's `timeline` output.")
    windows = document.get("windows") or []
    fps = float((windows[0].get("fps") if windows else None) or FPS)
    return document, side, windows, fps


def _speakers(timeline):
    """[(name, recording ids)] for every character who has lip-sync recordings."""
    owners = list(timeline.assets.by_kind(KIND_IMAGE)) + list(timeline.assets.by_kind(KIND_VIDEO))
    for assets in (timeline.local_refs or {}).values():
        owners.extend(a for a in assets if a.kind != KIND_AUDIO)
    found, seen = [], set()
    for owner in owners:
        if owner.asset_id in seen:
            continue
        seen.add(owner.asset_id)
        ids = lipsync.owner_voice_ids(timeline, owner.asset_id)
        if ids:
            found.append((owner.name, ids))
    return found


def _placements(document, side, asset_ids=None):
    """Every lip-sync recording's place on the film clock, as pulse_audio rows."""
    pieces = []
    for window in side.plan.windows:
        for ref in list(window.files) + list(getattr(window, "drive_files", None) or []):
            if ref.kind != KIND_AUDIO or ref.synthetic or ref.audio_role != AUDIO_ROLE_LIP_SYNC:
                continue
            for piece in (getattr(ref, "pieces", None) or [ref]):
                if asset_ids is None or piece.asset_id in asset_ids:
                    pieces.append(dict(piece.to_dict(), asset_id=piece.asset_id))
    owner_of = {rid: name for name, ids in _speakers(side.timeline) for rid in ids}
    labels = {s.get("shot_id"): s.get("label") or "" for s in document.get("shots") or []}
    return pa.placement_rows(pieces, speaker_of=lambda p: owner_of.get(p.get("asset_id"), ""),
                             shot_label_of=labels.get)


def _dialogue(document, side, windows, fps, asset_ids=None, final=True):
    """The recordings on the film clock, film-length; silence when there are none."""
    track = render.reference_audio_track(side.plan.windows, windows, side, fps=fps,
                                         asset_ids=asset_ids, final=final)
    return track if track is not None else media.silent_audio(pa.film_seconds(windows))


def _shots(document):
    """The document's shots with their film-clock spans, for stems.json."""
    out, start = [], 0.0
    for shot in document.get("shots") or []:
        seconds = float(shot.get("duration_seconds") or 0.0)
        out.append({"index": int(shot.get("index", len(out))) + 1,
                    "label": shot.get("label") or "", "start": round(start, 3),
                    "end": round(start + seconds, 3)})
        start += seconds
    return out


def _measure(audio):
    """(integrated LUFS, sample peak dBFS); None where silent or pyloudnorm is absent."""
    waveform = audio["waveform"][0].detach().to("cpu").float()
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    peak_db = round(20.0 * math.log10(peak), 2) if peak > 0 else None
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(int(audio["sample_rate"]))
        lufs = float(meter.integrated_loudness(waveform.numpy().T))
    except Exception:  # not installed, or too short to gate
        return None, peak_db
    return (round(lufs, 2) if math.isfinite(lufs) else None), peak_db


# ── Pulse Audio Timeline ────────────────────────────────────────────────────

class PulseAudioTimeline:
    """The timeline's dialogue as one track on the film clock, before or after rendering."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("PULSE_TIMELINE", {"tooltip":
                    "Pulse Slate's `timeline`. Every lip-sync recording in it -- bin rows "
                    "and Pulse Voices -- placed where Pulse places it, trimmed and gated to "
                    "its shot."}),
                "schema_version": schema_widget(),
                "speaker": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "Empty: everyone. @Name: that character's recordings only -- a "
                    "dialogue stem of their own."}),
                "recordings": (["final", "drive"], {"default": "final", "tooltip":
                    "final: each Pulse Voice's clean take where one is connected -- what "
                    "goes in the film. drive: the recordings that moved the mouths."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("dialogue", "placements", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The timeline's dialogue, placed, trimmed and gated to its shots, as "
                   "one film-length track -- everyone's or one character's -- with a "
                   "table of who is heard when. Hear and save it before rendering.")

    def execute(self, timeline, schema_version, speaker, recordings, unique_id=None):
        document, side, windows, fps = _timeline_parts(timeline)
        asset_ids = None
        if (speaker or "").strip():
            owner = lipsync.resolve_owner(side.timeline, speaker)
            asset_ids = lipsync.owner_voice_ids(side.timeline, owner.asset_id)
        track = _dialogue(document, side, windows, fps, asset_ids, recordings == "final")
        rows = _placements(document, side, asset_ids)
        report = pa.placement_report(rows, pa.film_seconds(windows))
        return (track, json.dumps(rows, indent=1), report)


# ── Pulse Audio Export ──────────────────────────────────────────────────────

class PulseAudioExport:
    """A rendered film's stems, as WAV files and stems.json, ready for a mix stage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip":
                    "The rendered film: Pulse Render's `video`, or a Resolve delivery of "
                    "it. Its run folder is where the stems are written, and where H3's own "
                    "track is read from."}),
                "schema_version": schema_widget(),
                "per_speaker": ("BOOLEAN", {"default": True, "tooltip":
                    "Also write each character's lines as a stem of their own, "
                    "dialogue_<Name>.wav."}),
                "sample_rate": (list(pa.SAMPLE_RATES), {"default": "48000", "tooltip":
                    "48000 for film and video work."}),
                "bit_depth": (list(pa.BIT_DEPTHS), {"default": "24", "tooltip":
                    "24-bit PCM is the delivery standard; 32f is 32-bit float."}),
                "handoff": (list(pa.HANDOFFS), {"default": "none", "tooltip":
                    "resolve_fairlight: also build a DaVinci Resolve project with the "
                    "picture on V1 and every stem on a named audio track, ready to mix in "
                    "Fairlight. Kept in your project list."}),
                "open_project": (list(bridge.OPEN_PROJECT_MODES),
                                 {"default": bridge.OPEN_PROJECT_STOP, "tooltip":
                    "For the hand-off. stop: if one of your projects is open in Resolve, "
                    "stop and say so. switch_and_restore: switch away and back."}),
                "when_done": (list(bridge.WHEN_DONE_MODES),
                              {"default": bridge.WHEN_DONE_QUIT, "tooltip":
                    "For the hand-off: quit a Resolve the node started, or leave it "
                    "running to open the project straight away."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "timeline": ("PULSE_TIMELINE", {"tooltip":
                    "Pulse Slate's `timeline`, for the dialogue stems and the shot list. "
                    "Without it only H3's sound, the mix and the stems wired below are "
                    "written."}),
                "music": ("AUDIO", {"tooltip": "A music stem, from your files or a "
                                               "separation model; fitted to the film's "
                                               "length, as every stem is."}),
                "sfx": ("AUDIO", {"tooltip": "An effects stem; fitted to the film."}),
                "foley": ("AUDIO", {"tooltip": "A Foley stem; fitted to the film."}),
                "ambience": ("AUDIO", {"tooltip": "An ambience stem; fitted to the film."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder", "report")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Writes a rendered film's stems -- the dialogue (and each speaker's), "
                   "H3's own sound, the mix and any stems wired in -- as WAV files with a "
                   "stems.json beside the film, and can hand them to DaVinci Resolve's "
                   "Fairlight as a project ready to mix.")

    def execute(self, video, schema_version, per_speaker, sample_rate, bit_depth, handoff,
                open_project, when_done, timeline=None, music=None, sfx=None, foley=None,
                ambience=None, unique_id=None):
        started = time.time()
        notes = []
        film, note = _source_file(video)
        if note:
            notes.append(note)
        run_dir = os.path.dirname(film)
        native = os.path.join(run_dir, bridge.NATIVE_FILM)
        native = native if os.path.isfile(native) else film
        out_dir = os.path.join(bridge.output_dir(native, folder_paths.get_output_directory()),
                               pa.STEMS_DIR)
        rate = int(sample_rate)
        width, height = video.get_dimensions()

        document = side = None
        windows, fps = [], float(FPS)
        if timeline is not None:
            document, side, windows, fps = _timeline_parts(timeline)
        mix = media.load_audio(native, sample_rate=rate)
        seconds = (pa.film_seconds(windows) if windows
                   else media.audio_length_seconds(mix) if mix is not None else 0.0)

        stems, written = [], []

        def keep(role, audio, source, speaker=None):
            # Every stem exactly as long as the film, so they line up in a mix stage:
            # H3's track can end a few milliseconds early, a decoded AAC mix late.
            if audio is None:
                return
            if seconds:
                audio = media.audio_span(media.resample_audio(audio, rate)
                                         if int(audio["sample_rate"]) != rate else audio,
                                         0.0, seconds)
            path = media.write_wav(os.path.join(out_dir, pa.stem_file(role, speaker)), audio,
                                   rate, bit_depth)
            if not path:
                return
            lufs, peak_db = _measure(audio)
            stems.append({"role": role, "speaker": speaker or "",
                          "file": os.path.basename(path), "source": source,
                          "sample_rate": rate, "channels": int(audio["waveform"].shape[1]),
                          "seconds": round(media.audio_length_seconds(audio), 3),
                          "lufs": lufs, "peak_db": peak_db})
            label = TRACK_LABELS[role] + (" " + speaker if speaker else "")
            written.append((label, path))

        if document is not None:
            keep("dialogue", _dialogue(document, side, windows, fps),
                 "the recordings, placed and gated by Pulse")
            if per_speaker:
                for name, ids in _speakers(side.timeline):
                    keep("dialogue", _dialogue(document, side, windows, fps, ids),
                         "%s's recordings only" % name, speaker=name)
        sound_file = os.path.join(run_dir, pa.SOUND_FILE)
        if os.path.isfile(sound_file):
            keep("h3_sound", media.load_audio(sound_file, sample_rate=rate),
                 "H3's own track, joined across windows")
        else:
            notes.append("no %s in %s -- the film was assembled before Pulse kept H3's own "
                         "track. Queue Pulse Render again (its segments are reused) to "
                         "write it." % (pa.SOUND_FILE, run_dir))
        for role, audio in (("music", music), ("sfx", sfx), ("foley", foley),
                            ("ambience", ambience)):
            keep(role, audio, "wired in")
        keep("mix", mix, "the film's own soundtrack")

        placements = _placements(document, side) if document is not None else []
        manifest = pa.stems_manifest(
            {"name": os.path.basename(run_dir), "fps": fps,
             "frames": sum(int(w.get("frames") or 0) for w in windows) or None,
             "seconds": round(seconds, 3), "width": width, "height": height,
             "picture": os.path.basename(native)},
            _shots(document) if document is not None else [], stems, placements, notes)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, pa.STEMS_JSON), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=1, ensure_ascii=False)

        if handoff == "resolve_fairlight":
            notes.append(self._handoff(run_dir, native, written, open_project, when_done,
                                       unique_id))

        lines = ["Pulse Audio Export -> %s" % out_dir]
        for stem in stems:
            lines.append("  %-24s %7.2f s  %s  %s  (%s)" % (
                stem["file"], stem["seconds"],
                "%6.1f LUFS" % stem["lufs"] if stem["lufs"] is not None else "     - LUFS",
                "peak %5.1f dB" % stem["peak_db"] if stem["peak_db"] is not None else "silent",
                stem["source"]))
        lines += ["  %s" % n for n in notes]
        lines.append("  %.0f s in all" % (time.time() - started))
        report = "\n".join(lines)
        log.info("[PulseStudio] %s", report)
        return (out_dir, report)

    @staticmethod
    def _handoff(run_dir, native, written, open_project, when_done, unique_id):
        """Build the Fairlight project through Resolve's own Python. Returns a note."""
        if os.name != "nt":
            raise RuntimeError("The Fairlight hand-off needs Windows and DaVinci Resolve "
                               "Studio.")
        try:
            python_exe = bridge.find_install(os.environ, os.path.isfile)[1]
        except bridge.ResolveBridgeError as exc:
            raise RuntimeError(str(exc)) from None
        resolved = sorted((f for f in os.listdir(run_dir) if f.endswith("_resolve.mp4")),
                          key=lambda f: os.path.getsize(os.path.join(run_dir, f)))
        picture = os.path.join(run_dir, resolved[-1]) if resolved else native
        name = "Pulse %s stems %s" % (os.path.basename(run_dir), time.strftime("%Y%m%d_%H%M"))

        def make_argv(quit_after, stop_file):
            return bridge.handoff_argv(
                python_exe, HANDOFF, picture, written, name, stop_file,
                switch_project=open_project == bridge.OPEN_PROJECT_SWITCH,
                quit_after=quit_after, wait=CONNECT_WAIT)

        session_notes = []
        result, _ = _resolve_session(make_argv, when_done, session_notes, unique_id)
        try:
            project = bridge.parse_handoff(result.stdout, result.stderr, result.returncode)
        except bridge.ResolveBridgeError as exc:
            raise RuntimeError(str(exc)) from None
        return ("handed to Resolve as the project %r (picture %s; %s)"
                % (project, os.path.basename(picture), "; ".join(session_notes) or "Resolve "
                   "was already open"))


NODE_CLASS_MAPPINGS = {"PulseAudioTimeline": PulseAudioTimeline,
                       "PulseAudioExport": PulseAudioExport}
NODE_DISPLAY_NAME_MAPPINGS = {"PulseAudioTimeline": "Pulse Audio Timeline",
                              "PulseAudioExport": "Pulse Audio Export"}
