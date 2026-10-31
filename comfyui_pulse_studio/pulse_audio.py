"""Pulse Audio: the film's sound as separate tracks on the film clock -- the decisions.

Pulse Audio Timeline and Pulse Audio Export (../audio_nodes.py) hand the sound
on as stems, the way a mix stage wants it. Two stems need no separation model:

  * **dialogue** -- the recordings themselves, laid on the film clock and gated
    to their shots exactly as Pulse Render muxes them
    (render.reference_audio_track), and one track per speaker when asked;
  * **H3's own sound** -- the track H3 generated, joined across the windows with
    the seam treatment. Pulse Render keeps it beside the film as SOUND_FILE.

Music, effects, Foley and ambience arrive on the export's own sockets: from a
separation or Foley model, or from your own files. Everything here is pure and
stdlib-only, tested without torch or ComfyUI.
"""

import re

__all__ = [
    "BIT_DEPTHS", "HANDOFFS", "SAMPLE_RATES", "SOUND_FILE", "STEMS_DIR", "STEMS_JSON",
    "STEM_ROLES", "film_seconds", "placement_report", "placement_rows", "safe_name",
    "stem_file", "stems_manifest",
]

#: H3's own track, joined across the windows, beside assembled.mp4 in a run folder.
SOUND_FILE = "assembled_sound.flac"
#: Where the export writes, inside the run folder, and its description of itself.
STEMS_DIR = "stems"
STEMS_JSON = "stems.json"
#: Every stem the export knows, in the order a mix stage lays them out.
STEM_ROLES = ("dialogue", "h3_sound", "music", "sfx", "foley", "ambience", "mix")
SAMPLE_RATES = ("48000", "44100")
#: "32f" is 32-bit float; the others are integer PCM.
BIT_DEPTHS = ("24", "16", "32f")
HANDOFFS = ("none", "resolve_fairlight")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(text):
    """A name that is safe in a file name on every system: 'Ada Lovelace' -> 'Ada_Lovelace'."""
    cleaned = _UNSAFE.sub("_", (text or "").strip().lstrip("@")).strip("._")
    return cleaned or "unnamed"


def stem_file(role, speaker=None):
    """'dialogue' -> 'dialogue.wav'; 'dialogue', '@Ada' -> 'dialogue_Ada.wav'."""
    if role not in STEM_ROLES:
        raise ValueError("stem role must be one of %s, got %r" % (", ".join(STEM_ROLES), role))
    return "%s_%s.wav" % (role, safe_name(speaker)) if speaker else "%s.wav" % role


def film_seconds(windows):
    """The film's length from the compiled document's windows (frames / fps each)."""
    return sum(float(w.get("frames") or 0) / float(w.get("fps") or 24) for w in windows or ())


def placement_rows(pieces, speaker_of=None, shot_label_of=None):
    """Where every recording is heard on the film clock, once each, in film order.

    `pieces` are the dicts of a compiled window's lip-sync references (FileRef or
    VoicePiece `to_dict()`): the recording's trimmed span sits at `audio_offset` on
    the film clock, and a `gate`, when present, is the shot span it may be heard in.
    A recording that spans several windows appears in each of them, so rows are
    deduplicated. `speaker_of(piece)` and `shot_label_of(shot_id)` name things.
    """
    rows, seen = [], set()
    for piece in pieces:
        gate = tuple(piece.get("gate") or ()) or None
        key = (piece.get("asset_id"), gate)
        if key in seen:
            continue
        seen.add(key)
        trim_start = float(piece.get("trim_start") or 0.0)
        trim_end = piece.get("trim_end")
        source_seconds = piece.get("source_seconds")
        offset = float(piece.get("audio_offset") or 0.0)
        if trim_end is not None:
            length = max(0.0, float(trim_end) - trim_start)
        elif source_seconds is not None:
            length = max(0.0, float(source_seconds) - trim_start)
        else:
            length = None
        film_in = offset if gate is None else max(offset, gate[0])
        ends = [e for e in ((offset + length) if length is not None else None,
                            gate[1] if gate else None) if e is not None]
        film_out = min(ends) if ends else None
        if film_out is not None and film_out <= film_in:
            continue  # gated out of its own shot: nothing of it is heard
        source_in = trim_start + (film_in - offset)
        rows.append({
            "name": piece.get("name") or "",
            "speaker": (speaker_of(piece) if speaker_of else "") or "",
            "file": piece.get("file") or "",
            "shot": (shot_label_of(piece.get("shot_id")) if shot_label_of and piece.get("shot_id")
                     else ""),
            "film_in": round(film_in, 3),
            "film_out": None if film_out is None else round(film_out, 3),
            "source_in": round(source_in, 3),
            "source_out": (None if film_out is None
                           else round(source_in + (film_out - film_in), 3)),
        })
    rows.sort(key=lambda r: (r["film_in"], r["name"]))
    return rows


def _seconds(value):
    return "  ?  " if value is None else "%6.2f" % value


def placement_report(rows, film_length):
    """The rows as a table: who is heard when, from which seconds of which file."""
    if not rows:
        return ("No lip-sync recording in this timeline: the dialogue track is %.2f s of "
                "silence." % film_length)
    lines = ["Dialogue on the film clock (%.2f s):" % film_length,
             "  film in - out      speaker      recording        file seconds  file"]
    for row in rows:
        lines.append("  %s - %s  %-12s %-16s %s - %s  %s%s" % (
            _seconds(row["film_in"]), _seconds(row["film_out"]),
            (row["speaker"] or "-")[:12], row["name"][:16], _seconds(row["source_in"]),
            _seconds(row["source_out"]), row["file"],
            ("  (%s)" % row["shot"]) if row["shot"] else ""))
    return "\n".join(lines)


def stems_manifest(film, shots, stems, placements=(), notes=()):
    """What stems.json says: the film, its shots, every stem written, and where
    each recording sits -- enough for a mix stage, or a script, to line it all up.

    `film`: name, fps, frames, seconds, width, height. `shots`: index, label,
    start, end. `stems`: role, speaker, file, source, sample_rate, channels,
    seconds, and lufs / peak_db when they were measured.
    """
    return {
        "schema": 1,
        "film": dict(film),
        "shots": [dict(s) for s in shots],
        "stems": [dict(s) for s in stems],
        "placements": [dict(p) for p in placements],
        "notes": list(notes),
    }
