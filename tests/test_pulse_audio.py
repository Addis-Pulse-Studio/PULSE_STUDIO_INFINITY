"""Pulse Audio: where the dialogue sits, what the export writes, and the Fairlight hand-off.

audio_nodes.py imports ComfyUI, so -- like nodes.py -- it is read with ast here.
What it decides lives in comfyui_pulse_studio/pulse_audio.py and runs here; the
hand-off helper is driven against a fake Resolve, as the Super Scale helper is
in test_delivery.py. The WAV writer runs where torch and PyAV are installed.
"""

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfyui_pulse_studio import pulse_audio as pa
from comfyui_pulse_studio import resolve_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "audio_nodes.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

# The Ref2VA dialogue graph's two lines, as the compiler hands them on.
ADA = {"asset_id": "ada_line", "name": "AdaLine1", "file": "Ada.wav", "trim_start": 0.0,
       "trim_end": 5.96, "audio_offset": 0.0, "source_seconds": 5.96}
BEN = {"asset_id": "ben_line", "name": "BenLine2", "file": "Bob.mp3", "trim_start": 0.0,
       "trim_end": 7.14, "audio_offset": 7.541666666666667, "source_seconds": 7.14}


class Names(unittest.TestCase):

    def test_safe_names(self):
        self.assertEqual(pa.safe_name("@Ada Lovelace"), "Ada_Lovelace")
        self.assertEqual(pa.safe_name("ጊዜ"), "unnamed")
        self.assertEqual(pa.safe_name("a/b\\c:d"), "a_b_c_d")

    def test_stem_files(self):
        self.assertEqual(pa.stem_file("dialogue"), "dialogue.wav")
        self.assertEqual(pa.stem_file("dialogue", "@Ada"), "dialogue_Ada.wav")
        self.assertEqual(pa.stem_file("h3_sound"), "h3_sound.wav")
        with self.assertRaises(ValueError):
            pa.stem_file("vocals")

    def test_the_film_is_as_long_as_its_windows(self):
        self.assertAlmostEqual(pa.film_seconds([{"frames": 362, "fps": 24}] * 2), 30.1667, 3)
        self.assertEqual(pa.film_seconds([]), 0.0)


class Placements(unittest.TestCase):

    def test_bin_lines_sit_where_their_offsets_put_them(self):
        speakers = {"ada_line": "Ada", "ben_line": "Ben"}
        rows = pa.placement_rows([BEN, ADA], speaker_of=lambda p: speakers[p["asset_id"]])
        self.assertEqual([(r["name"], r["speaker"], r["film_in"], r["film_out"],
                           r["source_in"], r["source_out"]) for r in rows],
                         [("AdaLine1", "Ada", 0.0, 5.96, 0.0, 5.96),
                          ("BenLine2", "Ben", 7.542, 14.682, 0.0, 7.14)])

    def test_a_gate_cuts_a_recording_to_its_shot(self):
        piece = dict(ADA, trim_end=None, source_seconds=17.08, gate=[0.0, 7.5417],
                     shot_id="s1")
        rows = pa.placement_rows([piece], shot_label_of={"s1": "Ada opens"}.get)
        self.assertEqual((rows[0]["film_in"], rows[0]["film_out"], rows[0]["source_out"],
                          rows[0]["shot"]), (0.0, 7.542, 7.542, "Ada opens"))

    def test_a_recording_late_in_its_gate_starts_where_it_starts(self):
        piece = dict(BEN, audio_offset=9.0, gate=[7.5417, 15.0833])
        row = pa.placement_rows([piece])[0]
        self.assertEqual((row["film_in"], row["film_out"], row["source_in"]), (9.0, 15.083, 0.0))

    def test_a_recording_that_spans_windows_is_listed_once(self):
        self.assertEqual(len(pa.placement_rows([ADA, dict(ADA), BEN])), 2)

    def test_a_recording_gated_out_of_its_shot_is_not_heard(self):
        piece = dict(ADA, gate=[10.0, 12.0])  # 0-5.96 on the film, gated to 10-12
        self.assertEqual(pa.placement_rows([piece]), [])

    def test_an_unknown_length_is_said_as_such(self):
        row = pa.placement_rows([dict(ADA, trim_end=None, source_seconds=None)])[0]
        self.assertIsNone(row["film_out"])
        self.assertIn("?", pa.placement_report([row], 15.08))

    def test_the_report(self):
        self.assertIn("silence", pa.placement_report([], 15.08))
        table = pa.placement_report(pa.placement_rows([ADA, BEN]), 15.08)
        self.assertIn("15.08 s", table)
        self.assertIn("AdaLine1", table)
        self.assertIn("Bob.mp3", table)

    def test_the_manifest(self):
        manifest = pa.stems_manifest({"name": "run", "fps": 24.0}, [{"index": 1}],
                                     [{"role": "dialogue"}], [{"name": "AdaLine1"}], ["a note"])
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual([manifest[k] for k in ("shots", "stems", "placements", "notes")],
                         [[{"index": 1}], [{"role": "dialogue"}], [{"name": "AdaLine1"}],
                          ["a note"]])
        json.dumps(manifest)


class TheHandoffsCommandLine(unittest.TestCase):

    def test_the_stems_in_track_order(self):
        argv = bridge.handoff_argv("py.exe", "handoff.py", "film.mp4",
                                   [("Dialogue", "d.wav"), ("H3 sound", "h.wav")], "Pulse x",
                                   "stop", quit_after=True)
        self.assertEqual(argv[:4], ["py.exe", "handoff.py", "--picture", "film.mp4"])
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "--stem"],
                         ["Dialogue=d.wav", "H3 sound=h.wav"])
        self.assertEqual(argv[-1], "--quit")
        self.assertNotIn("--switch-project", argv)

    def test_nothing_or_a_bad_label_is_refused(self):
        with self.assertRaises(bridge.ResolveBridgeError):
            bridge.handoff_argv("py", "h", "f", [], "n", "s")
        with self.assertRaises(bridge.ResolveBridgeError):
            bridge.handoff_argv("py", "h", "f", [("a=b", "x.wav")], "n", "s")

    def test_the_project_it_reports(self):
        self.assertEqual(bridge.parse_handoff('x\nPULSE_RESULT {"project": "Pulse r stems"}\n',
                                              "", 0), "Pulse r stems")
        with self.assertRaises(bridge.ResolveBridgeError):
            bridge.parse_handoff("done\n", "", 0)
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.parse_handoff("", "resolve_superscale: Resolve has your project 'Cut' open, "
                                     "... or pass --switch-project to switch away\n", 1)
        self.assertIn("switch_and_restore", str(caught.exception))


def _helper():
    path = ROOT / "tools" / "resolve_handoff.py"
    spec = importlib.util.spec_from_file_location("resolve_handoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = _helper()


class _Item:
    def __init__(self, path, resolution="3840x2160"):
        self.properties = {"File Path": path, "Resolution": resolution, "FPS": "24.000"}

    def GetClipProperty(self, key=None):
        return self.properties.get(key)


class _Resolve:
    """The calls handoff() makes, answered the way Resolve 21.1's API documents."""

    def __init__(self, open_project="Untitled Project", existing=(), refuse_append_at=None):
        self.calls, self.open_project = [], open_project
        self.existing, self.tracks = list(existing), 1
        self.refuse_append_at, self.appends = refuse_append_at, 0

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name,) + args)
            return {"GetProjectManager": self, "GetCurrentProject": self,
                    "GetName": self.open_project, "CreateProject": self,
                    "GetMediaPool": self, "CreateEmptyTimeline": self,
                    "GetProjectListInCurrentFolder": self.existing,
                    "GetStartFrame": 86400}.get(name, True)
        return call

    def ImportMedia(self, files):
        self.calls.append(("ImportMedia", files))
        return [_Item(f) for f in reversed(files)]  # order is not promised

    def GetTrackCount(self, kind):
        return self.tracks

    def AddTrack(self, kind, layout):
        self.calls.append(("AddTrack", kind, layout))
        self.tracks += 1
        return True

    def AppendToTimeline(self, infos):
        self.calls.append(("AppendToTimeline", infos))
        self.appends += 1
        return self.appends != self.refuse_append_at

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


class TheHandoff(unittest.TestCase):
    STEMS = [("Dialogue", os.path.join("run", "stems", "dialogue.wav")),
             ("H3 sound", os.path.join("run", "stems", "h3_sound.wav")),
             ("Mix", os.path.join("run", "stems", "mix.wav"))]
    PICTURE = os.path.join("run", "assembled_2160p_resolve.mp4")

    def test_the_picture_on_v1_and_a_named_track_per_stem(self):
        fake = _Resolve()
        name = HANDOFF.handoff(fake, self.PICTURE, self.STEMS, "Pulse run stems",
                               say=lambda *a: None)
        self.assertEqual(name, "Pulse run stems")
        appends = [c[1][0] for c in fake.named("AppendToTimeline")]
        self.assertEqual((appends[0]["mediaType"], appends[0]["trackIndex"],
                          appends[0]["recordFrame"]), (HANDOFF.VIDEO_ONLY, 1, 86400))
        self.assertEqual(appends[0]["mediaPoolItem"].GetClipProperty("File Path"), self.PICTURE)
        self.assertEqual([(a["mediaType"], a["trackIndex"], a["recordFrame"],
                           os.path.basename(a["mediaPoolItem"].GetClipProperty("File Path")))
                          for a in appends[1:]],
                         [(HANDOFF.AUDIO_ONLY, 1, 86400, "dialogue.wav"),
                          (HANDOFF.AUDIO_ONLY, 2, 86400, "h3_sound.wav"),
                          (HANDOFF.AUDIO_ONLY, 3, 86400, "mix.wav")])
        self.assertEqual(fake.named("SetTrackName"), [("SetTrackName", "audio", 1, "Dialogue"),
                                                      ("SetTrackName", "audio", 2, "H3 sound"),
                                                      ("SetTrackName", "audio", 3, "Mix")])
        self.assertEqual(len(fake.named("AddTrack")), 2)
        settings = fake.named("SetSettings")[0][1]
        self.assertEqual((settings["timelineResolutionWidth"], settings["timelineFrameRate"]),
                         ("3840", "24"))
        self.assertTrue(fake.named("SaveProject"))
        self.assertFalse(fake.named("DeleteProject"))

    def test_never_the_name_of_a_project_you_have(self):
        self.assertEqual(HANDOFF.project_name("P", {"P", "P 2"}), "P 3")
        fake = _Resolve(existing=["Pulse run stems"])
        self.assertEqual(HANDOFF.handoff(fake, self.PICTURE, self.STEMS, "Pulse run stems",
                                         say=lambda *a: None), "Pulse run stems 2")

    def test_a_half_built_project_is_deleted(self):
        fake = _Resolve(refuse_append_at=3)
        with self.assertRaises(HANDOFF.rs.SuperScaleError) as caught:
            HANDOFF.handoff(fake, self.PICTURE, self.STEMS, "Pulse run stems", say=lambda *a: None)
        self.assertIn("H3 sound", str(caught.exception))
        self.assertEqual(fake.named("DeleteProject"), [("DeleteProject", "Pulse run stems")])

    def test_a_project_of_yours_is_left_alone_or_loaded_again(self):
        with self.assertRaises(HANDOFF.rs.SuperScaleError):
            HANDOFF.handoff(_Resolve(open_project="Client cut"), self.PICTURE, self.STEMS, "P",
                            say=lambda *a: None)
        fake = _Resolve(open_project="Client cut")
        HANDOFF.handoff(fake, self.PICTURE, self.STEMS, "P", switch_project=True,
                        say=lambda *a: None)
        self.assertEqual(fake.calls[-1], ("LoadProject", "Client cut"))

    def test_a_stem_is_label_equals_file(self):
        self.assertEqual(HANDOFF.parse_stem("H3 sound=D:\\run\\a=b.wav"),
                         ("H3 sound", "D:\\run\\a=b.wav"))
        for bad in ("dialogue.wav", "=x.wav", "Dialogue="):
            with self.assertRaises(HANDOFF.rs.SuperScaleError):
                HANDOFF.parse_stem(bad)

    def test_main_reports_the_project(self):
        import io
        out = io.StringIO()
        with mock.patch.object(HANDOFF.rs, "connect", return_value=_Resolve()), \
                mock.patch.object(HANDOFF.os.path, "isfile", return_value=True), \
                mock.patch.object(sys, "stdout", out):
            code = HANDOFF.main(["--picture", self.PICTURE, "--stem", "Dialogue=d.wav",
                                 "--name", "Pulse run stems"])
        self.assertEqual(code, 0)
        self.assertEqual(bridge.parse_handoff(out.getvalue(), "", 0), "Pulse run stems")


class TheNodes(unittest.TestCase):
    """The glue in audio_nodes.py, read with ast."""

    def _class(self, name):
        return next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == name)

    def _calls(self, node):
        return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                for c in ast.walk(node) if isinstance(c, ast.Call)}

    def test_the_export_is_an_output_node_and_the_timeline_is_not(self):
        def flag(name):
            assigns = {t.id: a.value for a in self._class(name).body if isinstance(a, ast.Assign)
                       for t in a.targets if isinstance(t, ast.Name)}
            return getattr(assigns.get("OUTPUT_NODE"), "value", False)
        self.assertIs(flag("PulseAudioExport"), True)
        self.assertIs(flag("PulseAudioTimeline"), False)

    def test_the_dialogue_is_pulses_own_placement(self):
        """The same function Pulse Render muxes the recordings with -- not a copy."""
        self.assertIn("render.reference_audio_track", SOURCE)
        timeline = self._calls(self._class("PulseAudioTimeline"))
        self.assertTrue({"resolve_owner", "owner_voice_ids", "_dialogue"} <= timeline)

    def test_the_export_writes_wavs_and_the_manifest_and_hands_off_through_the_session(self):
        export = self._calls(self._class("PulseAudioExport"))
        self.assertTrue({"write_wav", "stems_manifest", "_resolve_session", "handoff_argv",
                         "parse_handoff"} <= export, export)

    def test_the_pack_registers_them(self):
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("from .audio_nodes import", init)
        for line in ('"PulseAudioTimeline": "Pulse Audio Timeline"',
                     '"PulseAudioExport": "Pulse Audio Export"'):
            self.assertIn(line, SOURCE)

    def test_pulse_render_keeps_h3s_own_track_before_a_recording_replaces_it(self):
        render = (ROOT / "render.py").read_text(encoding="utf-8")
        self.assertIn("media.write_segment_audio(os.path.join(directory, SOUND_FILE)", render)
        self.assertLess(render.index("SOUND_FILE), audio_out)"),
                        render.index("if options.use_reference_audio:"))


try:
    import av  # noqa: F401
    import torch  # noqa: F401
    HAVE_AV = True
except Exception:  # pragma: no cover - the bare test environment
    HAVE_AV = False


@unittest.skipUnless(HAVE_AV, "needs torch and PyAV (the ComfyUI environment)")
class TheWavWriter(unittest.TestCase):

    def test_a_24_bit_48k_wav_reads_back(self):
        import math
        import av
        import media
        import torch
        t = torch.arange(22050) / 44100.0
        tone = torch.sin(2 * math.pi * 440 * t).repeat(2, 1).unsqueeze(0) * 0.5
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tone.wav")
            self.assertEqual(media.write_wav(path, {"waveform": tone, "sample_rate": 44100},
                                             48000, "24"), path)
            with av.open(path) as container:
                stream = container.streams.audio[0]
                self.assertEqual((stream.codec_context.name, stream.rate, stream.channels),
                                 ("pcm_s24le", 48000, 2))
                self.assertAlmostEqual(float(container.duration) / 1e6, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
