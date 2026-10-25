"""A film can open where a previous run left off (carry-in).

Pulse Infinity renders a story one beat per queue run, and each beat has to open
on the last frame of the beat before it. That machinery already existed *between*
the windows of one render; carry-in lets window 0 continue a previous run in the
same way. These tests pin both halves of the promise:

  * a carried-in window 0 compiles exactly as a continuation window does -- the
    carried frame is <Picture 1>, the branch is the continuation branch, the
    prompt says "continuing directly from";
  * a timeline that carries nothing compiles, keys and names its run folder
    exactly as it did before carry-in existed. Every segment already on disk
    depends on that.

render.py and nodes.py import torch, so their halves are pinned by
source-contract checks and by lifting the pure helpers out with `ast`, as the
neighbouring suites do.
"""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

from comfyui_pulse_studio.assets import KIND_IMAGE, Asset, AssetBin
from comfyui_pulse_studio.compiler import (
    CARRY_AUDIO_ID,
    CARRY_IMAGE_ID,
    GUIDE_FIRST,
    CarryPolicy,
    _continues,
    compile_timeline,
)
from comfyui_pulse_studio.constants import BRANCH_FL2VA, BRANCH_REF2VA
from comfyui_pulse_studio.pulse_timeline import (
    build_timeline,
    global_block,
    ref_descriptor,
    shot_block,
    window_block,
)
from comfyui_pulse_studio.segcache import (
    MANIFEST_NAME,
    carry_in_descriptor,
    carry_in_problems,
    cache_key,
    derive_run_id,
    resolve_carry_source,
)
from comfyui_pulse_studio.timeline import Shot, Timeline

ROOT = Path(__file__).resolve().parents[1]
CARRY = carry_in_descriptor("pulse_infinity/clock_garden/beats/beat_0002", "a1", "ab" * 32)


def _film(branch=BRANCH_REF2VA, continuation=None, carry_in=None, guided=False,
          seconds=5.17):
    assets = []
    if branch == BRANCH_REF2VA:
        assets = [Asset("face_alice", KIND_IMAGE, name="Alice", file="example_face.png",
                        description="a girl in a torn blue apron dress")]
    timeline = Timeline(
        assets=AssetBin(assets),
        shots=[Shot("beat-3", start=0.0, duration=seconds,
                    prompt="@Alice steps through the gate into the garden.")],
        duration_seconds=seconds, branch=branch,
        continuation_branch=continuation or branch)
    timeline.guided_carry = guided
    timeline.carry_in = carry_in
    return timeline


class TestWindowZeroContinuesWhatItIsGiven(unittest.TestCase):
    def test_without_carry_in_window_0_opens_fresh(self):
        self.assertFalse(_continues(_film(), 0))
        self.assertTrue(_continues(_film(), 1))
        window = compile_timeline(_film(), carry=CarryPolicy("image")).windows[0]
        self.assertFalse(any(f.synthetic for f in window.files))
        self.assertNotIn("continuing directly from", window.prompt)

    def test_the_carried_frame_is_picture_1_and_the_cast_moves_up(self):
        window = compile_timeline(_film(carry_in=CARRY),
                                  carry=CarryPolicy("image", audio=False)).windows[0]
        self.assertEqual(window.files[0].asset_id, CARRY_IMAGE_ID)
        self.assertEqual(window.files[0].tag, "<Picture 1>")
        face = [f for f in window.files if f.asset_id == "face_alice"][0]
        self.assertEqual(face.tag, "<Picture 2>")
        self.assertIn("continuing directly from `<Picture 1>`", window.prompt)

    def test_the_audio_tail_rides_the_first_audio_socket_when_asked_for(self):
        window = compile_timeline(_film(carry_in=CARRY),
                                  carry=CarryPolicy("image", audio=True)).windows[0]
        carried = [f for f in window.files if f.asset_id == CARRY_AUDIO_ID]
        self.assertEqual([f.tag for f in carried], ["<Audio 1>"])

    def test_nothing_is_carried_when_the_carry_policy_says_none(self):
        window = compile_timeline(_film(carry_in=CARRY),
                                  carry=CarryPolicy("none", audio=False)).windows[0]
        self.assertFalse(any(f.synthetic for f in window.files))

    def test_fl2va_continuation_pins_the_carried_frame(self):
        film = _film(continuation=BRANCH_FL2VA, carry_in=CARRY)
        window = compile_timeline(film, carry=CarryPolicy("image")).windows[0]
        self.assertEqual(window.branch, BRANCH_FL2VA)
        self.assertEqual(window.anchors.get("first_frame"), CARRY_IMAGE_ID)
        self.assertIn("Continue the ongoing action naturally", window.prompt)

    def test_ref_guided_carry_pins_window_0_with_a_guide(self):
        film = _film(carry_in=CARRY, guided=True)
        window = compile_timeline(film, carry=CarryPolicy("image")).windows[0]
        self.assertEqual(window.anchors.get(GUIDE_FIRST), CARRY_IMAGE_ID)
        # The guide holds the frame; carrying it again would spend a picture slot.
        self.assertFalse(any(f.asset_id == CARRY_IMAGE_ID for f in window.files))

    def test_carry_in_is_never_serialised(self):
        film = _film(carry_in=CARRY)
        self.assertNotIn("carry_in", film.to_dict())
        self.assertEqual(json.dumps(film.to_dict(), sort_keys=True),
                         json.dumps(_film().to_dict(), sort_keys=True))


def _document(carry_in=None):
    shots = [shot_block("shot-a", 0, label="Gate", visual="Alice opens the gate.",
                        duration_seconds=5.17, resolved_prompt="<Subject 1> opens the gate.")]
    windows = [window_block(0, ["shot-a"], 124, seed=7, steps=8, sampler="euler",
                            scheduler="hyperflow", prompt="p0", carry_in=carry_in),
               window_block(1, ["shot-a"], 124, seed=8, steps=8, sampler="euler",
                            scheduler="hyperflow", continuity_in="none", prompt="p1")]
    refs = [ref_descriptor(1, "image", "Alice", "socket", sha256="ab" * 32)]
    return build_timeline(global_block(style="cinematic"), refs, shots, windows, budget={})


class TestNothingMovesWithoutCarryIn(unittest.TestCase):
    """The keys every existing segment is stored under.

    Computed from this document before carry-in existed. If either changes, every
    cached segment on every user's disk re-renders on the next queue.
    """

    def test_the_keys_and_the_run_folder_are_the_ones_on_disk(self):
        doc = _document()
        self.assertEqual(
            cache_key(doc, doc["windows"][0], "model-fp", "patch-fp"),
            "59e281b640054b3141e7ca40901dbcdb37aa47ba6a7446e63bdfedb983650601")
        self.assertEqual(
            cache_key(doc, doc["windows"][1], "model-fp", "patch-fp",
                      extra=[["upstream", "f" * 64]]),
            "bcca1481fed15689bba4d8d15658dc2b5865fb92e01f0a972e2fc10e92727741")
        self.assertEqual(derive_run_id(doc), "52e8062c7a4d")

    def test_a_window_block_says_carry_in_only_when_it_has_one(self):
        self.assertNotIn("carry_in", window_block(0, ["s"], 124))
        self.assertEqual(window_block(0, ["s"], 124, carry_in=CARRY)["carry_in"], CARRY)

    def test_carry_in_reaches_the_key_only_through_upstream(self):
        """Renaming the folder a beat lives in must not re-render the next beat;
        changing what that beat rendered must."""
        moved = dict(CARRY, run_dir="elsewhere/beat_0002")
        a, b = _document(carry_in=CARRY), _document(carry_in=moved)
        self.assertEqual(cache_key(a, a["windows"][0], "m", "p"),
                         cache_key(b, b["windows"][0], "m", "p"))
        self.assertNotEqual(
            cache_key(a, a["windows"][0], "m", "p", extra=[["upstream", "1" * 64]]),
            cache_key(a, a["windows"][0], "m", "p", extra=[["upstream", "2" * 64]]))


class TestTheCarrySourceIsFoundOrRefused(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output = self._tmp.name
        self.run = os.path.join(self.output, "pulse_infinity", "clock_garden", "beats",
                                "beat_0002", "a1")
        os.makedirs(self.run)
        self.key = "ab" * 32
        self.entry = {"cache_key": self.key, "status": "complete", "width": 1344,
                      "height": 768, "frames": 124, "video_path": "seg.mp4",
                      "audio_path": "seg.audio.flac", "last_frame_path": "seg.last.png"}
        for name in ("seg.mp4", "seg.audio.flac", "seg.last.png"):
            Path(self.run, name).write_bytes(b"x")
        self._write({"schema": 1, "segments": [self.entry]})

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, data):
        Path(self.run, MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")

    def test_the_segment_is_found(self):
        manifest, entry, problems = resolve_carry_source(self.output, CARRY, 1344, 768)
        self.assertEqual(problems, [])
        self.assertEqual(entry["cache_key"], self.key)
        self.assertTrue(manifest.resolve(entry, "last_frame_path").endswith("seg.last.png"))

    def test_windows_separators_resolve_the_same_folder(self):
        windows = carry_in_descriptor("pulse_infinity\\clock_garden\\beats\\beat_0002",
                                      "a1", self.key)
        self.assertEqual(windows, CARRY)
        self.assertEqual(resolve_carry_source(self.output, windows, 1344, 768)[2], [])

    def test_a_missing_folder_is_said_plainly(self):
        moved = dict(CARRY, run_id="a9")
        self.assertIn("no manifest.json", resolve_carry_source(self.output, moved)[2][0])

    def test_a_missing_segment_is_refused(self):
        other = dict(CARRY, cache_key="cd" * 32)
        self.assertIn("holds no segment", resolve_carry_source(self.output, other)[2][0])

    def test_missing_files_are_refused(self):
        os.unlink(os.path.join(self.run, "seg.last.png"))
        self.assertIn("missing files", resolve_carry_source(self.output, CARRY)[2][0])

    def test_a_different_canvas_is_refused(self):
        problems = resolve_carry_source(self.output, CARRY, 768, 1344)[2]
        self.assertIn("cannot change canvas", problems[0])

    def test_a_corrupt_manifest_is_reported_and_left_where_it_is(self):
        """Read-only: this folder belongs to the run being continued."""
        Path(self.run, MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        problems = resolve_carry_source(self.output, CARRY)[2]
        self.assertIn("cannot be read", problems[0])
        self.assertTrue(os.path.exists(os.path.join(self.run, MANIFEST_NAME)))
        self.assertFalse(os.path.exists(os.path.join(self.run, "manifest.corrupt.json")))

    def test_folders_outside_the_output_directory_are_refused_before_opening(self):
        for bad in ("../../etc", "/abs/path", "C:\\Users\\x", "a/../b", "a//b"):
            problems = carry_in_problems(dict(CARRY, run_dir=bad))
            self.assertTrue(problems, bad)
            self.assertEqual(resolve_carry_source(self.output, dict(CARRY, run_dir=bad))[0],
                             None)

    def test_a_malformed_descriptor_is_refused(self):
        self.assertTrue(carry_in_problems(None))
        self.assertTrue(carry_in_problems({"run_dir": "x", "run_id": "a1"}))
        self.assertTrue(carry_in_problems(dict(CARRY, cache_key="not-hex!")))


class TestOnlyTheFirstShotContinues(unittest.TestCase):
    """nodes._carry_in_of, lifted by ast -- nodes.py imports torch."""

    @staticmethod
    def _carry_in_of():
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "_carry_in_of"], type_ignores=[])
        namespace = {"carry_in_problems": carry_in_problems,
                     "carry_in_descriptor": carry_in_descriptor}
        exec(compile(module, "nodes.py", "exec"), namespace)
        return namespace["_carry_in_of"]

    def test_the_first_shot_carries_the_film_in(self):
        carry_in, notes = self._carry_in_of()([{"shot_id": "s1", "continue_from": CARRY}])
        self.assertEqual(carry_in, CARRY)
        self.assertEqual(notes, [])

    def test_a_later_shot_cannot(self):
        carry_in, notes = self._carry_in_of()([
            {"shot_id": "s1"}, {"shot_id": "s2", "label": "Two", "continue_from": CARRY}])
        self.assertIsNone(carry_in)
        self.assertIn("only the first shot", notes[0])

    def test_an_unusable_source_opens_fresh_and_says_so(self):
        carry_in, notes = self._carry_in_of()([
            {"shot_id": "s1", "continue_from": {"run_dir": "../x"}}])
        self.assertIsNone(carry_in)
        self.assertIn("opens fresh", notes[0])


class TestTheExecutorHonoursIt(unittest.TestCase):
    """render.py and nodes.py need torch; their contracts are pinned over source."""

    @staticmethod
    def _source(name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_window_0_chains_to_the_segment_it_continues(self):
        source = self._source("render.py")
        extras = source.split("def window_key_extras(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("if previous_key and (carried_in or consumes_carry(compiled)):", extras)
        self.assertIn('previous_key = carry_in.get("cache_key") if carry_in else None', source)
        self.assertIn("carried_in=(position == 0 and bool(carry_in))", source)

    def test_window_0_is_filled_from_the_run_it_continues(self):
        source = self._source("render.py")
        prelude = source.split("# ── the loop ──", 1)[1].split(
            "for position, window in enumerate(windows_doc):", 1)[0]
        self.assertIn("carry_from_segment(", prelude)
        self.assertIn("wants_video=window_wants(plan_windows[0], CARRY_VIDEO_ID)", prelude)
        self.assertIn("wants_audio=window_wants(plan_windows[0], CARRY_AUDIO_ID)", prelude)
        self.assertIn("not first_decision.reuse", prelude)

    def test_an_unresolvable_source_stops_the_render(self):
        source = self._source("render.py")
        self.assertIn("raise CarryInError(message)", source)
        self.assertIn("(ReuseOnlyMiss, CarryInError)", self._source("nodes.py"))

    def test_an_fl2va_pin_without_its_frame_is_refused_not_sampled_unpinned(self):
        source = self._source("render.py")
        fl2va = source.split('if anchors.get("first_frame") == CARRY_IMAGE_ID:', 1)[1]
        fl2va = fl2va.split("first = media.fit_image(carry_frame", 1)[0]
        self.assertIn("if carry_frame is None:", fl2va)
        self.assertIn("raise SocketGapError(", fl2va)

    def test_the_slate_never_conditions_a_carried_film_itself(self):
        source = self._source("nodes.py")
        short = source.split("# ── the short path:", 1)[1].split("# ── the long path:", 1)[0]
        self.assertLess(short.index("if carry_in:"), short.index("if len(plan.windows) == 1:"))


if __name__ == "__main__":
    unittest.main()
