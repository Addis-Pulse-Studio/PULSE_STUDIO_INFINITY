"""`text_to_video`: a whole film on fl2va, from the words alone.

The first T2V graphs got there with a key nobody can see -- `"branch": "fl2va"`
typed into timeline_data -- under continuity `none`. Every window then sampled on
fl2va: window 1 unpinned, each later one pinned to the previous window's last
frame. The mode says the same thing where it can be seen, and these pin that it
says exactly the same thing:

  * it is appended to the project combo, and the PulseShot combo keeps its list;
  * it needs no second checkpoint and no start_image;
  * window 1 has no anchor unless shot 1 brings a start_image, and every later
    window opens on the carried frame;
  * a bin is refused, in words that name the mode;
  * the hidden key still compiles, and to the same windows.
"""

import ast
import unittest
from pathlib import Path

from comfyui_pulse_studio import compile_timeline
from comfyui_pulse_studio.assets import KIND_IMAGE
from comfyui_pulse_studio.compiler import CARRY_IMAGE_ID, CarryPolicy
from comfyui_pulse_studio.constants import BRANCH_FL2VA, BRANCH_REF2VA
from comfyui_pulse_studio.pulse_timeline import (
    CONTINUITY_INHERIT,
    CONTINUITY_KEYFRAME_PAIRS,
    CONTINUITY_LAST_FRAME,
    CONTINUITY_MODES,
    CONTINUITY_NONE,
    CONTINUITY_REF_GUIDED,
    CONTINUITY_T2V,
    SHOT_CONTINUITY_MODES,
    ContinuityError,
    check_continuity,
    continuity_branches,
    resolve_continuity,
    shot_block,
)
from comfyui_pulse_studio.timeline import Timeline

SHOT = 181 / 24.0


def _apply_continuity_branch():
    """nodes._apply_continuity_branch, lifted by ast: nodes.py imports torch."""
    tree = ast.parse((Path(__file__).parent.parent / "nodes.py").read_text(encoding="utf-8"))
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name == "_apply_continuity_branch"], type_ignores=[])
    namespace = {"continuity_branches": continuity_branches, "CONTINUITY_T2V": CONTINUITY_T2V}
    exec(compile(module, "nodes.py", "exec"), namespace)
    return namespace["_apply_continuity_branch"]


def _film(n_shots=4, assets=(), branch=BRANCH_REF2VA, continuity=CONTINUITY_T2V,
          start_image=None):
    tl = Timeline(assets=list(assets),
                  shots=[{"id": "s%d" % i, "start": (i - 1) * SHOT, "duration": SHOT,
                          "prompt": "The woman (S1) walks on and talks."}
                         for i in range(1, n_shots + 1)],
                  duration_seconds=n_shots * SHOT, fps=24, window_seconds=15.0,
                  branch=branch)
    if start_image:
        tl.shot_anchors = {"s1": {"first": start_image}}
    _apply_continuity_branch()(tl, continuity)
    return tl


def _plan(tl):
    return compile_timeline(tl, policy="shot_aligned", carry=CarryPolicy(mode="image"))


class TheMode(unittest.TestCase):

    def test_it_is_appended_to_the_project_combo(self):
        self.assertEqual(CONTINUITY_MODES[-1], CONTINUITY_T2V)
        self.assertEqual(CONTINUITY_MODES[:4], (CONTINUITY_NONE, CONTINUITY_LAST_FRAME,
                                                CONTINUITY_KEYFRAME_PAIRS, CONTINUITY_REF_GUIDED))

    def test_the_shot_combo_keeps_its_list(self):
        self.assertEqual(SHOT_CONTINUITY_MODES,
                         (CONTINUITY_INHERIT, CONTINUITY_NONE, CONTINUITY_LAST_FRAME,
                          CONTINUITY_KEYFRAME_PAIRS, CONTINUITY_REF_GUIDED))
        with self.assertRaises(ContinuityError):
            resolve_continuity(CONTINUITY_NONE, CONTINUITY_T2V)
        self.assertEqual(resolve_continuity(CONTINUITY_T2V, CONTINUITY_INHERIT), CONTINUITY_T2V)

    def test_it_needs_no_second_checkpoint_and_no_start_image(self):
        shots = [shot_block("s%d" % i, i) for i in range(3)]
        self.assertEqual(check_continuity(shots, CONTINUITY_T2V, fl2va_connected=False), [])


class TheBranches(unittest.TestCase):

    def test_every_mode_maps_as_the_node_layer_always_did(self):
        cases = {
            (CONTINUITY_NONE, BRANCH_REF2VA): (BRANCH_REF2VA, BRANCH_REF2VA, False),
            (CONTINUITY_NONE, BRANCH_FL2VA): (BRANCH_FL2VA, BRANCH_FL2VA, False),
            (CONTINUITY_LAST_FRAME, BRANCH_REF2VA): (BRANCH_REF2VA, BRANCH_FL2VA, False),
            (CONTINUITY_KEYFRAME_PAIRS, BRANCH_REF2VA): (BRANCH_FL2VA, BRANCH_FL2VA, False),
            (CONTINUITY_REF_GUIDED, BRANCH_REF2VA): (BRANCH_REF2VA, BRANCH_REF2VA, True),
            (CONTINUITY_T2V, BRANCH_REF2VA): (BRANCH_FL2VA, BRANCH_FL2VA, False),
            (CONTINUITY_T2V, BRANCH_FL2VA): (BRANCH_FL2VA, BRANCH_FL2VA, False),
        }
        for (mode, opening), expected in cases.items():
            with self.subTest(mode=mode, opening=opening):
                self.assertEqual(continuity_branches(mode, opening), expected)

    def test_the_node_layer_sets_what_the_mapping_says(self):
        tl = _film(continuity=CONTINUITY_T2V)
        self.assertEqual((tl.branch, tl.continuation_branch, tl.guided_carry),
                         (BRANCH_FL2VA, BRANCH_FL2VA, False))
        self.assertTrue(tl.text_to_video)
        self.assertFalse(_film(continuity=CONTINUITY_NONE).text_to_video)


class TheFilm(unittest.TestCase):

    def test_window_1_is_the_words_alone_and_later_windows_open_on_the_carried_frame(self):
        plan = _plan(_film())
        self.assertTrue(plan.ok, plan.problems)
        self.assertEqual([w.branch for w in plan.windows], [BRANCH_FL2VA, BRANCH_FL2VA])
        first, second = plan.windows
        self.assertNotIn("first_frame", first.anchors)
        self.assertNotIn("last_frame", first.anchors)
        self.assertEqual(second.anchors.get("first_frame"), CARRY_IMAGE_ID)

    def test_the_prompts_follow_the_anchors(self):
        first, second = _plan(_film()).windows
        self.assertTrue(first.prompt.startswith("integrated_multimodal_description:\n"))
        self.assertTrue(second.prompt.startswith(
            "For the target video, at 0.00 seconds into the target video, <Picture 1> "
            "(from [Shot 1]) is fully referenced.\n\n"))

    def test_a_start_image_on_shot_1_pins_window_1(self):
        first = _plan(_film(start_image="socket:shot.s1.start_image")).windows[0]
        self.assertEqual(first.anchors.get("first_frame"), "socket:shot.s1.start_image")
        self.assertTrue(first.prompt.startswith("For the target video, at 0.00 seconds"))

    def test_a_bin_is_refused_in_words_that_name_the_mode(self):
        face = {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "ada.png"}
        plan = _plan(_film(assets=[face]))
        self.assertFalse(plan.ok)
        problems = " ".join(plan.problems)
        self.assertIn("continuity is text_to_video", problems)
        self.assertIn("Pulse Voice", problems)

    def test_the_hidden_key_keeps_its_own_words(self):
        face = {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "ada.png"}
        plan = _plan(_film(assets=[face], branch=BRANCH_FL2VA, continuity=CONTINUITY_NONE))
        self.assertIn("Switch to ref2va or empty the bin", " ".join(plan.problems))

    def test_the_hidden_key_compiles_to_the_same_windows(self):
        mode = _plan(_film(n_shots=8)).windows
        key = _plan(_film(n_shots=8, branch=BRANCH_FL2VA, continuity=CONTINUITY_NONE)).windows
        self.assertEqual(len(mode), 4)
        self.assertEqual([(w.branch, w.anchors, w.frame_count, w.prompt) for w in mode],
                         [(w.branch, w.anchors, w.frame_count, w.prompt) for w in key])


if __name__ == "__main__":
    unittest.main()
