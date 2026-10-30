"""The compiled prompt in the layout MiniMax's own guides prescribe.

`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` and `..._base_en.md` on
huggingface.co/MiniMaxAI/MiniMax-H3, and Pika H3 Flex's prompt guidance, which
repeats them. Four things the compiler used to say its own way:

  * a reference summary opens with a task-type prefix -- `[reference generation +
    audio reuse]` -- not with the film's window count;
  * a picture's partial-retention word is `partially_preserved` (`partially_copy`
    is the audio word);
  * a base-mode prompt opens with the image-alignment sentence, on its own, in one
    of three fixed templates, the end mark being the duration to two decimals;
  * lip-sync instructions do not go in overall_soundscape.

Speaker ids in retention_analysis are pinned in test_speakers.py.
"""

import unittest

from comfyui_pulse_studio import compile_timeline
from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset
from comfyui_pulse_studio.compiler import CarryPolicy
from comfyui_pulse_studio.constants import (
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_TIMBRE,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
)
from comfyui_pulse_studio.timeline import Timeline

SHOT = 181 / 24.0
FACE = {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "ada.png",
        "description": "a woman with curly black hair in a rust-orange jacket"}


def _section(prompt, name):
    return prompt.split(name + ":\n", 1)[1].split("\n\n", 1)[0]


def _reference_film(n_shots=2, role=AUDIO_ROLE_LIP_SYNC, retention="fully_preserved",
                    guided=False):
    tl = Timeline(assets=[dict(FACE, retention=retention)],
                  shots=[{"id": "s%d" % i, "start": (i - 1) * SHOT, "duration": SHOT,
                          "prompt": "@Ada talks to camera.", "speakers": ["ada"]}
                         for i in range(1, n_shots + 1)],
                  duration_seconds=n_shots * SHOT, fps=24, window_seconds=15.0)
    if role:
        for i in range(1, n_shots + 1):
            tl.local_refs["s%d" % i] = [Asset("aud_s%d" % i, KIND_AUDIO, name="Line%d" % i,
                                              file="", audio_role=role, audio_offset=(i - 1) * SHOT,
                                              source_seconds=5.0, take=True)]
    tl.guided_carry = guided
    return tl


def _base_film(n_shots=2, anchors=None, driven=True):
    tl = Timeline(assets=[], shots=[{"id": "s%d" % i, "start": (i - 1) * SHOT, "duration": SHOT,
                                     "prompt": "The woman talks to camera."}
                                    for i in range(1, n_shots + 1)],
                  duration_seconds=n_shots * SHOT, fps=24, window_seconds=15.0,
                  branch=BRANCH_FL2VA, continuation_branch=BRANCH_FL2VA)
    tl.shot_anchors = dict(anchors or {})
    if driven:
        for i in range(1, n_shots + 1):
            tl.local_refs["s%d" % i] = [Asset("aud_s%d" % i, KIND_AUDIO, name="Line%d" % i,
                                              file="", audio_role=AUDIO_ROLE_LIP_SYNC,
                                              audio_offset=(i - 1) * SHOT, source_seconds=5.0,
                                              take=True)]
    return tl


def _plan(tl, carry="image", audio=False):
    return compile_timeline(tl, policy="shot_aligned", carry=CarryPolicy(mode=carry, audio=audio))


class TheSummaryOpensWithATaskType(unittest.TestCase):

    def test_a_lip_synced_reference_window_is_generation_plus_audio_reuse(self):
        prompt = _plan(_reference_film(n_shots=2)).windows[0].prompt
        self.assertTrue(_section(prompt, "summary").startswith(
            "[reference generation + audio reuse] The target video shows <Subject 1>"))

    def test_a_borrowed_timbre_is_audio_reference(self):
        prompt = _plan(_reference_film(role=AUDIO_ROLE_TIMBRE)).windows[0].prompt
        self.assertTrue(_section(prompt, "summary").startswith(
            "[reference generation + audio reference]"))

    def test_pictures_alone_are_reference_generation(self):
        prompt = _plan(_reference_film(role=None)).windows[0].prompt
        self.assertTrue(_section(prompt, "summary").startswith("[reference generation] "))

    def test_no_window_count_reaches_the_model(self):
        plan = _plan(_reference_film(n_shots=4))
        self.assertEqual(len(plan.windows), 2)
        for window in plan.windows:
            self.assertNotIn("[window", window.prompt)
            self.assertTrue(_section(window.prompt, "summary").startswith("["))

    def test_a_continuation_window_still_reads_as_generation_from_its_anchor(self):
        second = _plan(_reference_film(n_shots=4)).windows[1].prompt
        summary = _section(second, "summary")
        self.assertTrue(summary.startswith("[reference generation + audio reuse]"))
        self.assertIn("continuing directly from `<Picture 1>`", summary)

    def test_a_guided_pin_is_keyframe_completion(self):
        second = _plan(_reference_film(n_shots=4, guided=True)).windows[1].prompt
        self.assertTrue(_section(second, "summary").startswith(
            "[reference generation + keyframe completion + audio reuse]"))

    def test_a_carried_clip_is_video_continuation(self):
        second = _plan(_reference_film(n_shots=4), carry="video").windows[1].prompt
        self.assertTrue(_section(second, "summary").startswith("[video continuation + "))

    def test_a_carried_audio_tail_is_audio_reuse(self):
        second = _plan(_reference_film(n_shots=4, role=None), audio=True).windows[1].prompt
        self.assertTrue(_section(second, "summary").startswith(
            "[reference generation + audio reuse]"))


class APicturesRetentionWord(unittest.TestCase):

    def test_partial_retention_of_a_picture_is_partially_preserved(self):
        prompt = _plan(_reference_film(retention="partially_copy")).windows[0].prompt
        retention = _section(prompt, "retention_analysis")
        self.assertIn(": partially_preserved - matches `<Picture 1>`", retention)
        self.assertNotIn("partially_copy - matches", retention)

    def test_full_retention_is_unchanged(self):
        prompt = _plan(_reference_film()).windows[0].prompt
        self.assertIn(": fully_preserved - matches `<Picture 1>`",
                      _section(prompt, "retention_analysis"))

    def test_an_audio_tail_keeps_the_audio_word(self):
        second = _plan(_reference_film(n_shots=4, role=None), audio=True).windows[1].prompt
        self.assertIn("(previous window's tail): partially_copy",
                      _section(second, "retention_analysis"))


class TheBaseModeAlignmentSentence(unittest.TestCase):

    def test_both_frames_use_the_fl2va_template_first_and_alone(self):
        anchors = {"s1": {"first": "socket:a"}, "s2": {"last": "socket:b"}}
        prompt = _plan(_base_film(anchors=anchors)).windows[0].prompt
        head, rest = prompt.split("\n\n", 1)
        self.assertEqual(
            head,
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) "
            "aligns with the 15.08-second mark of the target video.")
        self.assertTrue(rest.startswith("integrated_multimodal_description:\n"))

    def test_a_first_frame_alone_uses_the_i2va_template(self):
        prompt = _plan(_base_film(anchors={"s1": {"first": "socket:a"}})).windows[0].prompt
        self.assertTrue(prompt.startswith(
            "For the target video, at 0.00 seconds into the target video, <Picture 1> "
            "(from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description:"))

    def test_a_last_frame_alone_uses_the_l2va_template(self):
        prompt = _plan(_base_film(anchors={"s2": {"last": "socket:b"}})).windows[0].prompt
        self.assertTrue(prompt.startswith(
            "<Picture 1> (from [Shot 2]) aligns with the 15.08-second mark of the target "
            "video.\n\nintegrated_multimodal_description:"))

    def test_the_end_mark_is_the_duration_of_this_window(self):
        # One shot of 124 frames: 5.1667 s, written 5.17 -- Pika's own example.
        tl = _base_film(n_shots=1, anchors={"s1": {"first": "socket:a", "last": "socket:b"}})
        tl.shots[0].duration = 124 / 24.0
        tl.duration_seconds = 124 / 24.0
        prompt = _plan(tl).windows[0].prompt
        self.assertIn("Picture 2 (from Shot 1) aligns with the 5.17-second mark", prompt)

    def test_an_unanchored_window_has_no_alignment_sentence(self):
        prompt = _plan(_base_film()).windows[0].prompt
        self.assertTrue(prompt.startswith("integrated_multimodal_description:\n"))
        self.assertNotIn("Picture", prompt)

    def test_a_continuation_pin_says_continue_in_the_description(self):
        second = _plan(_base_film(n_shots=4)).windows[1].prompt
        self.assertTrue(second.startswith("For the target video, at 0.00 seconds"))
        description = _section(second, "integrated_multimodal_description")
        self.assertTrue(description.startswith(
            "Continue the ongoing action naturally from the opening frame's pose and framing"))


class LipSyncIsNotSoundscape(unittest.TestCase):

    def test_the_drive_note_rides_in_the_description(self):
        prompt = _plan(_base_film()).windows[0].prompt
        self.assertIn("is a recorded voice", _section(prompt, "integrated_multimodal_description"))
        self.assertNotIn("recorded voice", _section(prompt, "overall_soundscape"))
        self.assertNotIn("lip movements", _section(prompt, "overall_soundscape"))

    def test_the_note_follows_the_shots_it_covers(self):
        description = _section(_plan(_base_film()).windows[0].prompt,
                               "integrated_multimodal_description")
        self.assertLess(description.index("[Shot 2]"), description.index("recorded voice"))


if __name__ == "__main__":
    unittest.main()
