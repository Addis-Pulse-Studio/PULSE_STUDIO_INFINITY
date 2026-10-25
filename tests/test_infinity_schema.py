"""The director contracts: what every backend must send, and what it gets back.

Every Pulse Infinity director -- Claude Code over the mailbox, Gemini in
Antigravity, a local model, a script -- is held to these shapes by one
validator. The examples are the mission spec's own, verbatim: if they stop
round-tripping, every agent briefed on the spec writes plans the loop rejects.
"""

import copy
import json
import unittest

from comfyui_pulse_studio.pulse_memory import (
    MAX_BEAT_SECONDS,
    MIN_BEAT_SECONDS,
    PLAN_OPTIONAL,
    PLAN_REQUIRED,
    PROVENANCE_INTENDED,
    DirectedShotPlan,
    ObservationResult,
    TurnResponse,
    json_schema,
    unwrap_json,
)
from comfyui_pulse_studio.pulse_memory.schema import mentions, word_count

SPEC_PLAN = {
    "shot_index": 1,
    "visual_prompt": "Cinematic shot of Alice turning away from the clock, stepping into "
                     "the dark garden path...",
    "negative_prompt": "standing still, repeating action, looking down at hands",
    "camera_motion": "slow dolly forward, low angle",
    "duration_seconds": 5.0,
    "dialogue": {"speaker": "Alice", "text": "The time has already stopped...",
                 "emotion_tone": "whispering, anxious"},
    "continuity_anchors": {"held_items": ["brass key in left hand"],
                           "costume": "torn blue apron dress",
                           "environment": "overgrown clock garden, twilight"},
}


def _plan(**changes):
    data = copy.deepcopy(SPEC_PLAN)
    data.update(changes)
    return data


class TestTheSpecPlanRoundTrips(unittest.TestCase):
    def test_the_example_reads_back_exactly(self):
        plan, problems = DirectedShotPlan.from_dict(SPEC_PLAN, cast_names=["Alice"])
        self.assertEqual(problems, [])
        self.assertEqual(plan.to_dict(), SPEC_PLAN)
        self.assertEqual(json.dumps(plan.to_dict(), sort_keys=True),
                         json.dumps(SPEC_PLAN, sort_keys=True))

    def test_five_seconds_renders_as_the_124_frame_floor_and_says_so(self):
        plan, _ = DirectedShotPlan.from_dict(SPEC_PLAN)
        self.assertEqual(plan.frames, 124)
        self.assertAlmostEqual(plan.rendered_seconds, 124 / 24.0)
        self.assertEqual(plan.duration_seconds, 5.0, "what the director asked for is kept")
        self.assertTrue(any("124 frames" in n for n in plan.notes))

    def test_durations_land_on_the_grid_inside_the_trained_range(self):
        for asked, frames in ((1.0, 124), (5.17, 124), (6.0, 158), (15.08, 362), (40.0, 362)):
            plan, _ = DirectedShotPlan.from_dict(_plan(duration_seconds=asked))
            self.assertEqual(plan.frames, frames, asked)
            self.assertEqual((plan.frames - 5) % 17, 0)
        self.assertAlmostEqual(MIN_BEAT_SECONDS, 124 / 24.0)
        self.assertAlmostEqual(MAX_BEAT_SECONDS, 362 / 24.0)

    def test_optional_fields_are_written_only_when_set(self):
        plan, _ = DirectedShotPlan.from_dict(_plan(beat="Alice leaves the clock behind",
                                                   scene="Clock garden", scene_change=False))
        out = plan.to_dict()
        self.assertEqual(out["beat"], "Alice leaves the clock behind")
        self.assertIs(out["scene_change"], False)
        self.assertNotIn("end_of_story", out)
        self.assertEqual(set(out) - set(PLAN_REQUIRED), {"beat", "scene", "scene_change"})

    def test_a_silent_beat_is_null_dialogue(self):
        plan, problems = DirectedShotPlan.from_dict(_plan(dialogue=None))
        self.assertEqual(problems, [])
        self.assertFalse(plan.spoken)
        self.assertIsNone(plan.to_dict()["dialogue"])


class TestWhatAPlanMustNotDo(unittest.TestCase):
    def test_a_hand_typed_tag_is_refused(self):
        _, problems = DirectedShotPlan.from_dict(_plan(
            visual_prompt="<Picture 1> walks into the garden"))
        self.assertTrue(any("<Picture 1>" in p for p in problems))

    def test_a_mention_outside_the_cast_is_refused(self):
        _, problems = DirectedShotPlan.from_dict(
            _plan(visual_prompt="@Alice hands the key to @Bob"), cast_names=["Alice"])
        self.assertTrue(any("@Bob" in p for p in problems))

    def test_a_new_character_may_be_cited_the_beat_they_arrive(self):
        _, problems = DirectedShotPlan.from_dict(
            _plan(visual_prompt="@Alice meets @Hatter at the table",
                  new_characters=[{"name": "Hatter", "appearance": "tall green top hat"}]),
            cast_names=["Alice"])
        self.assertEqual(problems, [])

    def test_an_unknown_speaker_is_refused(self):
        _, problems = DirectedShotPlan.from_dict(
            _plan(dialogue={"speaker": "Queen", "text": "Off with it", "emotion_tone": ""}),
            cast_names=["Alice"])
        self.assertTrue(any("Queen" in p for p in problems))

    def test_a_line_too_long_for_its_beat_is_refused_with_the_numbers(self):
        long_line = " ".join(["word"] * 40)       # 40 words in 5.17 s is ~464 wpm
        _, problems = DirectedShotPlan.from_dict(
            _plan(dialogue={"speaker": "Alice", "text": long_line, "emotion_tone": ""}))
        self.assertTrue(any("words a minute" in p and "at most 17 fit" in p for p in problems))

    def test_the_same_line_fits_a_longer_beat(self):
        line = " ".join(["word"] * 40)
        _, problems = DirectedShotPlan.from_dict(_plan(
            duration_seconds=15.0,
            dialogue={"speaker": "Alice", "text": line, "emotion_tone": ""}))
        self.assertEqual(problems, [])

    def test_missing_and_mistyped_fields_are_all_reported_at_once(self):
        _, problems = DirectedShotPlan.from_dict({"shot_index": 0, "duration_seconds": "long",
                                                  "camera_motion": 3})
        joined = "\n".join(problems)
        for fragment in ("shot_index", "visual_prompt", "duration_seconds", "camera_motion"):
            self.assertIn(fragment, joined)

    def test_a_spoken_line_needs_a_speaker(self):
        _, problems = DirectedShotPlan.from_dict(_plan(
            dialogue={"speaker": "", "text": "Hello", "emotion_tone": ""}))
        self.assertTrue(any("speaker" in p for p in problems))

    def test_runaway_text_is_capped(self):
        plan, problems = DirectedShotPlan.from_dict(_plan(visual_prompt="x " * 5000))
        self.assertTrue(any("characters" in p for p in problems))
        self.assertLessEqual(len(plan.visual_prompt), 2000)


class TestCountingWords(unittest.TestCase):
    def test_delivery_tags_are_not_words(self):
        self.assertEqual(word_count("[whisper] The time has stopped"), 4)

    def test_other_scripts_count(self):
        self.assertEqual(word_count("ጊዜው ቆሟል"), 2)

    def test_mentions_in_order_without_duplicates(self):
        self.assertEqual(mentions("@Alice and @[Mad Hatter] then @Alice"),
                         ["Alice", "Mad Hatter"])


class TestObservations(unittest.TestCase):
    def test_a_full_observation_round_trips(self):
        data = {"shot_index": 3, "summary": "Alice pushes the gate open.",
                "completed_actions": ["pushed the heavy gate open"],
                "characters": {"Alice": {"posture": "midway through the gate",
                                         "inventory_add": ["brass key"]}},
                "environment": {"lighting": "amber twilight"},
                "stationary": False, "forbid": ["looking at the key again"]}
        obs, problems = ObservationResult.from_dict(data)
        self.assertEqual(problems, [])
        self.assertEqual(obs.to_dict(), data)
        self.assertTrue(obs.made_progress)

    def test_a_beat_that_changed_nothing_made_no_progress(self):
        obs, _ = ObservationResult.from_dict({"shot_index": 4, "summary": "She waits.",
                                              "completed_actions": [], "stationary": True})
        self.assertFalse(obs.made_progress)

    def test_at_most_three_completed_actions(self):
        _, problems = ObservationResult.from_dict({
            "shot_index": 1, "summary": "", "completed_actions": ["a", "b", "c", "d"]})
        self.assertTrue(any("at most 3" in p for p in problems))

    def test_a_director_that_cannot_see_reports_intent_not_fact(self):
        plan, _ = DirectedShotPlan.from_dict(_plan(beat="Alice leaves the clock behind"))
        obs = ObservationResult.from_intent(plan)
        self.assertEqual(obs.provenance, PROVENANCE_INTENDED)
        self.assertEqual(obs.completed_actions, ["Alice leaves the clock behind"])
        self.assertEqual(obs.to_dict()["provenance"], PROVENANCE_INTENDED)


class TestTurns(unittest.TestCase):
    def test_a_turn_needs_an_observation_after_the_opening(self):
        _, problems = TurnResponse.from_dict({"plan": SPEC_PLAN})
        self.assertTrue(any("observation" in p for p in problems))

    def test_the_opening_turn_may_send_the_plan_bare(self):
        turn, problems = TurnResponse.from_dict(SPEC_PLAN, needs_observation=False)
        self.assertEqual(problems, [])
        self.assertEqual(turn.plan.to_dict(), SPEC_PLAN)
        self.assertIsNone(turn.observation)

    def test_a_character_the_observation_introduces_can_be_cited_in_the_plan(self):
        turn, problems = TurnResponse.from_dict({
            "observation": {"shot_index": 1, "summary": "A cat appears.",
                            "completed_actions": ["a grinning cat appeared"],
                            "new_characters": [{"name": "Cat", "appearance": "striped"}]},
            "plan": _plan(shot_index=2, visual_prompt="@Cat grins at @Alice")},
            cast_names=["Alice"])
        self.assertEqual(problems, [])
        self.assertEqual(turn.plan.shot_index, 2)


class TestReadingWhatModelsWrite(unittest.TestCase):
    def test_a_fenced_block_in_prose(self):
        data, problems = unwrap_json("Here you go:\n```json\n{\"plan\": {\"shot_index\": 2}}\n```\n")
        self.assertEqual((data, problems), ({"plan": {"shot_index": 2}}, []))

    def test_a_think_block_then_an_object_with_trailing_words(self):
        data, _ = unwrap_json("<think>the gate { is open</think> {\"a\": \"}\"} done")
        self.assertEqual(data, {"a": "}"})

    def test_a_claude_cli_result_envelope(self):
        data, _ = unwrap_json(json.dumps({"type": "result", "result": "{\"plan\": {}}"}))
        self.assertEqual(data, {"plan": {}})

    def test_a_tool_use_block_in_a_content_list(self):
        data, _ = unwrap_json([{"type": "text", "text": "Submitting."},
                               {"type": "tool_use", "name": "submit_turn",
                                "input": {"plan": {"shot_index": 1}}}])
        self.assertEqual(data, {"plan": {"shot_index": 1}})

    def test_a_contract_with_a_result_field_is_not_mistaken_for_an_envelope(self):
        data, _ = unwrap_json({"plan": {"shot_index": 1}, "result": "ok"})
        self.assertIn("plan", data)

    def test_nothing_usable_says_so(self):
        self.assertIsNone(unwrap_json("I could not decide.")[0])
        self.assertIsNone(unwrap_json("")[0])
        self.assertIsNone(unwrap_json(None)[0])


class TestTheSchemasMatchTheValidators(unittest.TestCase):
    """The JSON Schemas agents are shown are generated from the same field lists
    the validator enforces -- a drift between them would brief an agent to send
    what the loop then rejects."""

    def test_the_plan_schema_requires_exactly_the_contract(self):
        schema = json_schema("plan")
        self.assertEqual(schema["required"], list(PLAN_REQUIRED))
        self.assertEqual(set(schema["properties"]), set(PLAN_REQUIRED) | set(PLAN_OPTIONAL))
        self.assertFalse(schema["additionalProperties"])

    def test_turn_and_open_differ_only_in_the_observation(self):
        self.assertEqual(json_schema("turn")["required"], ["observation", "plan"])
        self.assertEqual(json_schema("open")["required"], ["plan"])

    def test_a_schema_is_a_copy(self):
        json_schema("plan")["required"].append("x")
        self.assertNotIn("x", json_schema("plan")["required"])

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            json_schema("memory")


if __name__ == "__main__":
    unittest.main()
