"""The loop and the stall: the two ways a self-directed film stops being a film.

At cfg 1.0 a negative prompt does nothing, so these failures are refused at the
plan, before any GPU time is spent. Each test is a plan a real model would
plausibly write, and the reason it must or must not get through.
"""

import unittest

from comfyui_pulse_studio.pulse_memory import (
    ActionBlacklist,
    DirectedShotPlan,
    ObservationResult,
    SceneMemory,
    check_plan,
    content_words,
    forbidden_phrase,
    jaccard,
)


def _plan(index, visual, beat="", **extra):
    data = {"shot_index": index, "visual_prompt": visual, "negative_prompt": "",
            "camera_motion": "", "duration_seconds": 5.17, "dialogue": None,
            "continuity_anchors": {"held_items": [], "costume": "", "environment": ""},
            "beat": beat}
    data.update(extra)
    plan, problems = DirectedShotPlan.from_dict(data)
    assert not problems, problems
    return plan


def _memory():
    memory = SceneMemory("Escape the labyrinth")
    memory.seed_cast([("Alice", "a girl in a torn blue apron dress, disheveled hair")])
    memory.environment.setting = "overgrown clock garden at twilight"
    return memory


def _commit(memory, index, beat, actions=(), **extra):
    plan = _plan(index, "@Alice, a girl in a torn blue apron dress, disheveled hair, in the "
                        "overgrown clock garden at twilight. " + beat, beat=beat, **extra)
    obs, _ = ObservationResult.from_dict({"shot_index": index, "summary": beat,
                                          "completed_actions": list(actions)})
    memory.commit(plan, obs)


class TestContentWords(unittest.TestCase):
    def test_tense_and_number_meet(self):
        self.assertEqual(content_words("opening the gates"), content_words("opened the gate"))
        self.assertEqual(content_words("she closes the door"),
                         content_words("closing doors"))

    def test_short_nouns_count(self):
        self.assertIn("key", content_words("the brass key"))

    def test_static_words_are_removed(self):
        words = content_words("Alice in her blue apron runs", minus=["blue apron dress"])
        self.assertEqual(words, content_words("Alice runs"))

    def test_jaccard_of_nothing_is_zero(self):
        self.assertEqual(jaccard([], ["a"]), 0.0)


class TestForbiddenPhrases(unittest.TestCase):
    def test_regular_verbs_become_what_not_to_do_again(self):
        self.assertEqual(forbidden_phrase("opened the brass lock"),
                         "opening the brass lock again")
        self.assertEqual(forbidden_phrase("carried the lamp"), "carrying the lamp again")
        self.assertEqual(forbidden_phrase("scanned the room"), "scanning the room again")

    def test_irregular_verbs_still_read(self):
        self.assertEqual(forbidden_phrase("took the key"), "took the key again")

    def test_again_is_not_doubled(self):
        self.assertEqual(forbidden_phrase("waved again"), "waving again")


class TestTheBlacklist(unittest.TestCase):
    def test_a_near_duplicate_refreshes_instead_of_taking_a_slot(self):
        blacklist = ActionBlacklist(size=4, ttl=2)
        self.assertTrue(blacklist.add("opening the gate again", 1))
        self.assertFalse(blacklist.add("opening the gates again", 3))
        self.assertEqual(len(blacklist.entries), 1)
        self.assertEqual(blacklist.entries[0].expires_beat, 5)

    def test_a_rewording_of_a_forbidden_action_is_caught(self):
        blacklist = ActionBlacklist()
        blacklist.add("opening the brass lock again", 1)
        self.assertTrue(blacklist.violated_by("Alice slowly opens the brass lock once more"))
        self.assertFalse(blacklist.violated_by("Alice runs along the hedge toward the tower"))

    def test_a_one_word_phrase_never_forbids_a_whole_vocabulary(self):
        blacklist = ActionBlacklist()
        blacklist.add("walking again", 1)
        self.assertEqual(blacklist.violated_by("Alice walks to the tower"), [])

    def test_round_trip(self):
        blacklist = ActionBlacklist(size=5, ttl=3)
        blacklist.add("opening the gate again", 2)
        again = ActionBlacklist.from_list(blacklist.to_list(), size=5, ttl=3)
        self.assertEqual(again.to_list(), blacklist.to_list())


class TestCheckPlan(unittest.TestCase):
    def test_a_fresh_beat_passes(self):
        memory = _memory()
        _commit(memory, 1, "Alice unlocks the brass gate", ["unlocked the brass gate"])
        plan = _plan(2, "@Alice sprints down the hedge corridor toward the clock tower",
                     beat="Alice races toward the clock tower")
        self.assertEqual(check_plan(plan, memory), [])

    def test_a_reworded_beat_is_refused(self):
        memory = _memory()
        _commit(memory, 1, "Alice unlocks the brass gate", ["unlocked the brass gate"])
        plan = _plan(2, "@Alice unlocks the gate made of brass",
                     beat="Alice unlocks the brass gate carefully")
        problems = check_plan(plan, memory)
        self.assertTrue(any("restates beat 1" in p for p in problems), problems)

    def test_restating_the_cast_and_setting_is_not_a_repeat(self):
        """Every prompt repeats the pinned look and the place verbatim, on purpose."""
        memory = _memory()
        _commit(memory, 1, "Alice unlocks the brass gate", ["unlocked the brass gate"])
        plan = _plan(2, "@Alice, a girl in a torn blue apron dress, disheveled hair, in the "
                        "overgrown clock garden at twilight, climbs the ivy wall",
                     beat="Alice climbs the ivy wall")
        self.assertEqual(check_plan(plan, memory), [])

    def test_the_same_prompt_twice_is_refused(self):
        memory = _memory()
        _commit(memory, 1, "Alice unlocks the brass gate")
        plan = _plan(2, memory.last_prompt, beat="something new entirely")
        self.assertTrue(any("word for word" in p for p in check_plan(plan, memory)))

    def test_restaging_a_completed_action_is_refused(self):
        memory = _memory()
        _commit(memory, 1, "Alice unlocks the gate", ["opened the brass lock"])
        plan = _plan(2, "@Alice opens the brass lock again with shaking hands",
                     beat="Alice tries the lock")
        self.assertTrue(any("forbidden action" in p for p in check_plan(plan, memory)))

    def test_a_spent_scene_must_cut(self):
        memory = _memory()
        for beat in range(1, 9):
            _commit(memory, beat, "Alice does new thing number %d in sequence" % beat,
                    ["did thing %d" % beat], scene="Garden" if beat == 1 else "")
        plan = _plan(9, "@Alice reaches the far hedge", beat="Alice reaches the far hedge")
        self.assertTrue(any("budget" in p for p in check_plan(plan, memory)))
        cut = _plan(9, "@Alice inside the tower stairwell", beat="Alice enters the tower",
                    scene="Tower", scene_change=True)
        self.assertEqual(check_plan(cut, memory), [])

    def test_a_stalled_film_must_change_something(self):
        memory = _memory()
        _commit(memory, 1, "Alice hesitates at the gate")
        _commit(memory, 2, "Alice hesitates near the gate")
        self.assertTrue(memory.stagnant)
        same = _plan(3, "@Alice lingers at the gate", beat="Alice hesitates by the gate")
        self.assertTrue(any("changed nothing" in p for p in check_plan(same, memory)))
        moving = _plan(3, "@Alice kicks the gate down and runs",
                       beat="Alice breaks out of the garden")
        self.assertEqual(check_plan(moving, memory), [])


if __name__ == "__main__":
    unittest.main()
