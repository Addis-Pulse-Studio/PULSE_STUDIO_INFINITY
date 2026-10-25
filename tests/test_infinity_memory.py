"""Scene memory: the ledger, the history, and how it survives on disk.

The memory is the only thing a generated film remembers between beats -- H3
renders every beat from its prompt and one carried frame. So these tests are
about the two ways it can fail a film: remembering wrongly (an intention taken
for a fact, a pinned costume overwritten, one character split into two) and
forgetting (a crash that empties it, a history that grows until the director's
prompt drowns).
"""

import json
import os
import tempfile
import unittest

from comfyui_pulse_studio.pulse_memory import (
    MEMORY_FILE,
    PROVENANCE_INTENDED,
    PROVENANCE_OBSERVED,
    DirectedShotPlan,
    ObservationResult,
    SceneMemory,
    load_memory,
    restore_snapshot,
    save_memory,
    snapshot_memory,
)
from comfyui_pulse_studio.pulse_memory import scene_memory as scene_memory_module

SPEC_MEMORY = {
    "story_goal": "Escape the labyrinth before sundown",
    "current_beat": 3,
    "characters": {"Alice": {"appearance": "blue apron dress, disheveled hair",
                             "inventory": ["brass key"],
                             "last_posture": "facing north, midway through garden gate"}},
    "environment": {"setting": "clock garden", "time_of_day": "dusk",
                    "lighting": "amber twilight with deep shadows"},
    "action_history": ["opened the brass lock", "pushed the heavy gate open"],
    "forbidden_actions": ["looking at key again", "opening the gate again",
                          "standing stationary at entrance"],
}


def _plan(index, beat="", **extra):
    data = {"shot_index": index, "visual_prompt": "@Alice moves on. " + beat,
            "negative_prompt": "", "camera_motion": "", "duration_seconds": 5.17,
            "dialogue": None, "continuity_anchors": {"held_items": [], "costume": "",
                                                     "environment": ""}}
    data.update(beat=beat, **extra)
    plan, problems = DirectedShotPlan.from_dict(data)
    assert not problems, problems
    return plan


def _seen(index, actions=(), **extra):
    data = {"shot_index": index, "summary": "", "completed_actions": list(actions)}
    data.update(extra)
    obs, problems = ObservationResult.from_dict(data)
    assert not problems, problems
    return obs


class TestTheContractIsKept(unittest.TestCase):
    def test_the_spec_example_reads_back_on_every_spec_key(self):
        out = SceneMemory.from_dict(SPEC_MEMORY).to_dict()
        pulse = out.pop("pulse")
        self.assertEqual(out, SPEC_MEMORY)
        self.assertEqual(pulse["memory_schema"], 1)

    def test_everything_the_loop_adds_lives_under_one_key(self):
        out = SceneMemory.from_dict(SPEC_MEMORY).to_dict()
        self.assertEqual(set(out) - set(SPEC_MEMORY), {"pulse"})

    def test_a_memory_this_module_wrote_reads_back_whole(self):
        memory = SceneMemory.from_dict(SPEC_MEMORY)
        memory.commit(_plan(4, "Alice runs for the clock tower", scene="Garden"),
                      _seen(4, ["ran for the clock tower"]))
        again = SceneMemory.from_dict(json.loads(json.dumps(memory.to_dict())))
        self.assertEqual(again.to_dict(), memory.to_dict())


class TestTheLedger(unittest.TestCase):
    def setUp(self):
        self.memory = SceneMemory("Escape the labyrinth")
        self.memory.seed_cast([("Alice", "a girl in a torn blue apron dress")])

    def test_a_cast_description_is_the_pinned_identity(self):
        alice = self.memory.characters["Alice"]
        self.assertEqual(alice.pinned_appearance, "a girl in a torn blue apron dress")
        self.assertTrue(alice.cast)

    def test_an_observed_change_moves_the_look_but_not_the_identity(self):
        self.memory.commit(_plan(1, "Alice tears her sleeve on the hedge"), _seen(
            1, ["tore her sleeve on the hedge"],
            characters={"Alice": {"appearance_change": "apron dress, left sleeve torn"}}))
        alice = self.memory.characters["Alice"]
        self.assertEqual(alice.appearance, "apron dress, left sleeve torn")
        self.assertEqual(alice.identity, "a girl in a torn blue apron dress")

    def test_variants_of_a_name_fold_into_one_entity(self):
        for variant in ("alice", "@Alice", "Alice.", "Alice Liddell", "Allice"):
            self.assertEqual(self.memory.resolve_name(variant), "Alice", variant)
        self.assertIsNone(self.memory.resolve_name("Bob"))

    def test_inventory_is_gained_and_lost(self):
        self.memory.commit(_plan(1, "Alice picks up the brass key"), _seen(
            1, ["picked up the brass key"], characters={"Alice": {"inventory_add": ["brass key"]}}))
        self.memory.commit(_plan(2, "Alice drops the key in the fountain"), _seen(
            2, ["dropped the key in the fountain"],
            characters={"Alice": {"inventory_remove": ["the brass key"]}}))
        self.assertEqual(self.memory.characters["Alice"].inventory, [])

    def test_injuries_accumulate_and_mood_is_replaced(self):
        self.memory.commit(_plan(1, "a"), _seen(1, ["x"], characters={
            "Alice": {"injuries": ["scraped knee"], "emotional_state": "anxious"}}))
        self.memory.commit(_plan(2, "b"), _seen(2, ["y"], characters={
            "Alice": {"injuries": ["bruised wrist"], "emotional_state": "determined"}}))
        alice = self.memory.characters["Alice"]
        self.assertEqual(alice.injuries, ["scraped knee", "bruised wrist"])
        self.assertEqual(alice.emotional_state, "determined")

    def test_the_spatial_topology_updates_only_what_changed(self):
        self.memory.environment.setting = "clock garden"
        self.memory.commit(_plan(1, "dusk falls"), _seen(1, ["the sun set"], environment={
            "lighting": "deep blue dusk", "camera_orientation": "behind Alice, low",
            "landmarks": ["the stopped clock tower"]}))
        env = self.memory.environment
        self.assertEqual(env.setting, "clock garden")
        self.assertEqual(env.lighting, "deep blue dusk")
        self.assertEqual(env.landmarks, ["the stopped clock tower"])

    def test_an_intention_is_never_recorded_as_a_fact(self):
        self.memory.commit(_plan(1, "Alice finds the exit", characters_present=["Alice"]))
        self.assertEqual(self.memory.characters["Alice"].provenance, PROVENANCE_INTENDED)
        self.assertFalse(self.memory.view_for_director()["recent_intentions"][0]["verified"])
        self.memory.commit(_plan(2, "Alice climbs the wall"),
                           _seen(2, ["climbed the wall"], characters={"Alice": {}}))
        self.assertEqual(self.memory.characters["Alice"].provenance, PROVENANCE_OBSERVED)

    def test_new_characters_are_pinned_on_arrival(self):
        self.memory.commit(_plan(1, "the Hatter arrives", new_characters=[
            {"name": "Hatter", "appearance": "tall green top hat, patched coat"}]))
        self.assertEqual(self.memory.characters["Hatter"].pinned_appearance,
                         "tall green top hat, patched coat")


class TestTheHistory(unittest.TestCase):
    def test_completed_actions_become_forbidden_for_a_while(self):
        memory = SceneMemory("goal", blacklist_size=8, blacklist_ttl=4)
        memory.commit(_plan(1, "open the lock"), _seen(1, ["opened the brass lock"]))
        self.assertEqual(memory.forbidden_actions, ["opening the brass lock again"])
        for beat in range(2, 5):
            memory.commit(_plan(beat, "beat %d" % beat), _seen(beat, ["did thing %d" % beat]))
        self.assertIn("opening the brass lock again", memory.forbidden_actions)
        memory.commit(_plan(5, "beat 5"), _seen(5, ["did thing 5"]))
        self.assertNotIn("opening the brass lock again", memory.forbidden_actions,
                         "an entry lapses ttl beats after it was added")

    def test_the_blacklist_is_a_ring(self):
        memory = SceneMemory("goal", blacklist_size=3, blacklist_ttl=99)
        for beat, verb in enumerate(("opened", "closed", "lifted", "dropped", "kicked"), 1):
            memory.commit(_plan(beat, verb), _seen(beat, ["%s the crate" % verb]))
        self.assertEqual(len(memory.forbidden_actions), 3)
        self.assertEqual(memory.forbidden_actions[0], "kicking the crate again")

    def test_nobody_moving_forbids_standing_still_there(self):
        memory = SceneMemory("goal")
        memory.environment.setting = "the garden gate"
        memory.commit(_plan(1, "waits"), _seen(1, [], stationary=True))
        self.assertIn("standing still at the garden gate", memory.forbidden_actions)

    def test_two_beats_without_progress_is_a_stall(self):
        memory = SceneMemory("goal")
        memory.commit(_plan(1, "a"), _seen(1, []))
        self.assertFalse(memory.stagnant)
        notes = memory.commit(_plan(2, "b"), _seen(2, []))
        self.assertTrue(memory.stagnant)
        self.assertTrue(any("changed nothing" in n for n in notes))
        memory.commit(_plan(3, "c"), _seen(3, ["broke the window"]))
        self.assertFalse(memory.stagnant)

    def test_scenes_are_counted_and_a_cut_starts_a_new_one(self):
        memory = SceneMemory("goal")
        memory.commit(_plan(1, "a", scene="Garden"), _seen(1, ["x"]))
        memory.commit(_plan(2, "b"), _seen(2, ["y"]))
        self.assertEqual((memory.scene, memory.scene_length), ("Garden", 2))
        memory.commit(_plan(3, "c", scene="Tower", scene_change=True), _seen(3, ["z"]))
        self.assertEqual((memory.scene, memory.scene_length), ("Tower", 1))

    def test_a_long_history_folds_into_chapters_and_the_view_stays_bounded(self):
        memory = SceneMemory("goal")
        for beat in range(1, 60):
            memory.commit(_plan(beat, "beat %d" % beat), _seen(beat, ["event %d" % beat]))
        self.assertTrue(memory.chapters)
        self.assertLessEqual(len(memory.action_history),
                             scene_memory_module.COMPACT_TRIGGER)
        view = memory.view_for_director()
        self.assertLessEqual(len(view["recent_events"]), 8)
        self.assertLessEqual(len(view["recent_intentions"]), 8)
        self.assertLess(len(memory.view_text()), 4000, "the director's prompt must not "
                        "grow with the film")

    def test_the_prose_view_names_what_matters(self):
        text = SceneMemory.from_dict(SPEC_MEMORY).view_text()
        for fragment in ("Escape the labyrinth", "@Alice", "holding brass key",
                         "clock garden", "Forbidden now: looking at key again"):
            self.assertIn(fragment, text)


class TestOnDisk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load(self):
        memory = SceneMemory.from_dict(SPEC_MEMORY)
        save_memory(self.directory, memory)
        loaded, notes = load_memory(self.directory)
        self.assertEqual(notes, [])
        self.assertEqual(loaded.to_dict(), memory.to_dict())

    def test_no_file_is_a_fresh_memory(self):
        memory, notes = load_memory(self.directory, story_goal="goal")
        self.assertEqual((memory.current_beat, memory.story_goal, notes), (0, "goal", []))

    def test_a_snapshot_restores_the_memory_a_beat_was_planned_from(self):
        memory = SceneMemory.from_dict(SPEC_MEMORY)
        snapshot_memory(self.directory, memory, 4)
        memory.commit(_plan(4, "run"), _seen(4, ["ran away"]))
        before = restore_snapshot(self.directory, 4)
        self.assertEqual(before.current_beat, 3)
        self.assertNotIn("ran away", before.action_history)
        self.assertIsNone(restore_snapshot(self.directory, 9))

    def test_a_corrupt_file_is_kept_and_the_newest_snapshot_loaded(self):
        memory = SceneMemory.from_dict(SPEC_MEMORY)
        snapshot_memory(self.directory, memory, 4)
        with open(os.path.join(self.directory, MEMORY_FILE), "w", encoding="utf-8") as fh:
            fh.write("{ half a file")
        loaded, notes = load_memory(self.directory)
        self.assertEqual(loaded.current_beat, 3)
        self.assertIn("scene_memory.corrupt.json", notes[0])
        self.assertTrue(os.path.exists(os.path.join(self.directory,
                                                    "scene_memory.corrupt.json")))


if __name__ == "__main__":
    unittest.main()
