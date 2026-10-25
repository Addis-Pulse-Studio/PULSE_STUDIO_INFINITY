"""The closed loop, end to end, with no GPU: plan, render, observe, commit, repeat.

The render is a stand-in -- a record of segments with a cache key, which is all
the session ever reads from one -- and the director is the scripted backend, so
the whole loop runs in milliseconds. What is under test is everything that has
to hold for a film of a hundred beats rendered overnight: state that survives a
crash, a node ComfyUI runs twice that commits once, an undo that leaves no trace
of what it undid.
"""

import json
import os
import tempfile
import unittest

from comfyui_pulse_studio.infinity_session import (
    CONTROL_FILE,
    SESSION_FILE,
    STATUS_COMPLETE,
    STATUS_PAUSED,
    InfinitySession,
    SessionError,
    peek_revision,
    safe_session_name,
)
from comfyui_pulse_studio.pulse_director import EvidencePack, InfinityConfig
from comfyui_pulse_studio.pulse_director.scripted_provider import ScriptedDirector
from comfyui_pulse_studio.pulse_memory import MEMORY_FILE

SCRIPT = "\n---\n".join([
    "beat: Alice leaves the clock behind\nscene: Clock garden\n"
    "visual: @Alice turns away from the stopped clock onto the garden path.\n"
    "hold: brass key in left hand\nwhere: overgrown clock garden, twilight",
    "beat: Alice reaches the hedge gate\n"
    "visual: @Alice hurries along the path to an iron gate in the hedge.",
    "beat: Alice unlocks the gate and slips through\n"
    "visual: @Alice turns the brass key in the lock and slips into the maze.",
    "beat: Alice climbs the maze wall\nvisual: @Alice scrambles up the ivy onto the wall.",
    "beat: Alice sees the exit tower\n"
    "visual: @Alice stands on the wall top and spots a lit tower beyond the hedges.",
    "beat: Alice jumps down toward the tower\n"
    "visual: @Alice leaps from the wall into soft grass and runs for the tower.",
])
CAST = [("Alice", "a girl in a torn blue apron dress")]


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class _Counting(ScriptedDirector):
    calls = 0

    def answer(self, request):
        type(self).calls += 1
        return super().answer(request)


class SessionCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output = self._tmp.name
        self.config = InfinityConfig.resolve({}, "scripted", {})
        _Counting.calls = 0

    def tearDown(self):
        self._tmp.cleanup()

    def session(self, name="Clock Garden!", max_beats=0):
        session = InfinitySession(self.output, name, self.config)
        session.configure("Escape the labyrinth before sundown", cast=CAST)
        session.set_limits(max_beats)
        return session

    def director(self, session, script=SCRIPT, cls=_Counting):
        return cls({"script": script}, session.root, session=session.name)

    def render(self, session, key=None):
        beat, attempt = session.pending_beat, session.attempt
        run_dir, run_id = session.run_location()
        record = {"last_key": key or ("%02x" % beat) * 32, "seconds": 5.17, "fps": 24,
                  "width": 1344, "height": 768,
                  "segments": [{"video": "%s/%s/seg.mp4" % (run_dir, run_id),
                                "audio": "", "frames": 124}]}
        session.record_render(beat, attempt, record)
        evidence = EvidencePack(session.root, beat,
                                last_frame="beats/beat_%04d/a%d/evidence/last.png" % (beat, attempt))
        return record, evidence

    def beat(self, session, director):
        plan = session.plan_for_pending(director)
        record, evidence = self.render(session)
        return plan, session.commit_rendered(director, evidence, record)


class TestTheLoop(SessionCase):
    def test_six_beats_play_through_and_the_film_completes(self):
        session = self.session()
        director = self.director(session)
        plans = []
        for _ in range(6):
            plan, outcome = self.beat(session, director)
            plans.append(plan.shot_index)
        self.assertEqual(plans, [1, 2, 3, 4, 5, 6])
        self.assertTrue(outcome.complete)
        self.assertEqual(session.status, STATUS_COMPLETE)
        self.assertEqual([r["beat"] for r in session.state["reel"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(session.memory.current_beat, 6)
        self.assertIn("brass key in left hand", session.memory.characters["Alice"].inventory)

    def test_the_session_is_a_folder_a_person_can_read(self):
        session = self.session()
        self.beat(session, self.director(session))
        for rel in (SESSION_FILE, MEMORY_FILE, "beats/beat_0001/plan.json",
                    "beats/beat_0001/a1/beat.json", "beats/beat_0002/plan.json",
                    "schemas/turn.schema.json", "memory_snapshots/beat_0001.json"):
            self.assertTrue(os.path.exists(os.path.join(session.root, rel)), rel)
        journal = os.listdir(os.path.join(session.root, "journal"))
        self.assertTrue(any(n.endswith(".accepted.json") for n in journal))
        self.assertEqual(os.path.basename(session.root), "Clock_Garden")

    def test_state_survives_a_restart(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        again = InfinitySession(self.output, "Clock Garden!", self.config)
        self.assertEqual(again.state["beat"], 1)
        self.assertEqual(again.pending_beat, 2)
        self.assertEqual(again.memory.to_dict(), session.memory.to_dict())
        self.assertEqual(again.plan_for_pending(director).shot_index, 2)

    def test_each_beat_continues_the_one_before(self):
        session = self.session()
        director = self.director(session)
        self.assertIsNone(session.continue_from())
        self.beat(session, director)
        carry = session.continue_from()
        self.assertEqual(carry, {"run_dir": "pulse_infinity/Clock_Garden/beats/beat_0001",
                                 "run_id": "a1", "cache_key": "01" * 32})

    def test_the_opening_can_be_written_by_hand(self):
        session = self.session()
        director = self.director(session)
        plan = session.plan_for_pending(director, opening=(
            "visual: @Alice wakes beneath the stopped clock\ncamera: slow push in"))
        self.assertEqual(plan.visual_prompt, "@Alice wakes beneath the stopped clock")
        self.assertEqual(_Counting.calls, 0, "a written opening asks the director nothing")

    def test_an_opening_that_breaks_the_contract_is_refused(self):
        session = self.session()
        with self.assertRaises(SessionError):
            session.plan_for_pending(self.director(session), opening="visual: <Picture 1> waits")

    def test_max_beats_ends_the_film_on_an_observation(self):
        session = self.session(max_beats=2)
        director = self.director(session)
        self.beat(session, director)
        _, outcome = self.beat(session, director)
        self.assertTrue(outcome.complete)
        self.assertIsNone(outcome.next_plan)
        self.assertEqual(session.status, STATUS_COMPLETE)


class TestNothingHappensTwice(SessionCase):
    def test_committing_the_same_render_again_changes_nothing(self):
        """ComfyUI's cache may evict the Observer and run it again in the same queue.

        By then the session has moved on to the next beat -- which is exactly what
        once made the second run commit beat 1's render as beat 2. The render says
        which beat it is; the session's pending beat does not.
        """
        session = self.session()
        director = self.director(session)
        session.plan_for_pending(director)
        record, evidence = self.render(session)
        session.commit_rendered(director, evidence, record, beat=1, attempt=1)
        calls, history = _Counting.calls, list(session.memory.action_history)
        for again_session in (session, InfinitySession(self.output, "Clock Garden!", self.config)):
            again = again_session.commit_rendered(director, evidence, record, beat=1, attempt=1)
            self.assertTrue(again.replayed)
            self.assertEqual(again_session.state["beat"], 1)
            self.assertEqual(again_session.pending_beat, 2)
            self.assertEqual(again_session.memory.action_history, history)
            self.assertEqual(len(again_session.state["reel"]), 1)
        self.assertEqual(_Counting.calls, calls)

    def test_the_render_record_says_which_beat_it_is(self):
        """record_render stamps beat and attempt into the record on disk."""
        session = self.session()
        director = self.director(session)
        session.plan_for_pending(director)
        self.render(session)
        on_disk = session.render_record(1, 1)
        self.assertEqual((on_disk["beat"], on_disk["attempt"]), (1, 1))
        outcome = session.commit_rendered(director, EvidencePack(session.root, 1), on_disk)
        self.assertEqual(outcome.beat, 1)

    def test_a_render_the_session_has_moved_past_is_refused(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        session.plan_for_pending(director)
        stale, evidence = self.render(session)
        session.control("replan")
        with self.assertRaises(SessionError):
            session.commit_rendered(director, evidence, stale, beat=2, attempt=1)

    def test_a_crash_after_the_director_answered_replays_the_answer(self):
        session = self.session()
        director = self.director(session)
        session.plan_for_pending(director)
        record, evidence = self.render(session)
        saved = _read_json(os.path.join(session.root, SESSION_FILE))
        session.commit_rendered(director, evidence, record)
        calls = _Counting.calls
        # The crash: session.json and the memory roll back, the journal does not.
        with open(os.path.join(session.root, SESSION_FILE), "w") as fh:
            json.dump(saved, fh)
        os.unlink(os.path.join(session.root, MEMORY_FILE))
        resumed = InfinitySession(self.output, "Clock Garden!", self.config)
        resumed.configure("Escape the labyrinth before sundown", cast=CAST)
        resumed.commit_rendered(self.director(resumed), evidence, record)
        self.assertEqual(_Counting.calls, calls, "the journaled answer is replayed, not re-asked")
        self.assertEqual(resumed.state["beat"], 1)

    def test_a_crash_between_memory_and_session_commits_the_beat_once(self):
        session = self.session()
        director = self.director(session)
        session.plan_for_pending(director)
        record, evidence = self.render(session)
        saved = _read_json(os.path.join(session.root, SESSION_FILE))
        session.commit_rendered(director, evidence, record)
        once = list(session.memory.action_history)
        with open(os.path.join(session.root, SESSION_FILE), "w") as fh:
            json.dump(saved, fh)          # the memory kept the beat; the session did not
        resumed = InfinitySession(self.output, "Clock Garden!", self.config)
        resumed.commit_rendered(self.director(resumed), evidence, record)
        self.assertEqual(resumed.memory.action_history, once)
        self.assertEqual(resumed.memory.current_beat, 1)

    def test_a_committed_beat_cannot_be_committed_from_another_render(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        record, evidence = self.render(session, key="ff" * 32)
        with self.assertRaises(SessionError):
            session.commit_rendered(director, evidence, record, beat=1, attempt=1)


class TestUndo(SessionCase):
    def test_a_reroll_renders_the_same_plan_again_with_new_seeds(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        plan2, _ = self.beat(session, director)
        shot_before = session.shot_id(2, 1)
        message = session.control("reroll")
        self.assertIn("beat 2 will render again", message)
        self.assertEqual((session.state["beat"], session.pending_beat, session.attempt), (1, 2, 2))
        self.assertEqual(session.plan_for_pending(director).to_dict(), plan2.to_dict())
        self.assertNotEqual(session.shot_id(), shot_before)
        self.assertEqual(session.memory.current_beat, 1, "memory is back before beat 2")
        self.assertEqual([r["beat"] for r in session.state["reel"]], [1])
        self.assertEqual(session.run_location(), ("pulse_infinity/Clock_Garden/beats/beat_0002",
                                                  "a2"))

    def test_a_rerolled_beat_is_observed_afresh_not_replayed(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        self.beat(session, director)
        session.control("reroll")
        calls = _Counting.calls
        record, evidence = self.render(session, key="ab" * 32)
        session.commit_rendered(director, evidence, record)
        self.assertGreater(_Counting.calls, calls)

    def test_a_replan_asks_the_director_again_with_the_note(self):
        session = self.session()
        director = self.director(session)
        self.beat(session, director)
        session.control("replan", note="Too slow; make her run.")
        self.assertIsNone(session.state["pending_plan"])
        seen = []

        class Listening(_Counting):
            def answer(self, request):
                seen.append(request.guidance)
                return super().answer(request)

        plan = session.plan_for_pending(self.director(session, cls=Listening))
        self.assertEqual(plan.shot_index, 1)
        self.assertTrue(any("make her run" in g for g in seen))

    def test_pause_and_control_file(self):
        session = self.session()
        os.makedirs(session.root, exist_ok=True)
        with open(os.path.join(session.root, CONTROL_FILE), "w") as fh:
            json.dump({"action": "pause"}, fh)
        self.assertEqual(session.take_control(), ["paused after beat 0"])
        self.assertEqual(session.status, STATUS_PAUSED)
        self.assertFalse(os.path.exists(os.path.join(session.root, CONTROL_FILE)))
        self.assertIn("resumed", session.control("resume"))

    def test_the_revision_token_moves_on_every_change_and_names_the_session(self):
        session = self.session()
        director = self.director(session)
        session.save()
        before = peek_revision(session.root)
        self.beat(session, director)
        after = peek_revision(session.root)
        self.assertNotEqual(before, after)
        self.assertTrue(after.startswith(session.uid + ":"))
        self.assertEqual(peek_revision(os.path.join(self.output, "nowhere")), "new")


class TestTheSessionIsOnDiskBeforeItAsks(SessionCase):
    def test_session_json_exists_when_the_director_is_first_asked(self):
        """The director finds its first turn through the session's folder."""
        session = self.session()
        seen = []

        class Checking(_Counting):
            def answer(self, request):
                seen.append(os.path.exists(os.path.join(session.root, SESSION_FILE)))
                return super().answer(request)

        session.plan_for_pending(self.director(session, cls=Checking))
        self.assertEqual(seen, [True])


class TestNames(unittest.TestCase):
    def test_session_names_are_safe_folders_everywhere(self):
        self.assertEqual(safe_session_name("Clock Garden!"), "Clock_Garden")
        self.assertEqual(safe_session_name("../../etc"), "etc")
        self.assertEqual(safe_session_name(""), "session")
        self.assertEqual(safe_session_name("CON"), "CON_")
        self.assertLessEqual(len(safe_session_name("x" * 200)), 48)


if __name__ == "__main__":
    unittest.main()
