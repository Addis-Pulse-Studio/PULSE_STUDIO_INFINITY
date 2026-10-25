"""The director's side of the desk: tools/pulse_desk.py and the workspace kit.

An agent answers ComfyUI's turns with this tool, so it is tested as an agent
uses it: wait for a turn, read the brief, submit an answer that is refused, fix
it, submit again. And the installer that writes the agent's workspace is tested
for the two things that matter there -- every placeholder filled, and nothing of
the person's own overwritten.
"""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from comfyui_pulse_studio.pulse_director import KIND_OPEN, Mailbox, TurnRequest

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DESK = _tool("pulse_desk")
SETUP = _tool("setup_director_workspace")

GOOD = {"plan": {"shot_index": 1, "visual_prompt": "@Alice wakes beneath the clock.",
                 "negative_prompt": "", "camera_motion": "slow push in",
                 "duration_seconds": 5.17, "dialogue": None,
                 "continuity_anchors": {"held_items": [], "costume": "", "environment": ""}}}


class DeskCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.session = os.path.join(self.root, "garden")
        os.makedirs(self.session)
        Path(self.session, "session.json").write_text(json.dumps(
            {"uid": "u", "status": "active", "beat": 0, "pending_beat": 1}))
        self.box = Mailbox(self.session)
        self.box.post(TurnRequest(KIND_OPEN, "garden", 1, story_goal="Escape the labyrinth",
                                  cast=[{"name": "Alice", "description": "a girl"}],
                                  constraints={"cast": ["Alice"], "may_end_story": False}),
                      {"backend": "claude_code", "capabilities": {"sees_frames": True}})

    def tearDown(self):
        self._tmp.cleanup()

    def run_desk(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = DESK.main(["--root", self.root] + list(argv))
        return code, out.getvalue()

    def answer_file(self, data):
        path = os.path.join(self.root, "answer.json")
        Path(path).write_text(json.dumps(data))
        return path


class TestWaitingAndReading(DeskCase):
    def test_next_prints_the_turn_as_an_agent_should_read_it(self):
        code, out = self.run_desk("next")
        self.assertEqual(code, 0)
        for fragment in ("TURN b0001-open-a1", "write the opening beat",
                         "Escape the labyrinth", "@Alice: a girl", "submit b0001-open-a1"):
            self.assertIn(fragment, out)

    def test_next_says_a_director_is_at_the_desk(self):
        self.run_desk("next")
        self.assertIsNotNone(self.box.heartbeat_age())

    def test_nothing_waiting_is_its_own_exit_code(self):
        self.box.submit("b0001-open-a1", GOOD)
        code, out = self.run_desk("next")
        self.assertEqual(code, 3)
        self.assertIn("no turn waiting", out)

    def test_json_mode_adds_absolute_paths(self):
        code, out = self.run_desk("next", "--json")
        message = json.loads(out)
        self.assertEqual(message["turn_id"], "b0001-open-a1")
        self.assertEqual(message["session_root"], self.session)


class TestAnswering(DeskCase):
    def test_a_broken_answer_is_refused_before_it_reaches_comfyui(self):
        bad = json.loads(json.dumps(GOOD))
        bad["plan"]["visual_prompt"] = "<Picture 1> wakes up"
        code, out = self.run_desk("submit", "b0001-open-a1", self.answer_file(bad))
        self.assertEqual(code, 1)
        self.assertIn("REJECTED", out)
        self.assertIn("<Picture 1>", out)
        self.assertFalse(os.path.exists(self.box.outbox_path("b0001-open-a1")))

    def test_a_good_answer_lands_on_the_desk(self):
        code, out = self.run_desk("submit", "b0001-open-a1", self.answer_file(GOOD))
        self.assertEqual(code, 0, out)
        with open(self.box.outbox_path("b0001-open-a1"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), GOOD)

    def test_prose_around_the_json_is_tolerated(self):
        path = os.path.join(self.root, "answer.txt")
        Path(path).write_text("Here is beat one:\n```json\n%s\n```" % json.dumps(GOOD))
        self.assertEqual(self.run_desk("submit", "b0001-open-a1", path)[0], 0)

    def test_ending_the_story_when_it_may_not_end_is_refused(self):
        early = json.loads(json.dumps(GOOD))
        early["plan"]["end_of_story"] = True
        code, out = self.run_desk("submit", "b0001-open-a1", self.answer_file(early))
        self.assertEqual(code, 1)
        self.assertIn("end_of_story", out)

    def test_control_and_schema(self):
        code, out = self.run_desk("control", "pause")
        self.assertEqual(code, 0)
        with open(os.path.join(self.session, "control.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["action"], "pause")
        code, out = self.run_desk("schema", "turn")
        self.assertEqual(json.loads(out)["required"], ["observation", "plan"])


class TestTheWorkspaceKit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = self._tmp.name
        self.sessions = "/mnt/d/AI/ComfyUI/output/pulse_infinity"

    def tearDown(self):
        self._tmp.cleanup()

    def install(self, **kwargs):
        return SETUP.install(self.workspace, self.sessions, out=lambda *_: None, **kwargs)

    def test_every_file_is_written_with_no_placeholder_left(self):
        written = self.install()
        for rel in ("CLAUDE.md", "AGENTS.md", ".claude/skills/pulse-director/SKILL.md",
                    ".agent/rules/pulse-director.md", ".agent/workflows/pulse-direct.md",
                    ".claude/settings.json", ".pulse_desk.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.workspace, rel)), rel)
        for path in written:
            self.assertNotIn("{{", Path(path).read_text(encoding="utf-8"), path)

    def test_the_skill_carries_the_same_brief_the_cli_directors_get(self):
        from comfyui_pulse_studio.pulse_director import DIRECTOR_BRIEF
        self.install()
        skill = Path(self.workspace, ".claude/skills/pulse-director/SKILL.md").read_text(
            encoding="utf-8")
        self.assertIn(DIRECTOR_BRIEF.strip().splitlines()[0], skill)
        self.assertTrue(skill.startswith("---\nname: pulse-director\n"))

    def test_antigravity_gets_windows_paths(self):
        self.install()
        agents = Path(self.workspace, "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("D:\\AI\\ComfyUI\\output\\pulse_infinity", agents)

    def test_settings_are_merged_not_replaced(self):
        os.makedirs(os.path.join(self.workspace, ".claude"))
        Path(self.workspace, ".claude", "settings.json").write_text(json.dumps(
            {"model": "opus", "permissions": {"allow": ["Bash(git status)"]}}))
        self.install()
        data = json.loads(Path(self.workspace, ".claude", "settings.json").read_text())
        self.assertEqual(data["model"], "opus")
        self.assertIn("Bash(git status)", data["permissions"]["allow"])
        self.assertIn(self.sessions, data["permissions"]["additionalDirectories"])
        self.assertIn("Read(/%s/**)" % self.sessions, data["permissions"]["allow"])
        self.install()   # twice changes nothing
        again = json.loads(Path(self.workspace, ".claude", "settings.json").read_text())
        self.assertEqual(again, data)

    def test_a_persons_own_file_is_kept_unless_forced(self):
        Path(self.workspace, "CLAUDE.md").write_text("my own notes")
        self.install()
        self.assertEqual(Path(self.workspace, "CLAUDE.md").read_text(), "my own notes")
        self.install(force=True)
        self.assertIn("director workspace", Path(self.workspace, "CLAUDE.md").read_text())

    def test_the_desk_finds_the_sessions_from_the_workspace(self):
        self.install()
        with open(os.path.join(self.workspace, ".pulse_desk.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["sessions_root"], self.sessions)


if __name__ == "__main__":
    unittest.main()
