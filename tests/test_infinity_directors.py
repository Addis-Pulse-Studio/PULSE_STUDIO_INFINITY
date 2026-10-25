"""The pluggable director: the toggle, the desk, the CLI, and the one contract.

Every backend -- Claude Code at the desk, Gemini in Antigravity, Ollama and pi
through their CLIs, the script -- answers through the same `run_turn`, so the
tests drive that loop with stand-ins for the far side: a thread playing the
agent at the desk, a stub runner playing the CLI, and, once, a real child
process. Nothing here needs a model, a GPU or the network.
"""

import ast
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from comfyui_pulse_studio.pulse_director import (
    BACKENDS,
    FROM_CONFIG,
    KIND_OPEN,
    KIND_TURN,
    REGISTRY,
    DirectorCancelled,
    DirectorError,
    DirectorRejected,
    DirectorTimeout,
    EvidencePack,
    InfinityConfig,
    Mailbox,
    TurnRequest,
    convert_path,
    create_director,
    default_constraints,
    find_config_file,
    parse_config_text,
    parse_script,
    render_prompt,
    run_turn,
)
from comfyui_pulse_studio.pulse_director.cli_transport import CliResult, build_argv, run_cli
from comfyui_pulse_studio.pulse_director.mailbox import MAX_RESPONSE_BYTES
from comfyui_pulse_studio.pulse_memory import DirectedShotPlan, SceneMemory

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """\
beat: Alice leaves the clock behind
scene: Clock garden
visual: @Alice turns away from the stopped clock and steps onto the dark garden path.
camera: slow dolly forward, low angle
say: Alice | whispering, anxious | The time has already stopped...
hold: brass key in left hand
wear: torn blue apron dress
where: overgrown clock garden, twilight
seconds: 5.0
---
beat: Alice reaches the hedge gate
visual: @Alice hurries along the path to an iron gate set in the hedge.
---
beat: Alice unlocks the gate and slips through
visual: @Alice turns the brass key in the gate's lock and slips through into the maze.
"""


def _memory():
    memory = SceneMemory("Escape the labyrinth before sundown")
    memory.seed_cast([("Alice", "a girl in a torn blue apron dress")])
    return memory


def _plan_dict(index, visual, beat=""):
    return {"shot_index": index, "visual_prompt": visual, "negative_prompt": "",
            "camera_motion": "", "duration_seconds": 5.17, "dialogue": None,
            "continuity_anchors": {"held_items": [], "costume": "", "environment": ""},
            "beat": beat}


class TestTheToggle(unittest.TestCase):
    def test_the_widget_wins_then_the_environment_then_the_file(self):
        file_data = {"director_backend": "ollama"}
        env = {"PULSE_DIRECTOR_BACKEND": "pi"}
        self.assertEqual(InfinityConfig.resolve(file_data, "scripted", env).backend, "scripted")
        self.assertEqual(InfinityConfig.resolve(file_data, FROM_CONFIG, env).backend, "pi")
        self.assertEqual(InfinityConfig.resolve(file_data, FROM_CONFIG, {}).backend, "ollama")
        self.assertEqual(InfinityConfig.resolve({}, FROM_CONFIG, {}).backend, "claude_code")

    def test_the_report_says_where_the_choice_came_from(self):
        config = InfinityConfig.resolve({}, FROM_CONFIG, {"PULSE_DIRECTOR_BACKEND": "pi"})
        self.assertEqual(config.describe(), "pi (from $PULSE_DIRECTOR_BACKEND)")

    def test_an_unknown_name_falls_through_and_says_so(self):
        config = InfinityConfig.resolve({"director_backend": "gpt"}, FROM_CONFIG, {})
        self.assertEqual(config.backend, "claude_code")
        self.assertTrue(any("'gpt'" in p and "claude_code" in p for p in config.problems))

    def test_a_file_overrides_any_subset_of_the_defaults(self):
        config = InfinityConfig.resolve({"loop": {"scene_budget": 5},
                                         "backends": {"ollama": {"model": "qwen3.8:27b"}}},
                                        FROM_CONFIG, {})
        self.assertEqual(config.loop("scene_budget"), 5)
        self.assertEqual(config.loop("max_attempts"), 3)
        self.assertEqual(config.backend_settings("ollama"), {"model": "qwen3.8:27b"})

    def test_json_is_read_here_and_yaml_only_through_a_loader(self):
        self.assertEqual(parse_config_text('{"director_backend": "pi"}', "c.json")[0],
                         {"director_backend": "pi"})
        data, problems = parse_config_text("director_backend: pi\n", "c.yaml")
        self.assertEqual(data, {})
        self.assertIn("no YAML reader", problems[0])
        fake_yaml = lambda text: {"director_backend": text.split(":")[1].strip()}
        self.assertEqual(parse_config_text("director_backend: pi\n", "c.yaml", fake_yaml)[0],
                         {"director_backend": "pi"})
        self.assertTrue(parse_config_text("[1, 2]", "c.json")[1])

    def test_the_file_is_found_by_env_then_in_the_user_directory(self):
        with tempfile.TemporaryDirectory() as user:
            self.assertIsNone(find_config_file(user, {}))
            Path(user, "pulse_infinity.yaml").write_text("director_backend: pi\n")
            self.assertTrue(find_config_file(user, {}).endswith("pulse_infinity.yaml"))
            self.assertEqual(find_config_file(user, {"PULSE_INFINITY_CONFIG": "/x.yaml"}),
                             "/x.yaml")

    def test_every_backend_is_registered(self):
        self.assertEqual(tuple(REGISTRY), BACKENDS)


class TestTheBackends(unittest.TestCase):
    def test_who_is_a_cloud_agent_and_who_sees_what(self):
        self.assertTrue(REGISTRY["claude_code"].capabilities.cloud)
        self.assertFalse(REGISTRY["claude_code"].capabilities.sees_video)
        self.assertTrue(REGISTRY["gemini_antigravity"].capabilities.sees_video)
        self.assertTrue(REGISTRY["gemini_antigravity"].capabilities.hears_audio)
        self.assertFalse(REGISTRY["ollama"].capabilities.cloud)
        self.assertFalse(REGISTRY["scripted"].capabilities.sees_frames)

    def test_mailbox_is_the_default_for_the_agents_and_cli_for_the_local_ones(self):
        with tempfile.TemporaryDirectory() as root:
            config = InfinityConfig.resolve({}, FROM_CONFIG, {})
            self.assertEqual(create_director(config, root).transport_name, "mailbox")
            for name, transport in (("gemini_antigravity", "mailbox"), ("ollama", "cli"),
                                    ("pi", "cli")):
                config = InfinityConfig.resolve({}, name, {})
                self.assertEqual(create_director(config, root).transport_name, transport, name)

    def test_a_text_only_local_model_does_not_claim_to_see(self):
        with tempfile.TemporaryDirectory() as root:
            config = InfinityConfig.resolve({"backends": {"ollama": {"vision": False}}},
                                            "ollama", {})
            director = create_director(config, root)
            self.assertFalse(director.capabilities.sees_frames)
            self.assertTrue(REGISTRY["ollama"].capabilities.sees_frames,
                            "an instance setting must not change the class")

    def test_the_description_says_when_frames_leave_the_machine(self):
        with tempfile.TemporaryDirectory() as root:
            director = create_director(InfinityConfig.resolve({}, FROM_CONFIG, {}), root)
            self.assertIn("leave this machine", director.describe())
            local = create_director(InfinityConfig.resolve({}, "ollama", {}), root)
            self.assertIn("runs on this machine", local.describe())


class TestTheScript(unittest.TestCase):
    def test_the_plain_format_becomes_plans(self):
        plans, problems = parse_script(SCRIPT)
        self.assertEqual(problems, [])
        self.assertEqual(len(plans), 3)
        first = plans[0]
        self.assertEqual(first["dialogue"], {"speaker": "Alice", "emotion_tone":
                                             "whispering, anxious",
                                             "text": "The time has already stopped..."})
        self.assertEqual(first["continuity_anchors"]["held_items"], ["brass key in left hand"])
        self.assertEqual(first["scene"], "Clock garden")
        plan, plan_problems = DirectedShotPlan.from_dict(first, cast_names=["Alice"])
        self.assertEqual(plan_problems, [])

    def test_a_json_list_is_taken_as_it_is(self):
        plans, _ = parse_script(json.dumps([_plan_dict(1, "x")]))
        self.assertEqual(plans[0]["visual_prompt"], "x")

    def test_a_beat_with_no_visual_is_a_problem(self):
        self.assertTrue(parse_script("beat: nothing to see")[1])

    def test_the_script_runs_out(self):
        with tempfile.TemporaryDirectory() as root:
            director = create_director(InfinityConfig.resolve({}, "scripted", {}), root,
                                       overrides={"script": SCRIPT})
            self.assertTrue(director.can_continue(3))
            self.assertFalse(director.can_continue(4))


class TestTheTurnLoop(unittest.TestCase):
    """run_turn against the scripted director, and against a scripted liar."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _scripted(self, script=SCRIPT):
        return create_director(InfinityConfig.resolve({}, "scripted", {}), self.root,
                               session="garden", overrides={"script": script})

    def _request(self, memory, kind, beat, last_plan=None, may_end=False):
        return TurnRequest(kind, "garden", beat, story_goal=memory.story_goal,
                           memory=memory.view_for_director(), memory_text=memory.view_text(),
                           last_plan=last_plan, constraints=default_constraints(
                               memory, may_end=may_end))

    def test_the_opening_turn_plans_beat_one(self):
        memory = _memory()
        turn, notes = run_turn(self._scripted(), self._request(memory, KIND_OPEN, 1), memory)
        self.assertEqual(turn.plan.shot_index, 1)
        self.assertIsNone(turn.observation)
        self.assertTrue(any("124 frames" in n for n in notes))

    def test_a_turn_observes_then_plans(self):
        memory = _memory()
        director = self._scripted()
        first, _ = run_turn(director, self._request(memory, KIND_OPEN, 1), memory)
        memory.commit(first.plan)
        turn, _ = run_turn(director, self._request(memory, KIND_TURN, 2,
                                                   last_plan=first.plan.to_dict()),
                           memory, last_plan=first.plan)
        self.assertEqual((turn.observation.shot_index, turn.plan.shot_index), (1, 2))
        self.assertEqual(turn.observation.provenance, "intended")

    def test_a_rejected_answer_is_asked_again_with_the_reasons(self):
        memory = _memory()
        answers = [{"plan": _plan_dict(1, "<Picture 1> walks off")},
                   {"plan": _plan_dict(1, "@Alice walks off", beat="Alice leaves")}]
        seen = []

        class Liar(type(self._scripted())):
            def answer(self, request):
                seen.append(list(request.rejection))
                return answers[len(seen) - 1]

        director = Liar({"script": SCRIPT}, self.root, session="garden")
        turn, _ = run_turn(director, self._request(memory, KIND_OPEN, 1), memory)
        self.assertEqual(turn.plan.visual_prompt, "@Alice walks off")
        self.assertEqual(seen[0], [])
        self.assertTrue(any("<Picture 1>" in p for p in seen[1]))

    def test_every_attempt_failing_is_refused_with_the_last_reasons(self):
        memory = _memory()

        class Stubborn(type(self._scripted())):
            def answer(self, request):
                return {"plan": _plan_dict(1, "@Bob appears")}

        with self.assertRaises(DirectorRejected) as caught:
            run_turn(Stubborn({"script": SCRIPT}, self.root), self._request(memory, KIND_OPEN, 1),
                     memory, max_attempts=2)
        self.assertTrue(any("@Bob" in p for p in caught.exception.problems))

    def test_the_plan_is_judged_after_the_beat_it_follows(self):
        """Beat 1 completed 'opened the brass lock'; a beat 2 that opens it again
        is a restaging even though memory has not committed beat 1 yet."""
        memory = _memory()
        last, _ = DirectedShotPlan.from_dict(_plan_dict(1, "@Alice at the lock", "lock"))

        class Repeater(type(self._scripted())):
            def answer(self, request):
                return {"observation": {"shot_index": 1, "summary": "She opens it.",
                                        "completed_actions": ["opened the brass lock"]},
                        "plan": _plan_dict(2, "@Alice opens the brass lock again", "again")}

        with self.assertRaises(DirectorRejected) as caught:
            run_turn(Repeater({"script": SCRIPT}, self.root),
                     self._request(memory, KIND_TURN, 2, last_plan=last.to_dict()),
                     memory, last_plan=last, max_attempts=1)
        self.assertTrue(any("forbidden action" in p for p in caught.exception.problems))
        self.assertEqual(memory.current_beat, 0, "judging must not commit")

    def test_ending_early_is_refused(self):
        memory = _memory()

        class Quitter(type(self._scripted())):
            def answer(self, request):
                plan = _plan_dict(1, "@Alice sits down", "the end")
                plan["end_of_story"] = True
                return {"plan": plan}

        with self.assertRaises(DirectorRejected):
            run_turn(Quitter({"script": SCRIPT}, self.root),
                     self._request(memory, KIND_OPEN, 1, may_end=False), memory, max_attempts=1)

    def test_the_spec_interface_works_on_its_own(self):
        memory = _memory()
        director = self._scripted()
        plan = director.direct_next_beat(memory, memory.story_goal)
        self.assertEqual(plan.shot_index, 1)
        memory.commit(plan)
        evidence = EvidencePack(self.root, 2, last_frame="beats/beat_0002/a1/last.png")
        observation = director.inspect_shot("", evidence, memory)
        self.assertEqual(observation.shot_index, 2)


class TestTheDesk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _request(self, attempt=1):
        return TurnRequest(KIND_OPEN, "garden", 1, attempt=attempt, story_goal="goal")

    def test_an_agent_at_the_desk_answers(self):
        box = Mailbox(self.root, poll_seconds=0.01, timeout_seconds=10)
        request = self._request()
        box.post(request)

        def agent():
            agent_box = Mailbox(self.root)
            while not agent_box.pending():
                time.sleep(0.01)
            message = agent_box.pending()[0]
            agent_box.beat_heart("test agent")
            agent_box.submit(message["turn_id"], {"plan": _plan_dict(1, "@Alice runs")})

        thread = threading.Thread(target=agent)
        thread.start()
        answer = box.wait(request.turn_id)
        thread.join()
        self.assertEqual(answer["plan"]["visual_prompt"], "@Alice runs")
        self.assertIsNotNone(box.heartbeat_age())

    def test_the_request_says_where_to_reply_and_paths_are_relative(self):
        box = Mailbox(self.root)
        path = box.post(self._request())
        message = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(message["protocol"], "pulse-desk/1")
        self.assertEqual(message["reply_to"], "desk/outbox/b0001-open-a1.json")

    def test_posting_again_leaves_the_first_request_alone(self):
        box = Mailbox(self.root)
        path = box.post(self._request())
        before = Path(path).read_text(encoding="utf-8")
        time.sleep(1.1)
        box.post(self._request())
        self.assertEqual(Path(path).read_text(encoding="utf-8"), before)

    def test_an_answer_is_written_once(self):
        box = Mailbox(self.root)
        box.post(self._request())
        box.submit("b0001-open-a1", {"plan": {}})
        with self.assertRaises(FileExistsError):
            box.submit("b0001-open-a1", {"plan": {}})
        with self.assertRaises(FileNotFoundError):
            box.submit("b0009-open-a1", {"plan": {}})

    def test_nobody_answering_times_out_and_says_who_to_start(self):
        clock = [0.0]
        box = Mailbox(self.root, poll_seconds=1, timeout_seconds=30,
                      sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                      clock=lambda: clock[0])
        box.post(self._request())
        with self.assertRaises(DirectorTimeout) as caught:
            box.wait("b0001-open-a1")
        self.assertIn("/pulse-director", str(caught.exception))

    def test_the_interrupt_is_honoured(self):
        box = Mailbox(self.root, poll_seconds=0.01, timeout_seconds=0,
                      should_cancel=lambda: True)
        box.post(self._request())
        with self.assertRaises(DirectorCancelled):
            box.wait("b0001-open-a1")

    def test_prose_on_the_desk_comes_back_as_text_for_the_rejection(self):
        box = Mailbox(self.root, poll_seconds=0.01)
        box.post(self._request())
        Path(box.outbox_path("b0001-open-a1")).parent.mkdir(parents=True, exist_ok=True)
        Path(box.outbox_path("b0001-open-a1")).write_text("I think she should run.")
        self.assertEqual(box.wait("b0001-open-a1"), "I think she should run.")

    def test_a_runaway_answer_is_refused_by_size(self):
        box = Mailbox(self.root, poll_seconds=0.01)
        box.post(self._request())
        Path(box.outbox_path("b0001-open-a1")).parent.mkdir(parents=True, exist_ok=True)
        Path(box.outbox_path("b0001-open-a1")).write_text("x" * (MAX_RESPONSE_BYTES + 1))
        self.assertIn("at most", box.wait("b0001-open-a1"))

    def test_accepted_and_rejected_pairs_are_filed(self):
        box = Mailbox(self.root)
        for attempt, accepted in ((1, False), (2, True)):
            request = self._request(attempt)
            box.post(request)
            box.submit(request.turn_id, {"plan": {}})
            if accepted:
                box.accept(request.turn_id)
            else:
                box.reject(request.turn_id, ["bad tag"])
        self.assertEqual(box.pending(), [])
        done = sorted(os.listdir(os.path.join(self.root, "desk", "done")))
        rejected = sorted(os.listdir(os.path.join(self.root, "desk", "rejected")))
        self.assertIn("b0001-open-a2.response.json", done)
        self.assertIn("b0001-open-a1.verdict.json", rejected)


class _StubRunner:
    def __init__(self, stdout="", returncode=0):
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, argv, stdin_text, timeout, should_cancel=None, cwd=None):
        self.calls.append((argv, stdin_text))
        return CliResult(argv, self.stdout, "boom" if self.returncode else "",
                         self.returncode, 0.1)


class TestTheCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _ollama(self, runner, **settings):
        config = InfinityConfig.resolve({"backends": {"ollama": settings}}, "ollama", {})
        return create_director(config, self.root, session="garden", runner=runner)

    def test_the_prompt_goes_on_stdin_and_never_into_argv(self):
        runner = _StubRunner(json.dumps({"plan": _plan_dict(1, "@Alice runs")}))
        memory = _memory()
        request = TurnRequest(KIND_OPEN, "garden", 1, story_goal="goal",
                              constraints=default_constraints(memory, may_end=False))
        turn, _ = run_turn(self._ollama(runner, model="qwen3.8:27b"), request, memory)
        argv, stdin = runner.calls[0]
        # Checked against Ollama 0.35: no word-wrap escapes, no printed reasoning,
        # and the model unloaded as soon as it answers so H3 gets the card back.
        self.assertEqual(argv, ["ollama", "run", "qwen3.8:27b", "--format", "json",
                                "--nowordwrap", "--hidethinking", "--keepalive", "0s"])
        self.assertIn("You are the director", stdin)
        self.assertFalse(any("director" in a for a in argv))
        self.assertEqual(turn.plan.visual_prompt, "@Alice runs")
        self.assertTrue(os.path.exists(os.path.join(self.root, "desk", "cli",
                                                    "b0001-open-a1.prompt.txt")))

    def test_a_failing_cli_is_an_error_with_its_stderr(self):
        memory = _memory()
        with self.assertRaises(DirectorError) as caught:
            run_turn(self._ollama(_StubRunner(returncode=2)),
                     TurnRequest(KIND_OPEN, "garden", 1), memory)
        self.assertIn("boom", str(caught.exception))

    def test_images_expand_and_unknown_placeholders_are_refused(self):
        self.assertEqual(build_argv(["x", "{images}", "--m", "{model}"],
                                    {"images": ["a.png", "b.png"], "model": "m"}),
                         ["x", "a.png", "b.png", "--m", "m"])
        with self.assertRaises(DirectorError):
            build_argv(["x", "{nope}"], {"model": "m"})

    def test_evidence_paths_are_respelled_for_the_reader(self):
        self.assertEqual(convert_path("D:\\AI\\out\\a.png", "wsl"), "/mnt/d/AI/out/a.png")
        self.assertEqual(convert_path("/mnt/d/AI/out/a.png", "windows"), "D:\\AI\\out\\a.png")
        self.assertEqual(convert_path("/x/y", "native"), "/x/y")

    def test_the_evidence_and_the_rejection_reach_the_prompt(self):
        memory = _memory()
        evidence = EvidencePack(self.root, 1, last_frame="beats/b/last.png",
                                contact_sheet="beats/b/contact.png")
        request = TurnRequest(KIND_TURN, "garden", 2, evidence=evidence,
                              rejection=["shot_index must be 2"])
        caps = REGISTRY["ollama"].capabilities
        prompt = render_prompt(request, caps)
        self.assertIn(os.path.join(self.root, "beats", "b", "contact.png"), prompt)
        self.assertIn("shot_index must be 2", prompt)
        self.assertIn("First observe beat 1", prompt)
        self.assertIn('"observation"', prompt)

    def test_a_real_child_process_answers_on_stdout(self):
        echo = [sys.executable, "-c",
                "import sys, json; d = sys.stdin.read(); "
                "print(json.dumps({'chars': len(d), 'text': 'ጊዜው ቆሟል'}))"]
        result = run_cli(echo, "hello\n", timeout=30)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout), {"chars": 6, "text": "ጊዜው ቆሟል"})

    def test_a_hung_child_is_killed_at_the_timeout(self):
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        started = time.monotonic()
        with self.assertRaises(DirectorTimeout):
            run_cli(sleeper, "", timeout=1, poll=0.1)
        self.assertLess(time.monotonic() - started, 20)

    def test_an_interrupt_kills_the_child(self):
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        with self.assertRaises(DirectorCancelled):
            run_cli(sleeper, "", timeout=0, should_cancel=lambda: True, poll=0.1)

    def test_a_missing_cli_says_what_to_install(self):
        with self.assertRaises(DirectorError) as caught:
            run_cli(["pulse-no-such-cli-xyz"], "", timeout=5)
        self.assertIn("not on PATH", str(caught.exception))


class TestOnlyOneModuleStartsProcesses(unittest.TestCase):
    """The pack is socket-free because its director CLIs are the only thing it
    runs, and they are run in one place. A second place is a second policy."""

    ALLOWED = {"comfyui_pulse_studio/pulse_director/cli_transport.py"}
    EXCLUDED = {"tests", ".github", "__pycache__", ".git", ".venv", "venv",
                "reference_purpose only", "example_workflows"}
    CALLS = {("os", "system"), ("os", "popen"), ("os", "startfile"),
             ("asyncio", "create_subprocess_exec"), ("asyncio", "create_subprocess_shell"),
             ("pty", "spawn")}

    def _sources(self):
        for path in PROJECT_ROOT.rglob("*.py"):
            parts = path.relative_to(PROJECT_ROOT).parts
            if not set(parts[:-1]) & self.EXCLUDED:
                yield path

    def test_subprocess_and_its_cousins_appear_in_cli_transport_only(self):
        found = []
        for path in self._sources():
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                hit = None
                if isinstance(node, ast.Import):
                    hit = next((a.name for a in node.names
                                if a.name.split(".")[0] in ("subprocess", "pty")), None)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    if (node.module or "").split(".")[0] in ("subprocess", "pty"):
                        hit = node.module
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if (node.value.id, node.attr) in self.CALLS or (
                            node.value.id == "os" and node.attr.startswith("spawn")):
                        hit = "%s.%s" % (node.value.id, node.attr)
                if hit and rel not in self.ALLOWED:
                    found.append("%s:%d: %s" % (rel, node.lineno, hit))
        self.assertEqual(found, [], "only pulse_director/cli_transport.py may start a "
                                    "process:\n  " + "\n  ".join(found))

    def test_the_allowed_module_really_is_the_one_that_does(self):
        source = (PROJECT_ROOT / next(iter(self.ALLOWED))).read_text(encoding="utf-8")
        self.assertIn("import subprocess", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
