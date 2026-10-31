"""Pulse Resolve Super Scale: what the node decides, without Resolve or ComfyUI.

resolve_nodes.py imports ComfyUI, so -- like nodes.py -- it is read with ast here.
What it decides lives in comfyui_pulse_studio/resolve_bridge.py and runs here:
where Resolve is, which film goes to it, where the result lands, the helper's
command line, and how what the helper prints is read. The helper is driven
against a fake Resolve in test_delivery.py, the processes are run for real in
test_infinity_directors.py, and the slot contract is held by test_js_guard.py
and test_patches.py.
"""

import ast
import os
import tempfile
import unittest
from pathlib import Path

from comfyui_pulse_studio import resolve_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "resolve_nodes.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

PROGRAM_FILES = "C:\\Program Files"
RESOLVE_DIR = PROGRAM_FILES + "\\Blackmagic Design\\DaVinci Resolve"
RESOLVE_EXE = RESOLVE_DIR + "\\Resolve.exe"
RESOLVE_PY = RESOLVE_DIR + "\\ResolvePython\\ResolvePython.exe"


def _node():
    return next(n for n in TREE.body
                if isinstance(n, ast.ClassDef) and n.name == "PulseResolveSuperScale")


def _function(name, within=None):
    return next(n for n in ast.walk(within or TREE)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node):
    return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(node) if isinstance(c, ast.Call)}


class WhereResolveIs(unittest.TestCase):

    def test_the_default_install_is_found(self):
        present = {RESOLVE_EXE, RESOLVE_PY}.__contains__
        self.assertEqual(bridge.find_install({"ProgramFiles": PROGRAM_FILES}, present),
                         (RESOLVE_EXE, RESOLVE_PY))
        # Python's os.environ on Windows spells every name in capitals.
        self.assertEqual(bridge.find_install({"PROGRAMFILES": PROGRAM_FILES}, present),
                         (RESOLVE_EXE, RESOLVE_PY))

    def test_the_variable_names_another_folder(self):
        present = {"E:\\Apps\\Resolve\\Resolve.exe",
                   "E:\\Apps\\Resolve\\ResolvePython\\ResolvePython.exe"}.__contains__
        env = {"ProgramFiles": PROGRAM_FILES, bridge.RESOLVE_DIR_ENV: '"E:\\Apps\\Resolve"'}
        self.assertEqual(bridge.find_install(env, present)[0], "E:\\Apps\\Resolve\\Resolve.exe")

    def test_a_missing_resolve_names_studio_and_the_variable(self):
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.find_install({"ProgramFiles": PROGRAM_FILES}, lambda path: False)
        self.assertIn("Studio", str(caught.exception))
        self.assertIn(bridge.RESOLVE_DIR_ENV, str(caught.exception))

    def test_a_variable_pointing_nowhere_is_named(self):
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.find_install({bridge.RESOLVE_DIR_ENV: "E:\\Nope"}, lambda path: False)
        self.assertIn("E:\\Nope", str(caught.exception))

    def test_a_resolve_without_its_own_python_is_named(self):
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.find_install({"ProgramFiles": PROGRAM_FILES}, {RESOLVE_EXE}.__contains__)
        self.assertIn(RESOLVE_PY, str(caught.exception))


class WhichFilmGoesToResolve(unittest.TestCase):

    def test_pulses_own_delivery_gives_way_to_the_native_film(self):
        run = os.path.join("output", "pulseslate", "abc")
        native = os.path.join(run, "assembled.mp4")
        for name in ("assembled_2160p_rtx.mp4", "assembled_1080p_fit.mp4",
                     "assembled_2160p_resolve.mp4"):
            self.assertEqual(bridge.native_film(os.path.join(run, name), {native}.__contains__),
                             (native, True))

    def test_without_the_native_film_beside_it_the_film_goes_as_it_is(self):
        path = os.path.join("output", "x", "assembled_2160p_rtx.mp4")
        self.assertEqual(bridge.native_film(path, lambda p: False), (path, False))

    def test_any_other_film_goes_as_it_is(self):
        for name in ("assembled.mp4", "my_cut.mp4", "assembled.mov"):
            path = os.path.join("input", name)
            self.assertEqual(bridge.native_film(path, lambda p: True), (path, False))

    def test_the_canvas_is_pulses_exact_canvas(self):
        self.assertEqual(bridge.delivery_canvas(1344, 768, "2160p"), (3840, 2160))
        self.assertEqual(bridge.delivery_canvas(768, 1344, "1440p"), (1440, 2560))
        self.assertEqual(bridge.delivery_canvas(992, 992, "2160p"), (2160, 2160))
        self.assertEqual(bridge.delivery_canvas(1344, 768, "1080p"), (1920, 1080))
        self.assertEqual(bridge.delivery_canvas(768, 1344, "1080p"), (1080, 1920))
        with self.assertRaises(ValueError):
            bridge.delivery_canvas(1344, 768, "720p")
        with self.assertRaises(ValueError):
            bridge.delivery_canvas(0, 768, "2160p")

    def test_a_film_that_already_fills_the_canvas_is_not_sent(self):
        self.assertTrue(bridge.already_delivered(3840, 2160, 3840, 2160))
        self.assertTrue(bridge.already_delivered(4096, 2160, 3840, 2160))
        # A scope film at full width still needs its height: cover, then crop.
        self.assertFalse(bridge.already_delivered(3840, 1600, 3840, 2160))
        self.assertFalse(bridge.already_delivered(1344, 768, 3840, 2160))


class WhereTheFilmLands(unittest.TestCase):

    def test_a_film_in_the_output_folder_gets_its_result_beside_it(self):
        with tempfile.TemporaryDirectory() as output:
            run = os.path.join(output, "pulseslate", "abc")
            self.assertEqual(bridge.output_dir(os.path.join(run, "assembled.mp4"), output), run)

    def test_a_film_from_anywhere_else_goes_to_pulse_resolve(self):
        with tempfile.TemporaryDirectory() as base:
            output = os.path.join(base, "output")
            for film in (os.path.join(base, "input", "clip.mp4"),
                         os.path.join(base, "temp", "pulse_resolve_1.mp4"),
                         # a sibling whose name merely starts the same way
                         os.path.join(base, "output_old", "assembled.mp4")):
                with self.subTest(film=film):
                    self.assertEqual(bridge.output_dir(film, output),
                                     os.path.join(output, "pulse_resolve"))


class UnfinishedFilms(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.film = os.path.join(self.root, "assembled_2160p_resolve.mp4")

    def tearDown(self):
        os.chmod(self.root, 0o755)
        self._tmp.cleanup()

    def _write(self, age=0.0):
        with open(self.film, "wb") as handle:
            handle.write(b"half a film")
        stamp = os.path.getmtime(self.film) - age
        os.utime(self.film, (stamp, stamp))
        return stamp + age

    def test_the_name_is_the_helpers(self):
        self.assertEqual(bridge.output_name(os.path.join("run", "assembled.mp4"), "2160p"),
                         "assembled_2160p_resolve.mp4")

    def test_a_film_this_render_wrote_is_removed(self):
        since = self._write() - 5
        self.assertTrue(bridge.discard_unfinished(self.film, since))
        self.assertFalse(os.path.exists(self.film))

    def test_an_older_film_of_the_same_name_is_kept(self):
        now = self._write(age=3600)
        self.assertTrue(bridge.discard_unfinished(self.film, now - 60))
        self.assertTrue(os.path.exists(self.film))

    def test_a_stamp_a_hair_before_the_render_began_is_still_this_renders(self):
        # Windows stamps files from a coarser clock than time.time() reads.
        since = self._write() + bridge.CLOCK_SLACK / 4
        self.assertTrue(bridge.discard_unfinished(self.film, since))
        self.assertFalse(os.path.exists(self.film))

    def test_nothing_there_is_nothing_left(self):
        self.assertTrue(bridge.discard_unfinished(self.film, 0))

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                     "needs POSIX permissions that bind")
    def test_a_film_that_cannot_be_removed_yet_is_reported(self):
        since = self._write() - 5
        os.chmod(self.root, 0o555)  # what a file Resolve holds open looks like
        self.assertFalse(bridge.discard_unfinished(self.film, since, wait=0.3, poll=0.1))
        self.assertTrue(os.path.exists(self.film))


class TheHelpersCommandLine(unittest.TestCase):

    def _argv(self, **kwargs):
        return bridge.helper_argv("py.exe", "helper.py", "film.mp4", "out", "stop", **kwargs)

    def test_the_defaults(self):
        self.assertEqual(self._argv(), [
            "py.exe", "helper.py", "film.mp4", "--target", "2160p", "--mode", "auto",
            "--sharpness", "0.5", "--nr", "0.5", "--codec", "h265", "--out", "out",
            "--wait", "180", "--stop-file", "stop"])

    def test_quit_and_switch_only_when_asked(self):
        self.assertEqual(self._argv(switch_project=True, quit_after=True)[-2:],
                         ["--switch-project", "--quit"])
        self.assertNotIn("--quit", self._argv())
        self.assertNotIn("--switch-project", self._argv())

    def test_the_strengths_are_written_plainly(self):
        argv = self._argv(sharpness=0.05, noise_reduction=1)
        self.assertEqual(argv[argv.index("--sharpness") + 1], "0.05")
        self.assertEqual(argv[argv.index("--nr") + 1], "1")

    def test_every_size_switched_on_is_one_target(self):
        argv = self._argv(targets=["1080p", "1440p", "2160p"])
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "--target"],
                         ["1080p", "1440p", "2160p"])
        self.assertEqual([label for label, _ in bridge.TIERS], ["1080p", "2K", "4K"])
        self.assertEqual([target for _, target in bridge.TIERS], list(bridge.TARGETS))

    def test_a_value_the_helper_would_refuse_is_refused_first(self):
        for kwargs in ({"targets": ["720p"]}, {"targets": []}, {"mode": "8x"}, {"codec": "prores"},
                       {"sharpness": 1.5}, {"noise_reduction": -0.1}):
            with self.subTest(**kwargs), self.assertRaises(bridge.ResolveBridgeError):
                self._argv(**kwargs)


class ReadingTheHelper(unittest.TestCase):

    def test_progress_lines(self):
        self.assertEqual(bridge.parse_progress("  45%"), 45)
        self.assertEqual(bridge.parse_progress("  100%"), 100)
        self.assertEqual(bridge.parse_progress("  45.5%"), 45)
        for line in ("", "rendering assembled_2160p_resolve.mp4 with H265 ...", "45",
                     "D:\\run\\assembled_2160p_resolve.mp4", "  250%"):
            self.assertIsNone(bridge.parse_progress(line), line)

    def test_success_names_each_file_this_run_wrote(self):
        out = ("DaVinci Resolve Studio 21.1.1.10\n"
               "assembled.mp4: 1344x768 -> Super Scale 2x Enhanced -> 1920x1080, 3840x2160\n"
               "  100%\n"
               'PULSE_RESULT {"1080p": "D:\\\\run\\\\a_1080p_resolve.mp4", '
               '"2160p": "D:\\\\run\\\\a_2160p_resolve.mp4"}\n'
               "D:\\run\\a_1080p_resolve.mp4\nD:\\run\\a_2160p_resolve.mp4\n")
        self.assertEqual(bridge.parse_result(out, "", 0, lambda p: 200.0, since=100.0),
                         {"1080p": "D:\\run\\a_1080p_resolve.mp4",
                          "2160p": "D:\\run\\a_2160p_resolve.mp4"})

    def test_an_old_file_is_never_handed_on_as_the_new_one(self):
        out = 'PULSE_RESULT {"2160p": "D:\\\\run\\\\a.mp4"}\n'
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.parse_result(out, "", 0, lambda p: 50.0, since=100.0)
        self.assertIn("older than this run", str(caught.exception))

    def test_a_stamp_within_the_clock_slack_is_this_runs(self):
        out = 'PULSE_RESULT {"2160p": "D:\\\\run\\\\a.mp4"}\n'
        stamp = 100.0 - bridge.CLOCK_SLACK / 4
        self.assertEqual(bridge.parse_result(out, "", 0, lambda p: stamp, since=100.0),
                         {"2160p": "D:\\run\\a.mp4"})

    def test_success_without_a_file_is_a_failure(self):
        with self.assertRaises(bridge.ResolveBridgeError):
            bridge.parse_result('PULSE_RESULT {"2160p": "D:\\\\run\\\\a.mp4"}\n', "", 0,
                                lambda p: None, since=100.0)
        for nothing in ("", "D:\\run\\a.mp4\n", "PULSE_RESULT {}\n", "PULSE_RESULT oops\n"):
            with self.assertRaises(bridge.ResolveBridgeError):
                bridge.parse_result(nothing, "", 0, lambda p: 200.0, since=100.0)

    def test_a_failure_is_told_in_the_helpers_words_with_the_nodes_widgets(self):
        err = ("resolve_superscale: Resolve has your project 'Client cut' open, and this "
               "renders in a temporary project of its own. Save and close it first, or pass "
               "--switch-project to switch away from it and load it again afterwards.\n")
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.parse_result("DaVinci Resolve Studio 21.1.1.10\n", err, 1,
                                lambda p: None, since=0)
        told = str(caught.exception)
        self.assertIn("Client cut", told)
        self.assertIn("set open_project to switch_and_restore", told)
        self.assertNotIn("--switch-project", told)
        self.assertFalse(told.startswith(bridge.ERROR_PREFIX))

    def test_a_crash_is_told_by_its_last_line(self):
        err = ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
               "AttributeError: 'NoneType' object has no attribute 'GetName'\n")
        with self.assertRaises(bridge.ResolveBridgeError) as caught:
            bridge.parse_result("", err, 1, lambda p: None, since=0)
        self.assertIn("exit code 1", str(caught.exception))
        self.assertIn("AttributeError", str(caught.exception))

    def test_a_resolve_that_never_answered_is_told_apart(self):
        dead = ("resolve_superscale: DaVinci Resolve did not answer within 120 s. Is it "
                "running -- the .bat starts it -- and is Preferences > System > General > "
                "External scripting using set to Local?\n")
        self.assertTrue(bridge.never_answered(dead))
        self.assertFalse(bridge.never_answered(
            "resolve_superscale: Resolve has your project 'Client cut' open, ...\n"))
        self.assertFalse(bridge.never_answered(""))

    def test_tasklist_says_whether_resolve_runs(self):
        self.assertEqual(bridge.TASKLIST_ARGV[:3],
                         ("tasklist", "/FI", "IMAGENAME eq Resolve.exe"))
        self.assertTrue(bridge.resolve_listed(
            '"Resolve.exe","14932","Console","1","2,145,332 K"\r\n'))
        self.assertTrue(bridge.resolve_listed('\r\n"RESOLVE.EXE","1","Console","1","1 K"'))
        for text in ("INFO: No tasks are running which match the specified criteria.\r\n",
                     "INFORMATION : Aucune tâche en service ne correspond aux critères.\r\n",
                     '"ResolveHelper.exe","1","Console","1","1 K"', ""):
            self.assertFalse(bridge.resolve_listed(text), text)


class TheNode(unittest.TestCase):
    """The glue in resolve_nodes.py, read with ast."""

    def test_it_is_an_output_node(self):
        assigns = {t.id: a.value for a in _node().body if isinstance(a, ast.Assign)
                   for t in a.targets if isinstance(t, ast.Name)}
        self.assertIs(assigns["OUTPUT_NODE"].value, True)

    def test_its_combos_are_the_bridges_lists(self):
        for name in ("TARGETS", "MODES", "CODECS", "OPEN_PROJECT_MODES", "WHEN_DONE_MODES"):
            self.assertIn("list(bridge.%s)" % name, SOURCE)

    def test_the_pack_registers_and_ships_it(self):
        self.assertIn('"PulseResolveSuperScale": "Pulse Resolve Super Scale"', SOURCE)
        self.assertIn("from .resolve_nodes import",
                      (ROOT / "__init__.py").read_text(encoding="utf-8"))
        self.assertIn('"resolve_nodes"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("include tools/resolve_superscale.py", manifest)

    def test_the_progress_bar_is_never_touched_from_the_reader_thread(self):
        """ComfyUI's progress hook raises on a cancel and clears the interrupt flag;
        called from the reader thread, that would lose the cancel."""
        session = _function("_resolve_session")
        self.assertNotIn("update_absolute", _calls(_function("heard", session)))
        self.assertIn("update_absolute", _calls(_function("should_cancel", session)))

    def test_the_progress_hooks_interrupt_is_taken_as_a_cancel(self):
        """ComfyUI's InterruptProcessingException is a BaseException. Caught as an
        Exception it would escape, and the helper would be killed before it could
        stop Resolve's render and delete its project."""
        should_cancel = _function("should_cancel", _function("_resolve_session"))
        handlers = [h for h in ast.walk(should_cancel) if isinstance(h, ast.ExceptHandler)]
        self.assertTrue(handlers)
        for handler in handlers:
            self.assertEqual(getattr(handler.type, "id", None), "BaseException")

    def test_a_cancel_clears_comfyuis_interrupt_flag(self):
        """Raised with the flag still set, the next prompt would stop at once."""
        self.assertIn("throw_exception_if_processing_interrupted",
                      _calls(_function("_raise_interrupted")))

    def _guarded(self):
        """The try in execute() that starts Resolve, and so has to see it out."""
        execute = _function("_resolve_session")
        guarded = [t for t in ast.walk(execute) if isinstance(t, ast.Try) and t.finalbody
                   and "_start_resolve" in _calls(ast.Module(body=t.body, type_ignores=[]))]
        self.assertEqual(len(guarded), 1)
        return execute, guarded[0]

    def test_resolve_starts_only_where_it_is_seen_out_again(self):
        """Every check runs before Resolve starts, and the start sits in the try whose
        `finally` quits it, so no failure leaves a headless Resolve behind."""
        execute, guarded = self._guarded()
        body = ast.get_source_segment(SOURCE, execute)
        self.assertLess(body.index("make_argv("), body.index("_start_resolve("))
        self.assertIn("bridge.helper_argv(",
                      ast.get_source_segment(SOURCE, _function("make_argv",
                                                               _function("_run_resolve"))))
        self.assertIn("spawn_detached", _calls(_function("_start_resolve")))
        self.assertIn("_see_resolve_out",
                      _calls(ast.Module(body=guarded.finalbody, type_ignores=[])))

    def test_a_resolve_whose_scripting_never_answered_is_restarted_once(self):
        """A cold start can leave Resolve's script server dead for the session."""
        _, guarded = self._guarded()
        loops = [n for n in ast.walk(ast.Module(body=guarded.body, type_ignores=[]))
                 if isinstance(n, ast.For)]
        self.assertEqual(len(loops), 1)
        self.assertEqual(ast.literal_eval(loops[0].iter), (1, 2))
        calls = _calls(loops[0])
        for name in ("_start_resolve", "run_lines", "never_answered", "_end_resolve"):
            self.assertIn(name, calls)

    def test_an_unfinished_film_is_removed_once_the_run_is_over(self):
        """Resolve held a stopped render's file open until it quit (21.1.1, live)."""
        _, guarded = self._guarded()
        self.assertIn("cleanup", _calls(ast.Module(body=guarded.finalbody, type_ignores=[])))
        self.assertIn("discard_unfinished",
                      _calls(_function("cleanup", _function("_run_resolve"))))

    def test_both_nodes_run_the_same_shared_code(self):
        for name in ("PulseResolveSuperScale", "PulseResolveDeliver"):
            cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertIn("_run_resolve", _calls(_function("execute", cls)), name)

    def test_a_size_switched_off_sends_nothing_its_save_node_could_run_on(self):
        """ComfyUI's ExecutionBlocker(None) stops a Save node silently: no error."""
        cls = next(n for n in TREE.body
                   if isinstance(n, ast.ClassDef) and n.name == "PulseResolveDeliver")
        body = ast.get_source_segment(SOURCE, _function("execute", cls))
        self.assertIn("ExecutionBlocker(None)", body)
        self.assertIn("bridge.TIERS", body)
        self.assertIn('"PulseResolveDeliver": "Pulse Resolve Deliver"', SOURCE)

    def test_memory_reclaim_is_asked_by_its_node_id_and_widget_values(self):
        """The ids and values a saved workflow holds, which that pack keeps stable."""
        self.assertIn('MEMORY_RECLAIM_NODE = "MemoryReclaimUnload"', SOURCE)
        for pair in ('"mode": "Always"', '"unload_target": "All Models"',
                     '"cache_purge": "Hard"', '"gc_collect": True'):
            self.assertIn(pair, SOURCE)


if __name__ == "__main__":
    unittest.main()
