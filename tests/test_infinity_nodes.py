"""Pulse Infinity's nodes: the contracts ComfyUI itself holds them to.

infinity_nodes.py imports torch and ComfyUI, so -- like nodes.py -- it is read
with ast here, and the few pieces of logic that decide something are lifted out
and run. Slot order, hidden inputs and the widget tables are covered by
test_js_guard and test_patches, which read this file alongside nodes.py.
"""

import ast
import unittest
from pathlib import Path

from comfyui_pulse_studio.pulse_memory import DirectedShotPlan

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "infinity_nodes.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _class(name):
    return next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == name)


def _method(cls, name):
    return next(f for f in _class(cls).body if isinstance(f, ast.FunctionDef) and f.name == name)


def _plan(spoken=True):
    plan, problems = DirectedShotPlan.from_dict({
        "shot_index": 2, "visual_prompt": "@Alice whispers to the clock.",
        "negative_prompt": "", "camera_motion": "close-up",
        "duration_seconds": 5.17,
        "dialogue": ({"speaker": "Alice", "text": "The time has already stopped...",
                      "emotion_tone": "whispering, anxious"} if spoken else None),
        "continuity_anchors": {"held_items": [], "costume": "", "environment": ""}})
    assert not problems, problems
    return plan


class TestTheTtsLine(unittest.TestCase):
    """What Fish Audio S2 is handed: the line with its delivery in brackets."""

    @staticmethod
    def _tts_line():
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "tts_line")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "infinity_nodes.py", "exec"),
             namespace)
        return namespace["tts_line"]

    def test_the_delivery_rides_in_brackets(self):
        self.assertEqual(self._tts_line()(_plan()),
                         "[whispering, anxious] The time has already stopped...")

    def test_a_silent_beat_has_no_line(self):
        self.assertEqual(self._tts_line()(_plan(spoken=False)), "")
        self.assertEqual(self._tts_line()(None), "")


class TestTheTtsBranchRunsOnlyWhenItIsNeeded(unittest.TestCase):
    """The Dynamic Shot's dialogue_audio is lazy: ComfyUI evaluates the TTS node
    behind it only when check_lazy_status asks. Asking for an unconnected input
    raises in ComfyUI, so absent-from-kwargs and present-as-None must differ."""

    @classmethod
    def setUpClass(cls):
        cls_node = _class("PulseDynamicShot")
        body = [f for f in cls_node.body if isinstance(f, ast.FunctionDef)
                and f.name in ("_wants_take", "check_lazy_status")]
        for f in body:
            f.decorator_list = [d for d in f.decorator_list
                                if not (isinstance(d, ast.Name) and d.id == "staticmethod")]
        shell = ast.ClassDef(name="Shot", bases=[], keywords=[], body=body, decorator_list=[])
        namespace = {"DIALOGUE_TTS": "external_tts"}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])),
                     "infinity_nodes.py", "exec"), namespace)
        shot = namespace["Shot"]
        shot._wants_take = staticmethod(shot._wants_take)
        cls.node = shot()

    def _infinity(self, **changes):
        handle = {"dialogue_mode": "external_tts", "speaker": "Alice",
                  "speaker_has_voice": True}
        handle.update(changes)
        return handle

    def test_a_speaking_beat_asks_for_its_take(self):
        self.assertEqual(self.node.check_lazy_status(
            infinity=self._infinity(), dialogue_audio=None), ["dialogue_audio"])

    def test_an_unconnected_input_is_never_requested(self):
        self.assertEqual(self.node.check_lazy_status(infinity=self._infinity()), [])

    def test_a_take_already_evaluated_is_not_asked_for_again(self):
        self.assertEqual(self.node.check_lazy_status(
            infinity=self._infinity(), dialogue_audio={"waveform": None}), [])

    def test_silent_beats_native_dialogue_and_voiceless_speakers_skip_tts(self):
        for changes in ({"speaker": ""}, {"dialogue_mode": "native"},
                        {"speaker_has_voice": False}):
            with self.subTest(**changes):
                self.assertEqual(self.node.check_lazy_status(
                    infinity=self._infinity(**changes), dialogue_audio=None), [])


class TestComfyUIsContracts(unittest.TestCase):
    def test_is_changed_takes_whatever_comfyui_passes(self):
        """ComfyUI calls IS_CHANGED with every widget as a keyword, plus UNIQUE_ID."""
        fn = _method("PulseInfinityLoopController", "IS_CHANGED")
        self.assertIsNotNone(fn.args.kwarg)
        self.assertTrue(any(isinstance(d, ast.Name) and d.id == "classmethod"
                            for d in fn.decorator_list))

    def test_is_changed_names_the_session_and_its_revision(self):
        body = ast.get_source_segment(SOURCE, _method("PulseInfinityLoopController",
                                                      "IS_CHANGED"))
        self.assertIn("peek_revision(root)", body)

    def test_the_session_is_a_widget_so_is_changed_can_read_it(self):
        """IS_CHANGED receives constants only; a linked session name would be None."""
        source = ast.get_source_segment(SOURCE, _class("PulseInfinityLoopController"))
        self.assertIn('"session": ("STRING"', source)

    def test_the_observer_is_an_output_node(self):
        assigns = [n for n in _class("PulseObserver").body if isinstance(n, ast.Assign)]
        self.assertTrue(any(getattr(t, "id", "") == "OUTPUT_NODE" for a in assigns
                            for t in a.targets))

    def test_the_four_nodes_are_registered_with_display_names(self):
        self.assertIn('"PulseInfinityLoopController": PulseInfinityLoopController', SOURCE)
        for name in ("Pulse Infinity Loop", "Pulse Dynamic Shot", "Pulse Infinity Render",
                     "Pulse Observer"):
            self.assertIn('"%s"' % name, SOURCE)

    def test_the_pack_merges_them_and_refuses_a_clash(self):
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("from .infinity_nodes import", init)
        self.assertIn("node ids defined twice", init)

    def test_frontend_events_never_broadcast(self):
        """A broadcast would let one tab's film requeue another tab's graph."""
        send = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_send")
        body = ast.get_source_segment(SOURCE, send)
        self.assertIn("if server.client_id is not None:", body)
        self.assertIn("server.send_sync(event, data, server.client_id)", body)

    def test_the_dynamic_shot_never_pins_a_start_image(self):
        """An explicit start_image outranks the carried frame (compiler anchors)."""
        body = ast.get_source_segment(SOURCE, _method("PulseDynamicShot", "execute"))
        self.assertIn('"start_image": None', body)


if __name__ == "__main__":
    unittest.main()
