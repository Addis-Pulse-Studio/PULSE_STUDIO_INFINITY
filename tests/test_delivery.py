"""Delivering at 2K and 4K: the canvases, and RTX Video Super Resolution's reach.

Windows render inside H3's ~1 MP budget, so every delivery size is reached after
assembly. `_exact` resamples with Lanczos; `_rtx` asks NVIDIA's RTX Video Super
Resolution (nvvfx) and falls back to Lanczos, saying why, wherever it cannot run.
The sizes and the rule are pure and tested here everywhere; the fallback itself
is run on a real file where torch and PyAV are installed (the ComfyUI env), and
skipped elsewhere.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from comfyui_pulse_studio.concat import (
    RTX_MAX_SCALE,
    rtx_upscale_problem,
    upscale_engine,
    upscale_size,
)
from comfyui_pulse_studio.constants import UPSCALE_MODES


class TheModes(unittest.TestCase):

    def test_the_new_modes_are_appended(self):
        self.assertEqual(UPSCALE_MODES[:3], ("off", "1080p_fit", "1080p_exact"))
        self.assertEqual(UPSCALE_MODES[3:], ("1440p_exact", "2160p_exact",
                                             "1080p_rtx", "1440p_rtx", "2160p_rtx"))

    def test_every_mode_but_off_has_a_size(self):
        for mode in UPSCALE_MODES[1:]:
            with self.subTest(mode=mode):
                self.assertEqual(len(upscale_size(1344, 768, mode)), 4)

    def test_the_engine_is_read_from_the_mode(self):
        self.assertEqual([upscale_engine(m) for m in UPSCALE_MODES],
                         ["lanczos"] * 5 + ["rtx"] * 3)


class TheCanvases(unittest.TestCase):

    def test_2k_and_4k_cover_then_crop(self):
        self.assertEqual(upscale_size(1344, 768, "1440p_exact"), (2560, 1464, 2560, 1440))
        self.assertEqual(upscale_size(1344, 768, "2160p_exact"), (3840, 2196, 3840, 2160))

    def test_portrait_turns_and_square_squares(self):
        self.assertEqual(upscale_size(768, 1344, "2160p_exact")[2:], (2160, 3840))
        self.assertEqual(upscale_size(992, 992, "1440p_exact")[2:], (1440, 1440))

    def test_rtx_rounds_the_cover_to_a_multiple_of_8(self):
        self.assertEqual(upscale_size(1344, 768, "2160p_rtx"), (3840, 2200, 3840, 2160))
        for mode in ("1080p_rtx", "1440p_rtx", "2160p_rtx"):
            for w, h in ((1344, 768), (768, 1344), (1536, 672), (992, 992)):
                with self.subTest(mode=mode, size=(w, h)):
                    self.assertTrue(all(v % 8 == 0 for v in upscale_size(w, h, mode)))

    def test_rtx_and_lanczos_deliver_the_same_canvas(self):
        for height in ("1080p", "1440p", "2160p"):
            self.assertEqual(upscale_size(1344, 768, height + "_rtx")[2:],
                             upscale_size(1344, 768, height + "_exact")[2:])

    def test_1080p_exact_is_what_it_always_was(self):
        self.assertEqual(upscale_size(1344, 768, "1080p_exact"), (1920, 1098, 1920, 1080))

    def test_a_size_nobody_offers_is_refused(self):
        for mode in ("4k", "2160p", "2160p_bicubic", "720p_rtx"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                upscale_size(1344, 768, mode)


class RtxVideoSuperResolutionsReach(unittest.TestCase):

    def test_h3s_canvases_reach_4k_inside_it(self):
        for w, h in ((1344, 768), (768, 1344), (1536, 672), (992, 992), (1152, 864)):
            scaled_w, scaled_h, _, _ = upscale_size(w, h, "2160p_rtx")
            with self.subTest(size=(w, h)):
                self.assertIsNone(rtx_upscale_problem(w, h, scaled_w, scaled_h))

    def test_past_four_times_is_lanczos(self):
        scaled_w, scaled_h, _, _ = upscale_size(832, 480, "2160p_rtx")
        problem = rtx_upscale_problem(832, 480, scaled_w, scaled_h)
        self.assertIn("up to %gx" % RTX_MAX_SCALE, problem)
        self.assertIn("4.62x", problem)

    def test_a_film_already_that_big_is_not_enlarged(self):
        self.assertIn("only enlarges", rtx_upscale_problem(1920, 1080, 1920, 1080))


def _write_film(path, width, height, frames=6):
    import av
    import numpy as np

    with av.open(path, mode="w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            ramp = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
            image = np.broadcast_to(ramp, (height, width, 3)).copy()
            image[:, :, 1] = (i * 40) % 256
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def _probe(path):
    import av

    with av.open(path) as film:
        stream = film.streams.video[0]
        count = sum(1 for _ in film.decode(stream))
        return stream.codec_context.width, stream.codec_context.height, count


class TheFallback(unittest.TestCase):
    """media.upscale_video on a real file, with nvvfx taken away."""

    def setUp(self):
        try:
            import av  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs torch and PyAV (run it in the ComfyUI environment)")
        self.directory = tempfile.mkdtemp()
        self.src = os.path.join(self.directory, "assembled.mp4")
        _write_film(self.src, 640, 360)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _upscale(self, mode):
        import media

        how = {}
        with mock.patch.dict(sys.modules, {"nvvfx": None}):
            out = media.upscale_video(
                self.src, os.path.join(self.directory, "assembled_%s.mp4" % mode), mode,
                report=how)
        return out, how

    def test_without_nvvfx_an_rtx_mode_is_lanczos_and_says_so(self):
        out, how = self._upscale("1080p_rtx")
        self.assertEqual(how["engine"], "lanczos")
        self.assertIn("nvvfx is not installed", how["fallback"])
        self.assertEqual(_probe(out), (1920, 1080, 6))

    def test_past_four_times_is_lanczos_and_says_so(self):
        out, how = self._upscale("2160p_rtx")   # 640x360 -> 3840x2160 is 6x
        self.assertIn("up to 4x", how["fallback"])
        self.assertEqual(_probe(out), (3840, 2160, 6))

    def test_a_lanczos_mode_reports_no_fallback(self):
        out, how = self._upscale("1440p_exact")
        self.assertEqual(how, {"engine": "lanczos"})
        self.assertEqual(_probe(out), (2560, 1440, 6))


# ── tools/resolve_superscale.py ─────────────────────────────────────────────

def _tool():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "tools" / "resolve_superscale.py"
    spec = importlib.util.spec_from_file_location("resolve_superscale", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


class ResolvesDecisions(unittest.TestCase):

    def test_the_canvas_is_pulses_own(self):
        self.assertEqual(TOOL.canvas(1344, 768, "2160p"), (3840, 2160))
        self.assertEqual(TOOL.canvas(768, 1344, "1440p"), (1440, 2560))
        self.assertEqual(TOOL.canvas(992, 992, "2160p"), (2160, 2160))
        self.assertEqual(TOOL.canvas(1344, 768, "2160p"),
                         upscale_size(1344, 768, "2160p_rtx")[2:])
        self.assertEqual(TOOL.canvas(1344, 768, "1080p"), (1920, 1080))
        with self.assertRaises(TOOL.SuperScaleError):
            TOOL.canvas(1344, 768, "720p")

    def test_auto_is_2x_enhanced_for_every_pulse_canvas(self):
        # measured the best of Resolve's modes, 1344x768 to 4K (ENHANCED_REACH)
        self.assertEqual(TOOL.superscale_for(1344, 768, 3840, 2160), (2, True, ""))
        self.assertEqual(TOOL.superscale_for(1344, 768, 2560, 1440), (2, True, ""))
        self.assertEqual(TOOL.superscale_for(768, 1344, 2160, 3840), (2, True, ""))
        self.assertEqual(TOOL.superscale_for(992, 992, 2160, 2160), (2, True, ""))
        self.assertEqual(TOOL.superscale_for(1280, 720, 3840, 2160), (2, True, ""))

    def test_a_small_film_takes_the_smallest_plain_multiplier_that_covers(self):
        # 2x would leave Resolve's resize 2x to cover: unmeasured, so 3x or 4x
        self.assertEqual(TOOL.superscale_for(960, 540, 3840, 2160), (4, False, ""))
        self.assertEqual(TOOL.superscale_for(1280, 720, 3840, 2160, "3x"), (3, False, ""))
        self.assertEqual(TOOL.superscale_for(1100, 620, 3840, 2160), (4, False, ""))
        self.assertEqual(TOOL.superscale_for(1280, 724, 3840, 2160), (2, True, ""))

    def test_past_4x_says_resolve_makes_up_the_rest(self):
        multiplier, enhanced, note = TOOL.superscale_for(832, 480, 3840, 2160)
        self.assertEqual((multiplier, enhanced), (4, False))
        self.assertIn("short of 3840x2160", note)

    def test_a_chosen_multiplier_is_taken_as_given(self):
        self.assertEqual(TOOL.superscale_for(1344, 768, 3840, 2160, "4x"), (4, False, ""))
        self.assertEqual(TOOL.superscale_for(1344, 768, 3840, 2160, "2x_enhanced"),
                         (2, True, ""))
        self.assertEqual(TOOL.superscale_for(1344, 768, 2560, 1440, "2x"), (2, False, ""))

    def test_resolves_clip_properties_are_read(self):
        self.assertEqual(TOOL.parse_resolution("1344x768"), (1344, 768))
        self.assertEqual(TOOL.parse_resolution("768 x 1344"), (768, 1344))
        with self.assertRaises(TOOL.SuperScaleError):
            TOOL.parse_resolution("")
        self.assertEqual(TOOL.timeline_rate("24.000"), "24")
        self.assertEqual(TOOL.timeline_rate("23.976"), "23.976")
        self.assertEqual(TOOL.timeline_rate("29.97"), "29.97")
        self.assertIsNone(TOOL.timeline_rate(""))

    def test_resolves_own_encoder_is_preferred(self):
        listed = {"H.264": "H264", "H.264 NVIDIA": "H264_NVIDIA", "H.265": "H265",
                  "H.265 NVIDIA": "H265_NVIDIA"}
        # Resolve's own encoder first: the API sets no bitrate, and its own H.265
        # writes more of one than NVIDIA's (34 against 22 Mbit/s at 4K, 21.1.1)
        self.assertEqual(TOOL.pick_codec(listed, "h265"), "H265")
        self.assertEqual(TOOL.pick_codec(listed, "h264"), "H264")
        self.assertEqual(TOOL.pick_codec({"H.265 NVIDIA": "H265_NVIDIA", "H.264": "H264"},
                                         "h265"), "H265_NVIDIA")
        self.assertEqual(TOOL.pick_codec({"H.264": "H264", "H.265": "H265"}, "h265"), "H265")
        self.assertEqual(TOOL.pick_codec({"H.264": "H264"}, "h265"), "H264")
        with self.assertRaises(TOOL.SuperScaleError):
            TOOL.pick_codec({"DNxHR": "dnxhr"}, "h265")

    def test_a_run_folder_gives_its_native_film(self):
        names = ["assembled_2160p_rtx.mp4", "assembled.mp4", "manifest.json"]
        self.assertEqual(os.path.basename(TOOL.choose_source("run", names)), "assembled.mp4")
        with self.assertRaises(TOOL.SuperScaleError):
            TOOL.choose_source("run", ["assembled_2160p_rtx.mp4"])

    def test_names_and_paths(self):
        self.assertEqual(TOOL.output_name(os.path.join("out", "assembled.mp4"), "2160p"),
                         "assembled_2160p_resolve.mp4")
        self.assertEqual(TOOL.windows_path("/mnt/d/Addis/run/assembled.mp4"),
                         "D:\\Addis\\run\\assembled.mp4")
        self.assertEqual(TOOL.windows_path("D:\\Addis\\run"), "D:\\Addis\\run")


class _Clip:
    def __init__(self, resolution, accepts=True):
        self.properties = {"Resolution": resolution, "FPS": "24.000"}
        self.accepts = accepts
        self.super_scale = None

    def GetClipProperty(self, key=None):
        return self.properties.get(key)

    def SetClipProperty(self, *args):
        self.super_scale = args
        return self.accepts


class _Resolve:
    """The calls superscale() makes, answered the way Resolve 21.1's API documents."""

    def __init__(self, resolution="1344x768", ends="Complete", accepts=True,
                 open_project="Untitled Project"):
        self.calls = []
        self.clip = _Clip(resolution, accepts)
        self.ends = ends
        self.polls = 2
        self.open_project = open_project

    def __getattr__(self, name):          # every object is this one fake
        def call(*args):
            self.calls.append((name,) + args)
            return self.answers.get(name, True)
        return call

    @property
    def answers(self):
        return {"GetProjectManager": self, "CreateProject": self, "GetMediaPool": self,
                "GetCurrentProject": self, "GetName": self.open_project,
                "ImportMedia": [self.clip], "CreateTimelineFromClips": object(),
                "GetRenderCodecs": {"H.265": "H265", "H.265 NVIDIA": "H265_NVIDIA"},
                "AddRenderJob": "job-1"}

    def IsRenderingInProgress(self):
        self.polls -= 1
        return self.polls >= 0

    def GetRenderJobStatus(self, job):
        return {"JobStatus": self.ends if self.polls < 0 else "Rendering",
                "CompletionPercentage": 50, "Error": "disk full"}

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


class ResolveIsDrivenAsDocumented(unittest.TestCase):

    def _run(self, fake, **kwargs):
        with mock.patch.object(TOOL.time, "sleep"):
            return TOOL.superscale(fake, os.path.join("films", "run", "assembled.mp4"),
                                   say=lambda *a: None, **kwargs)

    def test_4k_is_2x_enhanced_on_a_cropping_canvas(self):
        fake = _Resolve()
        out = self._run(fake)
        self.assertEqual(out, os.path.join("films", "run", "assembled_2160p_resolve.mp4"))
        settings = fake.named("SetSettings")[0][1]
        self.assertEqual((settings["timelineResolutionWidth"],
                          settings["timelineResolutionHeight"]), ("3840", "2160"))
        self.assertEqual(settings["timelineInputResMismatchBehavior"], "scaleToCrop")
        self.assertEqual(settings["timelineFrameRate"], "24")
        self.assertEqual(fake.clip.super_scale, ("Super Scale", 2, 0.5, 0.5))
        # a plain list of paths: Resolve 21.1.1 imports nothing from [{"FilePath": ...}]
        self.assertEqual(fake.named("ImportMedia")[0][1],
                         [os.path.join("films", "run", "assembled.mp4")])
        self.assertEqual(fake.named("SetCurrentRenderFormatAndCodec")[0][1:],
                         ("mp4", "H265"))
        render = fake.named("SetRenderSettings")[0][1]
        self.assertTrue(render["ExportAudio"])
        self.assertEqual((render["FormatWidth"], render["FormatHeight"]), (3840, 2160))
        self.assertEqual(fake.named("StartRendering")[0][1], ["job-1"])

    def test_1440p_is_2x_enhanced_with_its_two_strengths(self):
        fake = _Resolve()
        self._run(fake, target="1440p", sharpness=0.3, noise_reduction=0.1)
        self.assertEqual(fake.clip.super_scale, ("Super Scale", 2, 0.3, 0.1))

    def test_the_project_is_deleted_afterwards(self):
        fake = _Resolve()
        self._run(fake)
        created = fake.named("CreateProject")[0][1]
        self.assertEqual([c[0] for c in fake.calls[-3:]],
                         ["DeleteRenderJob", "CloseProject", "DeleteProject"])
        self.assertEqual(fake.named("DeleteProject")[0][1], created)

    def test_a_failed_render_is_said_and_still_cleaned_up(self):
        fake = _Resolve(ends="Failed")
        with self.assertRaises(TOOL.SuperScaleError) as raised:
            self._run(fake)
        self.assertIn("Failed", str(raised.exception))
        self.assertIn("disk full", str(raised.exception))
        self.assertTrue(fake.named("DeleteProject"))

    def test_a_project_of_yours_is_left_alone(self):
        fake = _Resolve(open_project="Client cut v3")
        with self.assertRaises(TOOL.SuperScaleError) as raised:
            self._run(fake)
        self.assertIn("--switch-project", str(raised.exception))
        self.assertFalse(fake.named("CreateProject"))

    def test_switching_away_from_it_loads_it_again(self):
        fake = _Resolve(open_project="Client cut v3")
        self._run(fake, switch_project=True)
        self.assertEqual(fake.calls[-1], ("LoadProject", "Client cut v3"))

    def test_resolves_own_untitled_project_is_nobodys(self):
        fake = _Resolve()
        self._run(fake)
        self.assertFalse(fake.named("LoadProject"))

    def test_resolve_without_super_scale_is_named(self):
        with self.assertRaises(TOOL.SuperScaleError) as raised:
            self._run(_Resolve(accepts=False))
        self.assertIn("Studio", str(raised.exception))


class SeveralSizesFromOneSession(unittest.TestCase):
    """Pulse Resolve Deliver: 1080p, 2K and 4K switched on render in one project."""

    def _run(self, fake, targets):
        with mock.patch.object(TOOL.time, "sleep"):
            return TOOL.deliver(fake, os.path.join("films", "run", "assembled.mp4"), targets,
                                say=lambda *a: None)

    def test_one_project_one_timeline_one_job_per_size(self):
        fake = _Resolve()
        out = self._run(fake, ["1080p", "1440p", "2160p"])
        run = os.path.join("films", "run")
        self.assertEqual(out, {t: os.path.join(run, "assembled_%s_resolve.mp4" % t)
                               for t in ("1080p", "1440p", "2160p")})
        self.assertEqual(len(fake.named("CreateProject")), 1)
        self.assertEqual(len(fake.named("ImportMedia")), 1)
        # The timeline is built at the largest size, Super Scale chosen for it ...
        settings = fake.named("SetSettings")[0][1]
        self.assertEqual((settings["timelineResolutionWidth"],
                          settings["timelineResolutionHeight"]), ("3840", "2160"))
        self.assertEqual(fake.clip.super_scale, ("Super Scale", 2, 0.5, 0.5))
        # ... and each size is a render job at its own size and name.
        sizes = [(r[1]["FormatWidth"], r[1]["FormatHeight"], r[1]["CustomName"])
                 for r in fake.named("SetRenderSettings") if "FormatWidth" in r[1]]
        self.assertEqual(sizes, [(1920, 1080, "assembled_1080p_resolve"),
                                 (2560, 1440, "assembled_1440p_resolve"),
                                 (3840, 2160, "assembled_2160p_resolve")])
        self.assertEqual(len(fake.named("AddRenderJob")), 3)
        self.assertEqual(len(fake.named("StartRendering")), 1)
        self.assertEqual(len(fake.named("DeleteRenderJob")), 3)
        self.assertEqual(fake.named("DeleteProject")[0][1], fake.named("CreateProject")[0][1])

    def test_a_vertical_film_gets_vertical_sizes(self):
        fake = _Resolve(resolution="768x1344")
        self._run(fake, ["1080p", "2160p"])
        sizes = [(r[1]["FormatWidth"], r[1]["FormatHeight"])
                 for r in fake.named("SetRenderSettings") if "FormatWidth" in r[1]]
        self.assertEqual(sizes, [(1080, 1920), (2160, 3840)])

    def test_a_repeated_or_unknown_size(self):
        fake = _Resolve()
        self.assertEqual(list(self._run(fake, ["2160p", "2160p"])), ["2160p"])
        for targets in ([], ["720p"]):
            with self.assertRaises(TOOL.SuperScaleError):
                self._run(_Resolve(), targets)

    def test_main_names_each_file_on_one_result_line(self):
        import io
        import json
        fake = _Resolve()
        out = io.StringIO()
        with mock.patch.object(TOOL, "connect", return_value=fake), \
                mock.patch.object(TOOL.os.path, "isfile", return_value=True), \
                mock.patch.object(TOOL.time, "sleep"), mock.patch.object(sys, "stdout", out):
            code = TOOL.main([os.path.join("films", "run", "assembled.mp4"),
                              "--target", "1080p", "--target", "2160p"])
        self.assertEqual(code, 0)
        result = [l for l in out.getvalue().splitlines() if l.startswith("PULSE_RESULT ")]
        self.assertEqual(len(result), 1)
        self.assertEqual(sorted(json.loads(result[0][len("PULSE_RESULT "):])),
                         ["1080p", "2160p"])


class _StoppedMidRender(_Resolve):
    """A render that is under way when the stop file appears, with a partial film
    on disk, as Pulse Resolve Super Scale's Cancel finds it."""

    def __init__(self, stop_file, partial=None, **kwargs):
        super().__init__(**kwargs)
        self.stop_file = stop_file
        self.partial = partial
        self.polls = 10

    def StartRendering(self, jobs):
        self.calls.append(("StartRendering", jobs))
        if self.partial:
            with open(self.partial, "wb") as handle:
                handle.write(b"half a film")
        open(self.stop_file, "w").close()
        return True


class ResolveStopsWhenAsked(unittest.TestCase):
    """--stop-file: how ComfyUI's Cancel reaches a render through the node."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.stop = os.path.join(self.root, "stop")
        self.film = os.path.join(self.root, "assembled.mp4")
        self.out = os.path.join(self.root, "assembled_2160p_resolve.mp4")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fake):
        with mock.patch.object(TOOL.time, "sleep"):
            return TOOL.superscale(fake, self.film, stop_file=self.stop, say=lambda *a: None)

    def test_a_render_under_way_is_stopped_and_cleaned_up(self):
        fake = _StoppedMidRender(self.stop, partial=self.out)
        with self.assertRaises(TOOL.SuperScaleError) as raised:
            self._run(fake)
        self.assertIn("stopped", str(raised.exception))
        self.assertTrue(fake.named("StopRendering"))
        self.assertFalse(os.path.exists(self.out), "the half-rendered film was left behind")
        self.assertEqual([c[0] for c in fake.calls[-3:]],
                         ["DeleteRenderJob", "CloseProject", "DeleteProject"])

    def test_a_film_resolve_still_holds_is_left_for_main_to_remove(self):
        fake = _StoppedMidRender(self.stop, partial=self.out)
        with mock.patch.object(TOOL, "discard_unfinished", return_value=False):
            with self.assertRaises(TOOL.SuperScaleError) as raised:
                self._run(fake)
        paths, since = raised.exception.unfinished
        self.assertEqual(paths, [self.out])
        # main() removes it only if it reads as this render's file.
        from comfyui_pulse_studio.resolve_bridge import CLOCK_SLACK
        self.assertLessEqual(since - CLOCK_SLACK, os.path.getmtime(self.out))

    def test_the_never_answered_message_is_the_one_the_node_looks_for(self):
        from comfyui_pulse_studio.resolve_bridge import never_answered
        silent = mock.Mock()
        silent.scriptapp.return_value = None
        with mock.patch.dict(sys.modules, {"DaVinciResolveScript": silent}), \
                mock.patch.object(TOOL.time, "sleep"):
            with self.assertRaises(TOOL.SuperScaleError) as raised:
                TOOL.connect(0)
        self.assertTrue(never_answered("resolve_superscale: %s\n" % raised.exception))

    def test_a_stop_before_the_render_makes_no_project(self):
        open(self.stop, "w").close()
        fake = _Resolve()
        with self.assertRaises(TOOL.SuperScaleError) as raised:
            self._run(fake)
        self.assertIn("stopped", str(raised.exception))
        self.assertFalse(fake.named("CreateProject"))

    def test_an_older_film_of_the_same_name_survives_a_failed_render(self):
        with open(self.out, "wb") as handle:
            handle.write(b"yesterday's film")
        day_ago = os.path.getmtime(self.out) - 86400
        os.utime(self.out, (day_ago, day_ago))
        with self.assertRaises(TOOL.SuperScaleError):
            self._run(_Resolve(ends="Failed"))
        self.assertTrue(os.path.exists(self.out))

    def test_waiting_for_resolve_ends_when_asked(self):
        open(self.stop, "w").close()
        silent = mock.Mock()
        silent.scriptapp.return_value = None
        with mock.patch.dict(sys.modules, {"DaVinciResolveScript": silent}), \
                mock.patch.object(TOOL.time, "sleep"):
            with self.assertRaises(TOOL.SuperScaleError) as raised:
                TOOL.connect(600, stop_file=self.stop)
        self.assertIn("stopped", str(raised.exception))


class TheNodeAndTheHelperAgree(unittest.TestCase):
    """Pulse Resolve Super Scale builds the helper's command line and reads its
    messages; both are checked here against the helper itself."""

    def test_the_lists_are_the_bridges(self):
        from comfyui_pulse_studio import resolve_bridge
        self.assertIs(TOOL.TARGETS, resolve_bridge.TARGETS)
        self.assertIs(TOOL.MODES, resolve_bridge.MODES)
        self.assertIs(TOOL.CODECS, resolve_bridge.CODECS)

    def test_the_helper_accepts_every_flag_the_node_passes(self):
        import io
        from comfyui_pulse_studio.resolve_bridge import helper_argv
        argv = helper_argv("ResolvePython.exe", "resolve_superscale.py",
                           os.path.join("nowhere", "assembled.mp4"), "out", "stop",
                           targets=["1080p", "1440p"], mode="2x_enhanced", sharpness=0.3,
                           noise_reduction=0.1, codec="h264", switch_project=True,
                           quit_after=True)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            code = TOOL.main(argv[2:])
        # Past argparse (which exits 2 on a flag it does not know) to the film check.
        self.assertEqual(code, 1)
        self.assertIn("no such film", err.getvalue())

    def test_its_messages_reach_the_node_in_the_nodes_words(self):
        from comfyui_pulse_studio.resolve_bridge import explain
        with self.assertRaises(TOOL.SuperScaleError) as raised, \
                mock.patch.object(TOOL.time, "sleep"):
            TOOL.superscale(_Resolve(open_project="Client cut v3"),
                            os.path.join("films", "run", "assembled.mp4"), say=lambda *a: None)
        told = explain(str(raised.exception))
        self.assertIn("set open_project to switch_and_restore", told)
        self.assertNotIn("--switch-project", told)

        silent = mock.Mock()
        silent.scriptapp.return_value = None
        with mock.patch.dict(sys.modules, {"DaVinciResolveScript": silent}), \
                mock.patch.object(TOOL.time, "sleep"):
            with self.assertRaises(TOOL.SuperScaleError) as raised:
                TOOL.connect(0)
        told = explain(str(raised.exception))
        self.assertIn("the node starts it", told)
        self.assertNotIn(".bat", told)


if __name__ == "__main__":
    unittest.main()
