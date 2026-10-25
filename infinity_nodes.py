"""ComfyUI nodes for Pulse Studio Infinity: a film that directs itself, beat by beat.

One queue run renders one beat:

    Pulse Infinity Loop      reads the session on disk, hands the planned beat on
    Pulse Dynamic Shot       turns the plan into a PulseShot payload for PulseSlate
    (PulseSlate)             compiles it, the cast and the recipe -- unchanged
    Pulse Infinity Render    renders it into the session, continuing the last beat
    Pulse Observer           shows the result to the director, commits memory,
                             stores the next plan, and keeps the reel

Queue the graph N times for N beats, or turn on auto-continue (the Loop node's
context menu) for an open-ended film. Every decision lives in the pure core --
comfyui_pulse_studio/infinity_session.py, pulse_director/, pulse_memory/ -- and is
covered headless; these nodes marshal sockets into it, and nothing more.

The nodes live here rather than in nodes.py so that nodes.py keeps the eleven
classes its tests count, and they follow the same slot contract: `schema_version`
at widget 0, widgets append-only, every class declaring `"hidden": HIDDEN_INPUTS`.

Standing rules, as in nodes.py: execute() never writes a widget, and the Loop and
Dynamic Shot nodes only ever do things that are safe to do twice -- ComfyUI's
RAM-pressure cache may run any node again in the same queue.
"""

import logging
import os

import folder_paths
from comfy_execution.graph import ExecutionBlocker

from . import media, render
from .comfyui_pulse_studio.assets import KIND_AUDIO
from .comfyui_pulse_studio.constants import (
    AUDIO_MODE_LOCK,
    AUDIO_MODE_REFERENCE,
    AUDIO_MODES,
    AUDIO_ROLE_LIP_SYNC,
    AUTO_SHOT_PAD_SECONDS,
    DEFAULT_REMIX_STRENGTH,
    FPS,
    MAX_WINDOW_FRAMES,
    REF_FIT_CANVAS,
    REF_FIT_MODES,
    SCHEMA_VERSION,
    SEAM_AUDIO,
    SEAM_DEFAULT,
    SEAM_MODES,
)
from .comfyui_pulse_studio.frames import align_frame_count, frames_to_seconds
from .comfyui_pulse_studio.infinity_session import (
    STATUS_COMPLETE,
    STATUS_PAUSED,
    InfinitySession,
    SessionError,
    peek_revision,
    safe_session_name,
)
from .comfyui_pulse_studio.pulse_director import (
    BACKENDS,
    FROM_CONFIG,
    DirectorError,
    EvidencePack,
    InfinityConfig,
    create_director,
    find_config_file,
    parse_config_text,
)
from .comfyui_pulse_studio.pulse_memory import store
from .comfyui_pulse_studio.pulse_memory.schema import DirectedShotPlan
from .comfyui_pulse_studio.pulse_timeline import SHOT_CONTINUITY_MODES, CONTINUITY_INHERIT
from .comfyui_pulse_studio.segcache import CACHE_MODES, CarryInError, ReuseOnlyMiss, segment_paths
from .comfyui_pulse_studio.sockets import SocketGapError
from .nodes import (
    HIDDEN_INPUTS,
    HIDDEN_INPUTS_WITH_PROMPT,
    MAX_SLATE_CAST,
    PulseVoice,
    _numbered,
    _unwrap_timeline,
    schema_widget,
)

log = logging.getLogger(__name__)

CATEGORY = "AddisPulse/Infinity"
DIALOGUE_NATIVE = "native"
DIALOGUE_TTS = "external_tts"
DIALOGUE_MODES = (DIALOGUE_NATIVE, DIALOGUE_TTS)
CARRY_MODES = ("auto", "always", "never")
TRISTATE = ("auto", "on", "off")
#: Backends whose model runs on this GPU, and so wants H3 out of VRAM first.
LOCAL_BACKENDS = ("ollama", "pi")


# ── shared plumbing ─────────────────────────────────────────────────────────

def _yaml_loader():
    try:
        import yaml
    except ImportError:  # pragma: no cover - every ComfyUI ships PyYAML
        return None
    return yaml.safe_load


def load_config(widget_backend=FROM_CONFIG):
    """pulse_infinity.yaml (or .json) from $PULSE_INFINITY_CONFIG or ComfyUI's user
    folder, merged over the defaults, with the director chosen. Never raises."""
    path = find_config_file(folder_paths.get_user_directory())
    data, problems = {}, []
    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            problems.append("%s could not be opened: %s" % (path, exc))
        else:
            data, problems = parse_config_text(text, path, yaml_loader=_yaml_loader())
    config = InfinityConfig.resolve(data, widget_backend, os.environ, path or "", problems)
    for problem in config.problems:
        log.warning("[PulseStudio] infinity config: %s", problem)
    return config


def _interrupted():
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:  # pragma: no cover - outside ComfyUI
        return False


def _send(event, data):
    """A frontend event for the tab that queued this prompt -- never a broadcast,
    so one tab's film cannot requeue another's."""
    try:
        from server import PromptServer
        server = PromptServer.instance
        if server.client_id is not None:
            server.send_sync(event, data, server.client_id)
    except Exception as exc:  # pragma: no cover - headless
        log.debug("[PulseStudio] could not send %s: %s", event, exc)


def _status_sender(unique_id):
    def send(text):
        log.info("[PulseStudio] %s", text)
        _send("pulse_infinity.status", {"node_id": str(unique_id), "text": text})
    return send


def _free_vram():
    """Unload ComfyUI's models before a local director runs on the same GPU."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        log.info("[PulseStudio] unloaded ComfyUI's models so the local director has "
                 "the GPU; they reload on the next beat")
    except Exception as exc:  # pragma: no cover
        log.warning("[PulseStudio] could not free VRAM before the director: %s", exc)


def _blocked(count, message):
    blocker = ExecutionBlocker(message)
    return tuple(blocker for _ in range(count))


def _cast_of(kwargs):
    """The wired PulseCharacters: [(name, description)], and their voices by name."""
    cast, voices = [], {}
    for _, character in _numbered(kwargs, "cast.cast_", MAX_SLATE_CAST):
        name = (character.get("name") or "").strip()
        if not name:
            continue
        cast.append((name, character.get("description") or ""))
        voice = character.get("voice")
        if voice and voice.get("audio") is not None:
            voices[name.casefold()] = voice
    return cast, voices


def _session(handle_or_name, config):
    return InfinitySession(folder_paths.get_output_directory(), handle_or_name, config)


def _director(session, config, unique_id, script=""):
    overrides = {"script": script} if (script or "").strip() else None
    return create_director(config, session.root, session=session.name,
                           should_cancel=_interrupted, on_status=_status_sender(unique_id),
                           overrides=overrides)


def tts_line(plan):
    """The line, in the bracketed-delivery form Fish Audio S2 reads: "[whispering] ..."."""
    if plan is None or not plan.spoken:
        return ""
    tone = plan.dialogue.emotion_tone.strip()
    return ("[%s] %s" % (tone, plan.dialogue.text)) if tone else plan.dialogue.text


# ── Pulse Infinity Loop ─────────────────────────────────────────────────────

class PulseInfinityLoopController:
    """The session: where the film is, and what beat renders on this queue run."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_SLATE_CAST + 1):
            optional["cast.cast_%d" % i] = ("PULSE_CAST", {"tooltip":
                "A Pulse Character. The director cites them as @Name, memory pins their "
                "description as their identity, and their voice is the reference for "
                "text-to-speech when they speak. Wire the same characters into Pulse "
                "Slate's cast, with role voice_timbre."})
        return {
            "required": {
                "schema_version": schema_widget(),
                "session": ("STRING", {"default": "clock_garden", "multiline": False,
                                       "tooltip":
                    "The film's folder under ComfyUI/output/pulse_infinity/. Same name, "
                    "same film: queue again to continue it. A new name starts a new one."}),
                "story_goal": ("STRING", {"default": "Escape the labyrinth before sundown",
                                          "multiline": True, "dynamicPrompts": False,
                                          "tooltip":
                    "Where the story is heading. The director plans every beat toward "
                    "it, and may end the film once it is reached on screen."}),
                "world": ("STRING", {"default": "", "multiline": True,
                                     "dynamicPrompts": False, "tooltip":
                    "The world, the tone, standing direction -- anything the director "
                    "should know on every beat. Shown to it as the user's direction."}),
                "opening_beat": ("STRING", {"default": "", "multiline": True,
                                            "dynamicPrompts": False, "tooltip":
                    "Optional. Write beat 1 yourself, as `visual: ...` lines (camera:, "
                    "say: Name | tone | line, hold:, wear:, where:, seconds:). Empty "
                    "lets the director open the film."}),
                "max_beats": ("INT", {"default": 12, "min": 0, "max": 9999, "tooltip":
                    "Stop after this many beats. 0 is open-ended: the film runs until "
                    "the director ends it or you stop it."}),
                "director_backend": ([FROM_CONFIG] + list(BACKENDS), {
                    "default": FROM_CONFIG, "tooltip":
                    "Who directs. 'from_config' uses $PULSE_DIRECTOR_BACKEND, then "
                    "director_backend in pulse_infinity.yaml, then claude_code."}),
                "script": ("STRING", {"default": "", "multiline": True,
                                      "dynamicPrompts": False, "tooltip":
                    "For the scripted director: every beat, separated by `---` lines, "
                    "in the same format as opening_beat. Ignored by other directors."}),
                "dialogue_mode": (list(DIALOGUE_MODES), {"default": DIALOGUE_NATIVE,
                                                         "tooltip":
                    "'native': H3 speaks each line itself, in the cast's voice timbre. "
                    "'external_tts': wire a TTS node from `tts_text` and `speaker_voice` "
                    "into the Dynamic Shot's dialogue_audio; it runs only on beats that "
                    "speak, and the mouth follows the recording."}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "tooltip":
                    "The `seed` output is this plus the beat and attempt -- for a TTS node "
                    "or anything else that wants a per-beat seed. H3's own window seeds "
                    "already move with every beat and every reroll."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": optional,
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("PULSE_INFINITY", "INT", "STRING", "AUDIO", "STRING")
    RETURN_NAMES = ("infinity", "seed", "tts_text", "speaker_voice", "status")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Pulse Studio Infinity's session. Each queue run renders the next beat "
                   "the director planned; queue N times for N beats.")

    @classmethod
    def IS_CHANGED(cls, session="", director_backend=FROM_CONFIG, **kwargs):
        """The session's `<uid>:<revision>`, read from disk at execution start.

        It moves on every committed beat and every control action, so N queued
        copies of this graph render N beats; it carries the uid, so a reset
        session never matches a value ComfyUI's cache still holds.
        """
        config = load_config(director_backend)
        root = os.path.join(folder_paths.get_output_directory(),
                            str(config.data.get("sessions_dir") or "pulse_infinity"),
                            safe_session_name(session))
        return peek_revision(root)

    def execute(self, schema_version, session, story_goal, world, opening_beat, max_beats,
                director_backend, script, dialogue_mode, base_seed, unique_id=None, **kwargs):
        config = load_config(director_backend)
        try:
            film = _session(session, config)
        except SessionError as exc:
            log.error("[PulseStudio] %s", exc)
            return _blocked(5, str(exc))
        cast, voices = _cast_of(kwargs)
        notes = film.configure(story_goal, world, cast=cast)
        film.set_limits(max_beats)
        notes += film.take_control()
        for name, voice in voices.items():
            if voice.get("role", AUDIO_ROLE_LIP_SYNC) == AUDIO_ROLE_LIP_SYNC:
                notes.append(
                    "%s's voice is a lip_sync recording on the film clock. In a film "
                    "rendered beat by beat the clock restarts every beat, so every beat "
                    "would hear its first seconds as speech -- set that Pulse Voice's "
                    "role to voice_timbre." % name)

        if film.status in (STATUS_COMPLETE, STATUS_PAUSED):
            message = "%s -- %s" % (film.describe(), "the film is complete"
                                    if film.status == STATUS_COMPLETE else "paused")
            log.info("[PulseStudio] %s", message)
            _send("pulse_infinity.%s" % film.status, {"session": film.name,
                                                       "beat": film.state["beat"]})
            film.save()
            return _blocked(5, None)

        try:
            director = _director(film, config, unique_id, script)
            if not director.can_continue(film.pending_beat):
                film.control("stop")
                return _blocked(5, None)
            if (not film.state.get("pending_plan") and config.backend in LOCAL_BACKENDS
                    and director.settings.get("free_vram_before_call", True)):
                _free_vram()
            plan = film.plan_for_pending(director, opening=opening_beat)
        except (DirectorError, SessionError) as exc:
            message = "Pulse Infinity could not plan beat %d: %s" % (film.pending_beat, exc)
            log.error("[PulseStudio] %s", message)
            return _blocked(5, message)
        # Both writes are idempotent: the goal, the limits and the cast as the
        # widgets now say, and the memory with any newly wired cast member in it.
        film.save()
        store.save_memory(film.root, film.memory)

        speaker = plan.dialogue.speaker.lstrip("@") if plan.spoken else ""
        voice = voices.get(speaker.casefold()) if speaker else None
        handle = {
            "session": film.name, "root": film.root, "config": config,
            "backend_widget": director_backend, "script": script,
            "beat": film.pending_beat, "attempt": film.attempt,
            "epoch": film.state["epoch"], "plan": plan.to_dict(),
            "shot_id": film.shot_id(), "continue_from": film.continue_from(),
            "run_location": film.run_location(), "cast": cast,
            "dialogue_mode": dialogue_mode, "speaker": speaker,
            "speaker_has_voice": voice is not None, "revision": film.revision_token,
            "director": director.describe(), "config_source": config.describe(),
        }
        seed = (int(base_seed) + 1000 * film.pending_beat + film.attempt) & 0x7FFFFFFF
        status = "\n".join([film.describe(), "director: " + config.describe(),
                            "           " + director.describe(),
                            "beat %d: %s" % (plan.shot_index, plan.beat or plan.visual_prompt[:160])]
                           + ["note: " + n for n in notes + plan.notes])
        log.info("[PulseStudio] %s", status.replace("\n", " | "))
        speaker_audio = voice.get("audio") if voice else media.empty_audio()
        return (handle, seed, tts_line(plan), speaker_audio, status)


# ── Pulse Dynamic Shot ──────────────────────────────────────────────────────

class PulseDynamicShot:
    """The director's plan for this beat, as a Pulse Shot payload."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "infinity": ("PULSE_INFINITY", {"tooltip": "From Pulse Infinity Loop."}),
                "schema_version": schema_widget(),
                "style_suffix": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "Appended to every beat's description -- a look you want held for "
                    "the whole film ('35mm, natural light, shallow focus')."}),
                "dialogue_language": ("STRING", {"default": "", "multiline": False,
                                                 "tooltip":
                    "The language the lines are spoken in, for H3's dialogue tag. "
                    "Empty uses the film's own (English unless the slate says otherwise)."}),
                "continuity": (list(SHOT_CONTINUITY_MODES), {"default": CONTINUITY_INHERIT,
                                                            "tooltip":
                    "This shot's continuity, as on Pulse Shot. 'inherit' follows the "
                    "slate, which for Infinity should be 'none': ref2va with the last "
                    "beat's final frame as <Picture 1>, the cast references still live."}),
                "carry": (list(CARRY_MODES), {"default": "auto", "tooltip":
                    "Open on the last frame of the previous beat? 'auto': yes, unless "
                    "the director cut to a new scene. 'always' / 'never' override it."}),
                "anchors_in_prompt": ("BOOLEAN", {"default": True, "tooltip":
                    "Restate the plan's continuity anchors -- what is held, the costume, "
                    "the place -- in the shot text. The renderer has no memory of its own."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "dialogue_audio": ("AUDIO", {"lazy": True, "tooltip":
                    "The spoken line from a TTS node (dialogue_mode external_tts). "
                    "Evaluated only on a beat that speaks, so the TTS node does not run "
                    "for silent beats. Becomes the shot's lip_sync voice."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("PULSE_SHOT", "STRING")
    RETURN_NAMES = ("shot", "visual")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Turns the director's plan for this beat into a Pulse Shot: camera, "
                   "action, continuity anchors, the line, and where it continues from. "
                   "Wire into Pulse Slate's shots.shot_1.")

    @staticmethod
    def _wants_take(infinity):
        return bool(infinity and infinity.get("dialogue_mode") == DIALOGUE_TTS
                    and infinity.get("speaker") and infinity.get("speaker_has_voice"))

    def check_lazy_status(self, **kwargs):
        """Ask for the TTS take only when this beat speaks and one is wired.

        An optional input that is not connected is absent from kwargs, and asking
        for it raises; one that is connected and not yet evaluated is present as
        None. Only the second is requested.
        """
        infinity = kwargs.get("infinity")
        if ("dialogue_audio" in kwargs and kwargs["dialogue_audio"] is None
                and self._wants_take(infinity)):
            return ["dialogue_audio"]
        return []

    def execute(self, infinity, schema_version, style_suffix, dialogue_language, continuity,
                carry, anchors_in_prompt, dialogue_audio=None, unique_id=None, **kwargs):
        plan, problems = DirectedShotPlan.from_dict((infinity or {}).get("plan"))
        if plan is None:
            message = "the Loop Controller handed over no usable plan: %s" % "; ".join(problems)
            return _blocked(2, message)

        parts = []
        if plan.camera_motion:
            parts.append(plan.camera_motion[0].upper() + plan.camera_motion[1:].rstrip(".") + ".")
        parts.append(plan.visual_prompt.strip())
        if anchors_in_prompt and plan.continuity_anchors.sentence():
            parts.append(plan.continuity_anchors.sentence())
        if style_suffix.strip():
            parts.append(style_suffix.strip())
        visual = " ".join(p for p in parts if p)

        speaker = infinity.get("speaker") or ""
        audio_line = ""
        if plan.spoken:
            tone = plan.dialogue.emotion_tone.strip()
            audio_line = '@%s%s: "%s"' % (speaker, (", " + tone) if tone else "",
                                          plan.dialogue.text.strip().replace('"', "'"))

        voice = None
        duration = plan.rendered_seconds
        if dialogue_audio is not None and self._wants_take(infinity):
            voice = PulseVoice().execute(
                schema_version=SCHEMA_VERSION, name="%sLine" % speaker,
                role=AUDIO_ROLE_LIP_SYNC, aligns_to="shot_start", offset_seconds=0.0,
                trim_start=0.0, trim_end=-1.0, speaker=speaker,
                description="%s speaking this beat's line" % speaker, gate_to_shot=True,
                language=dialogue_language, audio=dialogue_audio)[0]
            take = media.audio_length_seconds(dialogue_audio) or 0.0
            # Long enough for the take and its pad, on the grid, inside one window.
            needed = align_frame_count(int(round((take + AUTO_SHOT_PAD_SECONDS) * FPS)))
            duration = frames_to_seconds(min(MAX_WINDOW_FRAMES, max(plan.frames, needed)))

        continue_from = infinity.get("continue_from")
        if carry == "never" or (carry == "auto" and plan.cuts):
            continue_from = None

        payload = {
            "shot_id": infinity["shot_id"],
            "label": plan.beat or "Beat %d" % plan.shot_index,
            "visual": visual,
            "audio_line": audio_line,
            "duration_seconds": duration,
            "continuity": continuity or CONTINUITY_INHERIT,
            "start_image": None, "end_image": None, "ref_images": [],
            "ref_audio": None, "ref_audio_mode": AUDIO_ROLE_LIP_SYNC,
            "voice": voice, "speaker": ("@" + speaker) if speaker else "",
            "continue_from": continue_from,
        }
        preview = "beat %d (%.2f s)%s\n%s%s" % (
            plan.shot_index, duration, "" if continue_from else " -- opens on a cut",
            visual, ("\n" + audio_line) if audio_line else "")
        return (payload, preview)


# ── Pulse Infinity Render ───────────────────────────────────────────────────

def _segment_record(output_dir, document, run_dir, run_id, session_root):
    """What render.run left for this beat, relative to the session folder."""
    directory = os.path.join(output_dir, *run_dir.split("/"), run_id)
    segments = []
    for window in document.get("windows") or []:
        key = window.get("cache_key")
        if not key:
            continue
        names = segment_paths(window["window_index"], key)
        entry = {"frames": int(window["frames"])}
        for field, name in (("video", "video_path"), ("audio", "audio_path"),
                            ("last_frame", "last_frame_path")):
            path = os.path.join(directory, names[name])
            entry[field] = (os.path.relpath(path, session_root).replace("\\", "/")
                            if os.path.exists(path) else "")
        segments.append(entry)
    windows = document.get("windows") or [{}]
    fps = int(windows[0].get("fps") or FPS)
    assembled = os.path.join(directory, "assembled.mp4")
    return {"last_key": windows[-1].get("cache_key") or "", "segments": segments,
            "fps": fps, "width": windows[0].get("width"), "height": windows[0].get("height"),
            "seconds": round(sum(s["frames"] for s in segments) / float(fps), 4),
            "clip": (os.path.relpath(assembled, session_root).replace("\\", "/")
                     if os.path.exists(assembled) else ""),
            "run_dir": run_dir, "run_id": run_id}


class PulseInfinityRender:
    """Pulse Render, pinned to the session: this beat's own run folder, continuing the last."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("PULSE_TIMELINE", {"tooltip":
                    "From the Pulse Slate whose shots.shot_1 is the Dynamic Shot."}),
                "infinity": ("PULSE_INFINITY", {"tooltip": "From Pulse Infinity Loop."}),
                "model": ("MODEL", {"tooltip":
                    "The ref2va checkpoint, every patch applied upstream (HyperFlow, "
                    "attention, memory), as for Pulse Render."}),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "schema_version": schema_widget(),
                "cache_mode": (list(CACHE_MODES), {"default": "auto", "tooltip":
                    "As on Pulse Render. 'auto' reuses this beat's segments when it is "
                    "re-run unchanged -- a resume after a crash renders nothing twice."}),
                "low_memory": ("BOOLEAN", {"default": True}),
                # 'audio', not Pulse Render's audio+colour: a beat is usually one
                # window, and the colour match can only correct a seam whose both
                # sides are still in memory -- never the one into the last beat.
                "seam_treatment": (list(SEAM_MODES), {"default": SEAM_AUDIO, "tooltip":
                    "Seams between this beat's own windows. The seam into the last "
                    "beat is joined in the reel (Pulse Observer's seam_treatment)."}),
                "audio_mode": (["auto"] + list(AUDIO_MODES), {"default": "auto", "tooltip":
                    "'auto': lock_source when the beat carries a TTS take, so the mouth "
                    "follows it; reference_only otherwise."}),
                "remix_strength": ("FLOAT", {"default": DEFAULT_REMIX_STRENGTH, "min": 0.0,
                                             "max": 1.0, "step": 0.05}),
                "ref_image_fit": (list(REF_FIT_MODES), {"default": REF_FIT_CANVAS}),
                "reference_audio": (list(TRISTATE), {"default": "auto", "tooltip":
                    "Mux the supplied take instead of H3's decoded track. 'auto': when "
                    "the beat carries one."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "model_fl2va": ("MODEL", {"tooltip":
                    "Only for continuity modes that pin a frame. Infinity's default, "
                    "'none', never samples on it."}),
            },
            "hidden": HIDDEN_INPUTS_WITH_PROMPT,
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "PULSE_BEAT", "STRING")
    RETURN_NAMES = ("video", "frames", "audio", "beat", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Renders this beat into the session folder, opening on the last frame "
                   "of the beat before it, and hands the result to Pulse Observer.")

    def execute(self, timeline, infinity, model, vae, audio_vae, schema_version, cache_mode,
                low_memory, seam_treatment, audio_mode, remix_strength, ref_image_fit,
                reference_audio, model_fl2va=None, unique_id=None, prompt=None):
        document, side = _unwrap_timeline(timeline)
        if document is None or not infinity:
            return _blocked(5, "Connect a Pulse Slate `timeline` and the Loop's `infinity`.")
        windows = document.get("windows") or []
        warnings = []
        expected = infinity.get("continue_from")
        carried = (windows[0].get("carry_in") if windows else None)
        if expected and not carried:
            warnings.append(
                "this beat should continue beat %d, but the timeline does not carry it in. "
                "Is Pulse Slate's shots.shot_1 wired from the Dynamic Shot?"
                % (infinity["beat"] - 1))
        if carried and windows and windows[0].get("branch") == "fl2va":
            warnings.append(
                "this beat continues the last one on the fl2va branch, which takes no "
                "references: no face or voice reaches it. Set Pulse Slate's continuity "
                "to 'none' to keep the cast references alive.")

        has_take = any(a.kind == KIND_AUDIO and a.audio_role == AUDIO_ROLE_LIP_SYNC
                       for refs in ((getattr(side.timeline, "local_refs", None) or {}).values()
                                    if side is not None and side.timeline is not None else ())
                       for a in refs)
        if audio_mode == "auto":
            mode = AUDIO_MODE_LOCK if has_take else AUDIO_MODE_REFERENCE
        else:
            mode = audio_mode
        use_reference = has_take if reference_audio == "auto" else reference_audio == "on"
        run_dir, run_id = infinity["run_location"]
        options = render.RenderOptions(
            cache_mode=cache_mode, run_dir=run_dir, run_id=run_id, save_segments=True,
            low_memory=low_memory, dry_run=False, prune_unused=False,
            use_reference_audio=use_reference, seam_treatment=seam_treatment,
            audio_mode=mode, remix_strength=remix_strength, ref_image_fit=ref_image_fit,
            want_frames=render.output_is_connected(prompt, unique_id, 1))
        try:
            result = render.run(document, side, model, vae, audio_vae,
                                model_fl2va=model_fl2va, options=options)
        except (ReuseOnlyMiss, CarryInError, SocketGapError) as exc:
            message = "Beat %d was not rendered: %s" % (infinity["beat"], exc)
            log.error("[PulseStudio] %s", message)
            return _blocked(5, message)

        output_dir = folder_paths.get_output_directory()
        record = _segment_record(output_dir, document, run_dir, run_id, infinity["root"])
        film = _session(infinity["session"], infinity["config"])
        film.record_render(infinity["beat"], infinity["attempt"], record)
        beat = {"session": infinity["session"], "root": infinity["root"],
                "beat": infinity["beat"], "attempt": infinity["attempt"], "record": record,
                "warnings": warnings + list(result.warnings)}
        report = "\n".join(["Pulse Infinity: beat %d, attempt %d -> %s/%s"
                            % (infinity["beat"], infinity["attempt"], run_dir, run_id)]
                           + ["!! " + w for w in warnings] + [result.report or ""])
        frames = result.frames if result.frames is not None else media.empty_images()
        audio = result.audio if result.audio is not None else media.empty_audio()
        video = result.video if result.video is not None else ExecutionBlocker(
            "The beat rendered but its segments could not be joined; see the report.")
        return (video, frames, audio, beat, report)


# ── Pulse Observer ──────────────────────────────────────────────────────────

def _evidence(film, beat, attempt, record, frames_wanted, sheet, images=None):
    """Stills of the beat on disk, for the director: frames, last frame, contact sheet."""
    folder = os.path.join(film.beat_dir(beat), "a%d" % attempt, "evidence")
    os.makedirs(folder, exist_ok=True)
    total = sum(int(s.get("frames") or 0) for s in record.get("segments") or [])
    count = max(2, int(frames_wanted))
    if images is not None and images.shape[0] > 0:
        total = int(images.shape[0])
    indices = sorted({int(round(i * (total - 1) / float(count - 1))) for i in range(count)}) \
        if total > 1 else [0]
    if images is not None and images.shape[0] > 0:
        stills = images[indices]
    else:
        paths = [os.path.join(film.root, *s["video"].split("/"))
                 for s in record.get("segments") or [] if s.get("video")]
        stills = media.sample_frames(paths, indices)
    rel = []
    if stills is not None:
        for n, image in enumerate(stills):
            path = os.path.join(folder, "f%02d.png" % (n + 1))
            media.write_png(path, image.unsqueeze(0))
            rel.append(film.relative(path))
    last = os.path.join(folder, "last.png")
    if stills is not None:
        media.write_png(last, stills[-1:].clone())
    fps = float(record.get("fps") or FPS)
    sheet_image = None
    sheet_rel = ""
    if sheet and stills is not None:
        sheet_image = media.contact_sheet(stills, columns=3, labels=[
            "%d  %.2fs" % (n + 1, i / fps) for n, i in enumerate(indices)])
        path = os.path.join(folder, "contact.png")
        media.write_png(path, sheet_image)
        sheet_rel = film.relative(path)
    clip = record.get("clip") or ""
    evidence = EvidencePack(film.root, beat, clip=clip,
                            audio=(record.get("segments") or [{}])[0].get("audio", ""),
                            last_frame=film.relative(last) if os.path.exists(last) else "",
                            frames=rel, contact_sheet=sheet_rel)
    return evidence, sheet_image


def _assemble_reel(film, seam):
    """reel/reel.mp4 from every committed beat, in order. Returns (path, problem)."""
    entries = film.state.get("reel") or []
    if not entries:
        return "", ""
    canvases = {(e.get("width"), e.get("height"), e.get("fps")) for e in entries}
    if len(canvases) > 1:
        return "", ("the beats are not all one canvas (%s); the reel is not assembled"
                    % ", ".join("%sx%s@%s" % c for c in sorted(canvases, key=str)))
    videos, counts, clips = [], [], []
    for entry in entries:
        fps = float(entry.get("fps") or FPS)
        for segment in entry.get("segments") or []:
            if not segment.get("video"):
                continue
            videos.append(os.path.join(film.root, *segment["video"].split("/")))
            counts.append(int(segment["frames"]))
            clip = (media.load_audio(os.path.join(film.root, *segment["audio"].split("/")))
                    if segment.get("audio") else None)
            clips.append((clip, int(segment["frames"]) / fps))
    # Every segment's sound trimmed or padded to exactly its frames -- a few samples
    # of drift per beat is a lip-sync error by beat fifty -- and a segment with no
    # sound becomes silence at the same rate as the rest, not at a default one.
    rate = next((c["sample_rate"] for c, _ in clips if c is not None), 44100)
    audio = [media.audio_span(c if c is not None else media.empty_audio(rate), 0.0, seconds)
             for c, seconds in clips]
    joined = media.concat_audio(audio, seam_treatment=seam)
    out = os.path.join(film.root, "reel", "reel.mp4")
    path = media.concat_videos(videos, out, frame_counts=counts,
                               fps=int(entries[0].get("fps") or FPS), audio=joined,
                               replace_audio=True)
    return path, ("" if path else "joining the beats into reel.mp4 failed; see the log")


class PulseObserver:
    """Shows the beat to the director, commits what it saw, stores the next plan."""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beat": ("PULSE_BEAT", {"tooltip": "From Pulse Infinity Render."}),
                "infinity": ("PULSE_INFINITY", {"tooltip": "From Pulse Infinity Loop."}),
                "schema_version": schema_widget(),
                "evidence_frames": ("INT", {"default": 5, "min": 2, "max": 12, "tooltip":
                    "How many evenly spaced stills the director is shown, first to last."}),
                "contact_sheet": ("BOOLEAN", {"default": True, "tooltip":
                    "Also lay the stills out in one captioned image -- one read shows "
                    "the whole beat's motion."}),
                "free_vram_before_director": (list(TRISTATE), {"default": "auto", "tooltip":
                    "Unload ComfyUI's models before the director thinks. 'auto': for a "
                    "local director (ollama, pi), which needs the same GPU."}),
                "seam_treatment": (list(SEAM_MODES), {"default": SEAM_DEFAULT, "tooltip":
                    "How the reel joins one beat's sound to the next."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "images": ("IMAGE", {"tooltip":
                    "Optional: the beat's frames after lip-sync correction (Pulse Lip-Sync "
                    "Paste). They are what the director is shown and what enters the reel."}),
                "audio": ("AUDIO", {"tooltip": "The sound to go with `images`."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("report", "evidence", "next_beat")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The director watches the beat, memory records what happened, and the "
                   "next beat is planned. Commits each beat once, however often it runs.")

    def execute(self, beat, infinity, schema_version, evidence_frames, contact_sheet,
                free_vram_before_director, seam_treatment, images=None, audio=None,
                unique_id=None, **kwargs):
        config = infinity["config"]
        film = _session(infinity["session"], config)
        film.configure(cast=infinity.get("cast") or [])
        record = beat["record"]
        number, attempt = int(beat["beat"]), int(beat["attempt"])
        if images is not None and audio is not None:
            # Lip-sync corrected: this beat's own file, and that is what the reel uses.
            final = os.path.join(film.beat_dir(number), "a%d" % attempt, "final.mp4")
            media.write_segment_video(final, images, audio, fps=int(record.get("fps") or FPS))
            final_audio = os.path.join(film.beat_dir(number), "a%d" % attempt, "final.flac")
            media.write_segment_audio(final_audio, audio)
            record = dict(record, clip=film.relative(final), segments=[{
                "video": film.relative(final), "audio": film.relative(final_audio),
                "frames": int(images.shape[0]), "last_frame": ""}])
        evidence, sheet = _evidence(film, number, attempt, record, evidence_frames,
                                    contact_sheet, images)
        try:
            director = _director(film, config, unique_id, infinity.get("script") or "")
            local = config.backend in LOCAL_BACKENDS
            if free_vram_before_director == "on" or (free_vram_before_director == "auto"
                                                     and local):
                _free_vram()
            outcome = film.commit_rendered(director, evidence, record, beat=number,
                                           attempt=attempt)
        except (DirectorError, SessionError) as exc:
            message = ("Beat %d rendered, but the director's turn failed: %s\nNothing was "
                       "committed; queue again to retry the turn." % (number, exc))
            log.error("[PulseStudio] %s", message)
            _send("pulse_infinity.error", {"session": film.name, "beat": number,
                                           "message": str(exc)})
            return {"ui": {"text": [message]}, "result": _blocked(3, message)}

        reel_note = ""
        every = max(1, int(config.loop("reel_every") or 1))
        if not outcome.replayed and (outcome.complete or film.state["beat"] % every == 0):
            path, problem = _assemble_reel(film, seam_treatment != "off")
            reel_note = ("reel: " + film.relative(path)) if path else ("reel: " + problem)

        lines = ["Pulse Infinity -- %s" % film.describe(),
                 "director: %s" % infinity.get("director", config.describe())]
        if outcome.observation is not None:
            obs = outcome.observation
            lines.append("beat %d seen (%s): %s" % (number, obs.provenance, obs.summary))
            if obs.completed_actions:
                lines.append("  completed: " + "; ".join(obs.completed_actions))
            if obs.continuity_issues:
                lines.append("  drift: " + "; ".join(obs.continuity_issues))
        if film.memory.forbidden_actions:
            lines.append("forbidden now: " + "; ".join(film.memory.forbidden_actions))
        next_text = ""
        if outcome.next_plan is not None:
            plan = outcome.next_plan
            next_text = "beat %d: %s" % (plan.shot_index, plan.beat or plan.visual_prompt)
            lines.append("next " + next_text)
            if plan.spoken:
                lines.append('  line: %s (%s): "%s"' % (plan.dialogue.speaker,
                                                        plan.dialogue.emotion_tone,
                                                        plan.dialogue.text))
        elif outcome.complete:
            lines.append("the film is complete")
        if reel_note:
            lines.append(reel_note)
        lines += ["note: " + n for n in outcome.notes] + ["!! " + w for w in
                                                          beat.get("warnings") or []]
        report = "\n".join(lines)
        _send("pulse_infinity.beat_done", {
            "session": film.name, "beat": number, "complete": outcome.complete,
            "continue": (not outcome.complete) and film.status == "active",
            "revision": film.revision_token})
        evidence_image = sheet if sheet is not None else media.empty_images()
        return {"ui": {"text": [report]}, "result": (report, evidence_image, next_text)}


NODE_CLASS_MAPPINGS = {
    "PulseInfinityLoopController": PulseInfinityLoopController,
    "PulseDynamicShot": PulseDynamicShot,
    "PulseInfinityRender": PulseInfinityRender,
    "PulseObserver": PulseObserver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulseInfinityLoopController": "Pulse Infinity Loop",
    "PulseDynamicShot": "Pulse Dynamic Shot",
    "PulseInfinityRender": "Pulse Infinity Render",
    "PulseObserver": "Pulse Observer",
}
