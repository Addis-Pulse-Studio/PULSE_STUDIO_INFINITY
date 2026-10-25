#!/usr/bin/env python3
"""Build the Pulse Studio Infinity example graphs.

    python3 tools/build_infinity_workflows.py [out_dir]

Writes two ComfyUI UI-format graphs (default: example_workflows/):

  PulseInfinity_Scripted.json    the closed loop with the offline scripted
                                 director and stock loaders only -- the graph to
                                 wire-check a box with, no model director needed
  PulseInfinity_ClosedLoop.json  HyperFlow 8-step on the full ref2va checkpoint,
                                 the director from pulse_infinity.yaml (Claude
                                 Code by default), and a Fish Audio S2 voice-clone
                                 branch for dialogue that only runs on beats that
                                 speak

Both are one beat per queue run: queue N times for N beats, or turn on
auto-continue in the Pulse Infinity Loop node's context menu.

The layout rules the Oct-2026 builder follows hold here too -- its Graph class,
loaders and model names are imported from it rather than copied, and that
builder is left untouched (tests/test_oct2026_workflows.py pins it to its own ten
graphs). Pulse widget arrays follow js/ps_widget_order.js; stock-node arrays are
the ones those nodes save on the target ComfyUI. Deterministic: rebuilding writes
byte-identical files.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_oct2026_workflows as oct  # noqa: E402

SCHEMA = oct.SCHEMA
EMPTY_BIN = oct.EMPTY_BIN
BASE_REF2VA = "minimax\\minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FACE = "example_pulse_subject_a.png"
VOICE = "example_pulse_voice.wav"
FISH_MODEL = "s2-pro-fp8"

STORY_GOAL = "Escape the labyrinth before sundown"
WORLD = ("A storybook clock garden at twilight: hedges cut into a maze, a stopped clock "
         "tower at its centre, amber light failing into blue. Live-action, 35mm, soft grain. "
         "Keep it gentle and wondrous; no horror.")
SCRIPT = """beat: Alice leaves the clock behind
scene: Clock garden
visual: @Alice turns away from the stopped clock and steps onto the dark garden path.
camera: slow dolly forward, low angle
say: Alice | whispering, anxious | The time has already stopped...
hold: brass key in left hand
wear: torn blue apron dress
where: overgrown clock garden, twilight
---
beat: Alice reaches the hedge gate
visual: @Alice hurries along the path to an iron gate set deep in the hedge.
camera: tracking shot beside her, waist height
hold: brass key in left hand
wear: torn blue apron dress
where: hedge maze edge, twilight
---
beat: Alice unlocks the gate and slips through
visual: @Alice turns the brass key in the gate's lock and slips through into the maze.
camera: close on the lock, then pull back as the gate swings
cut: no
hold: brass key in left hand
wear: torn blue apron dress
where: inside the hedge maze, deepening dusk"""

DESCRIPTION = ("a girl of about ten with long fair hair, wearing a torn blue apron dress "
               "and black shoes")

SCRIPTED_NOTE = """# Pulse Studio Infinity -- the closed loop, scripted

**One queue run renders one beat.** Queue this graph three times (or set the batch
count to 3) and the film plays through the three beats in the Loop node's `script`.

Each run: the **Loop** node reads the session folder, the **Dynamic Shot** turns
the planned beat into a shot, **Pulse Slate** compiles it with the cast, **Pulse
Infinity Render** renders it -- opening on the last frame of the beat before -- and
the **Observer** shows the result to the director, commits it to the scene memory
and plans the next beat.

This graph uses the **scripted** director: no model, nothing leaves the machine.
It is how you check the wiring and the GPU before handing the film to a real
director. Change `director_backend` to `from_config` (or `claude_code`) and the
director in pulse_infinity.yaml takes over -- see PulseInfinity_ClosedLoop.

Everything lands in `ComfyUI/output/pulse_infinity/<session>/`: the scene memory,
each beat's render folder and stills, the Director Desk, and the reel of every
committed beat. The Observer's report says what it saw and what comes next.

Settings that matter, already set: Pulse Slate continuity **none** (the last
beat's final frame rides in as a reference, the cast references stay live) and
carry_audio **off** (each beat's sound is its own; the reel joins them). The cast
voice is a Pulse Voice with role **voice_timbre**: a lip_sync recording sits on a
film clock that restarts every beat.
"""

CLOSED_LOOP_NOTE = """# Pulse Studio Infinity -- a film that directs itself

**One queue run renders one beat.** Queue it, or right-click the Loop node and
turn on **auto-continue**: the next beat is queued when this one is committed,
until the director ends the story or `max_beats` is reached. Pause, resume,
reroll or replan from the same menu.

**The director** is chosen by `director_backend` (default `from_config`): the
PULSE_DIRECTOR_BACKEND environment variable, else `director_backend` in
`ComfyUI/user/pulse_infinity.yaml`, else **claude_code**. Claude Code and
Antigravity direct through the Director Desk -- files in the session folder --
so start one first: `/pulse-director` (or `/loop /pulse-director`) in Claude
Code, `/pulse-direct` in Antigravity, in the director workspace that
`tools/setup_director_workspace.py` installs. Ollama and pi run through their
own command lines. With a cloud director the beat's stills (and for Gemini the
clip) are read by that agent; the pack itself sends nothing.

**Models:** the full ref2va checkpoint with the HyperFlow 8-step adapter, on the
`hyperflow_8step` recipe (cfg 1.0 -- negative prompts are enforced by the
director's validator, not by the sampler).

**Dialogue:** `dialogue_mode` native lets H3 speak each line in the cast's voice
timbre. Set it to **external_tts** and the Fish Audio S2 voice-clone node speaks
the line in the speaking character's own recording; it runs only on beats that
speak (keep_model_loaded is off: S2 needs about 20 GB). For mouth correction,
put Pulse Lip-Sync Segment -> LatentSync -> Pulse Lip-Sync Paste after the render
and wire the corrected images and audio into the Observer.
"""


def with_widget_inputs(node, names):
    """Mark inputs that are widgets fed by a link, as the frontend saves them."""
    for slot in node["inputs"]:
        if slot["name"] in names:
            slot["widget"] = {"name": slot["name"]}
    return node


def loop_node(g, pos, backend, script, dialogue_mode, max_beats, session, cast_slots=1):
    inputs = [("cast.cast_%d" % i, "PULSE_CAST") for i in range(1, cast_slots + 2)]
    return g.node("PulseInfinityLoopController",
                  [SCHEMA, session, STORY_GOAL, WORLD, "", max_beats, backend, script,
                   dialogue_mode, 0],
                  inputs=inputs,
                  outputs=[("infinity", "PULSE_INFINITY"), ("seed", "INT"),
                           ("tts_text", "STRING"), ("speaker_voice", "AUDIO"),
                           ("status", "STRING")],
                  pos=pos, size=(460, 820), title="Pulse Infinity Loop -- the session")


def dynamic_shot(g, pos):
    return g.node("PulseDynamicShot", [SCHEMA, "", "", "inherit", "auto", True],
                  inputs=[("infinity", "PULSE_INFINITY"), ("dialogue_audio", "AUDIO")],
                  outputs=[("shot", "PULSE_SHOT"), ("visual", "STRING")],
                  pos=pos, size=(400, 260), title="Pulse Dynamic Shot -- this beat")


def character(g, pos):
    return g.node("PulseCharacter", [SCHEMA, "Alice", DESCRIPTION, "fully_preserved"],
                  inputs=[("face", "IMAGE"), ("body", "IMAGE"), ("voice", "PULSE_VOICE")],
                  outputs=[("character", "PULSE_CAST")], pos=pos, size=(380, 260),
                  title="Alice -- face, description, voice")


def voice(g, pos):
    return g.node("PulseVoice",
                  [SCHEMA, "AliceVoice", "voice_timbre", "film_clock", 0.0, 0.0, -1.0,
                   "", "Alice's speaking voice", True, ""],
                  inputs=[("audio", "AUDIO"), ("final_audio", "AUDIO")],
                  outputs=[("voice", "PULSE_VOICE")], pos=pos, size=(380, 330),
                  title="Alice's voice -- voice_timbre, not lip_sync")


def slate(g, pos, steps=20, sampler="res_multistep", scheduler="simple"):
    inputs = [("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"),
              ("model_fl2va", "MODEL"), ("ref_video", "IMAGE"), ("ref_video_audio", "AUDIO"),
              ("ref_music", "AUDIO"), ("recipe", "PULSE_RECIPE"),
              ("recipe_fl2va", "PULSE_RECIPE"), ("refs.ref_image_1", "IMAGE"),
              ("shots.shot_1", "PULSE_SHOT"), ("shots.shot_2", "PULSE_SHOT"),
              ("voices.voice_1", "PULSE_VOICE"), ("cast.cast_1", "PULSE_CAST"),
              ("cast.cast_2", "PULSE_CAST")]
    widgets = [SCHEMA, EMPTY_BIN,
               "style: Live-action storybook fantasy, 35mm, soft grain, twilight light.\n"
               "soundscape: Wind through hedges, distant ticking that has stopped, "
               "soft footsteps on gravel.\nmusic: A sparse music-box motif.",
               "", 5.17, "16:9 landscape", 1344, 768, steps, sampler, scheduler, 1.0, 7,
               "fixed", "shot_aligned", 15.0, "crop", "image", False, 4.0, "match", 12.0,
               3.0, 3, "none"]
    return g.node("PulseSlate", widgets, inputs=inputs,
                  outputs=[("timeline", "PULSE_TIMELINE"), ("positive", "CONDITIONING"),
                           ("latent", "LATENT"), ("combined_audio", "AUDIO"),
                           ("images", "IMAGE"), ("compiled_prompt", "STRING")],
                  pos=pos, size=(700, 1500), title="Pulse Slate -- compiles the beat")


def infinity_render(g, pos):
    return g.node("PulseInfinityRender",
                  [SCHEMA, "auto", True, "audio", "auto", 0.35, "canvas", "auto"],
                  inputs=[("timeline", "PULSE_TIMELINE"), ("infinity", "PULSE_INFINITY"),
                          ("model", "MODEL"), ("vae", "VAE"), ("audio_vae", "VAE"),
                          ("model_fl2va", "MODEL")],
                  outputs=[("video", "VIDEO"), ("frames", "IMAGE"), ("audio", "AUDIO"),
                           ("beat", "PULSE_BEAT"), ("report", "STRING")],
                  pos=pos, size=(420, 330), title="Pulse Infinity Render -- this beat")


def observer(g, pos):
    return g.node("PulseObserver", [SCHEMA, 5, True, "auto", "audio"],
                  inputs=[("beat", "PULSE_BEAT"), ("infinity", "PULSE_INFINITY"),
                          ("images", "IMAGE"), ("audio", "AUDIO")],
                  outputs=[("report", "STRING"), ("evidence", "IMAGE"),
                           ("next_beat", "STRING")],
                  pos=pos, size=(420, 300), title="Pulse Observer -- the director's turn")


def preview_text(g, pos, title, size=(560, 420)):
    return g.node("PreviewAny", [], inputs=[("source", "*")],
                  outputs=[("STRING", "STRING")], pos=pos, size=size, title=title)


def wire_loop(g, model, clip, video_vae, audio_vae, loop, shot, slate_node, render_node,
              observer_node, cast):
    g.link(cast, "character", loop, "cast.cast_1")
    g.link(cast, "character", slate_node, "cast.cast_1")
    g.link(loop, "infinity", shot, "infinity")
    g.link(shot, "shot", slate_node, "shots.shot_1")
    g.link(model, 0, slate_node, "model")
    g.link(clip, "CLIP", slate_node, "clip")
    g.link(video_vae, "VAE", slate_node, "vae")
    g.link(audio_vae, "VAE", slate_node, "audio_vae")
    g.link(slate_node, "timeline", render_node, "timeline")
    g.link(loop, "infinity", render_node, "infinity")
    g.link(model, 0, render_node, "model")
    g.link(video_vae, "VAE", render_node, "vae")
    g.link(audio_vae, "VAE", render_node, "audio_vae")
    g.link(render_node, "beat", observer_node, "beat")
    g.link(loop, "infinity", observer_node, "infinity")


def common_outputs(g, loop, render_node, observer_node, x, y, prefix):
    save = g.node("SaveVideo", ["video/" + prefix, "auto", "auto", "auto"],
                  inputs=[("video", "VIDEO")], outputs=[("video", "VIDEO")], pos=(x, y),
                  size=(560, 520), title="This beat")
    g.link(render_node, "video", save, "video")
    sheet = g.node("PreviewImage", [], inputs=[("images", "IMAGE")], pos=(x, y + 560),
                   size=(560, 420), title="What the director was shown")
    g.link(observer_node, "evidence", sheet, "images")
    report = preview_text(g, (x + 600, y), "Observer: what it saw, what comes next")
    g.link(observer_node, "report", report, "source")
    status = preview_text(g, (x + 600, y + 460), "Loop: session and director")
    g.link(loop, "status", status, "source")


def cast_nodes(g, x, y):
    face = oct.load_image(g, FACE, (x, y), "Alice's face (placeholder -- use your own)")
    audio = g.node("LoadAudio", [VOICE], outputs=[("AUDIO", "AUDIO")], pos=(x, y + 380),
                   size=(320, 136), title="Alice's voice (placeholder)")
    voice_node = voice(g, (x + 360, y + 380))
    g.link(audio, "AUDIO", voice_node, "audio")
    cast = character(g, (x + 360, y))
    g.link(face, "IMAGE", cast, "face")
    g.link(voice_node, "voice", cast, "voice")
    return cast


def build_scripted():
    g = oct.Graph("PulseInfinity_Scripted")
    oct.note(g, SCRIPTED_NOTE, (-620, 0), size=(580, 980))
    model = oct.unet(g, BASE_REF2VA, (0, 0), "H3 ref2va (base)")
    clip, video_vae, audio_vae = oct.loaders(g, 0, 120)
    cast = cast_nodes(g, 0, 460)
    loop = loop_node(g, (760, 0), "scripted", SCRIPT, "native", 3, "clock_garden_scripted")
    shot = dynamic_shot(g, (760, 860))
    slate_node = slate(g, (1260, 0))
    render_node = infinity_render(g, (2000, 0))
    observer_node = observer(g, (2000, 380))
    wire_loop(g, model, clip, video_vae, audio_vae, loop, shot, slate_node, render_node,
              observer_node, cast)
    common_outputs(g, loop, render_node, observer_node, 2460, 0, "pulse_infinity_scripted")
    return g


def build_closed_loop():
    g = oct.Graph("PulseInfinity_ClosedLoop")
    oct.note(g, CLOSED_LOOP_NOTE, (-620, 0), size=(580, 1100))
    unet = oct.unet(g, oct.REF2VA_FULL, (0, 0), "H3 ref2va (full -- HyperFlow needs it)")
    model = oct.hyperflow(g, (0, 110), "HyperFlow 8-step adapter")
    g.link(unet, "MODEL", model, "model")
    clip, video_vae, audio_vae = oct.loaders(g, 0, 250)
    cast = cast_nodes(g, 0, 590)
    loop = loop_node(g, (760, 0), "from_config", "", "native", 12, "clock_garden")
    shot = dynamic_shot(g, (760, 860))
    recipe = oct.recipe(g, "hyperflow_8step", (1260, 1540), "Recipe: HyperFlow 8-step")
    slate_node = slate(g, (1260, 0))
    g.link(recipe, "recipe", slate_node, "recipe")
    render_node = infinity_render(g, (2000, 0))
    observer_node = observer(g, (2000, 380))
    wire_loop(g, model, clip, video_vae, audio_vae, loop, shot, slate_node, render_node,
              observer_node, cast)

    # Fish Audio S2 speaks the line in the speaking character's own recording. It
    # sits behind the Dynamic Shot's lazy dialogue_audio, so with dialogue_mode
    # native -- or on a silent beat -- it never runs.
    tts = g.node("FishS2VoiceCloneTTS",
                 [FISH_MODEL, "", "auto", "auto", "auto", "auto", 0, 200, 0.8, 0.8, 1.1, 0,
                  "fixed", False, True, False, ""],
                 inputs=[("reference_audio", "AUDIO"), ("text", "STRING"), ("seed", "INT")],
                 outputs=[("audio", "AUDIO")], pos=(760, 1160), size=(400, 520),
                 title="Fish Audio S2 -- the line, in the speaker's voice")
    with_widget_inputs(tts, ("text", "seed"))
    g.link(loop, "speaker_voice", tts, "reference_audio")
    g.link(loop, "tts_text", tts, "text")
    g.link(loop, "seed", tts, "seed")
    g.link(tts, "audio", shot, "dialogue_audio")
    common_outputs(g, loop, render_node, observer_node, 2460, 0, "pulse_infinity")
    return g


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for graph in (build_scripted(), build_closed_loop()):
        path = os.path.join(out_dir, graph.name + ".json")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(graph.to_json(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        written.append(path)
    return written


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(HERE),
                                                         "example_workflows")
    for path in build(out_dir):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
