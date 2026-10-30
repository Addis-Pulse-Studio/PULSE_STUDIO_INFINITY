# Pulse Studio — MiniMax H3 for ComfyUI

Direct a [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) render from a
shot timeline and an asset bin, on one node, without ever typing a reference tag
number.

Three things it does that the existing directors don't:

- **An Asset Bin that cannot misnumber a reference tag.** You never type
  `<Picture 3>`; you write `@Mimi`, and the ordinal is computed from live bin
  order at compile time.
- **A Retake Scissor** that patches a bad three seconds without re-rendering
  fifteen.
- **Still Mode**, so you can make your reference images in the same window you
  use them.
- **Pulse Studio Infinity**, a film that directs itself: H3 renders a beat, a
  director -- Claude Code, Gemini in Antigravity, a local model, or a script --
  looks at what came out, a scene memory records it, and the next beat opens on
  the last frame of this one. See [below](#pulse-studio-infinity--a-film-that-directs-itself).

Plus compiler correctness, which is invisible when it's right and destroys trust
when it's wrong.

> **POWERED BY MINIMAX H3.** This repository contains no model weights. The
> weights carry the
> [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3),
> a separate agreement governing their use and any output generated with them.

---

## Install

Clone into your ComfyUI installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Addis-Pulse-Studio/comfyui-pulse-studio
```

Restart ComfyUI. There is nothing to `pip install`: the pack has no
dependencies of its own, and the tensor layer uses `torch`, `numpy`, `Pillow`
and `av`, all of which a working ComfyUI already has. See
[requirements.txt](requirements.txt) for why that stays true.

**ComfyUI 0.30.0 or newer**, which is the version this was developed and tested
against. It has to be a build carrying `comfy_extras/nodes_minimax_h3.py` and
`comfy/ldm/minimax/` — without those, MiniMax H3 is not there to direct. There
is no mechanism for a custom node to enforce a host version, so this is
documentation rather than a check.

Then load one of the graphs in `example_workflows/`:

| graph | what it shows |
|---|---|
| `PulseSlate_Single.json` | The short path: one window, sampled by your own graph. Start here. |
| `PulseSlate_LongForm.json` | The long path: many windows, `PulseRender`, the segment cache. |
| `PulseSlate_Cast.json` | The Asset Bin — named references, descriptions, retention, per-shot audio. |
| `PulseSlate_Voice.json` | Lip sync: one narration, one `PulseVoice`, every window — and what the report says when the recording does not cover the film. |
| `PulseSlate_Retake.json` | Re-render a bad span of a finished film, in place. |
| `PulseInfinity_Scripted.json` | Pulse Studio Infinity's closed loop on the offline scripted director: queue it three times, three beats. |
| `PulseInfinity_ClosedLoop.json` | The closed loop with HyperFlow 8-step, your director (Claude Code by default) and a Fish Audio S2 dialogue branch. Needs ComfyUI-HyperFlow and ComfyUI-FishAudioS2. |

Every one of them opens with no third-party pack installed. The placeholder
references `PulseSlate_Cast.json` cites are copied into ComfyUI's `input/` folder
the first time this pack loads, so its bin opens resolved rather than pointing at
files you have not got — they are flat colour and a sine tone, and the point is
to replace them.

## Updating

New work lands on `main`, so updating is a pull in place:

```bash
cd ComfyUI/custom_nodes/comfyui-pulse-studio
git pull
```

Restart ComfyUI afterwards — the pack is read at startup, and a running server
goes on serving the code it loaded. Nothing else is needed: there are still no
dependencies to reinstall, and workflows you have already saved keep loading,
since widget values migrate by name and a slot that changed type is dropped
rather than silently mis-wired.

## Models

Not redistributed here. Download from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and place
them as below — the `minimax/` subfolder is what the example workflows expect:

| file | goes in | approx |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax/` | 20 GB |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax/` | 20 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/minimax/` | 15 GB |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/minimax/` | 4.9 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/minimax/` | 578 MB |

Those are the exact files the example graphs were built and tested against, and
the sizes are measured from that install. Nothing here is tied to a precision —
if you run a different quantisation of the same weights, point the loader
widgets at it and the rest of the graph is unchanged.

**On Linux and macOS the five model widgets will load red.** ComfyUI compares a
saved model value against a list it builds with the host's path separator, so
the graphs — authored and verified on Windows — carry `minimax\name.safetensors`
and no spelling is portable. Click each loader and pick the same file; the graph
is unchanged otherwise. Windows users see no such prompt.

The two DiT files are the ref2va and fl2va branches from
[§ Verified model constraints](#verified-model-constraints) — you need
whichever branches your timeline actually uses, and **only** `ref2va` for a
plain reference-driven render. Loading both is ~42 GB.

**Weights carry the MiniMax H3 Community License**, a separate agreement from
this pack's Apache-2.0 that governs your use of the model and of what you
generate with it. Read it before shipping client work.

### The distilled checkpoints, and graphs for them

`python3 tools/build_oct2026_workflows.py <folder> [media.json]` writes ten graphs
into a folder of your choosing — 30 s and 60 s of each family below. They are
written rather than shipped because they name model files and media your install
has to have, and HyperFlow needs the third-party ComfyUI-HyperFlow pack. The media
names in the script are placeholders; a `media.json` with the same keys (`MEDIA` at
the top of the script) names your own files without editing it.

| family | checkpoint | format |
|---|---|---|
| `PS_FastH3Omni_Dialogue` | [FastH3-Omni 8-step](https://huggingface.co/FastVideo/FastH3-Omni-Comfy) (pruned ref2va) + VSA | 16:9 two-hander |
| `PS_HyperFlow_Dialogue` | ref2va (non-pruned) + [HyperFlow](https://huggingface.co/VideoRebirthAI/hyperflow) | the same script, for an A/B |
| `PS_FastH3Omni_ContinuousTake` | FastH3-Omni 8-step | 9:16, one speaker |
| `PS_HyperFlow_ContinuousTake` | ref2va (non-pruned) + HyperFlow | 16:9, the same script |
| `PS_TaoMate_Vertical` | fl2va (non-pruned) + [TaoMate 3-step](https://huggingface.co/TaoLiveAIGC/TaoMate-H3-FL2VA) | 9:16 keyframe series |

Every graph renders all of its windows on **one** checkpoint, and keeps its faces
and recordings in Pulse Slate's Asset Bin — one place to attach a reference. Each
has a READ FIRST note saying what to replace.

## Never type a tag number

> ### Drag assets onto the node body. Reference them by name.
>
> `@Mimi`, `@Image1`, `@[Cafe wide]` — never `<Picture 2>`.

H3's reference tags are ordinals assigned by socket position. Type one by hand
and it goes stale the moment the bin changes, and the render **still succeeds** —
describing the wrong picture, with no error anywhere. This node computes every
ordinal at compile time from live bin order, and reports a hand-typed tag as an
error rather than trusting it. The mechanism, and the reference-video soundtrack
case that makes it worse, is in [The tag problem](#the-tag-problem-and-why-this-node-exists).

## Two paths, by duration

Two paths and a graph for each, never one graph with a muted branch. Pick the
file that matches the length you are making.

| duration | graph | what happens | status |
|---|---|---|---|
| **≤ 15 s** — one window | `PulseSlate_Single.json` | `PulseSlate` hands back `positive` and `latent`. The graph carries sampler → decode → mux. | **Verified on hardware.** |
| **> 15 s** — many windows | `PulseSlate_LongForm.json` | `PulseSlate` hands a `PULSE_TIMELINE` to `PulseRender`, which renders one window per H3 call, caches each to disk, and stitches. | **Runs on hardware** — a >15 s multi-window render completed 2026-08-10. The seam-by-seam listening pass is still not written down, so treat seam quality as unconfirmed. |

That last column is stated plainly because a stitched seam is the kind of thing
that passes every test and still sounds wrong. The path runs end to end; what
nobody has written down is whether the seams hold up to listening.

Clearing it is a specific, ordered piece of work rather than a vague "try it":
[`docs/HARDWARE_VERIFICATION.md`](docs/HARDWARE_VERIFICATION.md) walks the long
render, every seam, the five cache behaviours and the `PulseBench` numbers in
one sitting, and has blanks to record what actually happened.

Past a single window, `PulseSlate` **blocks** `positive` and `latent` rather than
handing back the last window alone. Until 3.0.0 it did hand them back, and a
still-wired sampler quietly re-rendered that one window and saved it as the whole
film: a 15-second timeline that split into 192 + 175 frames produced a 7-second
file, with no error anywhere, because 175 frames is a perfectly valid latent.

## Pulse Studio Infinity — a film that directs itself

A third path, for films nobody scripted beat by beat. **One queue run renders one
beat**:

1. **Pulse Infinity Loop** reads the session folder and hands on the beat the
   director planned.
2. **Pulse Dynamic Shot** turns the plan into a shot for Pulse Slate's
   `shots.shot_1`: camera, action, continuity anchors, the line.
3. **Pulse Slate** compiles it with your cast and recipe, unchanged.
4. **Pulse Infinity Render** renders it into the session, opening on the last
   frame of the previous beat.
5. **Pulse Observer** shows the result to the director. The director's
   observation goes into the scene memory, the next plan is stored, and the beat
   joins the reel.

Queue the graph N times for N beats, or turn on **auto-continue** in the Loop
node's context menu. Start with `PulseInfinity_Scripted.json` (the offline
scripted director, three beats), then `PulseInfinity_ClosedLoop.json`
(HyperFlow 8-step, your director, a Fish Audio S2 dialogue branch).

**One key picks the director:** `director_backend` — `claude_code` (default),
`gemini_antigravity`, `ollama`, `pi` or `scripted`. Set it on the Loop node, with
`PULSE_DIRECTOR_BACKEND`, or in `ComfyUI/user/pulse_infinity.yaml`. Start that
file from [`pulse_infinity.example.yaml`](pulse_infinity.example.yaml).
- **Claude Code and Antigravity** direct through the **Director Desk**, files in
  the session folder. `tools/setup_director_workspace.py` installs a workspace
  with the `pulse-director` skill and the Antigravity rules. Then run
  `/loop /pulse-director` in Claude Code, or `/pulse-direct` in Antigravity.
- **Ollama and pi** run through their own command lines.

**Every answer is held to one contract.** That contract is the mission's
`DirectedShotPlan`, read back byte for byte, with one validator. Refused plans
go back to the director with the reasons:
- a hand-typed `<Picture N>`;
- a cast member who does not exist;
- a line too long for its beat;
- a restaged action;
- a reworded beat;
- a stall;
- a scene past its budget.

HyperFlow samples at cfg 1.0, where a negative prompt is never evaluated, so
this validator is where "don't repeat that" is enforced.

**Nothing happens twice.** Director exchanges are journaled and replayed, a
render is committed once, and reroll and replan restore the memory from before
the beat.

The full guide is [`docs/INFINITY.md`](docs/INFINITY.md). It covers the session
folder, the protocol, the contracts, continuity, dialogue, controls and
troubleshooting.

### The segment cache

`PulseRender` writes every window to `ComfyUI/output/<run_dir>/<run_id>/` as it
finishes and fsyncs the manifest before starting the next one.

- **Kill it at window 9 of 12 and requeue** → 0–8 load from disk, 9–11 render.
- **Edit one shot** → only the window holding it re-renders, at its original seed.
- **Change the seed, the steps, or the upstream patch chain** → everything goes,
  because all three change what comes out.

Seeds are derived from the *set of shots in a window*, so inserting a shot at the
top no longer rerolls every window after it.

**Run `dry_run` first.** It produces the report and nothing else — no sampling, no
decode, no file writes. A wrong reference binding renders successfully and hands
you a well-formed film of the wrong person; the report's per-shot ordinal map is
how you catch that before spending the GPU time.

## Model patches — wire them upstream

The pack consumes a patched `MODEL`. It does not patch anything itself and takes
no dependency on the packs that do. The documented chain:

```
UNETLoader
  → SpectrumApplyMiniMaxH3        (history_storage: system_ram)
  → PathchSageAttentionKJ         (sets Sage as the fallback backend)
  → MiniMax H3 Scheduled Sol Attention Patch
  → MiniMax H3 Chunk FeedForward
  → PulseSlate / PulseRender
```

You wire this yourself — **no graph ships with it pre-wired.** A variant that did,
`PulseSlate_Starter_SpectrumSage.json`, shipped in 3.0.0 and was dropped
afterwards so that the one example left needs no third-party pack to load clean.
The chain needs [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3),
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) and
[ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn); none is
required by Pulse Studio itself. `PulseSlate_LongForm.json`'s own note carries
the same chain and the same warning.

**The Sol-Attn node must come after the Sage patch.** Applied after it, Sol adopts
the Sage forward as its fallback, so ineligible steps and short sequences run
memory-efficient Sage while eligible steps run Sol-Attn. Applied *before* it, the
Sage patch shadows the Sol node entirely and it does nothing — and the graph still
runs, still produces output, and gives you none of the speedup. Nothing warns you,
which is why it is stated here and again in the workflow's own note.

Wire it into **both** model inputs: `model` carries your references and
`model_fl2va` carries the first/last anchors. Set the Sol node's
`sink_conditioning` to `exact_kv_and_rows` — Pulse Slate pairs each character
image with its own audio track, and that pairing lives in exactly the query rows
that setting keeps dense. Nothing else needs hand-management: `PulseRender`
detects whatever patches it is handed and folds them into the cache key by
itself, so changing `tau_start` re-renders the affected segments instead of
silently reusing segments rendered at the old sparsity.

This order is documented rather than enforced. Until 2026-08-11 a test asserted
it against the shipped variant; that graph is gone, and nothing checks the
ordering for you now.

Nothing in this pack vendors, wraps or reimplements any of those kernels. Its job
is to detect them, fold them into the cache key, report the chain, and ship
workflows wired in the right order.

**Upstream is not a style preference.** `PulseRender` samples with the model
handed *to* it, so a patch applied downstream of it does nothing at all — and a
patch that changes between two runs of the same timeline would otherwise let the
cache reuse segments across it, giving you a film whose shots do not match.

**`system_ram` is not a speed setting.** On a 32 GB card, Spectrum storing its
history in system RAM is what makes a 362-frame window *fit at all*. Set it to
`vram` and a full-length window is likely to fail with an out-of-memory error
rather than run slowly.

The node inspects the model it is handed and warns — on the node face and in the
console — when it finds no attention patch or no offload patch, naming both. It
never blocks: running unpatched is a legitimate choice, and the warning exists so
that it is a choice rather than an accident.

## cfg stays at 1.0

H3's own reference pipeline has no negative conditioning anywhere; it uses
`BasicGuider`, which takes a single conditioning input. `cfg = 1.0` is that
native path and it is the default here.

Above 1.0 the node switches to `CFGGuider` with an **empty** negative prompt,
because there is nowhere for a real one to come from. That is offered because it
was asked for, not because it is recommended. Leave it at 1.0 unless you are
deliberately experimenting.

## No network

**The pack makes no outbound request** — not at import, not during execution,
not from the widget layer. No telemetry, no model auto-download, no CDN script,
no webfont, no remote image.

This is enforced by a source scan over both languages that runs in CI, not by
policy: outbound HTTP clients, sockets and auto-download are banned in the
Python, and absolute-origin URLs, CDN loaders, webfonts and request constructors
are banned in the JavaScript. Registering an inbound aiohttp route is explicitly
allowed — that is ComfyUI's own server, and answering the frontend is not egress.

Everything you drop in the bin stays on your machine.

**Pulse Infinity's director is the one exception, and only if you choose it.** The
pack still opens no socket: Claude Code and Antigravity direct through a folder of
files (the Director Desk), and Ollama and pi through their own command lines,
started with no shell from one module that a test confines process creation to.
But directing means looking: a cloud agent you point at a session folder reads the
evidence frames it finds there, so with `claude_code` or `gemini_antigravity`
active, those frames leave the machine *through that agent*. `ollama`, `pi` on a
local model, and `scripted` keep everything local. The Loop Controller's report
names the active director and says which kind it is.

**Pulse Resolve Super Scale starts DaVinci Resolve** on this machine, from the same
one module, and talks to it only through Resolve's own Python. Resolve's
scripting library links Windows' socket library, which is one reason it is
never loaded into ComfyUI. Nothing leaves the machine.

## Known issues

- **Workflows saved by any pre-2.0.0 build will not load.** Widget order changed
  once, deliberately, before anything was published; a file without a schema
  version in slot 0 is refused by name rather than guessed at. There is no
  migration and there will not be one — the values in those files are already
  shifted, so a "successful" load would render something wrong instead of
  failing. Rebuild from `example_workflows/PulseSlate_LongForm.json`. The full
  reasoning is in [CHANGELOG.md](CHANGELOG.md).
- **Path B's seams are unconfirmed** — the render runs; nobody has written down
  how the window boundaries sound. See the duration table above.

---

## Where you type

The Director node carries **two multiline boxes on its face**, plus an exposed
control panel — nothing is hidden behind a gear.

**GLOBAL PROMPT** — art style, lighting, camera rules, identity locks, score.
Compiles into `subject_definitions` and `retention_analysis`. Optional labels at
the start of a line: `style:` `identity:` `retention:` `soundscape:` `music:`.
Unlabelled text is treated as style.

**SHOT PROMPT** — one shot per line, each opening with `[Shot N]` or a
`[MM:SS.mmm]` timecode. Shots without a timecode spread evenly between the ones
that have them, so you can stamp only the moments you care about. Quoted `"text"`
becomes dialogue.

Both boxes preserve newlines through a workflow save/load, and **queuing never
writes back to them** — see the erasure note below.

If a widget ever does strip the line breaks, `normalize_prompt_text` restores
them before parsing (`] [` and runs of two or more spaces become newlines). That
is belt-and-braces behind the multiline widgets, not a replacement for them.

### Assets

Drag images, video or audio **onto the node body**. No external loader nodes.
Each drop gets a short alias (`Image1`, `Video1`, `Audio1`) you rename inline —
you never type a filename. The meter refuses a drop that would break the budget
rather than failing at queue time.

Two fields on each row decide how much the model is told about a reference:

- **description** — a one-line description promotes the reference from a bare
  `<Picture i>` citation to a `<Subject N>` definition, which is the shape
  MiniMax's own reference format asks for and what identity consistency is built
  on. Clear it and the same image is cited plainly; both are legal. Audio has no
  `<Subject N>` form, so the field is refused there rather than written where it
  can never reach the model.
- **retention** — `fully_preserved` or `partially_copy`, written *verbatim* into
  the prompt's retention section. A closed list rather than free text: the value
  is a sentence the model reads literally, and a typo in one is not an error, it
  is a slightly worse render.

---

## The tag problem, and why this node exists

H3's reference tags are positional. From `comfy/text_encoders/minimax.py`:

```python
counters = {"image": 0, "audio": 0, "video": 0}
for item in minimax_ref_items:
    counters[kind] += 1
    add_text("<Picture %d>: " % counters["image"])
```

An ordinal means nothing more than *"the nth of its type in socket order."* Insert
an image at the top of your bin and every `<Picture N>` after it shifts by one —
while the prompt text referencing them stays exactly as you typed it.

The render still succeeds. It just describes the wrong pictures, and nothing
anywhere reports an error.

The subtle case is worse. A reference video's soundtrack is appended to
`ref_items` **before** its own `<Video k>` entry and **before** every standalone
audio, so ticking "use its soundtrack" on one video silently renumbers every
`<Audio j>` in your prompt:

```
bin: [video A]  [audio VO]                →  VO is <Audio 1>
bin: [video A + soundtrack]  [audio VO]   →  VO is <Audio 2>
```

This node's answer is that **no human ever types an ordinal**. You reference
assets by name (`@Mimi`) or id (`{{mimi}}`); the compiler resolves them against
current bin order on every compile. Renumbering cannot desynchronise, because
there is only ever one number and it is computed, not stored. Hand-typed tags are
detected and reported rather than silently trusted.

---

## Verified model constraints

Everything below was read out of ComfyUI's source, not inferred. All of it is
encoded in `comfyui_pulse_studio/constants.py` with the file and symbol it came from.

**Keyframes exist at two positions only.** `comfy/ldm/minimax/model.py`,
`PackedLayout.__init__`:

```python
if pixel_index == 0:                                cond_t = float(text_len)
elif frame_count is not None and pixel_index == frame_count - 1:
                                                    cond_t = ...
else: raise ValueError("only first/last keyframe anchors are supported")
```

This is RoPE position math inside the DiT, not a node-level guard. **A custom node
cannot route around it.** Per-shot visual control in H3 is textual — `[Shot N]`
timestamps inside the prompt. Arbitrary-timecode image anchors are not a missing
feature here; they are impossible.

**References and anchors are mutually exclusive.** Two nodes, two checkpoints,
disjoint inputs:

| | anchors | references | checkpoint |
|---|---|---|---|
| `MiniMaxH3ImageToVideo` | `first_frame`, `last_frame` | — | fl2va |
| `MiniMaxH3ReferenceToVideo` | — | 9 img / 3 vid / 3 audio | ref2va |

You cannot hold a character by reference *and* anchor a frame in the same render.
Loading both checkpoints is ~42 GB, so the choice is a memory decision too.

**Budget.** ≤9 images, ≤3 videos (2–15s each), ≤3 standalone audio, **≤12 files
total**. The 12-file cap genuinely binds — 9+3+3 = 15 sockets exist.

A reference video's soundtrack takes its own `<Audio j>` ordinal, separate from
the three standalone slots, so **six audio references are reachable inside the
documented budget**: three videos with their soundtracks enabled, plus three
standalone. The soundtrack rides inside the video's own file and costs nothing
against the 12.

**The audio cap is the one limit here that is not the model's.** Images at 9 and
videos at 3 are Autogrow socket templates in
`comfy_extras/nodes_minimax_h3.py`; so is audio at 3 — but nothing below it
agrees. `PackedLayout` appends one `ref_audio` segment per reference block in a
plain loop, with no count check, and the tokenizer just increments a counter.
The socket cap is enforced by graph validation, and this pack calls the
reference encoder **in-process**, so it never applies to what the node passes.

`audio_ref_ceiling` (3–9, default 3) exposes that. At 3 nothing changes. Above
3 the bin accepts up to nine standalone audio references, the file total rises
by the same amount, the meter turns amber, and the node says once per render
that it is past what MiniMax and ComfyUI document. It will run. Nobody has
published a result at nine, and every reference rides all your sampling steps —
so treat it as an experiment you measure, not a setting you leave on.

**Frame grid.** Frames snap to `17k + 5`, floor 5, trained ceiling 362 (~15.08s at
24fps). `MAX_WINDOW_FRAMES` is configuration, not a constant: when MiniMax raises
the ceiling, one integer moves.

**Trained floor.** `MiniMaxH3ImageToVideo`'s own tooltip gives the trained range
as ~124–362. Every emitted window is therefore **≥124 frames**: a short tail is
merged backwards into its predecessor, and where that would exceed the ceiling
the partition is rebalanced into fewer, longer windows. The one exception is a
render whose *total* is under 124 frames, which passes through as a single short
window — there is nothing to merge it with. The floor outranks your chosen window
length, and any override is reported.

**Audio is reference, not sync.** `ref_audios` is standalone reference
conditioning — voice timbre and sonic character, with no frame-level alignment.
There is no audio-driven latent path, so **lip-sync and beat-matched motion are
not achievable through the public node API.** Nothing this node emits promises
either; there is a test asserting so.

---

## Nodes

### Pulse Slate · MiniMax H3

Compiles a timeline into a storyboard prompt and reference set, and stops. It
does not sample.

- **One window** → hands back `positive` / `latent` for your own sampler.
- **Longer than one window** → those two outputs are blocked; take the `timeline`
  output to a **Pulse Render** node.

The prompt it writes is laid out the way MiniMax's own guides lay it out
(`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` and `..._base_en.md` on
huggingface.co/MiniMaxAI/MiniMax-H3). A reference window's summary opens with
its task type — `[reference generation + audio reuse] The target video shows …`
— where it used to open with `[window 2 of 4]`, which the model was never taught.
`retention_analysis` names a voice's owner without a speaker id, and a picture
kept in part is `partially_preserved` (`partially_copy` is the audio word; the
bin still stores it). A First/Last-Frame window opens with the guide's
image-alignment sentence, on its own line, its end mark the window's length to
two decimals, and its lip-sync instruction sits after the shots it covers rather
than in `overall_soundscape`. Segments rendered before this re-render once,
because their prompts changed.

Windows are partitioned `balanced` by default: 16s at a 15s ceiling becomes two
~8s windows, not 15s + 1s. A 1s trailing window is below H3's trained floor and
reliably looks broken. `fill` is available when you mean windows of a literal
length, and `shot_aligned` puts the seams **on shot cuts** wherever the frame
grid allows it — a shot that straddles a seam is compiled into *both* windows,
described twice and hashed into both window seeds, so aligning is what stops that
happening. It cannot hit every cut: cumulative position after k windows is
17K + 5k, so a seam lands on a cut at frame p only when p ≡ 5k (mod 17). It takes
the largest subset it can hit, gives up `window_seconds` to hit them, and names
the cuts it could not reach in the report.

`speech_aware` listens. It reads where every lip-sync recording is speaking and
puts each seam on a cut where the grid reaches one, otherwise into a pause, and
never through a word: a seam mid-syllable has one window close the mouth and the
next one open it again, which reads as a stutter however good either side is.
Windows stay between 5 and 15 s. Every other policy now reports a seam that lands
inside speech.

`aspect_ratio` picks the canvas, and each preset is fitted to H3's 1,032,192 px
budget at load time rather than hardcoded, so the table cannot drift from the
budget: 16:9 is 1344×768, 9:16 is 768×1344, 21:9 is 1536×672, 1:1 is 992×992,
4:3 is 1152×864. `custom` uses the `width`/`height` widgets instead.

`continuity` chooses how windows join: `none`, `last_frame_carry`,
`keyframe_pairs`, `ref_guided_carry` or `text_to_video`. `last_frame_carry` and
`keyframe_pairs` pin a frame, which is the fl2va checkpoint specifically, so they
need `model_fl2va` and **fail at compile time** without it rather than falling
back to something that looks like it worked. `ref_guided_carry` (experimental)
keeps every window on the reference checkpoint and pins the previous window's
last frame with a guide instead.

`text_to_video` puts the whole film on fl2va, from the words alone: window 1 is
unpinned (unless shot 1 has a `start_image`), and each later window opens on the
previous window's last frame. Its one checkpoint is the fl2va one, on `model` —
`model_fl2va` is not needed, and is used instead when wired. fl2va takes no
references, so the bin stays empty and each line's voice goes on its shot's
Pulse Voice. It replaces the `"branch": "fl2va"` that the first T2V graphs typed
into `timeline_data` by hand; that key still works under `none`, and compiles to
the same windows.

**One checkpoint per film, unless you mean two.** fl2va takes no references, so
a film whose first window is a reference window and whose later windows are fl2va
sees the faces and voices in window 1 only — after it, identity is whatever the
carried frame and the prompt can hold, and the voice is a description. With
`none` or `ref_guided_carry` nothing samples on `model_fl2va` at all, so leave it
unwired: it is ~20 GB resident for nothing, and the report says so.

The **`recipe`** socket takes a [Pulse Recipe](#pulse-recipe): how this film's
checkpoint wants to be sampled (FastH3-Omni, HyperFlow, TaoMate, Pika H3 Flex
or the base models). It replaces the steps / sampler / scheduler / cfg widgets, which then
only apply when nothing is wired. `recipe_fl2va` does the same for the fl2va
windows. The **`cast`** sockets take [Pulse Character](#pulse-character) nodes,
for a graph that prefers wires to the Asset Bin.

### Pulse Shot

One node per shot. Its text, its length, its continuity override, its own
first/last frames and its own scene-local references — and because those frames
are real `IMAGE` inputs, a shot can open on a frame generated elsewhere in the
same graph rather than on a file you had to export first.

Connecting any shot socket makes the **shot text box inactive**. The two are never
merged and neither is silently preferred; `compiled_prompt` says which won, at the
top, every time.

Each shot's identity is written once and never changes, which is what keeps its
seed and its cached segment attached to it when you reorder the film.

#### `ref_audio`, and what `ref_audio_mode` decides

H3 **always** generates its own audio track. What a reference recording changes is
what the model does alongside it, and the two options are genuinely different jobs:

- **`lip_sync`** (default) — the character's mouth matches your recording. The
  prompt says so explicitly, naming the tag, because that sentence *is* the
  mechanism: the tokenizer emits only the marker `<Audio j>: ` and the waveform
  never reaches the language model, so wiring the socket tells the model nothing
  on its own. The clip is also trimmed to the window's exact span — a 30-second
  file against a 9.42-second window asks the model to align two different
  stretches of time, and the result is a mouth that tracks nothing.
- **`voice_timbre`** — the model speaks this shot's own dialogue and borrows only
  the character of the voice. No alignment, no trim.

**Which seconds get trimmed: this socket reads the film clock.** A `lip_sync`
clip is cut to the window it rides in, and `ref_audio` takes those seconds
starting at the window's position **on the whole film's clock**. So this socket
expects a recording that runs alongside the entire film — one narration track,
wired into every shot that speaks. It is the scope that is scene-local, not the
timing: `@Voice` on shot 3 is invisible to shot 2, but its *seconds* are still
counted from the film's zero.

A take recorded for one shot begins at that shot, not at the film. Left on this
socket it is cut at an offset past its own end, and the window is handed silence
— no error, no failed render, just a mouth that does not move. **Use
[Pulse Voice](#pulse-voice) instead**, which says where the recording starts.

Either way the report now measures it. A recording that does not cover its window
is named with the shortfall in seconds:

> `@Voice` covers 0.00-12.40s on the film clock and window 3 covers 24.50-36.75s.
> The two do not meet, so nothing of this recording reaches this window.

and one that merely runs out early gets the partial form. Both then say what the
gap actually looks like, which depends on the shots: with no dialogue the mouth
is genuinely still, and with dialogue it is not still at all — the `<d>` block
tells the model to say those words, so it generates speech and the mouth moves
while `use_reference_audio` mixes the reference's silence over the top. A talking
head on a dead track, fixed by cutting the words rather than by hunting for a
longer recording. A recording nothing
could measure is never diagnosed: unknown is not the same as empty.

A **trailing** gap on the **last** window is held to a whole second rather than a
quarter. A narration stopping a fraction of a second before the final frame is
ordinary — the recording ends when the speaking ends, and the grid rounds the
last window up to a legal frame count regardless. A gap anywhere else, and
lead-in silence anywhere at all, keeps the quarter-second rule: a silence in the
middle of a film is a mistake, and a recording that starts late is late wherever
it happens.

The mode is part of the cache key, so switching it re-renders rather than handing
back a segment made under the other instruction.

On a `lip_sync` shot the audio H3 generates is a re-synthesis of your recording —
close, but not your take. `PulseRender.use_reference_audio` muxes the original
into the finished film instead. The generated track is still written to every
segment's `.flac`, so it is a mux choice you can flip and requeue without
re-rendering anything.

**A `lip_sync` reference on a shot with no dialogue produces no mouth movement**,
and nothing warns you: there is no speech to match. Put the recording on the shot
that speaks.

The inverse is reported. A shot carrying both a quoted line and a `lip_sync`
reference gives the model two answers to "what is she saying" — the `<d>` block
instructs it to speak those words, and the recording says whatever it says. The
compiler names it in the report rather than refusing, because a quote that *is*
the recording's transcript is legitimate and only you know that. Otherwise drop
the quote and let the audio carry the words; keep the `@Voice` tag in the line,
which is what tells the model whose shot the recording belongs to.

#### `speaker`, and why a two-hander needs it

The socket carries a waveform and an ordinal. It does not carry an owner — the
tokenizer emits `<Audio j>: ` and the waveform never reaches the language model
— so on a shot with one character there is nothing to confuse and on a shot with
two there is nothing to go on. ComfyUI ships a known bug of exactly that shape
([Comfy-Org/ComfyUI#15454](https://github.com/Comfy-Org/ComfyUI/issues/15454)):
the intended character's lips move, and the other character's accent comes out
of them.

MiniMax's reference format binds a voice to a face with a **global speaker id**.
Type the character's `@Name` into `speaker` — a bin asset, or this shot's own
`@Ref1` — and the compiler:

- assigns that character `(S1)`, `(S2)`, … **at their first line in the film**,
  and keeps it for every later shot and every later window. The id is what tells
  the model that the person talking after a cut is the person who talked before
  it, so it is assigned once across the whole timeline rather than per window;
- stamps it on that character in the shots they actually speak in, and nowhere
  else — `<Subject 1> (S1) crosses to the counter`. A silent character in the
  background stays unstamped; an id there reads as a cue to give them a line;
- binds this shot's `ref_audio` to them by name, so the prompt says
  `` `<Audio 1>` is the speech <Subject 1> (S1) is saying `` rather than "this
  character";
- adds a `retention_analysis` line in MiniMax's audio vocabulary —
  `fully_copy` for `lip_sync`, `reference` for `voice_timbre`.

A character with a description in the bin becomes `<Subject N> (S1)`. One
without becomes `<Picture N> (S1)`, which is much weaker and still binds the
voice to a face.

Leave it blank on a shot where nobody speaks, and on a one-hander where there is
nothing to confuse — the prompt is then byte-for-byte what it was before the
field existed, and your cache is not invalidated. A name that matches nothing is
reported, never guessed at: binding a voice to the wrong face is worse than
leaving it unbound.

#### A voice in the Asset Bin

A recording dropped in the bin has no shot to read a speaker off, so it says who
it belongs to itself. The audio row in the panel carries two controls:

- **role** — `lip_sync` or `voice_timbre`, the same two jobs `ref_audio_mode`
  picks between. Until this shipped there was no way to say it on a bin
  recording, so every one of them compiled as a timbre reference and only a
  `PulseShot`'s own socket could ask for lip sync.
- **whose voice** — the character, picked by name and **stored as an asset id**.
  A binding to "reference 3" would follow whatever landed in slot 3 after the
  next bin edit and report nothing; a binding to an id follows the character
  through renames, reorders and renumbering.

That is what makes a film with three characters and three voice files work:
each recording names its own owner, each owner keeps one speaker id, and the
prompt says which is which. Supplying somebody's voice makes them a speaker even
if no shot ever named them — they are numbered after everyone who has a line, so
adding a voice reference cannot renumber a character already on screen.

Binding to a picture that is not in the bin, or to another recording, is refused
rather than written into the prompt. A voice belongs to somebody the model can
see.

A bin recording sits on the **film clock**, and the row says where:

- **at** — the film second the recording starts. 0 is with the film, which is
  what a narration spanning the timeline wants.
- **from** / **to** — which seconds of the file play. Empty **to** is the end of
  the file.

So one take file can carry several lines from the bin: a row per line, each cut
out of the file and placed at its shot, with no Load Audio or Pulse Voice wired
anywhere. The example two-handers are built exactly like that. Every lip-sync
line of one character rides **one** `<Audio j>` in each window, so the meter
counts voices, not lines: six lines between two characters read 2/3 audio, and
the panel tags every line of a voice with that voice's number.

What the bin also gets is measurement: the file's length is read from its header
at compile time, so a recording that does not cover the windows it is trimmed to
is named in the report rather than rendering as silence. A window holding two or
more voices is judged on all of them together — each is silent while the other
speaks, by design — and only a stretch nobody speaks in is reported.

Where both apply — a shot's own `ref_audio` that also carries an explicit
`voice_of` — the explicit binding wins. The author naming a character beats the
wiring implying one.

**Carry-over will not evict a bound voice first.** A continuation window's
carry-over claims the front of the audio group, so the last user recording is
dropped when the group is full. Chosen by bin position that is a coin flip, and
a character keeping their picture and their lines while losing their voice from
window 2 onward reads as drift rather than as a missing reference. Unbound clips
go first, then `voice_timbre`, then `lip_sync` — losing an alignment
desynchronises a mouth, which is visible. The drop is still reported either way,
and the survivors keep their bin order so nothing renumbers that did not have to.

### Pulse Voice

One recording, and where on the film clock it belongs. Output is a `PULSE_VOICE`
that goes into **Pulse Slate**'s `voices.voice_1..3` — visible to every window —
or into one **Pulse Shot**'s `voice` input, for that shot alone.

It exists because an `AUDIO` socket carries a waveform and nothing else: not a
role, not an owner, and not a position in time. All three had to be said
somewhere, and until this node the first was a widget on `PulseShot`, the second
was a field only a bin asset had, and the third could not be said at all.

It also collapses the pattern the socket forced. A narration covering the whole
film used to need one wire per shot — four identical wires to one `LoadAudio` on
a four-window film — because a shot's own socket was the only route to
`lip_sync`. `PulseSlate.ref_music` and `ref_video_audio` carry no role: the first
is the non-diegetic score and the second is a video's paired soundtrack, so
neither can ask for a mouth to move. One `PulseVoice` on the slate reaches every
window.

| widget | what it decides |
| --- | --- |
| `name` | the `@handle` to cite it by in a shot. Never an ordinal — those are computed. |
| `role` | `lip_sync` or `voice_timbre`, the same two jobs `ref_audio_mode` picks between. |
| `aligns_to` | **`film_clock`**: the first sample is the film's second zero — a narration spanning the timeline. **`shot_start`**: the first sample is the start of the shot it is wired to — a per-shot take. |
| `offset_seconds` | added on top of whichever base `aligns_to` chose, for a take with room tone in front of it. Negative moves it earlier. |
| `trim_start` / `trim_end` | which part of the file to use at all. `-1` on `trim_end` means "to the end". |
| `speaker` | whose voice it is, as an `@Name` from the Asset Bin. Gets them a stable `(S1)` and makes the prompt name them instead of saying "this character". |
| `description` | what the recording is, in a few words. Rides into `subject_definitions` beside the reference. |

`aligns_to` is a choice rather than a number because "this recording starts when
the shot starts" should be something the graph *states*, not arithmetic the
author redoes every time a duration changes. On Pulse Slate there is no shot to
start at, so `shot_start` there is reported and read as `film_clock`.

The offset reaches the segment cache key, and only when it is non-zero — two
renders identical in every other field are different films, and the content
digest cannot tell them apart because it hashes the whole file either way. A
project whose recordings all start with the film keeps every key already on disk.

On a shot carrying both, `voice` wins over `ref_audio` and the report says so.
`PulseRender.use_reference_audio` mixes **every** lip-sync recording in a window
into the finished film, each placed at its own offset — not just the first one it
finds, which is what a two-hander used to lose a character to.

An optional **`final_audio`** socket takes a clean take for the finished film, such
as a dry stem or an unprocessed master. `audio` still drives the mouth. When
`use_reference_audio` is on, `final_audio` is what gets muxed, with the same trim
and at the same seconds. The two must line up sample for sample, and the report
says so when their lengths differ.

`example_workflows/PulseSlate_Voice.json` is the whole thing in one graph.

### Pulse Recipe

How one checkpoint wants to be sampled, said once and wired into Pulse Slate's
`recipe` (reference windows) or `recipe_fl2va` (First/Last-Frame windows). The
distilled checkpoints do not agree with each other, and the wrong settings render
without complaint and look like the model being bad:

| preset | steps · sampler · schedule | for |
|---|---|---|
| `fasth3_omni_8step` | 8 · euler · simple | FastH3-Omni (pruned ref2va). Patch it with Model Attention Backend → Model Sparse Attention (vsa, keep 10). |
| `taomate_fl2va_3step` | 3 · euler · sigmas `1.0, 0.961165, 0.853333, 0.0` | the full fl2va checkpoint with the TaoMate LoRA at 1.0 |
| `hyperflow_8step` | 8 · euler · `hyperflow` | either branch, non-pruned, with ComfyUI-HyperFlow's LoRA loader |
| `pika_h3_flex_6step` | 6 · lcm · sigmas `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0` | either branch, with Pika H3 Flex's ref2va or fl2va LoRA at 1.0 (the pruned checkpoints are fine: it patches attention and MLP only) |
| `h3_base` | 20 · res_multistep · simple | the undistilled checkpoints |
| `custom` | the widgets, or a hand-written `sigmas` | anything else |

cfg is 1.0 on all of them. A recipe that cannot be sampled at all — sigmas that
run upward or stop short of 0, a `hyperflow` schedule at anything but 8 steps —
is refused at compile time rather than inside the first window's sampler. One
that is merely off-recipe for the model it samples — FastH3's stacked PDD heads
away from 8 Euler steps or without VSA, the `hyperflow` schedule on a model with
no HyperFlow adapter, Pika's schedule with no LoRA on the model or on top of
HyperFlow — is named in the report.

Pika samples `lcm`: a fresh noise draw at every step, which its six intervals
were distilled against; Pika's README says plain six-step Euler is not the same.
`lcm` is on every sampler list in the pack (the slate, `custom`, Retake, Still),
appended so the values saved graphs hold stay where they were.

### Pulse Character

One character as one node: `name`, `description`, `retention`, a `face`, an
optional full-body `body` and an optional `voice` from a Pulse Voice, bound to one
another when it is made. Into Pulse Slate's `cast` sockets. The body view is cited
inside the character's one `<Subject N>`, never as a second person.

It is the wired alternative to the Asset Bin. Use one or the other for a film —
references attached in two places are the ones that drift apart.

### Pulse Render

Executes a `PULSE_TIMELINE`, reusing every window already on disk. See
[the segment cache](#the-segment-cache).

- `cache_mode` — `auto` reuses what is unchanged; `force_rerender` ignores the
  cache; `reuse_only` refuses to render anything and aborts naming the first
  missing window, for assembling a final cut without regenerating a frame.
- `dry_run` — the report and nothing else.
- `low_memory` — 8-bit frame accumulation and VRAM released between windows. The
  finished video is assembled by stream-copying the segment files, so a
  twelve-window film is never held in RAM.
- `prune_unused` — off by default. Segments from earlier edits are what make
  flipping back to an earlier edit free, so nothing is ever deleted automatically.
- `seam_treatment` — what to do where two windows meet. The container-level seam
  is solved by construction (placement is computed from frame counts, never
  measured from timestamps), but the two sides are two independently generated
  takes: a score that restarts, a level that steps, a grade that drifts.
  `audio+colour`, the default, level-matches and tapers each audio join and grades
  a window's opening towards the frame the previous one ended on, decaying over 12
  frames — a flat correction across a whole window does not remove a cut, it moves
  it to the next seam. `audio` does the first only. `off` leaves both untreated,
  which is what you want when you are A/B-ing it, and colour matching is the kind
  of correction that occasionally makes things worse. A window following a
  *cached* segment is never colour-matched — the segment either side of it is
  already encoded, so only one side of that seam can move — and the report says so
  rather than quietly doing half the job.

- `audio_mode` sets what a `lip_sync` recording does to the window it rides in.
  - `reference_only` is the default and matches every render before this widget
    existed. The recording is a reference block that the model reads and
    re-voices.
  - `lock_source` also writes the window's lip-sync mix into the *target* audio
    latent and masks it at 0. The real waveform then drives the mouth at every
    step, and H3 never regenerates it. Pair it with `use_reference_audio` to put
    the exact takes in the film; the report warns if you don't.
  - `remix_source` re-noises the locked mix to `remix_strength`. HyperFlow wasn't
    trained on partly masked rows, so use `lock_source` with it.

  The mode reaches the cache key only when it isn't `reference_only`, and only on
  windows that carry a lip-sync recording. Every key already on disk stays put.

- `ref_image_fit` — `canvas` crops each reference picture to the render's aspect,
  as always. `keep_aspect` hands it over whole and lets H3 scale it, so a portrait
  character sheet keeps its head and its feet.
- `upscale` — the finished film at 1080, 1440 or 2160. Windows render inside
  H3's pixel budget either way; neither FastH3-Omni nor TaoMate renders 1080
  natively. `1080p_fit` keeps the aspect (1344×768 → 1890×1080). `1080p_exact`,
  `1440p_exact` and `2160p_exact` deliver 1920×1080, 2560×1440 or 3840×2160
  (turned for portrait), resampled with Lanczos, which adds no detail the render
  did not have. `1080p_rtx`, `1440p_rtx` and `2160p_rtx` deliver the same canvases
  through NVIDIA's RTX Video Super Resolution — see
  [Delivering at 2K / 4K](#delivering-at-2k--4k). Either way the film is upscaled
  one frame at a time, its audio copied untouched, so a long film is never held
  in memory at the larger size, and it lands beside `assembled.mp4` as
  `assembled_<mode>.mp4`.

Every approximation patched onto the incoming model is detected and folded into
the cache key. Sol-Attn, Spectrum and EasyCache all change what the same prompt at
the same seed produces, and a cache that ignored them would hand you a film whose
shots were rendered at different sparsities without saying so.

### Delivering at 2K / 4K

H3 renders inside a ~1 MP budget, so every delivery size is reached after the film
is assembled. Every route delivers the same canvas: they cover it and
centre-crop, so 16:9 at 1344×768 becomes exactly 3840×2160, losing a few rows at
the top and bottom.

| route | where | what it is |
|---|---|---|
| `1440p_exact`, `2160p_exact` | Pulse Render | A Lanczos resample. Fast, and adds no detail. |
| `1080p_rtx`, `1440p_rtx`, `2160p_rtx` | Pulse Render | NVIDIA RTX Video Super Resolution (quality ULTRA) on the GPU, one frame at a time: a learned upscale that restores detail on the way up. |
| [Pulse Resolve Super Scale](#pulse-resolve-super-scale) | DaVinci Resolve Studio, in the graph after Pulse Render | Resolve's Super Scale, 2× Enhanced, with Resolve's own resize for the rest of the way, rendered to H.265. As much fine detail as RTX and closer to the film it came from (measured below). |
| `tools/resolve_superscale.bat` | DaVinci Resolve Studio, by hand after the render | The same Super Scale, by drag and drop. |

**RTX** needs an RTX card and `nvvfx`, which NVIDIA's RTX nodes for ComfyUI
install (`pip install nvidia-vfx`). Without either, past 4×, or on an SDK error,
an `_rtx` mode resamples with Lanczos instead, and the report says which ran and
why.

**Resolve, by hand.** Drag a film, or a Pulse run folder, onto
`tools/resolve_superscale.bat`; from a run folder it takes the native
`assembled.mp4`, never an already-upscaled copy. Or run
`resolve_superscale.bat <film or folder> --target 1440p`. It does five things:

1. Starts Resolve headless (`-nogui`) if it isn't running, and quits it again
   afterwards.
2. Builds a throwaway project with a timeline at the target canvas, input sizing
   `scaleToCrop`.
3. Sets Super Scale on the clip. `auto` is 2× Enhanced (sharpness and noise
   reduction 0.5), with Resolve's own resize covering what is left — up to 1.5×,
   which takes every Pulse canvas to 1440p or 4K. A smaller film than that gets
   3× or 4×.
4. Renders `<name>_<target>_resolve.mp4` beside the film, with Resolve's own
   H.265 encoder and the film's audio.
5. Deletes the project.

On an RTX 5090 a 30 s film went to 4K in 16 s once Resolve was up. Starting Resolve
headless adds about half a minute.

How the routes compare: one Pulse frame (1344×768, a face), taken to 4K each way.
The Laplacian's variance on the face is the fine detail; scaling the 4K frame back
down and comparing it with the original is how faithful the upscale is (luma,
Resolve Studio 21.1.1):

| 4K from 1344×768 | fine detail | back to the source |
|---|---|---|
| Lanczos (`2160p_exact`) | 10.0 | 43.6 dB |
| RTX VSR (`2160p_rtx`) | 15.3 | 38.6 dB |
| Super Scale 3× | 9.9 | 42.8 dB |
| Super Scale 4× | 10.2 | 44.7 dB |
| **Super Scale 2× Enhanced** (the default) | **15.8** | **45.2 dB** |
| Super Scale 2× Enhanced, sharpness 1.0, NR 0.2 | 92.6 | 35.1 dB |

By eye RTX smooths skin to wax; 2× Enhanced at full sharpness turns eyes and skin
crunchy. Raise `--sharpness` in small steps, if at all.

Through the API Resolve takes no quality setting for an MP4 codec, so the encoder
sets the bitrate: at 4K its own H.265 wrote 34 Mbit/s, its NVIDIA H.265 22 and its
H.264 58. The helper uses its own H.265; `--codec h264` gives the larger file.

The options are `--mode auto|2x|2x_enhanced|3x|4x`, `--sharpness` and `--nr` for
2× Enhanced, `--codec h265|h264`, `--out` and `--keep-project`. Resolve holds one
project at a time, so if one of yours is open the helper stops and says so;
`--switch-project` lets it switch to its temporary project and load yours again
afterwards.

It needs two things, set up once:

- DaVinci Resolve **Studio**. Super Scale is a Studio feature.
- Preferences › System › General › External scripting using: **Local**.

It talks only to the Resolve on this machine, through Resolve's own scripting
module, and runs under Resolve's own Python. ComfyUI never imports Resolve's
module: the node below runs this same helper as a child process.

### Pulse Resolve Super Scale

The Resolve route as a node, so the 2K/4K film arrives in the same queue run as
the render. Wire Pulse Render's `video` into it. Its own `video` output goes on
to Save Video or a preview like any other, and the file is written beside
`assembled.mp4` as `assembled_<target>_resolve.mp4`.

```
Pulse Render ──video──▶ Pulse Resolve Super Scale ──video──▶ Save Video
```

When it runs, it:

1. Frees the GPU, because Resolve shares it with ComfyUI (`free_vram`, on by
   default). With [ComfyUI-Memory-Reclaim](https://github.com/Addis-Pulse-Studio/ComfyUI-Memory-Reclaim)
   installed it runs that pack's *All Models* + *Hard* purge, which also hands
   PyTorch's cache back to the driver, and its before/after numbers go into the
   report. Without it, ComfyUI's own unload. The models reload on the next run.
2. Starts Resolve headless if it isn't running.
3. Runs `tools/resolve_superscale.py` under Resolve's own Python: the same helper
   as the `.bat`, with the same canvas, `auto` mode and encoder. Its progress
   drives the node's progress bar.
4. Quits a Resolve it started (`when_done`), so the GPU is free again before the
   next render. A Resolve you already had open is left as it is.

The widgets are the helper's options: `target`, `mode`, `sharpness`,
`noise_reduction`, `codec`, and `open_project` (`stop`, or `switch_and_restore`
for the helper's `--switch-project`).

- **Which film.** Pulse Render with its own `upscale` set hands on
  `assembled_<mode>.mp4`. The node sends the native `assembled.mp4` beside it
  instead, as the `.bat` does, and the report says so. Any other VIDEO works too.
  A plain file goes as it is, and frames or a trimmed clip are written out
  first. The result lands beside a film in ComfyUI's output folder, and in
  `output/pulse_resolve/` for one from anywhere else. A film that already fills
  the canvas is refused.
- **Cancel** stops the render. The helper stops Resolve's render, removes the
  unfinished file and deletes its temporary project. A Resolve the node started
  is quit just as after a finished render.
- **Time.** Starting Resolve adds about half a minute to every run.
  `leave_running` keeps a headless Resolve up between runs, but it then holds
  GPU memory the next H3 render wants.
- **Another folder.** If Resolve isn't in
  `C:\Program Files\Blackmagic Design\DaVinci Resolve`, set `PULSE_RESOLVE_DIR`
  to its folder and restart ComfyUI.
- **Windows only**, with the one-time setup above: Resolve Studio, and External
  scripting set to Local.

To free the GPU in the graph instead, put Memory Reclaim's *Advanced Model
Unload* (All Models, Hard) on the `video` wire between Pulse Render and this
node, and turn `free_vram` off.

### Pulse Lip-Sync Segment / Pulse Lip-Sync Paste

Per-character correction for when `lock_source` isn't enough, such as a fast line,
a profile, or a small face. A lip-sync model (LatentSync and the like) run over the
whole film with the whole mix repaints whatever face it finds to whoever is
talking. The second character's lines then land on the first character's mouth.

**Segment** hands the model one character. It takes the timeline, Pulse Render's
`frames` and an `@Name`:
- It finds that character's voices: a PulseVoice whose `speaker` names them, or a
  voice wired to a shot they speak in.
- It keeps only the frames where their own recording is audible (`threshold_db`),
  plus `handle_seconds`. A handle stops where another character becomes audible,
  so it never repaints their mouth at a cut.
- With `region = box`, it keeps only their face region, for a two-shot.
- It cuts only their recording, relabelled 24 → 25 fps for LatentSync.

**Paste** writes the result back at exactly those frames and that region, with a
feathered edge. Chain one pair per character.
`example_workflows/PulseSlate_Dialogue_HyperFlow.json` ships the chain bypassed.

### Pulse Bench

Reads `PulseRender` manifests back and prints seconds-per-frame and peak VRAM
grouped by patch chain. Sol-Attn and Spectrum address *different* memory, and
community reports on Spectrum's speed effect disagree — so this pack measures
rather than recommends. Point it at a run folder after a render and compare rows.

### Pulse Retake · MiniMax H3

Mark a bad span; the node pins the exact frame **before** the cut as
`first_frame` and the exact frame **after** it as `last_frame`, renders only the
gap, and stitches.

This is the one job where H3's two-anchor limit is an advantage rather than a
compromise — two anchors is exactly what a patch needs.

Patch length must sit on the frame grid, so the **cut snaps to fit** rather than
being rejected after you've made it. The stitch is length-preserving by
construction and asserted at runtime: a mismatch would shift every frame after
the patch and desync the audio. `keep_base_audio` defaults on, because a
re-rendered patch invents its own score and will not match the surrounding track.

### Pulse Still · MiniMax H3

A still is a 5-frame render where you keep one frame — same conditioning nodes,
same compiler, same reference marshalling.

- `frame_pick` (0–4): your source is pinned at frame 0, so 0 hugs the original
  and 4 drifts furthest. This is the edit-strength dial, exposed rather than
  hidden.
- `canvas_from_reference`: fits the source's aspect into H3's 1,032,192 px budget
  on the /32 grid. The long edge rounds down; the short edge takes whichever grid
  step is *nearest* the true ratio and still inside the budget. Flooring both axes
  independently loses up to a step on each and the two losses compound rather than
  cancel — 16:9 came out 1344×736, 4% under budget at an actual ratio of 1.826.
  It is the same function the `aspect_ratio` presets use, so a 1920×1080 reference
  and the "16:9 landscape" preset cannot resolve to different canvases.

Deliberately a mode, not a second product. No layers, no masks, no inpainting.
The moment it grows a brush it has become a different application.

---

### Pulse Infinity Loop · Dynamic Shot · Infinity Render · Observer

The closed loop. See [Pulse Studio Infinity](#pulse-studio-infinity--a-film-that-directs-itself)
and [`docs/INFINITY.md`](docs/INFINITY.md).

**Pulse Infinity Loop** is the session. It is identified by its `session` name,
under `ComfyUI/output/pulse_infinity/`.
- Widgets: `story_goal`, `world` (standing direction), an optional
  `opening_beat`, `max_beats` (0 for open-ended), `director_backend`, a
  `script` for the scripted director, and `dialogue_mode`.
- Inputs: the `cast` sockets (Pulse Characters).
- Outputs: the `infinity` handle, a per-beat `seed`, `tts_text` (the line in
  Fish S2's `[delivery] text` form), `speaker_voice`, and `status`.

**Pulse Dynamic Shot** turns the planned beat into a Pulse Shot payload for the
slate.
- It carries the camera, the action, the continuity anchors and the quoted
  line bound to its speaker.
- It continues the last beat unless the director cut.
- A lazy `dialogue_audio` input runs a TTS node only on beats that speak.

**Pulse Infinity Render** is Pulse Render pinned to the session's beat folder.
- It continues the previous beat.
- `audio_mode` and `reference_audio` are `auto`: the beat locks and muxes a TTS
  take when it carries one.

**Pulse Observer** is an output node.
- It shows the director five stills, the last frame and a contact sheet of the
  beat (or lip-sync-corrected frames).
- It commits the observation, stores the next plan and rebuilds the reel.
- It commits each beat once, however many times ComfyUI runs it.

## Architecture

```
comfyui_pulse_studio/          headless core — stdlib only, no torch, no comfy
  constants.py          every value, with the source symbol it came from
  frames.py             the 17k+5 grid; window partitioning
  assets.py             the bin, the budget, and the ONE place ordinals are assigned
  binops.py             UI-facing ops; renumber preview
  timeline.py           the data model that rides in the hidden JSON widget
  compiler.py           timeline -> storyboard prompt + ordered file list
  parsing.py            the two prompt boxes -> shots and project fields
  widget_state.py       merge-not-replace document edits (the erasure fix)
  sockets.py            Autogrow dict shape: contiguous, 0-based, gapless
  retake.py             cut geometry, anchor legality, stitch integrity
  still.py              canvas fitting, frame_pick, branch selection
  patches.py            what the incoming model is missing — duck-typed, imports nothing
  pulse_timeline.py     the PULSE_TIMELINE document, shot ids, window seeds, continuity
  fingerprint.py        every approximation on the model → one canonical hash
  segcache.py           cache key, manifest, durable writes, reuse-vs-render
  report.py             the §9 report; packed-sequence geometry
  bench.py              manifest timings grouped by patch chain
media.py                torch/PIL/PyAV loading — the only tensor code
render.py               the executor: the window loop, the disk writes, the assembly
nodes.py                ComfyUI binding — node faces only
js/ps_widget_guard.js   the prompt-widget write trap (isolated so it is testable)
js/ps_widget_order.js   WIDGET_NAMES[node][version] — the slot contract's name table
js/ps_sockets.js        growing socket groups; links restored by name, not by index
js/ps_warnings.js       paints the patch warning on the node face; owns no widget slot
js/pulse_slate.js       the node face: prompt cards, asset bin, thumbnails
tests/js/               Node tests for the JavaScript that can abort a load
```

`segcache.py` is deliberately free of ComfyUI imports even though it writes files
(spec §15.3): the entire resume behaviour — cache keys, manifest durability,
reuse-vs-render — is exercised on a machine with no GPU. `render.py` holds the
half that cannot be, so the boundary between "tested" and "verified on hardware"
runs along a file rather than through one.

Nothing in `comfyui_pulse_studio/` imports torch or comfy — and that is enforced by an
AST test, not a convention, so it survives the phase where the timeline canvas
wants to decode a preview frame. It is what lets the correctness live somewhere
testable: the whole suite runs in a few seconds with no GPU, no ComfyUI, and no
pip install.

### Window continuity

Two mechanisms hold a multi-window film together. Both originate upstream; see
[Credits](#credits).

**Chunked windows.** A timeline longer than one H3 call is cut into chained
windows at a fixed frame ceiling. `frames.py` owns the partitioning policies, the
124-frame trained floor, the backwards tail merge and the rebalancing that keeps
window lengths even.

**Audio carry-over.** On each continuation window, the previous window's decoded
audio tail is fed back through the reference audio sockets, so the window
continues the existing score and room tone rather than generating an unrelated
one. The tail length is a tunable (`carry_audio_seconds`, default 4 s). A reused
window decodes nothing, so its successor's carry-over is rebuilt from the cached
segment's own PNG and FLAC rather than from tensors in memory.

### Two erasure classes, both closed structurally

**Prompt erasure.** The bin panel used to own the timeline JSON, with a
`catch { return {assets: []} }` fallback that discarded every other field. Any
path reaching it wrote an assets-only document over the real one. Fixed by
moving prompts into their own widgets, making parse failure an error rather than
a default, and doing every edit as a server-side *merge*. Queuing cannot write to
a prompt widget; there is a regression test that types into both boxes with an
empty timeline, queues, and asserts the text survives.

**Socket erasure.** `io.Autogrow` inputs arrive as a **dict** keyed
`ref_image_0`, `ref_image_1`, … and the tokenizer numbers by position in it. A
file that fails to load would leave a hole, and a hole shifts every tag after it.
Unloadable references are therefore dropped *before* tags are assigned, so
ordinals stay dense by construction; `check_socket_groups` is the assertion under
that, and a decode failure at render time stops the run rather than misnumbering.

The JS panel never computes an ordinal itself — it asks the server. A second
implementation of the numbering rule in JavaScript is exactly how the two would
drift apart, and a drifted tag renders successfully while describing the wrong
picture.

---

## Tests

```bash
python3 run_tests.py        # the whole suite, no Python dependencies
python3 run_tests.py -v
node tests/js/test_widget_guard.mjs   # also run by the suite when node exists
node tests/js/test_widget_order.mjs
node tests/js/test_sockets.mjs
```

CI runs exactly that, on Python 3.10, 3.11 and 3.12, on every push and pull
request — plus a job that imports the pure core on a machine with nothing
installed, so a stray `torch` import cannot hide behind another step. See
[.github/workflows/ci.yml](.github/workflows/ci.yml).

Covers frame quantisation (including behavioural parity with core's own
`align_frame_count`), timestamp monotonicity, tag renumbering under
add/remove/reorder/soundtrack-toggle, budget enforcement, window partitioning,
retake geometry over an exhaustive sweep of reachable cuts, canvas fitting, and
end-to-end invariants such as *every tag cited in a prompt must correspond to a
socket that window actually carries*.

Lip-sync timing has its own file. It asserts the span arithmetic in both
directions — that a recording starting after a window opens leads in with
silence rather than being slid forward, and that every span is exactly the
window's length however far outside the recording it falls — and it pins the
zero-offset path byte for byte, because that is what every render already on
disk was cut with.

Also asserted, over all five shipped graphs rather than over one of them: stored
widget values stay aligned with `INPUT_TYPES` (a widget inserted in the middle
would silently shift every value after it), the stored canvas matches the preset
the graph selects, every link is backfilled in both directions, every emitted
window is at least 124 frames, no graph carries a muted branch or a floating
`PulseShot`, no graph cites a reference file the pack does not ship, and no graph
wires a DiT checkpoint nothing in it ever samples with — ~20 GB resident for a
model no window uses.

The segment cache gets its own file, covering each acceptance behaviour in the
spec's §7.5 — resume after a kill, one-shot edits invalidating one window, seed
and `steps` changes invalidating all of them, a requeue with no changes rendering
nothing — plus the properties underneath them: that a cache key is stable across
processes, that a manifest entry without its file is not trusted, that a corrupt
manifest is moved aside rather than overwritten, and that inserting a shot does
not move the seed of any window whose shot set is unchanged.

Patch detection is tested against stub patchers for each of the four packs,
present and absent, including the one property that would quietly disable the
whole cache: **nothing in the descriptor may carry a memory address**, or two
runs of the same graph would key differently and re-render for ever.

The JavaScript that can abort a workflow load is tested too. `protectWidget` is
idempotent across reloads, declines a non-configurable descriptor rather than
throwing, and **wraps** any existing accessor instead of replacing it — replacing
it would sever the widget from ComfyUI's serialization and lose the typed prompt
on save. Every LiteGraph hook is wrapped in `try`, because a throw in
`onConfigure` is what produces *"Loading aborted due to error reloading workflow
data"*.

---

## Maintainers

- Behailu Weldeyohannes — [@Behailu-Weldeyohannes](https://github.com/Behailu-Weldeyohannes), [@behailuaisolutions](https://github.com/behailuaisolutions)

## Acknowledgements

Developed with assistance from Claude, Anthropic's AI assistant.

---

## Credits

Pulse Studio Infinity's closed loop adapts ideas published in Video-Rebirth's
[HyperFlow Infinity](https://github.com/Video-Rebirth/hyperflow-infinity):
- scene-graph memory deltas;
- rejecting a beat that restates an earlier one;
- scene budgets;
- stagnation detection;
- retrying with the rejection reason;
- journaled director calls.

No code or prompt text from that project is included; see [NOTICE](NOTICE).

This project is a fork of
[muse-collective-26/MiniMaxH3-Director-Seed-Hunt](https://github.com/muse-collective-26/MiniMaxH3-Director-Seed-Hunt),
licensed under the MIT License. The upstream copyright notice is reproduced in
[NOTICE](NOTICE), and derived modules carry an attribution header.

Two design elements originate upstream:

- Chunking a long timeline into chained fixed-ceiling windows.
- Audio carry-over across a window seam.

Their implementation in this project is described under
[Architecture](#window-continuity).

## License

**Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Upstream is MIT, which is compatible with an Apache-2.0 project provided
attribution survives; upstream's copyright notice is reproduced in full in
[NOTICE](NOTICE) as that licence requires. muse-collective MIT code **is**
present in this tree — stated plainly there so the question is settled on the
record.

Model weights are **not** redistributed here and carry the MiniMax H3 Community
License, which is a separate agreement governing the weights and their use.

Source provenance rules for contributors — what may be read, what may not be
copied, and the named GPL-3.0 projects that rule covers — are in
[CONTRIBUTING.md](CONTRIBUTING.md).
