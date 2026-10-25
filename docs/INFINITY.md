# Pulse Studio Infinity

A film that directs itself, one beat at a time. MiniMax H3 renders a beat; a
director -- Claude Code, Gemini in Antigravity, a local model through Ollama or
pi, or a script -- looks at what actually came out, updates a scene memory, and
plans the next beat; the next queue run renders it, opening on the last frame of
the one before. Sixty seconds, ten minutes, or until the story ends.

The director layer and the memory never learn which diffusion weights render a
beat, and the render layer never learns which model directs it. Both sides meet
in one JSON contract, enforced by one validator.

- [How a beat happens](#how-a-beat-happens)
- [Quick start](#quick-start)
- [Choosing the director](#choosing-the-director)
- [The director workspace](#the-director-workspace)
- [The session folder](#the-session-folder)
- [The contracts](#the-contracts)
- [The Director Desk protocol](#the-director-desk-protocol-pulse-desk1)
- [Keeping the story moving](#keeping-the-story-moving)
- [Continuity between beats](#continuity-between-beats)
- [Dialogue](#dialogue)
- [Controls](#controls)
- [Things that never happen twice](#things-that-never-happen-twice)
- [Privacy](#privacy)
- [Troubleshooting](#troubleshooting)
- [Limits and follow-ups](#limits-and-follow-ups)

## How a beat happens

One ComfyUI queue run is one beat:

```
Pulse Infinity Loop ──infinity──► Pulse Dynamic Shot ──shot──► Pulse Slate ──timeline──► Pulse Infinity Render
   reads the session,                the plan as a              the cast, the            renders into
   plans the beat once               PulseShot payload          recipe, compiles         beats/beat_NNNN/aK,
   (opening / script / director)                                                         continuing beat N-1
                                                                                                │ beat
                                                              Pulse Observer ◄──────────────────┘
                                         stills → director → observation + next plan → memory, reel
```

Queue the graph N times for N beats -- each queued copy re-reads the session,
because the Loop node's `IS_CHANGED` is the session's `<uid>:<revision>` -- or turn
on **auto-continue** in the Loop node's context menu for an open-ended film.

## Quick start

1. Open `example_workflows/PulseInfinity_Scripted.json`. It uses the **scripted**
   director (no model, nothing leaves the machine) and stock loaders.
2. Queue it three times. Three beats render; the Observer's report shows what was
   committed and what comes next; `ComfyUI/output/pulse_infinity/clock_garden_scripted/reel/`
   holds the joined film.
3. Open `example_workflows/PulseInfinity_ClosedLoop.json` for the real thing:
   HyperFlow 8-step, the director from your config, and a Fish Audio S2 branch for
   dialogue.

## Choosing the director

One key, `director_backend`, resolved in this order -- the first that names a
backend wins, and the Loop node's `status` says which and why:

1. the Loop node's `director_backend` widget, unless it is `from_config`;
2. the environment variable `PULSE_DIRECTOR_BACKEND`;
3. `director_backend` in `pulse_infinity.yaml` (or `.json`) -- in ComfyUI's
   `user/` folder, or wherever `PULSE_INFINITY_CONFIG` points;
4. `claude_code`.

Start from `pulse_infinity.example.yaml` in the pack.

| Backend | Reached through | Sees | Where it runs |
|---|---|---|---|
| `claude_code` (default) | the Director Desk; or `claude -p` | the contact sheet and last frame | cloud agent |
| `gemini_antigravity` | the Director Desk; or `gemini -p` | the full clip with its sound, then the stills | cloud agent |
| `ollama` | `ollama run <model> --format json` | the stills, for a vision model | this machine |
| `pi` | `pi -p`; or the Director Desk | the stills, for a vision model | depends on its model |
| `scripted` | in process | nothing -- it takes its own plan as the observation | this machine |

The pack opens no socket for any of them. The Desk is files; the CLI backends
are started with no shell, from one module, with the prompt on stdin.

**Ollama and the GPU.** H3 holds about 21 GB of a 32 GB card. With
`free_vram_before_call` (on for `ollama` and `pi`) the Loop node and the
Observer unload ComfyUI's models before the director runs; they reload on the
next beat. HyperFlow's `auto` mode is bypass on an int8 checkpoint, so a reload
does not repeat its five-minute first patch.

**ComfyUI on Windows, Claude Code in WSL.** The Desk handles this by itself
(every path in a request is relative to the session folder). For the headless
`cli` transport, run Claude through `wsl.exe` and set `path_style: wsl` -- see the
example config.

## The director workspace

An agent needs instructions and a tool. Install both into a folder:

```bash
python3 tools/setup_director_workspace.py D:/PULSE_STUDIO_INFINITY --comfyui D:/AI_Tools/ComfyUI_Conda/ComfyUI
```

It writes `CLAUDE.md` and the `pulse-director` skill for Claude Code (plus a
`.claude/settings.json` that adds the sessions folder and allows the desk tool),
`AGENTS.md` and `.agent/rules` / `.agent/workflows/pulse-direct.md` for
Antigravity, and `.pulse_desk.json`, which tells the desk tool where the
sessions are. The skill's brief is generated from the same text CLI directors
are fed (`pulse_director/prompts.py`). Existing files are kept unless `--force`.

Then, in Claude Code in that folder: `/pulse-director` for one turn, or
`/loop /pulse-director` to direct until the film ends. In Antigravity:
`/pulse-direct`.

The desk tool, `tools/pulse_desk.py`, is stdlib-only and imports the pack's own
validator:

```
pulse_desk.py next --wait 540        wait for a turn and print it
pulse_desk.py submit <turn> <file>   validate, then answer (REJECTED lists every problem)
pulse_desk.py status | sessions | memory [--json] | schema turn|plan|observation|open
pulse_desk.py control pause|resume|stop|reroll|replan [--note "..."]
```

## The session folder

`ComfyUI/output/pulse_infinity/<session>/` -- every path inside these files is
relative to it:

| | |
|---|---|
| `session.json` | uid, goal, status, the committed beat, the pending beat and attempt, the undo epoch, the revision |
| `scene_memory.json` | what the film remembers (below) |
| `memory_snapshots/beat_NNNN.json` | the memory before each beat -- what undo restores |
| `beats/beat_NNNN/plan.json` | the plan the beat was rendered from |
| `beats/beat_NNNN/aK/` | attempt K: the normal Pulse Render run folder (manifest, segments), `beat.json`, `evidence/` (stills, last frame, contact sheet) |
| `desk/inbox`, `outbox`, `done`, `rejected` | the Director Desk |
| `journal/` | every director exchange; accepted answers are replayed on a re-run |
| `schemas/` | the JSON Schemas a director answers to |
| `reel/` | every committed beat, joined |
| `control.json` | a pending control request, consumed by the next queue run |

## The contracts

**The plan** (`DirectedShotPlan`) -- the mission's director output contract,
read back byte for byte:

```json
{
  "shot_index": 1,
  "visual_prompt": "Cinematic shot of Alice turning away from the clock, stepping into the dark garden path...",
  "negative_prompt": "standing still, repeating action, looking down at hands",
  "camera_motion": "slow dolly forward, low angle",
  "duration_seconds": 5.0,
  "dialogue": {"speaker": "Alice", "text": "The time has already stopped...", "emotion_tone": "whispering, anxious"},
  "continuity_anchors": {"held_items": ["brass key in left hand"], "costume": "torn blue apron dress", "environment": "overgrown clock garden, twilight"}
}
```

Optional, written back only when set: `beat` (one line: the story step),
`scene`, `scene_change` (true opens on a cut), `new_characters`,
`characters_present`, `end_of_story`. Durations are rounded to the nearest frame
and snapped up to H3's 17k+5 grid inside the trained 124-362 frames
(5.0 s renders as 124 frames, 5.17 s, and the report says so). Refused: hand-typed
`<Picture N>` tags, `@Name` for someone not in the cast, an unknown speaker, a
line faster than 200 words a minute for its beat.

**The observation** -- what the beat showed, as changes: `summary`,
`completed_actions` (at most three story events), per-character `posture`,
`inventory_add` / `inventory_remove`, `appearance_change`, `injuries`,
`emotional_state`; `environment` changes; `stationary`; `forbid`;
`continuity_issues`.

**The turn** -- `{"observation": {...}, "plan": {...}}`; the opening turn is
`{"plan": {...}}`. `pulse_desk.py schema turn` prints the JSON Schema.

**The memory** -- `scene_memory.json` writes the mission's state contract with
its keys unchanged (`story_goal`, `current_beat`, `characters` with
`appearance` / `inventory` / `last_posture`, `environment` with `setting` /
`time_of_day` / `lighting`, `action_history`, `forbidden_actions`). The entity
ledger adds, per character, `pinned_appearance` (the identity line, restated
in every prompt and never overwritten by an observation), `costume_tags`,
`injuries`, `emotional_state`, first and last beat seen, `cast` and
`provenance`; the place gains `camera_orientation` and `landmarks`. The loop's
own bookkeeping lives under one `pulse` key.

## The Director Desk protocol (`pulse-desk/1`)

- ComfyUI writes `desk/inbox/<turn>.json`, once. `<turn>` is
  `b0004-turn-a1` -- the beat being planned, the kind, the attempt -- with an
  `-e<epoch>-` after a reroll or replan.
- The request carries `kind` (`open`, `turn`, `observe`, `plan`), `beat`,
  `story_goal`, `memory` and `memory_text` (a bounded view), `last_plan`,
  `evidence` and `look_at` (relative paths, most telling first), `constraints`,
  `rejection` (the previous attempt's problems), `cast`, `pacing`, `guidance`,
  `capabilities`, `answer_schema` and `reply_to`.
- The director writes `desk/outbox/<turn>.json`, once (`pulse_desk.py submit`).
- ComfyUI validates it. Accepted: both files move to `desk/done/`. Rejected:
  to `desk/rejected/` with a `.verdict.json`, and attempt `a2` is posted with the
  problems -- never a rewrite of `a1`, because replacing a file the other side has
  open fails on Windows. After `max_attempts` the beat is refused (or, with
  `on_invalid: accept`, kept with its problems in the notes).
- Both sides poll; WSL's inotify does not see writes made from Windows.
- `desk/heartbeat.json` says a director is waiting; without one, the node's
  status says "nobody is at the desk -- start /pulse-director".
- Answers are untrusted: capped at 256 KB, parsed tolerantly (fences, prose, a
  `<think>` block, a tool_use block, a CLI result envelope), validated strictly,
  and no path in them is ever followed.

## Keeping the story moving

At cfg 1.0 -- HyperFlow's only setting -- a negative prompt is never evaluated.
So the two classic failures of self-directed video are refused at the plan,
before any GPU time:

- **The loop.** Every completed action becomes a forbidden one ("opened the brass
  lock" → "opening the brass lock again") for four beats, in a ring of eight. A
  plan is refused if it restages one (75% of its content words), restates one of
  the last eight beats (content-word Jaccard over 0.5), or repeats the last
  prompt verbatim. The cast's appearance and the setting are removed before
  comparing -- every prompt restates them on purpose.
- **The stall.** Two beats with no completed action and no change of state is a
  stall; the next plan must name a visible change in its `beat`, or cut. A scene
  that has run eight beats must cut.

A refused plan goes back to the director with the reasons. A turn is also
judged against the memory *as it will be* once its own observation is committed,
so a plan that restages the action the director just reported finishing is
caught on the same turn.

These ideas -- scene-graph deltas, beat-repeat rejection, scene budgets,
stagnation detection, retry with the reason -- come from Video-Rebirth's
HyperFlow Infinity. The implementation and the words are this pack's own.

## Continuity between beats

Window 2 of a render has always continued window 1. Infinity makes window 1 of
a beat continue the last window of the beat before (`carry-in`): the Dynamic
Shot's payload carries `continue_from` -- the previous beat's run folder and
segment key -- and the slate compiles window 1 as a continuation, so its
carry-over sockets are filled from that segment's files and its key chains to it.

The slate's own `continuity` setting decides the seam. Use **`none`**: ref2va,
the previous last frame as `<Picture 1>`, the cast's face references still live,
one checkpoint. `ref_guided_carry` also pins the frame with a guide. The modes
that pin a frame on fl2va (`last_frame_carry`, `keyframe_pairs`) drop every
reference -- no face, no voice -- and say so. When the director sets
`scene_change: true`, the beat opens on a clean cut instead.

Keep the slate's `carry_audio` **off**: its 4 s tail outlasts the 0.4 s after a
spoken line and would carry speech into the next beat.

## Dialogue

- `dialogue_mode: native` -- H3 speaks the `<d>` line itself, in the speaking cast
  member's voice timbre. Wire each cast voice as a Pulse Voice with role
  **`voice_timbre`**: a `lip_sync` voice sits on the film clock, which restarts
  every beat, and the Loop node warns about it.
- `dialogue_mode: external_tts` -- the Loop node's `tts_text` (the line in Fish
  Audio S2's `[delivery] text` form), `speaker_voice` (the speaker's own
  recording) and `seed` feed a TTS node; its audio goes into the Dynamic Shot's
  lazy `dialogue_audio`, which ComfyUI evaluates only on beats that speak. The
  take becomes the shot's lip-sync voice, the beat is lengthened to fit it, and
  Pulse Infinity Render locks it into the audio latent (`audio_mode: auto`) and
  muxes it (`reference_audio: auto`).
- For mouth correction: Pulse Lip-Sync Segment → LatentSync → Pulse Lip-Sync
  Paste after the render, and the corrected images and audio into the
  Observer's `images` / `audio`. The director is shown the corrected frames and
  the reel uses them.

## Controls

From the Loop node's context menu, from `pulse_desk.py control`, or by writing
`control.json`:

- **pause / resume** -- the next queue run does nothing while paused.
- **stop** -- the film is complete.
- **reroll** -- render the last beat again from the same plan (a new attempt, so
  new seeds); the memory goes back to before it.
- **replan** -- ask the director for a new plan for the last beat, with your note.

## Things that never happen twice

ComfyUI's RAM-pressure cache can run a node again in the same queue, and a crash
can stop one half-way. So:

- a director exchange is journaled under the session's undo epoch, the beat, the
  kind and the render it observed, and replayed rather than asked again;
- a render is committed once; committing it again returns the first outcome;
- a render the session has moved past (it was undone, or overtaken) is refused;
- a crash between saving the memory and saving the session restores the
  pre-beat snapshot before committing again;
- each beat attempt has its own run folder, so a resumed render reuses every
  segment it finished.

## Privacy

The pack sends nothing anywhere. A Director Desk session is files on your disk.
But directing means looking: with `claude_code` or `gemini_antigravity`, the
agent you run reads the beat's stills (and, for Gemini, the clip) and they leave
the machine through it. `ollama`, `pi` on a local model and `scripted` keep
everything local. The Loop node's status names the active director and says
which kind it is.

## Troubleshooting

- **"no answer ... nobody has checked the desk"** -- start the director:
  `/pulse-director` in Claude Code, `/pulse-direct` in Antigravity, in the
  workspace you installed. Queue again; the request is still on the desk.
- **The film does not advance when queued again** -- it is paused or complete
  (the Loop node's status says which), or the last turn failed: the Observer's
  report says why, and queueing again retries it.
- **"this beat should continue beat N, but the timeline does not carry it in"** --
  Pulse Slate's `shots.shot_1` is not wired from the Dynamic Shot.
- **A director keeps being rejected** -- read `desk/rejected/*.verdict.json`; the
  same problems are in the next request's `rejection`.
- **Ollama runs out of memory** -- keep `free_vram_before_call` on, and use a
  smaller tag or `OLLAMA_KEEP_ALIVE=0`.

## Limits and follow-ups

- Seams are a carried reference frame, not a carried latent tail. HyperFlow
  Infinity and H3-Motion-Context pin the previous segment's last seven latents;
  that is the next step here.
- One beat is one queue run. A single node that loops inside one execution would
  run headless without the browser; auto-continue needs the tab open (or queue a
  batch).
- Whether Antigravity's agent can watch an mp4 from disk depends on its tools;
  the request always lists the stills as well.
