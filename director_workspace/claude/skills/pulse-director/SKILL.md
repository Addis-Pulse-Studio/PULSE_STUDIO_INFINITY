---
name: pulse-director
description: Direct a Pulse Studio Infinity film from the Director Desk -- wait for ComfyUI's next turn, look at the beat it rendered, and submit the observation and the next shot plan. Use when asked to direct, run the desk, or continue a Pulse Infinity film; run it under /loop to direct indefinitely.
---

# Pulse Director

You direct a film that MiniMax H3 renders one beat at a time inside ComfyUI.
After each beat, ComfyUI posts a turn on the Director Desk -- files in the
session folder -- and waits for your answer. You answer with JSON only.

## One turn

1. Wait for the next turn (blocks up to nine minutes, and tells ComfyUI a
   director is at the desk while it waits):

       {{PYTHON}} {{DESK}} next --wait 540

   If it prints `no turn waiting`, say so in one line and stop -- under `/loop`
   you will be woken for the next one.
2. Read what it prints: the kind of turn, the story goal, the cast you may cite,
   the scene memory, the constraints (forbidden actions, scene budget, the
   longest line that fits), the plan the last beat was rendered from, and any
   reasons your previous answer was rejected.
3. Look at the evidence with the Read tool -- the contact sheet first (five
   captioned stills across the beat), then the last frame. Judge what is really
   on screen, not what was planned.
4. Write the answer to a scratch file. A turn is
   `{"observation": {...}, "plan": {...}}`; the opening turn is `{"plan": {...}}`.
   `{{PYTHON}} {{DESK}} schema turn` prints the exact shape.
5. Submit it:

       {{PYTHON}} {{DESK}} submit <turn_id> <answer.json>

   It runs the same validator ComfyUI will. If it prints `REJECTED`, fix every
   listed problem and submit again. Never edit files in the session folder by
   hand.
6. Report one line: "beat N: <what happens>".

Other commands: `status`, `memory`, `show <turn_id>`, and
`control pause|resume|stop|reroll|replan --note "..."` (applied at the next queue).

## The brief

{{BRIEF}}

## Observing

{{OBSERVER_BRIEF}}

## A plan, for reference

```json
{
  "shot_index": 2,
  "visual_prompt": "@Alice turns away from the stopped clock and steps onto the dark garden path, gravel crunching under her shoes.",
  "negative_prompt": "standing still, looking down at hands",
  "camera_motion": "slow dolly forward, low angle",
  "duration_seconds": 5.17,
  "dialogue": {"speaker": "Alice", "text": "The time has already stopped...", "emotion_tone": "whispering, anxious"},
  "continuity_anchors": {"held_items": ["brass key in left hand"], "costume": "torn blue apron dress", "environment": "overgrown clock garden, twilight"},
  "beat": "Alice leaves the clock behind",
  "scene": "Clock garden",
  "scene_change": false
}
```
