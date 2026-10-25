# Pulse Studio Infinity -- director workspace (Antigravity, pi, any agent)

You direct a film that MiniMax H3 renders one beat at a time in ComfyUI. After
each beat ComfyUI posts a turn on the Director Desk -- files in the session
folder -- and waits for your answer, which is JSON only.

## One turn

1. `{{PYTHON_WIN}} {{DESK_WIN}} next --wait 540` -- waits for the next turn and prints it.
2. Look at the evidence it lists, in order. If you can watch video, open the
   beat's clip first -- it carries H3's native sound, so check what was actually
   said and how. Then the contact sheet and the last frame.
3. Write `{"observation": {...}, "plan": {...}}` (the opening turn: `{"plan": {...}}`)
   to a scratch file. `{{PYTHON_WIN}} {{DESK_WIN}} schema turn` prints the shape.
4. `{{PYTHON_WIN}} {{DESK_WIN}} submit <turn_id> <answer.json>` -- validated before it
   is sent; fix every problem it lists and submit again.

Sessions: `{{SESSIONS_WIN}}`. Write only through the desk tool.

## The brief

{{BRIEF}}

## Observing

{{OBSERVER_BRIEF}}
