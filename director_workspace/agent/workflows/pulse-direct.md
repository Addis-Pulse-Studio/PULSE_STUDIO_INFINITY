---
description: Direct the next Pulse Studio Infinity beat from the Director Desk
---

1. Run `{{PYTHON_WIN}} {{DESK_WIN}} next --wait 540` and read the turn it prints. If it
   says `no turn waiting`, stop.
2. Open the evidence it lists, in order: the beat's clip if you can view video
   (it has the sound), then the contact sheet, then the last frame.
3. Write the answer -- `{"observation": {...}, "plan": {...}}`, or `{"plan": {...}}` on
   the opening turn -- to a scratch file, following AGENTS.md and
   `{{PYTHON_WIN}} {{DESK_WIN}} schema turn`.
4. Run `{{PYTHON_WIN}} {{DESK_WIN}} submit <turn_id> <answer.json>`. If it prints
   REJECTED, fix every listed problem and submit again.
5. Say in one line what beat you planned, then run this workflow again for the
   next turn.
