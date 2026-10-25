# Pulse Studio Infinity -- director workspace

This folder is where a Claude Code session directs Pulse Studio Infinity films.
ComfyUI renders each beat with MiniMax H3 and asks the director, through files,
what the beat showed and what happens next.

- Direct: run the `pulse-director` skill (`/pulse-director`), or `/loop /pulse-director`
  to keep directing until the film ends.
- The desk tool: `{{PYTHON}} {{DESK}}` -- `next`, `submit`, `status`, `memory`,
  `schema`, `control`.
- Sessions live in `{{SESSIONS}}` -- one folder per film: `scene_memory.json`,
  `beats/`, `desk/`, `journal/`, `reel/`. Read anything there; write only through
  the desk tool.
- The pack: `{{PACK}}`. Do not edit it from here.

Answer turns with JSON only, cite cast members as @Name, never type `<Picture N>`
tags, and never restage a forbidden action. The skill has the full brief.
