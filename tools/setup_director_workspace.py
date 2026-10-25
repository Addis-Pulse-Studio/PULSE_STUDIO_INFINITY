#!/usr/bin/env python3
"""Make a folder into a Pulse Studio Infinity director workspace.

    python3 tools/setup_director_workspace.py <workspace> --comfyui <ComfyUI folder>

Writes, from the templates in director_workspace/ and the director brief in
comfyui_pulse_studio/pulse_director/prompts.py (so the skill an agent reads and
the prompt a CLI director is fed can never say different things):

    CLAUDE.md, .claude/skills/pulse-director/SKILL.md   Claude Code
    .claude/settings.json                               the sessions folder as an
                                                        additional directory, and
                                                        the desk tool allowed
    AGENTS.md, .agent/rules/, .agent/workflows/         Antigravity (and pi)
    .pulse_desk.json                                    where the sessions are,
                                                        for tools/pulse_desk.py

An existing file is kept unless --force; .claude/settings.json is merged, never
replaced. Run from WSL, it writes Linux paths for Claude Code and Windows paths
for Antigravity, which runs on Windows.
"""

import argparse
import json
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

from comfyui_pulse_studio.pulse_director.prompts import (  # noqa: E402
    DIRECTOR_BRIEF,
    OBSERVER_BRIEF,
    convert_path,
)

TEMPLATES = os.path.join(PACK, "director_workspace")
#: template path -> installed path. Dot-folders are spelled without the dot in the
#: pack, so Claude Code working on the pack itself never mistakes them for its own.
LAYOUT = {
    "CLAUDE.md": "CLAUDE.md",
    "AGENTS.md": "AGENTS.md",
    "claude/skills/pulse-director/SKILL.md": ".claude/skills/pulse-director/SKILL.md",
    "agent/rules/pulse-director.md": ".agent/rules/pulse-director.md",
    "agent/workflows/pulse-direct.md": ".agent/workflows/pulse-direct.md",
}


def _on_wsl():
    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _windows(path):
    return convert_path(path, "windows") if path.startswith("/mnt/") else path


def values(sessions_root):
    desk = os.path.join(PACK, "tools", "pulse_desk.py")
    posix = os.name != "nt"
    return {
        "{{PYTHON}}": "python3" if posix else "python",
        "{{DESK}}": desk,
        "{{SESSIONS}}": sessions_root,
        "{{PACK}}": PACK,
        "{{PYTHON_WIN}}": "python",
        "{{DESK_WIN}}": _windows(desk),
        "{{SESSIONS_WIN}}": _windows(sessions_root),
        "{{BRIEF}}": DIRECTOR_BRIEF.strip(),
        "{{OBSERVER_BRIEF}}": OBSERVER_BRIEF.strip(),
    }


def render(text, table):
    for key, value in table.items():
        text = text.replace(key, value)
    return text


def _rule_path(path):
    """A permission rule's absolute path: //mnt/d/... for /mnt/d/..., as Claude Code reads it."""
    return "/" + path if path.startswith("/") else path.replace("\\", "/")


def settings(existing, sessions_root, table):
    data = dict(existing or {})
    permissions = dict(data.get("permissions") or {})
    extra = list(permissions.get("additionalDirectories") or [])
    if sessions_root not in extra:
        extra.append(sessions_root)
    allow = list(permissions.get("allow") or [])
    for rule in ("Bash(%s %s:*)" % (table["{{PYTHON}}"], table["{{DESK}}"]),
                 "Read(%s/**)" % _rule_path(sessions_root)):
        if rule not in allow:
            allow.append(rule)
    permissions.update(additionalDirectories=extra, allow=allow)
    data["permissions"] = permissions
    return data


def install(workspace, sessions_root, force=False, out=print):
    workspace = os.path.abspath(workspace)
    table = values(sessions_root)
    written = []
    for source, target in LAYOUT.items():
        destination = os.path.join(workspace, *target.split("/"))
        if os.path.exists(destination) and not force:
            out("kept   %s (exists; --force to replace)" % target)
            continue
        with open(os.path.join(TEMPLATES, *source.split("/")), "r", encoding="utf-8") as fh:
            text = render(fh.read(), table)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        written.append(destination)
        out("wrote  %s" % target)

    settings_path = os.path.join(workspace, ".claude", "settings.json")
    existing = {}
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(settings(existing, sessions_root, table), fh, indent=2)
        fh.write("\n")
    written.append(settings_path)
    out("merged .claude/settings.json")

    desk_path = os.path.join(workspace, ".pulse_desk.json")
    with open(desk_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"sessions_root": sessions_root, "sessions_root_windows":
                   table["{{SESSIONS_WIN}}"], "pack": PACK}, fh, indent=2)
        fh.write("\n")
    written.append(desk_path)
    out("wrote  .pulse_desk.json")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workspace")
    parser.add_argument("--comfyui", help="the ComfyUI folder (its output/ holds the sessions)")
    parser.add_argument("--sessions-root", help="the sessions folder itself, instead")
    parser.add_argument("--sessions-dir", default="pulse_infinity",
                        help="sessions_dir from pulse_infinity.yaml (default pulse_infinity)")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    args = parser.parse_args(argv)
    if args.sessions_root:
        root = args.sessions_root
    elif args.comfyui:
        root = os.path.join(args.comfyui, "output", args.sessions_dir)
    else:
        parser.error("give --comfyui <ComfyUI folder> or --sessions-root <folder>")
    if _on_wsl() and len(root) > 2 and root[1] == ":":
        root = convert_path(root, "wsl")
    install(args.workspace, os.path.abspath(root), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
