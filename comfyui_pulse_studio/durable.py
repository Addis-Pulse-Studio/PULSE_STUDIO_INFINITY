"""Durable file writes: temp file, fsync, atomic replace, fsync directory.

One implementation of the write order spec §7.4 settled for the segment
manifest, so every record that has to survive a crash is written the same way:

    temp file in the target's own directory -> write -> flush -> fsync
    -> os.replace onto the target -> fsync the directory

A reader therefore sees either the old file or the new one, never half of
either. The temp file sits in the target's directory because `os.replace` is
atomic only within one filesystem, and a temp file in the system temp folder can
be on another drive -- on the Windows boxes this pack mostly runs on, `C:` while
the output folder is on `D:`.

The segment manifest was the only such record until Pulse Infinity, which keeps
four more: the session, the scene memory, the desk's messages and the journal of
every director exchange. A second copy of these eight lines would be a second
place for the order to drift.

Deliberately free of ComfyUI and torch imports, like the rest of the core.
"""

import json
import os
import tempfile

__all__ = ["fsync_dir", "write_text_durably", "write_json_durably"]


def write_text_durably(path, text, prefix=".pulse-", suffix=".tmp"):
    """Write `text` to `path` so that a crash leaves the old file or the new one.

    `prefix` names the temp file, so a caller can recognise its own leftovers --
    a crash between the write and the replace leaves one behind. Returns `path`.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:  # pragma: no cover
            pass
        raise
    fsync_dir(directory)
    return path


def write_json_durably(path, data, prefix=".pulse-", indent=2):
    """`write_text_durably` for a JSON document, keys sorted so diffs stay small.

    `ensure_ascii` is off: these records hold prompts and dialogue, and a line of
    Amharic escaped to `\\u12a0...` is unreadable to the person opening the file to
    see what the director decided.
    """
    payload = json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False)
    return write_text_durably(path, payload, prefix=prefix, suffix=".json")


def fsync_dir(directory):
    """fsync a directory so a rename is durable. No-op where unsupported."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (OSError, AttributeError):  # pragma: no cover - Windows
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystems that refuse
        pass
    finally:
        os.close(fd)
