"""Keeping a generated film moving: the action blacklist and the repeat checks.

A model directing one short shot at a time falls into two failures, and both are
well documented in self-play video systems:

  * **the loop** -- the same action restaged with new words ("she opens the
    gate", "she pushes the gate open", "the gate swings open as she pushes");
  * **the stall** -- characters frozen in place while the camera and the mood
    keep changing, every beat a reaction to the beat before.

At cfg 1.0 (HyperFlow's only setting) a negative prompt is never evaluated, so
neither can be fixed by telling H3 what not to draw. They are fixed upstream,
by refusing the plan, and the director is told exactly why and asked again.

WHAT IS COMPARED
----------------
Content words: lowercased, three letters or more (a "key" or a "map" is often
the whole beat), stopwords dropped, crudely stemmed so "opening", "opened" and
"opens" meet, and "gate", "gates" and "gated" too. And only *action* words: every
word of the cast's appearance, the costume and the setting is removed first.
Those are restated verbatim in every prompt on purpose (the renderer has no
memory), so comparing whole prompts would call every beat a repeat of the last.

The thresholds -- Jaccard over 0.5 against any of the last eight beats, 75% of a
forbidden action's words reappearing -- are the ones HyperFlow Infinity found to
separate a new beat from a reworded one. The ideas are credited there; this is
our own implementation of them.

Stdlib only, like the rest of the core.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

__all__ = [
    "REPEAT_JACCARD",
    "FORBIDDEN_OVERLAP",
    "BlacklistEntry",
    "ActionBlacklist",
    "content_words",
    "jaccard",
    "forbidden_phrase",
    "check_plan",
]

#: Above this content-word Jaccard, a beat restates an earlier one.
REPEAT_JACCARD = 0.5
#: This share of a forbidden action's words reappearing restages it.
FORBIDDEN_OVERLAP = 0.75
#: How many earlier beats a new one is compared against.
REPEAT_WINDOW = 8

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

STOPWORDS = frozenset("""
    the and for she her hers his him its are was were but not off out our you who
    all any can has had how now one two too yet per via own way why may let get got
    put set did does done into onto from with that this these those then than them
    they their there here what when where which while whose will would could should
    about above after again against also along among another around away back
    because been before behind being below beneath beside between both down during
    each either even ever every further have having just like more most much near
    other over same shot some such through toward towards under until upon very
    within without your camera frame slowly slow quickly quick gently softly still
    begins begin starts start continues continue moment
""".split())


def _stem(word: str) -> str:
    """A crude suffix strip, so tense and number stop hiding a repeat.

    Not a linguist's stemmer, and it does not need to be: both sides of every
    comparison go through the same function, so all it must do is send the forms
    of one word to one string. The final "e" goes too, which is what makes
    "close" / "closed" / "closing" and "gate" / "gates" agree.
    """
    if word.endswith("ingly") and len(word) > 7:
        word = word[:-5]
    elif word.endswith("ing") and len(word) >= 6:
        word = word[:-3]
    elif word.endswith(("ied", "ies")) and len(word) >= 5:
        word = word[:-3] + "y"
    elif word.endswith("ed") and len(word) >= 5:
        word = word[:-2]
    elif word.endswith("es") and len(word) >= 5 and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        word = word[:-2]
    elif word.endswith("ly") and len(word) >= 6:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) >= 4:
        word = word[:-1]
    # "stepping" / "stepped" -> "stepp" -> "step". Not s or l: "grass", "ball".
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiousl":
        word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_words(text: str, minus: Iterable[str] = ()) -> frozenset:
    """The stemmed content words of `text`, less every word of `minus`."""
    words = {_stem(w) for w in (m.group(0).casefold() for m in _WORD_RE.finditer(text or ""))
             if len(w) >= 3 and w not in STOPWORDS}
    if minus:
        drop = set()
        for other in minus:
            drop |= content_words(other)
        words -= drop
    return frozenset(words)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def forbidden_phrase(action: str) -> str:
    """A completed action, as the thing not to do again.

    "opened the brass lock" -> "opening the brass lock again". Only the first
    word is touched, and only when it ends in -ed: that is every regular verb, and
    an irregular one ("took the key") still reads correctly with "again".
    """
    action = (action or "").strip().rstrip(".")
    if not action:
        return ""
    head, _, rest = action.partition(" ")
    lower = head.casefold()
    if lower.endswith("ied") and len(lower) > 4:
        head = head[:-3] + "ying"
    elif lower.endswith("ed") and len(lower) > 4 and not lower.endswith("eed"):
        head = head[:-2] + "ing"
    phrase = (head + (" " + rest if rest else "")).strip()
    return phrase if phrase.casefold().endswith(" again") else phrase + " again"


class BlacklistEntry:
    """One forbidden action, with the beat it was added and the beat it expires."""

    __slots__ = ("phrase", "added_beat", "expires_beat", "source")

    def __init__(self, phrase: str, added_beat: int, expires_beat: int,
                 source: str = "observed") -> None:
        self.phrase = phrase
        self.added_beat = int(added_beat)
        self.expires_beat = int(expires_beat)
        self.source = source

    def to_dict(self) -> dict:
        return {"phrase": self.phrase, "added_beat": self.added_beat,
                "expires_beat": self.expires_beat, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> Optional["BlacklistEntry"]:
        try:
            return cls(str(data["phrase"]), int(data["added_beat"]),
                       int(data["expires_beat"]), str(data.get("source") or "observed"))
        except (KeyError, TypeError, ValueError):
            return None


class ActionBlacklist:
    """A ring buffer of the last N things that must not happen again.

    Bounded twice: at most `size` entries (the oldest falls off), and each one
    lapses `ttl` beats after it was added unless the observer sees it again. An
    unbounded list would eventually forbid the story itself -- after forty beats,
    "walking", "looking" and "speaking" are all on it.
    """

    __slots__ = ("size", "ttl", "entries")

    def __init__(self, size: int = 8, ttl: int = 4,
                 entries: Optional[list[BlacklistEntry]] = None) -> None:
        self.size = max(1, int(size))
        self.ttl = max(1, int(ttl))
        self.entries = list(entries or [])

    def add(self, phrase: str, beat: int, source: str = "observed") -> bool:
        """Forbid `phrase` from `beat` on. A near-duplicate refreshes the old entry
        instead of taking a second slot. Returns True when it was new."""
        phrase = (phrase or "").strip()
        words = content_words(phrase)
        if not phrase or not words:
            return False
        for entry in self.entries:
            if jaccard(words, content_words(entry.phrase)) >= 0.8:
                entry.expires_beat = int(beat) + self.ttl
                return False
        self.entries.append(BlacklistEntry(phrase, beat, int(beat) + self.ttl, source))
        del self.entries[:-self.size]
        return True

    def expire(self, beat: int) -> list[str]:
        """Drop what has lapsed by `beat`. Returns the phrases dropped."""
        gone = [e.phrase for e in self.entries if e.expires_beat <= int(beat)]
        self.entries = [e for e in self.entries if e.expires_beat > int(beat)]
        return gone

    def active(self) -> list[str]:
        """The forbidden phrases, newest first -- the spec's `forbidden_actions`."""
        return [e.phrase for e in reversed(self.entries)]

    def violated_by(self, text: str, minus: Iterable[str] = ()) -> list[str]:
        """The forbidden actions `text` restages."""
        words = content_words(text, minus)
        hits = []
        for entry in self.entries:
            wanted = content_words(entry.phrase) - {"again"}
            if len(wanted) >= 2 and len(wanted & words) / float(len(wanted)) >= FORBIDDEN_OVERLAP:
                hits.append(entry.phrase)
        return hits

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, items: Optional[list], size: int = 8, ttl: int = 4) -> "ActionBlacklist":
        entries = [e for e in (BlacklistEntry.from_dict(i) for i in (items or [])
                               if isinstance(i, dict)) if e is not None]
        return cls(size, ttl, entries)


def check_plan(plan, memory, max_scene_beats: int = 8,
               repeat_window: int = REPEAT_WINDOW) -> list[str]:
    """Why `plan` would loop or stall the film, as problems to hand the director.

    `memory` is a SceneMemory (duck-typed, so this module does not import it).
    Every check compares action words only -- see the module docstring for why
    the cast's appearance and the setting are removed first.
    """
    problems = []
    static = memory.static_words()
    prompt = plan.visual_prompt or ""

    if memory.last_prompt and prompt.strip().casefold() == memory.last_prompt.strip().casefold():
        problems.append("the visual_prompt repeats the previous beat's word for word -- "
                        "write the next beat, not the last one")

    intent = plan.beat or prompt
    words = content_words(intent, static)
    for earlier in memory.beats[-repeat_window:]:
        if jaccard(words, content_words(earlier.get("intent", ""), static)) > REPEAT_JACCARD:
            problems.append("this beat restates beat %s (%r) -- advance the story instead"
                            % (earlier.get("beat"), (earlier.get("intent") or "")[:120]))
            break

    for phrase in memory.blacklist.violated_by(prompt + " " + (plan.beat or ""), static):
        problems.append("this beat restages a forbidden action: %r" % (phrase,))

    new_scene = plan.cuts or (plan.scene and plan.scene != memory.scene)
    if memory.scene and memory.scene_length >= max_scene_beats and not new_scene:
        problems.append("scene %r has run %d beats, its whole budget -- set scene_change "
                        "true and move the story to a new place or situation"
                        % (memory.scene, memory.scene_length))

    if memory.stagnant and not new_scene:
        recent = [content_words(b.get("intent", ""), static) for b in memory.beats[-3:]]
        if not plan.beat or any(jaccard(words, r) >= 0.25 for r in recent):
            problems.append("the last %d beats changed nothing -- this one must complete a "
                            "visible change of situation (say which in `beat`), or set "
                            "scene_change true" % memory.stagnant_streak)
    return problems
