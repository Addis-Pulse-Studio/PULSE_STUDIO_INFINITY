"""Pulse Infinity's memory and its contracts. Stdlib only.

  schema       -- DirectedShotPlan, ObservationResult, TurnResponse, unwrap_json,
                  json_schema: what every director backend must send
  scene_memory -- SceneMemory: the entity ledger, spatial topology and history
  antiloop     -- the action blacklist and the repeat / stall checks
  store        -- scene_memory.json on disk, snapshots, crash recovery
"""

from .antiloop import ActionBlacklist, check_plan, content_words, forbidden_phrase, jaccard
from .scene_memory import MEMORY_SCHEMA, CharacterState, Environment, SceneMemory
from .schema import (
    MAX_BEAT_SECONDS,
    MIN_BEAT_SECONDS,
    PLAN_OPTIONAL,
    PLAN_REQUIRED,
    PROVENANCE_INTENDED,
    PROVENANCE_OBSERVED,
    ContinuityAnchors,
    Dialogue,
    DirectedShotPlan,
    NewCharacter,
    ObservationResult,
    TurnResponse,
    json_schema,
    unwrap_json,
)
from .store import MEMORY_FILE, load_memory, restore_snapshot, save_memory, snapshot_memory

__all__ = [
    "ActionBlacklist",
    "CharacterState",
    "ContinuityAnchors",
    "Dialogue",
    "DirectedShotPlan",
    "Environment",
    "MAX_BEAT_SECONDS",
    "MEMORY_FILE",
    "MEMORY_SCHEMA",
    "MIN_BEAT_SECONDS",
    "NewCharacter",
    "ObservationResult",
    "PLAN_OPTIONAL",
    "PLAN_REQUIRED",
    "PROVENANCE_INTENDED",
    "PROVENANCE_OBSERVED",
    "SceneMemory",
    "TurnResponse",
    "check_plan",
    "content_words",
    "forbidden_phrase",
    "jaccard",
    "json_schema",
    "load_memory",
    "restore_snapshot",
    "save_memory",
    "snapshot_memory",
    "unwrap_json",
]
