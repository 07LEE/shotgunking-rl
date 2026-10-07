"""Private JSONL traces for inspectable live assistance sessions."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .suggestions import ACTION_NAMES


TRACE_SCHEMA = "runtime-action-trace-v1"
MANIFEST_SCHEMA = "runtime-run-manifest-v1"


class ActionTraceWriter:
    """Append one durable JSON record for every attempted runtime action."""

    def __init__(self, directory="data/runtime_traces", session_id=None):
        now = datetime.now(timezone.utc)
        self.session_id = session_id or now.strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = Path(directory) / f"{self.session_id}.jsonl"
        self.manifest_path = Path(directory) / f"{self.session_id}.manifest.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        payload = {
            "schema": TRACE_SCHEMA,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(payload), ensure_ascii=False) + "\n")
            stream.flush()
        return payload

    def write_manifest(self, manifest):
        payload = {
            "schema": MANIFEST_SCHEMA,
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **manifest,
        }
        self.manifest_path.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload


def file_sha256(path):
    """Return a local artifact hash without loading the whole file into memory."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_action_trace(
    *,
    episode,
    step,
    mode,
    rank,
    floor,
    weapon_type,
    weapon_name,
    observation,
    action_mask,
    q_values,
    recommendations,
    selected_action,
    selection_source,
    loaded_ammo,
    max_ammo,
    reserve_ammo,
    next_observation,
    reward,
    terminated,
    truncated,
    info,
    accepted_for_replay,
):
    """Build a JSON-safe record from the exact pre- and post-action states."""
    observation = np.asarray(observation)
    next_observation = np.asarray(next_observation)
    selected_action = int(selected_action)
    executed_action = info.get("action")
    return {
        "episode": int(episode),
        "step": int(step),
        "mode": mode,
        "scope": {
            "rank": int(rank),
            "floor": int(floor),
            "weapon_type": int(weapon_type),
            "weapon_name": weapon_name,
        },
        "observation": {
            "board": observation[:64].reshape(8, 8),
            "ammo": {
                "loaded": int(loaded_ammo),
                "capacity": int(max_ammo),
                "reserve": int(reserve_ammo),
            },
        },
        "decision": {
            "action_mask": np.asarray(action_mask, dtype=np.int8),
            "q_values": q_values,
            "recommendations": [
                {
                    "action": item.action,
                    "name": item.name,
                    "q_value": item.q_value,
                    "reason": item.reason,
                }
                for item in recommendations
            ],
            "selected_action": selected_action,
            "selected_name": ACTION_NAMES.get(
                selected_action, f"Action {selected_action}"
            ),
            "selection_source": selection_source,
        },
        "result": {
            "executed_action": executed_action,
            "executed_name": (
                ACTION_NAMES.get(int(executed_action), f"Action {executed_action}")
                if executed_action is not None
                else None
            ),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "transition_verified": bool(
                info.get("input_succeeded", True)
                and info.get("transition_ready", True)
            ),
            "accepted_for_replay": bool(accepted_for_replay),
            "info": info,
            "next_board": next_observation[:64].reshape(8, 8),
        },
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
