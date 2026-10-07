"""Private JSONL traces for inspectable live assistance sessions."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .suggestions import ACTION_NAMES


TRACE_SCHEMA = "runtime-action-trace-v1"


class ActionTraceWriter:
    """Append one durable JSON record for every attempted runtime action."""

    def __init__(self, directory="data/runtime_traces", session_id=None):
        now = datetime.now(timezone.utc)
        self.session_id = session_id or now.strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = Path(directory) / f"{self.session_id}.jsonl"
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
