"""Runtime state calculations and operation results shared by control code."""

from .actions import (
    ShotTarget,
    action_destination,
    available_moves,
    detect_promotion,
    select_shot_target,
)
from .state import (
    ACTION_DIRECTIONS,
    build_action_mask,
    build_observation,
    build_threat_matrix,
    update_enemy_turns,
)
from .rewards import RewardResult, calculate_step_reward
from .transitions import TerminalResult, apply_terminal_rules, update_countdown
from .io import CaptureResult, InputResult, IOStatus
from .execution import ActionCompletion, CompletionStatus, verify_action_completion

__all__ = [
    "ACTION_DIRECTIONS",
    "ActionCompletion",
    "CaptureResult",
    "CompletionStatus",
    "InputResult",
    "IOStatus",
    "RewardResult",
    "ShotTarget",
    "TerminalResult",
    "action_destination",
    "apply_terminal_rules",
    "available_moves",
    "build_action_mask",
    "build_observation",
    "build_threat_matrix",
    "calculate_step_reward",
    "detect_promotion",
    "select_shot_target",
    "update_countdown",
    "update_enemy_turns",
    "verify_action_completion",
]
