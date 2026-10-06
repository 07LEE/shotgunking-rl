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
from .transitions import (
    RuntimeScreenState,
    ScreenAssessment,
    TerminalResult,
    apply_screen_state_rules,
    apply_terminal_rules,
    classify_runtime_screen,
    update_countdown,
)
from .io import CaptureResult, InputResult, IOStatus
from .execution import ActionCompletion, CompletionStatus, verify_action_completion
from .scope import AssistanceScope, assess_assistance_scope, format_assistance_scope

__all__ = [
    "ACTION_DIRECTIONS",
    "ActionCompletion",
    "AssistanceScope",
    "CaptureResult",
    "CompletionStatus",
    "InputResult",
    "IOStatus",
    "RewardResult",
    "RuntimeScreenState",
    "ScreenAssessment",
    "ShotTarget",
    "TerminalResult",
    "action_destination",
    "apply_terminal_rules",
    "apply_screen_state_rules",
    "assess_assistance_scope",
    "available_moves",
    "build_action_mask",
    "build_observation",
    "build_threat_matrix",
    "calculate_step_reward",
    "classify_runtime_screen",
    "detect_promotion",
    "format_assistance_scope",
    "select_shot_target",
    "update_countdown",
    "update_enemy_turns",
    "verify_action_completion",
]
