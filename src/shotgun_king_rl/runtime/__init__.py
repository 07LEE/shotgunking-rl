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
from .consistency import ObservationConsistency, compare_policy_observation
from .scope import AssistanceScope, assess_assistance_scope, format_assistance_scope
from .suggestions import (
    ACTION_KEYS,
    ACTION_NAMES,
    SuggestedAction,
    SuggestionView,
    build_suggestion_view,
    format_suggestion_view,
)
from .tracing import (
    ActionTraceWriter,
    MANIFEST_SCHEMA,
    TRACE_SCHEMA,
    build_action_trace,
    file_sha256,
)
from .modes import RuntimeMode, resolve_runtime_mode

__all__ = [
    "ACTION_DIRECTIONS",
    "ACTION_KEYS",
    "ACTION_NAMES",
    "ActionCompletion",
    "ActionTraceWriter",
    "AssistanceScope",
    "CaptureResult",
    "CompletionStatus",
    "InputResult",
    "IOStatus",
    "MANIFEST_SCHEMA",
    "ObservationConsistency",
    "RewardResult",
    "RuntimeScreenState",
    "RuntimeMode",
    "ScreenAssessment",
    "ShotTarget",
    "SuggestedAction",
    "SuggestionView",
    "TerminalResult",
    "TRACE_SCHEMA",
    "action_destination",
    "apply_terminal_rules",
    "apply_screen_state_rules",
    "assess_assistance_scope",
    "available_moves",
    "build_action_mask",
    "build_action_trace",
    "build_observation",
    "build_threat_matrix",
    "build_suggestion_view",
    "calculate_step_reward",
    "classify_runtime_screen",
    "compare_policy_observation",
    "detect_promotion",
    "format_assistance_scope",
    "format_suggestion_view",
    "file_sha256",
    "resolve_runtime_mode",
    "select_shot_target",
    "update_countdown",
    "update_enemy_turns",
    "verify_action_completion",
]
