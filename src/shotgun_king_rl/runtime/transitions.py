"""Pure episode and observed-screen transition decisions."""

from dataclasses import dataclass
from enum import Enum


class RuntimeScreenState(str, Enum):
    """Screen states that determine whether a transition is safe to consume."""

    GAMEPLAY = "gameplay"
    TRANSITION = "transition"
    VICTORY = "victory"
    DEFEAT = "defeat"
    OBSERVATION_FAILURE = "observation_failure"


@dataclass(frozen=True)
class ScreenAssessment:
    """One explicit classification of the latest captured game screen."""

    state: RuntimeScreenState
    reason: str


@dataclass(frozen=True)
class TerminalResult:
    """Reward and terminal state after applying legacy end conditions."""

    reward: float
    terminated: bool
    outcome: str | None


def classify_runtime_screen(
    *,
    capture_succeeded,
    king_count,
    enemy_count,
    retry_detected=False,
    card_selection_detected=False,
    transition_exhausted=False,
):
    """Classify a capture without treating a missing king as immediate defeat."""
    if not capture_succeeded:
        return ScreenAssessment(
            RuntimeScreenState.OBSERVATION_FAILURE,
            "capture_failed",
        )
    if retry_detected:
        return ScreenAssessment(RuntimeScreenState.DEFEAT, "retry_popup")
    if card_selection_detected:
        return ScreenAssessment(RuntimeScreenState.VICTORY, "card_selection")
    if king_count == 1:
        if enemy_count == 0:
            return ScreenAssessment(RuntimeScreenState.VICTORY, "board_cleared")
        return ScreenAssessment(RuntimeScreenState.GAMEPLAY, "board_ready")
    if transition_exhausted:
        return ScreenAssessment(
            RuntimeScreenState.OBSERVATION_FAILURE,
            f"unrecognized_board:king_count={king_count}",
        )
    return ScreenAssessment(
        RuntimeScreenState.TRANSITION,
        f"board_not_ready:king_count={king_count}",
    )


def apply_screen_state_rules(
    reward,
    enemy_count,
    assessment,
    countdown_expired=False,
):
    """Apply terminal rewards only to explicitly classified screen states."""
    if assessment.state is RuntimeScreenState.DEFEAT:
        return TerminalResult(
            reward=-15.0 - float(enemy_count),
            terminated=True,
            outcome="retry_popup_defeat",
        )
    if assessment.state is RuntimeScreenState.VICTORY:
        return TerminalResult(
            reward=reward + 10.0,
            terminated=True,
            outcome=(
                "card_victory"
                if assessment.reason == "card_selection"
                else "board_victory"
            ),
        )
    if assessment.state is RuntimeScreenState.GAMEPLAY and countdown_expired:
        return TerminalResult(
            reward=-15.0,
            terminated=True,
            outcome="countdown_defeat",
        )
    return TerminalResult(reward=reward, terminated=False, outcome=None)


def update_countdown(active, trigger_step, current_step, enemy_count):
    """Return the countdown trigger and whether its defeat limit expired."""
    if not active:
        return trigger_step, False
    if trigger_step is None and 0 < enemy_count <= 6:
        trigger_step = current_step
    expired = trigger_step is not None and current_step - trigger_step >= 12
    return trigger_step, expired


def apply_terminal_rules(
    reward,
    enemy_count,
    king_present,
    popup_detected,
    card_selection_detected,
    countdown_expired,
):
    """Apply the existing terminal rules in their original evaluation order."""
    terminated = False
    outcome = None
    if countdown_expired:
        reward = -15.0
        terminated = True
        outcome = "countdown_defeat"

    if popup_detected:
        reward = -15.0 - float(enemy_count)
        terminated = True
        outcome = "retry_popup_defeat"
    elif enemy_count == 0 and king_present:
        reward += 10.0
        terminated = True
        outcome = "board_victory"
    elif not king_present and card_selection_detected:
        reward += 10.0
        terminated = True
        outcome = "card_victory"
    elif not king_present and enemy_count > 0:
        reward = -15.0 - float(enemy_count)
        terminated = True
        outcome = "missing_king_defeat"

    return TerminalResult(reward=reward, terminated=terminated, outcome=outcome)
