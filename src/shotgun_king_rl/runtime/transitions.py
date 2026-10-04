"""Pure episode transition decisions for the legacy environment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalResult:
    """Reward and terminal state after applying legacy end conditions."""

    reward: float
    terminated: bool
    outcome: str | None


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
