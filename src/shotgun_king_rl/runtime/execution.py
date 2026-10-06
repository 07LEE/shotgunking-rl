"""Verify requested game actions against the next observed state."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CompletionStatus(str, Enum):
    """Whether an observed state confirms that an input took effect."""

    CONFIRMED = "confirmed"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActionCompletion:
    """Evidence that the game applied one dispatched action."""

    status: CompletionStatus
    reason: str

    @property
    def confirmed(self):
        return self.status is CompletionStatus.CONFIRMED


def verify_action_completion(
    action,
    before_board,
    after_board,
    before_ammo,
    after_ammo,
    destination=None,
    shot_was_melee=False,
):
    """Confirm movement from the king position and weapon actions from ammunition."""
    if not _valid_board(before_board) or not _valid_board(after_board):
        return ActionCompletion(CompletionStatus.UNKNOWN, "invalid_board")

    if action in range(8) or action in range(10, 18):
        if destination is None or not _valid_square(destination):
            return ActionCompletion(CompletionStatus.UNKNOWN, "missing_destination")
        if np.count_nonzero(after_board == 1) == 1 and after_board[destination] == 1:
            return ActionCompletion(CompletionStatus.CONFIRMED, "king_reached_destination")
        if np.array_equal(before_board, after_board):
            return ActionCompletion(CompletionStatus.UNCHANGED, "board_unchanged")
        return ActionCompletion(CompletionStatus.UNKNOWN, "king_destination_unverified")

    if action == 8:
        if not _valid_ammo(before_ammo) or not _valid_ammo(after_ammo):
            return ActionCompletion(CompletionStatus.UNKNOWN, "ammo_unavailable")
        if after_ammo[0] > before_ammo[0] or after_ammo[1] < before_ammo[1]:
            return ActionCompletion(CompletionStatus.CONFIRMED, "reload_visible_in_ammo")
        if after_ammo == before_ammo:
            return ActionCompletion(CompletionStatus.UNCHANGED, "ammo_unchanged")
        return ActionCompletion(CompletionStatus.UNKNOWN, "reload_change_unverified")

    if action == 9:
        if shot_was_melee and destination is not None and _valid_square(destination):
            if np.count_nonzero(after_board == 1) == 1 and after_board[destination] == 1:
                return ActionCompletion(CompletionStatus.CONFIRMED, "melee_move_visible")
        if _valid_ammo(before_ammo) and _valid_ammo(after_ammo):
            if after_ammo[0] < before_ammo[0]:
                return ActionCompletion(CompletionStatus.CONFIRMED, "shot_visible_in_ammo")
            if after_ammo == before_ammo and np.array_equal(before_board, after_board):
                return ActionCompletion(CompletionStatus.UNCHANGED, "state_unchanged")
        return ActionCompletion(CompletionStatus.UNKNOWN, "shot_change_unverified")

    return ActionCompletion(CompletionStatus.UNKNOWN, "unsupported_action")


def _valid_board(board):
    return isinstance(board, np.ndarray) and board.shape == (8, 8)


def _valid_ammo(ammo):
    return isinstance(ammo, tuple) and len(ammo) == 2 and all(type(value) is int and value >= 0 for value in ammo)


def _valid_square(square):
    return isinstance(square, tuple) and len(square) == 2 and all(type(value) is int and 0 <= value < 8 for value in square)
