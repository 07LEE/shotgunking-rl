"""Pure action selection calculations for the legacy environment."""

from dataclasses import dataclass

import numpy as np

from .state import ACTION_DIRECTIONS


@dataclass(frozen=True)
class ShotTarget:
    """An enemy selected along one of the eight firing rays."""

    direction: tuple[int, int]
    distance: int
    piece: int
    threatens_king: bool

    @property
    def effective_distance(self):
        diagonal = self.direction[0] != 0 and self.direction[1] != 0
        return self.distance * (1.414 if diagonal else 1.0)


def action_destination(king_position, action):
    """Return the board destination for a one- or two-cell move action."""
    two_cell = action in range(10, 18)
    direction_index = action - 10 if two_cell else action
    multiplier = 2 if two_cell else 1
    row_offset, col_offset = ACTION_DIRECTIONS[direction_index]
    return (
        king_position[0] + row_offset * multiplier,
        king_position[1] + col_offset * multiplier,
    )


def available_moves(board, threat, king_position, move_range_bonus=0):
    """Return collision-free moves and the safe subset in action order."""
    candidates = list(range(8))
    if move_range_bonus > 0:
        candidates.extend(range(10, 18))

    valid = []
    safe = []
    for action in candidates:
        target_row, target_col = action_destination(king_position, action)
        if not (0 <= target_row < 8 and 0 <= target_col < 8):
            continue
        if action in range(10, 18):
            row_offset, col_offset = ACTION_DIRECTIONS[action - 10]
            middle_row = king_position[0] + row_offset
            middle_col = king_position[1] + col_offset
            if board[middle_row, middle_col] != 0:
                continue
        if board[target_row, target_col] != 0:
            continue
        valid.append(action)
        if threat[target_row, target_col] == 0:
            safe.append(action)
    return valid, safe


def detect_promotion(previous_board, current_board):
    """Detect the legacy pawn-to-piece transition on the bottom rank."""
    for col in range(8):
        if current_board[7, col] not in (3, 4, 5, 6):
            continue
        for previous_col in (col - 1, col, col + 1):
            if 0 <= previous_col < 8 and (
                previous_board[6, previous_col] == 2
                or previous_board[7, previous_col] == 2
            ):
                return True
    return False


def select_shot_target(board, king_position, royal_guard_active=False):
    """Select the closest threatening target, then the closest other target."""
    best = None
    knights_present = np.any(board == 3)
    for row_offset, col_offset in ACTION_DIRECTIONS:
        for distance in range(1, 8):
            row = king_position[0] + row_offset * distance
            col = king_position[1] + col_offset * distance
            if not (0 <= row < 8 and 0 <= col < 8):
                break
            piece = int(board[row, col])
            if piece >= 2:
                if piece == 6 and royal_guard_active and knights_present:
                    break
                diagonal = row_offset != 0 and col_offset != 0
                straight = row_offset == 0 or col_offset == 0
                threatens = (
                    (piece == 2 and row_offset == -1 and diagonal and distance == 1)
                    or (piece == 4 and diagonal)
                    or (piece == 5 and straight)
                    or (piece == 6 and (straight or diagonal))
                )
                candidate = ShotTarget(
                    direction=(row_offset, col_offset),
                    distance=distance,
                    piece=piece,
                    threatens_king=threatens,
                )
                if best is None or (
                    candidate.threatens_king and not best.threatens_king
                ) or (
                    candidate.threatens_king == best.threatens_king
                    and candidate.distance < best.distance
                ):
                    best = candidate
                break
            if piece == 1:
                break
    return best
