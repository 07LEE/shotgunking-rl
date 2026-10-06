"""Deterministic state calculations with no screen capture or game input."""

import numpy as np

from .actions import ACTION_DIRECTIONS, available_moves, select_shot_target


def build_threat_matrix(board):
    """Return the attacked cells for the legacy numeric board format."""
    threat = np.zeros((8, 8), dtype=np.int32)
    straight_directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal_directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    knight_offsets = (
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    )

    for enemy_row in range(8):
        for enemy_col in range(8):
            piece = board[enemy_row, enemy_col]
            if piece < 2:
                continue
            if piece == 2:
                for col_offset in (-1, 1):
                    row, col = enemy_row + 1, enemy_col + col_offset
                    if 0 <= row < 8 and 0 <= col < 8:
                        threat[row, col] = 1
            elif piece == 3:
                for row_offset, col_offset in knight_offsets:
                    row = enemy_row + row_offset
                    col = enemy_col + col_offset
                    if 0 <= row < 8 and 0 <= col < 8:
                        threat[row, col] = 1
            elif piece in (4, 5, 6):
                directions = ()
                if piece in (5, 6):
                    directions += straight_directions
                if piece in (4, 6):
                    directions += diagonal_directions
                for row_offset, col_offset in directions:
                    for distance in range(1, 8):
                        row = enemy_row + row_offset * distance
                        col = enemy_col + col_offset * distance
                        if not (0 <= row < 8 and 0 <= col < 8):
                            break
                        threat[row, col] = 1
                        if board[row, col] != 0:
                            break
    return threat


def build_action_mask(
    observation,
    action_count,
    tracked_king,
    loaded_ammo,
    max_ammo,
    reserve_ammo,
    range_limit,
    royal_guard_active=False,
    move_range_bonus=0,
    melee_damage=0.0,
):
    """Return valid legacy actions from an observation and scalar state."""
    mask = np.zeros(action_count, dtype=np.float32)
    if observation is None:
        return mask

    board = observation[:64].reshape(8, 8)
    threat = observation[64:128].reshape(8, 8)
    king_positions = np.argwhere(board == 1)
    if len(king_positions) != 1:
        return mask
    king_row, king_col = (int(value) for value in king_positions[0])

    valid_moves, safe_moves = available_moves(
        board,
        threat,
        (king_row, king_col),
        move_range_bonus=move_range_bonus,
    )
    permitted_moves = safe_moves if safe_moves else valid_moves
    for action in permitted_moves:
        if action < action_count:
            mask[action] = 1.0

    mask[8] = float(loaded_ammo < max_ammo and reserve_ammo > 0)
    shot_target = select_shot_target(
        board,
        (king_row, king_col),
        royal_guard_active=royal_guard_active,
        range_limit=range_limit,
    )
    melee_available = (
        shot_target is not None
        and shot_target.distance == 1
        and melee_damage > 0
    )
    mask[9] = float(
        shot_target is not None and (loaded_ammo > 0 or melee_available)
    )
    return mask


def build_observation(
    board,
    threat,
    loaded_ammo,
    reserve_ammo,
    damage,
    range_limit,
    spread,
    extra_turn_active,
    hp_matrix,
    turn_matrix,
):
    """Assemble the legacy 281-value observation vector."""
    status = np.zeros(20, dtype=np.float32)
    status[0] = float(extra_turn_active)
    return np.concatenate(
        [
            np.asarray(board, dtype=np.float32).reshape(-1),
            np.asarray(threat, dtype=np.float32).reshape(-1),
            np.array([loaded_ammo, reserve_ammo], dtype=np.float32),
            np.array([damage, range_limit, spread], dtype=np.float32),
            status,
            np.asarray(hp_matrix, dtype=np.float32).reshape(-1),
            np.asarray(turn_matrix, dtype=np.float32).reshape(-1),
        ]
    )


def update_enemy_turns(
    previous_board,
    current_board,
    previous_turns,
    enemy_specs,
    preserve_turns=False,
):
    """Advance estimated enemy turn counters after one observed transition."""
    updated = {}
    for current_row in range(8):
        for current_col in range(8):
            piece = int(current_board[current_row, current_col])
            if piece < 2:
                continue
            previous_positions = [
                (row, col)
                for row in range(8)
                for col in range(8)
                if int(previous_board[row, col]) == piece
            ]
            matched = min(
                previous_positions,
                key=lambda position: abs(current_row - position[0]) + abs(current_col - position[1]),
                default=None,
            )
            if matched != (current_row, current_col) or matched not in previous_turns:
                updated[(current_row, current_col)] = enemy_specs[piece]["turn"]
                continue
            previous_turn = previous_turns[matched]
            if preserve_turns:
                updated[(current_row, current_col)] = previous_turn
            elif previous_turn <= 0.0:
                updated[(current_row, current_col)] = enemy_specs[piece]["turn"]
            else:
                updated[(current_row, current_col)] = max(0.0, previous_turn - 1.0)
    return updated
