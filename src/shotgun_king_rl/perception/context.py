"""Validate observed side/terrain context without inferring missing values."""

import numpy as np

from ..state_schema import MOAT_ROW, SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY
from .types import BoardContextReading


def valid_board_context(context, board):
    if not isinstance(context, BoardContextReading) or type(context.moat_active) is not bool:
        return False
    sides = context.piece_sides
    if (not isinstance(sides, np.ndarray) or sides.shape != (8, 8)
            or not np.issubdtype(sides.dtype, np.integer) or sides.dtype == np.bool_
            or not np.all(np.isin(sides, (SIDE_EMPTY, SIDE_ALLY, SIDE_ENEMY)))):
        return False
    return bool(np.all(sides[board == 0] == SIDE_EMPTY)
                and np.all(sides[board != 0] != SIDE_EMPTY)
                and np.all(sides[board == 1] == SIDE_ALLY))


def encode_sided_board(board, context):
    """Keep enemies positive and distinguish allied pieces from controlled king."""
    result = np.asarray(board).copy()
    allied = (context.piece_sides == SIDE_ALLY) & (board != 1)
    result[allied] *= -1
    return result


def terrain_grid(context):
    """Derive the fixed moat row only from a known card state."""
    if type(context.moat_active) is not bool:
        raise ValueError('Moat state must be confirmed')
    terrain = np.zeros((8, 8), dtype=np.float32)
    if context.moat_active:
        terrain[MOAT_ROW] = 1
    return terrain
