"""Enemy piece specifications and normalization utilities.

This module stores default attributes (health, turn interval) of enemy pieces
and generates normalized observation matrices for reinforcement learning.
"""

try:
    import numpy as np
except ImportError:
    np = None

# Static specifications mapping for enemy pieces: {piece_id: {"hp": default_hp, "turn": default_turn}}
# 2: Pawn, 3: Knight, 4: Bishop, 5: Rook, 6: Queen/King
ENEMY_SPECS = {
    2: {"hp": 5.0, "turn": 5.0},
    3: {"hp": 3.0, "turn": 3.0},
    4: {"hp": 3.0, "turn": 3.0},
    5: {"hp": 4.0, "turn": 4.0},
    6: {"hp": 4.0, "turn": 4.0},
}

# Max values for normalization to prevent scale instability in neural network inputs
MAX_HP = 10.0
MAX_TURN = 5.0


def get_enemy_specs_matrices(state_matrix, enemy_turns_dict=None, rank=1):
    """Generates normalized HP and Turn matrices for detected enemy pieces.

    Args:
        state_matrix: 8x8 numpy array representing piece positions.
        enemy_turns_dict: Dictionary mapping (row, col) coordinates to raw turns left.
        rank: Target story mode difficulty level.

    Returns:
        A tuple of (hp_matrix, turn_matrix) as 8x8 float32 numpy arrays.
    """
    if np is None:
        return None, None

    # Calculate dynamic specs based on cumulative rank penalties
    hp_pawn = 5.0 + (1.0 if rank >= 20 else 0.0)
    hp_knight = 3.0 + (1.0 if rank >= 16 else 0.0)
    hp_bishop = 3.0 + (1.0 if rank >= 15 else 0.0)

    hp_rook = 4.0
    if rank >= 9: hp_rook += 1.0
    if rank >= 13: hp_rook += 1.0
    if rank >= 19: hp_rook += 1.0

    hp_boss = 4.0
    if rank >= 7: hp_boss += 1.0
    if rank >= 14: hp_boss += 1.0
    if rank >= 17: hp_boss += 1.0  # Queen spec
    if rank >= 18: hp_boss += 1.0

    hp_specs = {
        2: hp_pawn,
        3: hp_knight,
        4: hp_bishop,
        5: hp_rook,
        6: hp_boss,
    }

    hp_matrix = np.zeros((8, 8), dtype=np.float32)
    turn_matrix = np.zeros((8, 8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            val = int(state_matrix[r, c])
            if val in ENEMY_SPECS:
                raw_hp = hp_specs.get(val, ENEMY_SPECS[val]["hp"])

                # Retrieve tracked turn if available, else fallback to default specification
                if enemy_turns_dict is not None and (r, c) in enemy_turns_dict:
                    raw_turn = float(enemy_turns_dict[(r, c)])
                else:
                    raw_turn = ENEMY_SPECS[val]["turn"]

                # Scale values down to [0.0, 1.0] range
                hp_matrix[r, c] = min(1.0, raw_hp / MAX_HP)
                turn_matrix[r, c] = min(1.0, raw_turn / MAX_TURN)

    return hp_matrix, turn_matrix
