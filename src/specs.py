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


def get_enemy_specs_matrices(state_matrix, enemy_turns_dict=None):
    """Generates normalized HP and Turn matrices for detected enemy pieces.

    Args:
        state_matrix: 8x8 numpy array representing piece positions.
        enemy_turns_dict: Dictionary mapping (row, col) coordinates to raw turns left.

    Returns:
        A tuple of (hp_matrix, turn_matrix) as 8x8 float32 numpy arrays.
    """
    if np is None:
        return None, None

    hp_matrix = np.zeros((8, 8), dtype=np.float32)
    turn_matrix = np.zeros((8, 8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            val = int(state_matrix[r, c])
            if val in ENEMY_SPECS:
                raw_hp = ENEMY_SPECS[val]["hp"]

                # Retrieve tracked turn if available, else fallback to default specification
                if enemy_turns_dict is not None and (r, c) in enemy_turns_dict:
                    raw_turn = float(enemy_turns_dict[(r, c)])
                else:
                    raw_turn = ENEMY_SPECS[val]["turn"]

                # Scale values down to [0.0, 1.0] range
                hp_matrix[r, c] = min(1.0, raw_hp / MAX_HP)
                turn_matrix[r, c] = min(1.0, raw_turn / MAX_TURN)

    return hp_matrix, turn_matrix
