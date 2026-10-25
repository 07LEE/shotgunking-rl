"""Shared identities for runtime observations and compatible checkpoints."""

QUEEN = 8
WHITE_KING = 9
SUPPORTED_BOARD_IDS = (0, 1, 2, 3, 4, 5, QUEEN, WHITE_KING)

SIDE_EMPTY = 0
SIDE_ALLY = 1
SIDE_ENEMY = 2
MOAT_ROW = 4
TERRAIN_OFFSET = 281

# Allied non-controlled pieces use negative IDs; the last grid is terrain.
OBSERVATION_SCHEMA = "board-context-345-v1"
OBSERVATION_SIZE = 345
CHECKPOINT_FORMAT = "shotgun-king-dqn-v1"
