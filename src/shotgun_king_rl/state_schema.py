"""Shared identities for runtime observations and compatible checkpoints."""

QUEEN = 8
WHITE_KING = 9
SUPPORTED_BOARD_IDS = (0, 1, 2, 3, 4, 5, QUEEN, WHITE_KING)

# The vector length is unchanged, but royal IDs and threat semantics differ.
OBSERVATION_SCHEMA = "distinct-royals-281-v1"
OBSERVATION_SIZE = 281
CHECKPOINT_FORMAT = "shotgun-king-dqn-v1"
