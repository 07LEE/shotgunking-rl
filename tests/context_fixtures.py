"""Explicit all-enemy, no-moat context for existing synthetic fixtures only."""

import numpy as np

from shotgun_king_rl.perception import BoardContextReading, Prediction


class NormalContextReader:
    def __init__(self, board_source):
        self.board_source = board_source

    def read(self, image):
        board = self.board_source() if callable(self.board_source) else self.board_source
        sides = np.where(board == 0, 0, np.where(board == 1, 1, 2)).astype(np.int8)
        return Prediction.accepted(BoardContextReading(sides, False), source='context:test')
