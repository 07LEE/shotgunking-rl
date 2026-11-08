"""Independent exact-image evidence for cell sides and the fixed moat row."""

import hashlib

import numpy as np

from ..state_schema import MOAT_ROW, SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY
from .types import Prediction


CELL_SIZE = 65
BOARD_SIZE = CELL_SIZE * 8


def _valid_board_image(image):
    return (isinstance(image, np.ndarray) and image.shape == (BOARD_SIZE, BOARD_SIZE, 3)
            and image.dtype == np.uint8)


def _key(image):
    return hashlib.sha256(image.tobytes()).digest()


class CellSideReader:
    """Match complete cell images without assuming sides from positions or colors.

    Matching is deliberately exact. New layouts of known cells can be read,
    but an unseen rendering, overlay or shape remains unknown.
    """

    source = 'piece-sides:reviewed-cells-v1'

    def __init__(self, references=()):
        self._labels = {}
        self.reference_cells = 0
        for image, sides in references:
            if (not _valid_board_image(image) or not isinstance(sides, np.ndarray)
                    or sides.shape != (8, 8) or not np.issubdtype(sides.dtype, np.integer)
                    or not np.all(np.isin(sides, (SIDE_EMPTY, SIDE_ALLY, SIDE_ENEMY)))):
                raise ValueError('Side references require a BGR board and known 8x8 side grid')
            for row in range(8):
                for col in range(8):
                    patch = image[row*CELL_SIZE:(row+1)*CELL_SIZE, col*CELL_SIZE:(col+1)*CELL_SIZE]
                    self._labels.setdefault(_key(patch), set()).add(int(sides[row, col]))
                    self.reference_cells += 1

    def read_board(self, image):
        if not _valid_board_image(image):
            return Prediction.unknown(source=self.source, reason='invalid_board_image')
        if not self._labels:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        sides = np.zeros((8, 8), dtype=np.int8)
        for row in range(8):
            for col in range(8):
                patch = image[row*CELL_SIZE:(row+1)*CELL_SIZE, col*CELL_SIZE:(col+1)*CELL_SIZE]
                labels = self._labels.get(_key(patch))
                if not labels:
                    return Prediction.unknown(source=self.source, reason=f'unmatched_cell:{row},{col}')
                if len(labels) != 1:
                    return Prediction.unknown(source=self.source, reason=f'conflicting_cell:{row},{col}')
                sides[row, col] = next(iter(labels))
        return Prediction.accepted(sides, source=self.source)


class MoatRowReader:
    """Require explicit known moat state for an identical fixed-row image.

    Other rows cannot determine moat state. A piece moving onto the moat row
    can still block recognition until that row rendering has a reference.
    """

    source = 'moat:reviewed-row-v1'

    def __init__(self, references=()):
        self._labels = {}
        self.reference_rows = 0
        for image, active in references:
            if not _valid_board_image(image) or type(active) is not bool:
                raise ValueError('Moat references require a BGR board and explicit boolean state')
            row = image[MOAT_ROW*CELL_SIZE:(MOAT_ROW+1)*CELL_SIZE]
            self._labels.setdefault(_key(row), set()).add(active)
            self.reference_rows += 1

    def read_board(self, image):
        if not _valid_board_image(image):
            return Prediction.unknown(source=self.source, reason='invalid_board_image')
        if not self._labels:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        labels = self._labels.get(_key(image[MOAT_ROW*CELL_SIZE:(MOAT_ROW+1)*CELL_SIZE]))
        if not labels:
            return Prediction.unknown(source=self.source, reason='unmatched_row')
        if len(labels) != 1:
            return Prediction.unknown(source=self.source, reason='conflicting_references')
        return Prediction.accepted(next(iter(labels)), source=self.source)
