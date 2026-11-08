"""Boundary fragments need floor, occupancy, and a recognized neighbor together."""

import unittest

import numpy as np

from shotgun_king_rl.perception import BoardContextReader, BoardContextReading
from shotgun_king_rl.perception.context_overlap import NeighborOverlapReader
from shotgun_king_rl.perception.context_cells import MoatRowReader
from shotgun_king_rl.perception.types import Prediction


class NeighborOverlapTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((520, 520, 3), 100, np.uint8)
        self.sides = np.zeros((8, 8), np.int8)
        self.sides[3, 3] = 1
        self.reference[3*65+10:3*65+49, 3*65+26:3*65+39] = 220
        self.reader = BoardContextReader([(self.reference, BoardContextReading(self.sides, False))])
        self.board = np.zeros((8, 8), np.int8)
        self.board[3, 3] = 1
        self.query = self.reference.copy()
        self.query[3*65-6:3*65+49, 3*65+26:3*65+39] = 220

    def predictions(self, image):
        return [[self.reader.side_reader.read_cell(image[y*65:(y+1)*65, x*65:(x+1)*65],
                 known_identity=1 if self.board[y, x] == 1 else -1) for x in range(8)] for y in range(8)]

    def accepts(self, image, board=None, predictions=None):
        return self.reader.overlap_reader.accepts_empty(image, 2, 3,
            self.board if board is None else board,
            self.predictions(image) if predictions is None else predictions)

    def test_connected_fragment_recovers_complete_context_without_mutating_inputs(self):
        class Frame:
            board_image = self.query
        original = self.query.copy()
        cell = self.reader.read_side_cells(self.query, board=self.board)[2][3]
        self.assertEqual(cell.value, 0)
        self.assertEqual(cell.source, self.reader.overlap_reader.source)
        self.assertFalse(self.reader.read_prepared(Frame()).is_accepted)
        result = self.reader.read_with_board(Frame(), self.board)
        self.assertTrue(result.is_accepted, result.reason)
        np.testing.assert_array_equal(result.value.piece_sides, self.sides)
        np.testing.assert_array_equal(self.query, original)
        self.assertIs(result.value.moat_active, False)

    def test_contaminated_empty_reference_cannot_force_a_negative_ghost(self):
        contaminated = self.reference.copy()
        # This reviewed empty tile contains the head of its own occupied neighbor.
        contaminated[65+46:130, 65+26:65+39] = 220
        contaminated[130:130+49, 65+26:65+39] = 220
        sides = self.sides.copy()
        sides[2, 1] = 2
        reader = NeighborOverlapReader([(self.reference, self.sides), (contaminated, sides)])
        query = self.reference.copy()
        query[3*65-10:3*65+49, 3*65+26:3*65+39] = 220
        patch = query[2*65:3*65, 3*65:4*65]
        nearest = reader._floor(patch, empty=True)
        self.assertFalse(nearest[-1].any())  # The closest patch creates a ghost.
        self.assertTrue(reader.accepts_empty(query, 2, 3, self.board, self.predictions(query)))

    def test_occupancy_unknown_neighbor_and_missing_floor_cannot_supply_empty(self):
        occupied = self.board.copy()
        occupied[2, 3] = 2
        self.assertFalse(self.accepts(self.query, board=occupied))
        predictions = self.predictions(self.query)
        predictions[3][3] = Prediction.unknown(source='test', reason='unmatched_cell')
        self.assertFalse(self.accepts(self.query, predictions=predictions))
        reader = NeighborOverlapReader()
        self.assertFalse(reader.accepts_empty(self.query, 2, 3, self.board, self.predictions(self.query)))

    def test_core_marker_disconnected_fragment_and_colored_arrow_stay_unknown(self):
        core = self.query.copy()
        core[2*65+28:2*65+37, 3*65+28:3*65+37] = 220
        self.assertFalse(self.accepts(core))
        gap = self.query.copy()
        gap[3*65:3*65+5, 3*65+26:3*65+39] = 100
        self.assertFalse(self.accepts(gap))
        arrow = self.query.copy()
        arrow[3*65-6:3*65, 3*65+26:3*65+39] = (30, 30, 220)
        self.assertFalse(self.accepts(arrow))
        isolated = self.query.copy()
        isolated[2*65+3:2*65+7, 3*65+3:3*65+7] = 220
        self.assertFalse(self.accepts(isolated))

    def test_large_occlusion_and_changed_floor_are_rejected(self):
        large = self.query.copy()
        large[3*65-21:3*65, 3*65:4*65] = 220
        self.assertFalse(self.accepts(large))
        foreign = self.query.copy()
        foreign[2*65:3*65, 3*65:4*65] += 40
        self.assertFalse(self.accepts(foreign))

    def test_conflicting_cell_labels_are_never_overridden(self):
        conflict = self.sides.copy()
        conflict[2, 3] = 2
        reader = BoardContextReader([(self.reference, BoardContextReading(self.sides, False)),
                                     (self.reference, BoardContextReading(conflict, False))])
        class Frame:
            board_image = self.reference
        result = reader.read_with_board(Frame(), self.board)
        self.assertFalse(result.is_accepted)
        self.assertTrue(result.reason.startswith('piece_sides:conflicting_cell:'))


class MoatFloorTests(unittest.TestCase):
    def test_changed_occupied_row_uses_two_agreeing_visible_floor_tiles(self):
        image = np.full((520, 520, 3), 100, np.uint8)
        sides = np.zeros((8, 8), np.int8)
        changed = image.copy()
        changed[4*65:5*65, :65] = 220
        empty = np.ones(8, bool)
        empty[0] = False
        for active in (False, True):
            reader = MoatRowReader([(image, active)], [(image, sides, active)])
            self.assertFalse(reader.read_board(changed).is_accepted)
            self.assertIs(reader.read_board(changed, empty_cells=empty).value, active)
            single = np.zeros(8, bool)
            single[1] = True
            self.assertFalse(reader.read_board(changed, empty_cells=single).is_accepted)
        conflict = MoatRowReader([(image, True)], [(image, sides, True), (image, sides, False)])
        self.assertFalse(conflict.read_board(changed, empty_cells=empty).is_accepted)

    def test_other_rows_and_unrecognized_floor_cannot_determine_moat(self):
        image = np.full((520, 520, 3), 100, np.uint8)
        sides = np.zeros((8, 8), np.int8)
        reader = MoatRowReader([(image, False)], [(image, sides, False)])
        image[4*65:5*65] = 180
        self.assertFalse(reader.read_board(image, empty_cells=np.ones(8, bool)).is_accepted)
        self.assertFalse(reader.read_board(image, empty_cells=np.ones(8, int)).is_accepted)
