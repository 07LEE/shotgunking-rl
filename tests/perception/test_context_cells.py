"""Cell evidence may move; unknown sides and moat state must stay independent."""

import unittest

import numpy as np

from shotgun_king_rl.perception import BoardContextReader, BoardContextReading
from shotgun_king_rl.perception.context_cells import CellSideReader, MoatRowReader


class ContextCellTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(31).integers(0, 256, (520, 520, 3), dtype=np.uint8)
        self.sides = np.zeros((8, 8), dtype=np.int8)
        self.sides[7, 3] = 1
        self.sides[0, 2] = 2

    def moved(self, first, second):
        result = self.image.copy()
        def cell(image, position):
            row, col = position
            return image[row*65:(row+1)*65, col*65:(col+1)*65]
        cell(result, first)[:] = cell(self.image, second)
        cell(result, second)[:] = cell(self.image, first)
        return result

    def test_known_allied_and_enemy_cells_move_to_new_positions(self):
        reader = CellSideReader([(self.image, self.sides)])
        for first, second in [((7, 3), (5, 3)), ((0, 2), (2, 6))]:
            with self.subTest(first=first):
                expected = self.sides.copy()
                expected[first], expected[second] = self.sides[second], self.sides[first]
                result = reader.read_board(self.moved(first, second))
                self.assertTrue(result.is_accepted)
                np.testing.assert_array_equal(result.value, expected)

    def test_changed_layout_keeps_known_moat_row(self):
        subject = BoardContextReader([(self.image, BoardContextReading(self.sides, False))])
        class Frame:
            board_image = self.moved((7, 3), (5, 3))
        result = subject.read_prepared(Frame())
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value.piece_sides[5, 3], 1)
        self.assertEqual(result.value.piece_sides[7, 3], 0)
        self.assertIs(result.value.moat_active, False)

    def test_foreign_rendering_and_overlay_never_reuse_prior_sides(self):
        reader = CellSideReader([(self.image, self.sides)])
        self.assertTrue(reader.read_board(self.image).is_accepted)
        changed = self.image.copy()
        changed[0:65, 0:65] = 0
        self.assertEqual(reader.read_board(changed).reason, 'unmatched_cell:0,0')
        self.assertTrue(reader.read_board(self.image).is_accepted)
        self.assertFalse(reader.read_board(None).is_accepted)

    def test_same_cell_with_conflicting_sides_is_unknown(self):
        other = self.sides.copy()
        other[0, 2] = 1
        reader = CellSideReader([(self.image, self.sides), (self.image, other)])
        result = reader.read_board(self.image)
        self.assertFalse(result.is_accepted)
        self.assertEqual(result.reason, 'conflicting_cell:0,2')

    def test_new_fixed_row_cannot_infer_moat_from_other_rows(self):
        reader = MoatRowReader([(self.image, True)])
        self.assertTrue(reader.read_board(self.moved((7, 3), (5, 3))).value)
        changed = self.image.copy()
        changed[4*65:5*65] = self.image[3*65:4*65]
        self.assertEqual(reader.read_board(changed).reason, 'unmatched_row')
        self.assertFalse(MoatRowReader().read_board(self.image).is_accepted)
        conflict = MoatRowReader([(self.image, True), (self.image, False)])
        self.assertEqual(conflict.read_board(self.image).reason, 'conflicting_references')

    def test_cell_evidence_without_moat_label_cannot_supply_complete_context(self):
        subject = BoardContextReader([(self.image, BoardContextReading(self.sides, None))])
        class Frame:
            board_image = self.image
        self.assertTrue(subject.side_reader.read_board(self.image).is_accepted)
        self.assertEqual(subject.read_prepared(Frame()).reason, 'moat:references_unavailable')
        self.assertEqual(subject.metadata()['references'], 0)
        self.assertEqual(subject.metadata()['side_reference_frames'], 1)

    def test_reference_and_return_arrays_are_owned(self):
        image, sides = self.image.copy(), self.sides.copy()
        reader = CellSideReader([(image, sides)])
        image[:] = 0
        sides[:] = 2
        result = reader.read_board(self.image)
        result.value[:] = 2
        np.testing.assert_array_equal(reader.read_board(self.image).value, self.sides)

    def test_invalid_reference_types_are_rejected(self):
        for image, sides in [(self.image[:65], self.sides), (self.image, np.zeros((8, 8), bool)),
                             (self.image, np.full((8, 8), -1)), (self.image, self.sides.astype(float))]:
            with self.assertRaises(ValueError):
                CellSideReader([(image, sides)])
        for active in (None, 0, 1, 'false'):
            with self.assertRaises(ValueError):
                MoatRowReader([(self.image, active)])
