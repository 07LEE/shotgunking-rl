"""Numeric proposals preserve ambiguity and require matching row context."""

import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.numeric import match_numeric, numeric_token, match_floor_shapes


class NumericProposalTests(unittest.TestCase):
    def test_floor_stroke_matching_rejects_conflicting_labels_and_blanks(self):
        template = np.zeros((36, 24), np.uint8)
        cv2.putText(template, '2', (2, 29), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        shifted = np.zeros_like(template)
        shifted[:, 1:] = template[:, :-1]
        self.assertEqual(match_floor_shapes(shifted, [(2, template)])[0], 2)
        self.assertIsNone(match_floor_shapes(shifted, [(2, template), (3, template)]))
        self.assertIsNone(match_floor_shapes(np.zeros_like(template), [(2, template)]))
        self.assertIsNone(match_floor_shapes(shifted, []))

    def image(self, text, shift=0):
        image = np.zeros((720, 1280, 3), np.uint8)
        cv2.putText(image, text, (300 + shift, 240), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
        return image

    def test_numeric_normalization_follows_horizontal_shift(self):
        first = numeric_token(self.image('4'))
        second = numeric_token(self.image('4', 4))
        self.assertIsNotNone(first)
        np.testing.assert_array_equal(first, second)

    def test_conflicting_labels_and_wrong_context_remain_unknown(self):
        token = numeric_token(self.image('4'))
        row = np.random.default_rng(12).integers(0, 255, (65, 84), dtype=np.uint8)
        self.assertEqual(match_numeric(token, row, [(4, token, row)])[0], 4)
        self.assertIsNone(match_numeric(token, row, [(4, token, row), (6, token, row)]))
        wrong = np.random.default_rng(13).integers(0, 255, row.shape, dtype=np.uint8)
        self.assertIsNone(match_numeric(token, wrong, [(4, token, row)]))
        self.assertIsNone(match_numeric(None, row, [(4, token, row)]))

    def test_different_number_and_blank_do_not_match(self):
        row = np.random.default_rng(12).integers(0, 255, (65, 84), dtype=np.uint8)
        self.assertIsNone(match_numeric(numeric_token(self.image('8')), row, [(4, numeric_token(self.image('4')), row)]))
        self.assertIsNone(numeric_token(np.zeros((720, 1280, 3), np.uint8)))
