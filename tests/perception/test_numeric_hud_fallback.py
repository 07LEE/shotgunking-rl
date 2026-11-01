"""Small rendering shifts can use reviewed shapes, but never another title."""

import unittest

import cv2
import numpy as np

from shotgun_king_rl.perception.numeric import numeric_token
from tests.support.hud import CASES, screen


class NumericHudFallbackTests(unittest.TestCase):
    def cases(self):
        return [case for case in CASES if case.name in ('attack', 'range', 'spread', 'melee_damage')]

    def shifted(self, image):
        return cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 0]]), (1280, 720))

    def test_shifted_rows_need_both_the_title_and_reviewed_value_shape(self):
        for case in self.cases():
            with self.subTest(field=case.name):
                reference = screen(case)
                template, token = case.patch(reference), numeric_token(reference, case.offset)
                ordinary = case.reader([(case.value, template)])
                reader = case.reader([(case.value, template)], numeric_examples=[(case.value, token, template)])
                changed = self.shifted(reference)
                self.assertFalse(ordinary.read(changed).is_accepted)
                self.assertEqual(reader.read(changed).value, case.value)
                self.assertFalse(reader.read(self.shifted(screen(case, seed=11))).is_accepted)
                self.assertFalse(reader.read(self.shifted(screen(case, digits=case.unseen or '8'))).is_accepted)

    def test_conflicting_numeric_labels_stay_unknown_and_owned_copies_are_used(self):
        for case in self.cases():
            with self.subTest(field=case.name):
                reference = screen(case)
                template, token = case.patch(reference), numeric_token(reference, case.offset)
                conflicting = case.reader(numeric_examples=[(case.value, token, template), (case.conflicting, token, template)])
                self.assertFalse(conflicting.read(self.shifted(reference)).is_accepted)
                reader = case.reader(numeric_examples=[(case.value, token, template)])
                token[:] = 0
                template[:] = 0
                self.assertEqual(reader.read(self.shifted(reference)).value, case.value)
