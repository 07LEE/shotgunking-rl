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

    def test_spread_keeps_dim_strokes_without_changing_other_fields(self):
        case = next(case for case in CASES if case.name == 'spread')
        reference = screen(case)
        dimmed = reference.copy()
        region = dimmed[350:420]
        region[region > 140] = 130
        self.assertIsNone(numeric_token(dimmed, case.offset))
        token = numeric_token(reference, case.offset, threshold=120)
        reader = case.reader(numeric_examples=[(case.value, token, case.patch(reference))])
        self.assertEqual(reader.read(dimmed).value, case.value)
        self.assertFalse(reader.read(screen(case, digits='180')).is_accepted)

    def test_dim_attack_overlap_requires_title_and_separates_three_from_eight(self):
        case = next(case for case in CASES if case.name == 'attack')
        references = [screen(case, digits=str(value)) for value in (3, 8)]
        for image in references:
            region = image[250:280, 270:310]
            region[region > 140] = 120
        examples = [(value, numeric_token(image, threshold=110), case.patch(image))
                    for value, image in zip((3, 8), references)]
        reader = case.reader(dim_numeric_examples=examples)
        for value in (3, 8):
            query = screen(case, digits=str(value))
            # Preserve title evidence but lower only numeric stroke brightness.
            region = query[250:280, 270:310]
            region[region > 140] = 120
            self.assertGreater(numeric_token(query, threshold=110).sum(), numeric_token(query).sum())
            self.assertEqual(reader.read(query).value, value)
        self.assertFalse(reader.read(screen(case, seed=11, digits='3')).is_accepted)
        self.assertFalse(reader.read(screen(case, digits='9')).is_accepted)
        conflict = case.reader(dim_numeric_examples=[examples[0], (8, examples[0][1], examples[0][2])])
        self.assertFalse(conflict.read(references[0]).is_accepted)
        examples[0][1][:] = 0
        examples[0][2][:] = 0
        self.assertEqual(reader.read(references[0]).value, 3)

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
