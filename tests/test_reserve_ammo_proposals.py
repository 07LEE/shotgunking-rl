"""Reserve proposals verify shell shapes independently of other HUD icons."""

import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.reserve_ammo import match_reserve, reserve_shell_count


class ReserveAmmoProposalTests(unittest.TestCase):
    def image(self, count):
        image = np.zeros((720, 1280, 3), np.uint8)
        image[40:48, 430:470] = 220
        for index in range(20):
            start = 384 + index * 16
            image[89:108, start:start + 8] = (30, 25, 20)
        for index in range(count):
            start = 384 + index * 16
            image[89:108, start:start + 8] = (0, 0, 255)
        return image

    def examples(self, count):
        return [(count, cv2.cvtColor(self.image(count)[78:111, 382:706], cv2.COLOR_BGR2GRAY))]

    def test_individual_shell_reference_supports_new_counts(self):
        for count in (1, 2, 3, 7, 8, 9, 20):
            with self.subTest(count=count):
                self.assertEqual(match_reserve(self.image(count), self.examples(1))[0], count)

    def test_zero_requires_reviewed_empty_slot(self):
        self.assertEqual(match_reserve(self.image(0), self.examples(0))[0], 0)
        self.assertIsNone(match_reserve(self.image(0), self.examples(1)))

    def test_blank_and_missing_gun_remain_unknown(self):
        self.assertIsNone(match_reserve(np.zeros((720, 1280, 3), np.uint8), self.examples(0)))
        image = self.image(2)
        image[34:75] = 0
        self.assertIsNone(match_reserve(image, self.examples(2)))

    def test_short_neighboring_grenade_marks_are_not_shells(self):
        image = self.image(2)
        image[95:102, 416:456] = (0, 0, 255)
        self.assertEqual(reserve_shell_count(image), 2)
        self.assertEqual(match_reserve(image, self.examples(1))[0], 2)

    def test_missing_templates_remain_unknown(self):
        self.assertIsNone(match_reserve(self.image(2), []))
