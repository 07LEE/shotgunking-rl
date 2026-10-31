"""Loaded ammo requires shell structure and reviewed HUD evidence."""

import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.loaded_ammo import loaded_shell_count, match_loaded


class LoadedAmmoProposalTests(unittest.TestCase):
    def image(self, count):
        image = np.zeros((720, 1280, 3), np.uint8)
        for index in range(2):
            image[52:72, 384 + index * 16:392 + index * 16] = (30, 25, 20)
        for index in range(count):
            image[52:72, 384 + index * 16:392 + index * 16] = (0, 0, 255)
        image[40:48, 430:470] = 220
        return image

    def examples(self, count):
        return [(count, cv2.cvtColor(self.image(count)[34:75, 382:550], cv2.COLOR_BGR2GRAY))]

    def test_zero_one_two_shells_require_corresponding_reference(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                self.assertEqual(match_loaded(self.image(count), self.examples(count))[0], count)
                self.assertIsNone(match_loaded(self.image(count), self.examples((count + 1) % 3)))

    def test_blank_or_missing_gun_is_unknown(self):
        self.assertIsNone(match_loaded(np.zeros((720, 1280, 3), np.uint8), self.examples(0)))
        image = self.image(2)
        image[:, 416:] = 0
        self.assertIsNone(match_loaded(image, self.examples(2)))

    def test_reserve_and_grenades_do_not_change_loaded_count(self):
        image = self.image(1)
        image[89:108, 384:704] = (0, 0, 255)
        self.assertEqual(loaded_shell_count(image), 1)
        self.assertEqual(match_loaded(image, self.examples(1))[0], 1)

    def test_rotated_gun_does_not_change_shell_match(self):
        image = self.image(2)
        image[:, 416:] = 0
        cv2.line(image, (420, 40), (470, 70), (220, 220, 220), 5)
        self.assertEqual(match_loaded(image, self.examples(2))[0], 2)

    def test_empty_slot_bodies_ignore_moving_upper_gun_edge(self):
        image = self.image(0)
        image[34:40, 382:416] = 220
        self.assertEqual(match_loaded(image, self.examples(0))[0], 0)
        self.assertIsNone(match_loaded(image, self.examples(1)))

    def test_absent_slot_bodies_do_not_imply_zero(self):
        image = self.image(0)
        image[30:79, 378:420] = 0
        self.assertIsNone(match_loaded(image, self.examples(0)))
