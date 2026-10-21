"""Game viewport detection across capture sizes and aspect ratios."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.vision.layout import detect_game_crop, normalize_game_screen
from shotgun_king_rl.vision.screens import _board_cell_means, check_gameplay_board


class ScreenLayoutTests(unittest.TestCase):
    def setUp(self):
        self.game = np.random.default_rng(11).integers(30, 220, (720, 1280, 3), dtype=np.uint8)

    def test_letterboxed_capture_is_cropped_without_distortion(self):
        source = np.empty((720, 1280, 3), dtype=np.uint8)
        source[:, :640] = (40, 90, 180)
        source[:, 640:] = (180, 90, 40)
        game = cv2.resize(source, (1600, 900), interpolation=cv2.INTER_NEAREST)
        capture = np.zeros((1000, 1600, 3), dtype=np.uint8)
        capture[50:950] = game

        self.assertEqual(detect_game_crop(capture), [0, 50, 1600, 950])
        normalized = normalize_game_screen(capture)

        self.assertEqual(normalized.shape, (720, 1280, 3))
        np.testing.assert_array_equal(normalized[360, 320], source[360, 320])
        np.testing.assert_array_equal(normalized[360, 960], source[360, 960])

    def test_pillarboxed_capture_is_cropped_without_distortion(self):
        game = cv2.resize(self.game, (1600, 900), interpolation=cv2.INTER_NEAREST)
        capture = np.zeros((900, 1800, 3), dtype=np.uint8)
        capture[:, 100:1700] = game

        self.assertEqual(detect_game_crop(capture), [100, 0, 1700, 900])
        self.assertEqual(normalize_game_screen(capture).shape, (720, 1280, 3))

    def test_unknown_nonstandard_capture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Game area not detected"):
            normalize_game_screen(np.zeros((800, 1200, 3), dtype=np.uint8))

    def test_gameplay_board_requires_checker_pattern(self):
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        colors = ((170, 180, 220), (90, 100, 150))
        for row in range(8):
            for col in range(8):
                screen[119 + row * 64:119 + (row + 1) * 64, 383 + col * 64:383 + (col + 1) * 64] = colors[(row + col) % 2]

        self.assertTrue(check_gameplay_board(screen))
        self.assertFalse(check_gameplay_board(np.zeros_like(screen)))

    def test_cell_means_preserve_coordinates_channels_and_strided_inputs(self):
        for image in (self.game, self.game.astype(np.float32) / 255):
            board = image[119:631, 383:895]
            self.assertFalse(board.flags.c_contiguous)
            expected = np.asarray([
                [board[r*64:(r+1)*64, c*64:(c+1)*64].mean(axis=(0, 1)) for c in range(8)]
                for r in range(8)
            ])
            np.testing.assert_array_equal(_board_cell_means(board), expected)

    def test_checker_separation_threshold_remains_inclusive(self):
        for separation, accepted in ((79, False), (80, True), (81, True)):
            screen = np.zeros((720, 1280, 3), dtype=np.uint8)
            for row in range(8):
                for col in range(8):
                    screen[119+row*64:119+(row+1)*64, 383+col*64:383+(col+1)*64] = (
                        (50 + separation * ((row+col) % 2), 50, 50)
                    )
            self.assertEqual(check_gameplay_board(screen), accepted)

    def test_checker_variation_threshold_remains_inclusive(self):
        for variation, accepted in ((34, True), (35, True), (36, False)):
            screen = np.zeros((720, 1280, 3), dtype=np.uint8)
            for row in range(8):
                for col in range(8):
                    value = 50 + 100 * ((row+col) % 2) + (variation if row < 4 else -variation)
                    screen[119+row*64:119+(row+1)*64, 383+col*64:383+(col+1)*64] = (value, 50, 50)
            self.assertEqual(check_gameplay_board(screen), accepted)


if __name__ == "__main__":
    unittest.main()
