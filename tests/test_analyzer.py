"""Small offline regression checks; no capture or game inputs."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyzer
from shotgun_king_rl.vision import analyzer as vision_analyzer


class ThinOutlineTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((65, 65), 220, dtype=np.uint8)
        # A narrow connected outline has little contour area but a clear signal.
        self.gray[20:45, 30:33] = 150
        self.image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

    def test_matching_thin_piece_is_not_discarded_as_noise(self):
        signal = vision_analyzer._extract_signal(self.gray).astype(np.float32)
        for piece_id in (3, 5):
            templates = {key: [] for key in range(1, 7)}
            templates[piece_id] = [signal]
            with self.subTest(piece_id=piece_id), patch.object(
                vision_analyzer, "_TEMPLATES", templates
            ), patch.object(vision_analyzer, "_TEMPLATES_LOADED", True):
                self.assertEqual(
                    vision_analyzer.classify_patch(self.image, self.gray), piece_id
                )

    def test_unmatched_small_signal_is_still_rejected(self):
        templates = {key: [] for key in range(1, 7)}
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ):
            self.assertEqual(vision_analyzer.classify_patch(self.image, self.gray), 0)


class TemplateContrastTests(unittest.TestCase):
    def test_rook_outline_matches_with_reversed_signal_contrast(self):
        template = np.zeros((45, 45), dtype=np.float32)
        template[8:35, 15:30] = 255
        template[5:10, 10:35] = 255
        templates = {key: [] for key in range(1, 7)}
        templates[5] = [template]
        gray = np.full((65, 65), 220, dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ), patch.object(
            vision_analyzer, "_extract_signal", return_value=(255 - template).astype(np.uint8)
        ):
            self.assertEqual(vision_analyzer.classify_patch(image, gray), 5)


@unittest.skipUnless((ROOT / "data/test_fixtures/board_01.json").exists() and (ROOT / "data/test_fixtures/board_01.png").exists() and (ROOT / "assets/pieces").is_dir(), "Private fixtures and piece templates are required")
class BoardRegressionTests(unittest.TestCase):
    def setUp(self):
        analyzer.reset_analyzer_cache()

    def test_saved_board_and_cached_repeat(self):
        fixture = ROOT / "data/test_fixtures/board_01.json"
        annotation = json.loads(fixture.read_text())
        image = cv2.imread(str(fixture.parent / annotation["image"]))
        self.assertIsNotNone(image)
        expected = np.asarray(annotation["board"])
        np.testing.assert_array_equal(analyzer.get_state_matrix(image), expected)
        np.testing.assert_array_equal(analyzer.get_state_matrix(image), expected)

    def test_uniform_tiles_do_not_become_pieces(self):
        for brightness in (100, 160, 220):
            with self.subTest(brightness=brightness):
                gray = np.full((65, 65), brightness, dtype=np.uint8)
                patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.assertEqual(analyzer.classify_patch(patch, gray), 0)


if __name__ == "__main__":
    unittest.main()
