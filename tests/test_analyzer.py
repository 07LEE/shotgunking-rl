"""Small offline regression checks; no capture or game inputs."""

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyzer


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
