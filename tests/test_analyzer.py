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


class TemplateFeatureCacheTests(unittest.TestCase):
    def tearDown(self):
        vision_analyzer._TEMPLATE_FEATURES.clear()

    def test_features_are_reused_and_replacement_is_recomputed(self):
        template = np.zeros((45, 45), dtype=np.float32)
        template[10:30, 15:25] = 255
        expected = cv2.morphologyEx(template, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        with patch.object(vision_analyzer.cv2, 'morphologyEx', wraps=cv2.morphologyEx) as morphology:
            edges, std, edges_std = vision_analyzer._template_features(template)
            again = vision_analyzer._template_features(template)
            self.assertIs(edges, again[0])
            self.assertEqual(morphology.call_count, 1)
            replacement = template.copy()
            replacement[5:10, 5:10] = 120
            vision_analyzer._template_features(replacement)
            self.assertEqual(morphology.call_count, 2)
        np.testing.assert_array_equal(edges, expected)
        self.assertEqual(std, template.std())
        self.assertEqual(edges_std, expected.std())

    def test_template_reload_discards_previous_features(self):
        template = np.zeros((45, 45), dtype=np.float32)
        vision_analyzer._template_features(template)
        with patch.object(vision_analyzer, '_TEMPLATES_LOADED', False), patch.object(
            vision_analyzer, '_TEMPLATES', {}
        ), patch.object(vision_analyzer.os.path, 'exists', return_value=False):
            vision_analyzer._load_templates()
        self.assertEqual(vision_analyzer._TEMPLATE_FEATURES, {})

    def test_cached_deviations_preserve_shift_scores_and_blank_rejection(self):
        rng = np.random.default_rng(10)
        template = rng.uniform(0, 255, (45, 45)).astype(np.float32)
        for query in (template.copy(), np.zeros_like(template)):
            score = vision_analyzer._shifted_template_score(query, template)
            cached = vision_analyzer._shifted_template_score(
                query, template, query_std=query.std(), template_std=template.std()
            )
            self.assertEqual(score, cached)


class ShiftedTemplateTests(unittest.TestCase):
    def signals(self, displacement):
        template = np.zeros((45, 45), dtype=np.float32)
        template[10:33, 18:20] = 255
        template[10:12, 14:26] = 255
        query = np.zeros_like(template)
        query[:, displacement:] = template[:, :-displacement]
        return query, template

    def test_thin_displaced_piece_matches_before_geometry_rejection(self):
        query, template = self.signals(2)
        self.assertLess(float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0]), .50)
        templates = {key: [] for key in range(1, 8)}
        templates[4] = [template]
        gray = np.full((65, 65), 220, dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with patch.object(vision_analyzer, '_TEMPLATES', templates), patch.object(
            vision_analyzer, '_TEMPLATES_LOADED', True
        ), patch.object(vision_analyzer, '_extract_signal', return_value=query.astype(np.uint8)):
            self.assertEqual(vision_analyzer.classify_patch(image, gray), 4)
            templates[2] = [query]
            # A confident existing match must not be replaced by the retry.
            self.assertEqual(vision_analyzer.classify_patch(image, gray), 2)

    def test_translation_search_is_bounded(self):
        query, template = self.signals(2)
        self.assertAlmostEqual(vision_analyzer._shifted_template_score(query, template), 1.0, places=5)
        query, template = self.signals(8)
        self.assertLess(vision_analyzer._shifted_template_score(query, template), .50)

    def test_featureless_queries_and_templates_cannot_match(self):
        query, template = self.signals(2)
        blank = np.zeros_like(query)
        self.assertEqual(vision_analyzer._shifted_template_score(blank, template), -1)
        self.assertEqual(vision_analyzer._shifted_template_score(query, blank), -1)


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


class PlayerKingGuardTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((65, 65), 220, dtype=np.uint8)
        self.gray[20:45, 20:40] = 40
        self.image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

    def test_dark_contour_cannot_override_mismatched_king_template(self):
        template = np.zeros((45, 45), dtype=np.float32)
        template[:5, :5] = 255
        templates = {key: [] for key in range(1, 7)}
        templates[1] = [template]
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ):
            self.assertNotEqual(vision_analyzer.classify_patch(self.image, self.gray), 1)

    def test_red_overlay_cannot_be_accepted_as_player_king(self):
        self.image[25:35, 25:35] = (0, 0, 255)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        templates = {key: [] for key in range(1, 7)}
        templates[1] = [vision_analyzer._extract_signal(gray).astype(np.float32)]
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ):
            self.assertNotEqual(vision_analyzer.classify_patch(self.image, gray), 1)

    def test_geometry_fallback_remains_available_without_king_templates(self):
        templates = {key: [] for key in range(1, 7)}
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ):
            self.assertEqual(vision_analyzer.classify_patch(self.image, self.gray), 1)


class SpecialKnightTests(unittest.TestCase):
    def tearDown(self):
        vision_analyzer.reset_analyzer_cache()

    def test_special_template_is_opt_in(self):
        gray = np.full((65, 65), 220, dtype=np.uint8)
        gray[20:45, 30:33] = 150
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        templates = {key: [] for key in range(1, 8)}
        templates[7] = [vision_analyzer._extract_signal(gray).astype(np.float32)]
        with patch.object(vision_analyzer, "_TEMPLATES", templates), patch.object(
            vision_analyzer, "_TEMPLATES_LOADED", True
        ):
            self.assertEqual(vision_analyzer.classify_patch(image, gray), 0)
            self.assertEqual(vision_analyzer.classify_patch(
                image, gray, include_special_knights=True
            ), 7)
            templates[3] = templates[7]
            self.assertEqual(vision_analyzer.classify_patch(
                image, gray, include_special_knights=True
            ), 3)

    def test_cached_board_is_not_shared_between_legacy_and_extended_modes(self):
        image = np.full((720, 1280, 3), 220, dtype=np.uint8)
        vision_analyzer.reset_analyzer_cache()
        def classify(*args, include_special_knights=False):
            return 7 if include_special_knights else 3
        with patch.object(vision_analyzer, "classify_patch", side_effect=classify):
            for extended in (False, True, False):
                board = vision_analyzer.get_state_matrix(
                    image, include_special_knights=extended
                )
                np.testing.assert_array_equal(board, np.full((8, 8), 7 if extended else 3))

    def test_dual_results_preserve_special_selection_before_legacy_alignment(self):
        gray = np.full((65, 65), 220, dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        signal = np.zeros((45, 45), dtype=np.uint8)
        signal[10:33, 18:20] = 255
        signal[10:12, 14:26] = 255
        shifted = np.zeros_like(signal)
        shifted[:, 2:] = signal[:, :-2]
        templates = {key: [] for key in range(1, 8)}
        templates[4] = [signal.astype(np.float32)]
        templates[7] = [shifted.astype(np.float32)]
        with patch.object(vision_analyzer, '_TEMPLATES', templates), patch.object(
            vision_analyzer, '_TEMPLATES_LOADED', True
        ), patch.object(vision_analyzer, '_extract_signal', return_value=shifted) as extract:
            self.assertEqual(vision_analyzer.classify_patch_modes(image, gray), (4, 7))
            self.assertEqual(extract.call_count, 1)

    def test_dual_results_keep_legacy_fallback_without_legacy_templates(self):
        gray = np.full((65, 65), 220, dtype=np.uint8)
        gray[20:45, 30:33] = 150
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        templates = {key: [] for key in range(1, 8)}
        templates[7] = [vision_analyzer._extract_signal(gray).astype(np.float32)]
        with patch.object(vision_analyzer, '_TEMPLATES', templates), patch.object(
            vision_analyzer, '_TEMPLATES_LOADED', True
        ):
            self.assertEqual(vision_analyzer.classify_patch_modes(image, gray), (0, 7))
            templates[3] = templates[7]
            self.assertEqual(vision_analyzer.classify_patch_modes(image, gray), (3, 3))

    def test_dual_board_classifies_each_cell_once_without_temporal_cache(self):
        image = np.full((720, 1280, 3), 220, dtype=np.uint8)
        with patch.object(vision_analyzer, 'classify_patch_modes', return_value=(3, 7)) as classify:
            legacy, extended = vision_analyzer.get_state_matrices(image)
            self.assertEqual(classify.call_count, 64)
            np.testing.assert_array_equal(legacy, np.full((8, 8), 3))
            np.testing.assert_array_equal(extended, np.full((8, 8), 7))
            legacy[:] = 0
            np.testing.assert_array_equal(extended, np.full((8, 8), 7))
            vision_analyzer.get_state_matrices(image)
            self.assertEqual(classify.call_count, 128)


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


class EmptyTileTests(unittest.TestCase):
    def test_uniform_tiles_do_not_become_pieces(self):
        king = np.zeros((45, 45), dtype=np.float32)
        king[12:35, 18:26] = 255
        with patch.object(vision_analyzer, '_TEMPLATES', {1: [king]}), patch.object(
            vision_analyzer, '_TEMPLATES_LOADED', True
        ):
            for brightness in (100, 160, 220):
                with self.subTest(brightness=brightness):
                    gray = np.full((65, 65), brightness, dtype=np.uint8)
                    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    self.assertEqual(analyzer.classify_patch(image, gray), 0)


if __name__ == "__main__":
    unittest.main()
