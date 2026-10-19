"""Frame-local preprocessing reuse and independent observation checks."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shotgun_king_rl.perception.frame import PreparedFrame, read_frame
from shotgun_king_rl.perception import OverlayScreenReader, PerceptionPipeline, Prediction, RedPixelAmmoReader, TemplatePieceReader
from shotgun_king_rl.vision import analyzer
from shotgun_king_rl.vision.layout import normalize_game_screen


class PreparedFrameTests(unittest.TestCase):
    def tearDown(self):
        analyzer.reset_analyzer_cache()

    def test_readers_share_normalization_crop_and_grayscale(self):
        image = np.full((720, 1280, 3), 100, dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        board[7, 4] = 1
        with patch("shotgun_king_rl.perception.frame.normalize_game_screen", wraps=normalize_game_screen) as normalize, patch.object(
            analyzer, "crop_chessboard", wraps=analyzer.crop_chessboard
        ) as crop, patch.object(cv2, "cvtColor", wraps=cv2.cvtColor) as gray, patch.object(
            analyzer, "get_state_matrices", return_value=(board, board)
        ):
            frame = PreparedFrame(image)
            for reader in (OverlayScreenReader(), TemplatePieceReader(), RedPixelAmmoReader()):
                read_frame(reader, frame)
            self.assertEqual(normalize.call_count, 1)
            self.assertEqual(crop.call_count, 1)
            grayscale_calls = [call for call in gray.call_args_list if call.args[1] == cv2.COLOR_BGR2GRAY]
            self.assertEqual(len(grayscale_calls), 2)  # Full frame and board, once each.

    def test_new_observation_never_reuses_previous_frame_pixels(self):
        image = np.full((720, 1280, 3), 100, dtype=np.uint8)
        first = PreparedFrame(image)
        old = first.board_gray.copy()
        image[:] = 200
        second = PreparedFrame(image)
        self.assertTrue(np.all(old == 100))
        self.assertTrue(np.all(first.board_gray == 100))
        self.assertTrue(np.all(second.board_gray == 200))

    def test_pipeline_shares_only_within_each_observation(self):
        image = np.full((720, 1280, 3), 100, dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        board[7, 4] = 1
        class Combat:
            def read(self, image):
                return Prediction.unknown(source="test", reason="unavailable")
        pipeline = PerceptionPipeline(OverlayScreenReader(), TemplatePieceReader(), RedPixelAmmoReader(), Combat())
        with patch("shotgun_king_rl.perception.frame.normalize_game_screen", wraps=normalize_game_screen) as normalize, patch.object(
            analyzer, "crop_chessboard", wraps=analyzer.crop_chessboard
        ) as crop, patch.object(analyzer, "get_state_matrices", return_value=(board, board)), patch(
            "shotgun_king_rl.perception.readers.screens.check_gameplay_board", return_value=True
        ):
            for _ in range(2):
                result = pipeline.observe(image)
                self.assertIsNotNone(result.board)
                self.assertFalse(result.ready_for_policy)
            self.assertEqual(normalize.call_count, 2)
            self.assertEqual(crop.call_count, 2)

    def test_unprepared_reader_receives_original_input_without_preprocessing(self):
        original = object()
        class Reader:
            def read(self, image):
                return image
        with patch("shotgun_king_rl.perception.frame.normalize_game_screen") as normalize:
            self.assertIs(read_frame(Reader(), PreparedFrame(original)), original)
            normalize.assert_not_called()

    def test_invalid_frame_keeps_reader_failure_reasons(self):
        frame = PreparedFrame(None)
        self.assertEqual(read_frame(OverlayScreenReader(), frame).reason, "invalid_image")
        self.assertEqual(read_frame(TemplatePieceReader(), frame).reason, "invalid_image")
        self.assertEqual(read_frame(RedPixelAmmoReader(), frame).reason, "ammo_not_read")

    def test_board_readers_share_one_scan_and_keep_modes_separate(self):
        image = np.full((720, 1280, 3), 100, dtype=np.uint8)
        with patch.object(analyzer, 'classify_patch_modes', return_value=(1, 7)) as classify, patch(
            'shotgun_king_rl.perception.readers.screens.check_gameplay_board', return_value=True
        ):
            frame = PreparedFrame(image)
            screen = read_frame(OverlayScreenReader(), frame)
            pieces = read_frame(TemplatePieceReader(), frame)
            self.assertEqual(classify.call_count, 64)
            self.assertTrue(screen.is_accepted)
            self.assertEqual(pieces.reason, 'player_king_count:0')
            np.testing.assert_array_equal(pieces.value, np.full((8, 8), 7))
            pieces.value[:] = 0
            np.testing.assert_array_equal(frame.boards[1], np.full((8, 8), 7))

    def test_explicit_overlay_does_not_start_board_classification(self):
        image = np.full((720, 1280, 3), 100, dtype=np.uint8)
        with patch.object(analyzer, 'get_state_matrices') as classify, patch(
            'shotgun_king_rl.perception.readers.screens.check_retry_popup', return_value=True
        ):
            result = read_frame(OverlayScreenReader(), PreparedFrame(image))
            self.assertTrue(result.is_accepted)
            classify.assert_not_called()
