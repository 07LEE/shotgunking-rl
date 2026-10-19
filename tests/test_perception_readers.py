"""Offline checks for separate piece, ammunition, and screen readers."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.perception import (
    OverlayScreenReader,
    PredictionStatus,
    RedPixelAmmoReader,
    ScreenKind,
    TemplatePieceReader,
)
from shotgun_king_rl.vision.ammo import extract_ammo_count, read_ammo_count


class PerceptionReaderTests(unittest.TestCase):
    def test_special_knight_is_preserved_but_rejected_for_policy(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        board[7, 4] = 1
        board[0, 0] = 7
        with patch(
            "shotgun_king_rl.perception.frame.analyzer.get_state_matrices",
            return_value=(board, board),
        ) as read_board:
            result = TemplatePieceReader().read(image)
        self.assertEqual(read_board.call_count, 1)
        self.assertEqual(read_board.call_args.kwargs["board_img"].shape, (520, 520, 3))
        self.assertEqual(read_board.call_args.kwargs["board_gray"].shape, (520, 520))
        np.testing.assert_array_equal(read_board.call_args.args[0], image)
        self.assertEqual(result.status, PredictionStatus.REJECTED)
        self.assertEqual(result.reason, "unsupported_runtime_piece:special_knight")
        np.testing.assert_array_equal(result.value, board)

    def test_ammo_reader_counts_calibrated_red_slots(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for index in range(3):
            start = 384 + index * 16
            image[52:72, start:start + 8] = (0, 0, 255)
        for index in range(5):
            start = 384 + index * 16
            image[89:108, start:start + 8] = (0, 0, 255)

        self.assertEqual(read_ammo_count(image), (3, 5))
        result = RedPixelAmmoReader().read(image)
        self.assertEqual(result.status, PredictionStatus.ACCEPTED)
        self.assertEqual((result.value.loaded, result.value.reserve), (3, 5))

    def test_ammo_reader_does_not_count_grenades_after_reserve_shells(self):
        for reserve in (0, 1, 6, 7, 8, 9):
            with self.subTest(reserve=reserve):
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                for index in range(reserve):
                    start = 384 + index * 16
                    image[93:107, start:start + 2] = (0, 0, 255)
                for index in range(reserve, min(reserve + 3, 20)):
                    start = 384 + index * 16
                    image[95:102, start:start + 8] = (0, 0, 255)

                self.assertEqual(read_ammo_count(image), (0, reserve))

    def test_ammo_reader_reports_failure_while_legacy_api_keeps_fallback(self):
        result = RedPixelAmmoReader().read(None)

        self.assertEqual(result.status, PredictionStatus.UNKNOWN)
        self.assertEqual(result.reason, "ammo_not_read")
        self.assertEqual(extract_ammo_count(None), (2, 8))

    def test_piece_reader_rejects_board_without_one_player_king(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        with patch(
            "shotgun_king_rl.perception.frame.analyzer.crop_chessboard",
            return_value=np.zeros((520, 520, 3), dtype=np.uint8),
        ), patch(
            "shotgun_king_rl.perception.frame.analyzer.get_state_matrices",
            return_value=(board, board),
        ):
            result = TemplatePieceReader().read(image)

        self.assertEqual(result.status, PredictionStatus.REJECTED)
        self.assertEqual(result.reason, "player_king_count:0")

    def test_overlay_reader_accepts_explicit_screen_only(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch(
            "shotgun_king_rl.perception.readers.screens.check_retry_popup",
            return_value=False,
        ), patch(
            "shotgun_king_rl.perception.readers.screens.check_card_selection_screen",
            return_value=True,
        ):
            card = OverlayScreenReader().read(image)

        self.assertEqual(card.status, PredictionStatus.ACCEPTED)
        self.assertEqual(card.value, ScreenKind.CARD_SELECTION)

        with patch(
            "shotgun_king_rl.perception.readers.screens.check_retry_popup",
            return_value=False,
        ), patch(
            "shotgun_king_rl.perception.readers.screens.check_card_selection_screen",
            return_value=False,
        ):
            unverified = OverlayScreenReader().read(image)

        self.assertEqual(unverified.status, PredictionStatus.UNKNOWN)
        self.assertEqual(unverified.reason, "gameplay_or_transition_unverified")

    def test_overlay_reader_accepts_gameplay_after_aspect_normalization(self):
        game = np.random.default_rng(7).integers(30, 220, (900, 1600, 3), dtype=np.uint8)
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        image[50:950] = game
        board = np.zeros((8, 8), dtype=int)
        board[7, 4] = 1

        def assert_normalized(screen, **prepared):
            self.assertEqual(screen.shape, (720, 1280, 3))
            return board, board

        with patch(
            "shotgun_king_rl.perception.readers.screens.check_retry_popup",
            return_value=False,
        ), patch(
            "shotgun_king_rl.perception.readers.screens.check_card_selection_screen",
            return_value=False,
        ), patch(
            "shotgun_king_rl.perception.readers.screens.check_gameplay_board",
            return_value=True,
        ), patch(
            "shotgun_king_rl.perception.frame.analyzer.get_state_matrices",
            side_effect=assert_normalized,
        ):
            result = OverlayScreenReader().read(image)

        self.assertEqual(result.status, PredictionStatus.ACCEPTED)
        self.assertEqual(result.value, ScreenKind.GAMEPLAY)


if __name__ == "__main__":
    unittest.main()
