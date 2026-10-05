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

    def test_ammo_reader_reports_failure_while_legacy_api_keeps_fallback(self):
        result = RedPixelAmmoReader().read(None)

        self.assertEqual(result.status, PredictionStatus.UNKNOWN)
        self.assertEqual(result.reason, "ammo_not_read")
        self.assertEqual(extract_ammo_count(None), (2, 8))

    def test_piece_reader_rejects_board_without_one_player_king(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        with patch(
            "shotgun_king_rl.perception.readers.analyzer.crop_chessboard",
            return_value=np.zeros((520, 520, 3), dtype=np.uint8),
        ), patch(
            "shotgun_king_rl.perception.readers.analyzer.get_state_matrix",
            return_value=board,
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


if __name__ == "__main__":
    unittest.main()
