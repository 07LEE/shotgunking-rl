"""Offline checks for independently replaceable perception readers."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.perception import (
    AmmoReading,
    PerceptionPipeline,
    Prediction,
    ScreenKind,
)


class FakeReader:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return self.prediction


class PerceptionPipelineTests(unittest.TestCase):
    def test_gameplay_state_is_ready_only_when_all_required_readers_pass(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
        ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))

        result = PerceptionPipeline(screen, board, ammo).observe(object())

        self.assertTrue(result.ready_for_policy)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.sources, ("screen:v1", "pieces:v1", "ammo:v1"))

    def test_non_gameplay_screen_skips_board_and_ammo_readers(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.CARD_SELECTION, source="screen:v1"))
        board = FakeReader(Prediction.unknown(source="pieces:v1", reason="not run"))
        ammo = FakeReader(Prediction.unknown(source="ammo:v1", reason="not run"))

        result = PerceptionPipeline(screen, board, ammo).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertEqual(result.screen_kind, ScreenKind.CARD_SELECTION)
        self.assertEqual(result.issues, ("screen:card_selection",))
        self.assertEqual(board.calls, 0)
        self.assertEqual(ammo.calls, 0)

    def test_unknown_ammo_blocks_policy_without_discarding_valid_board(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
        ammo = FakeReader(Prediction.unknown(source="ammo:v1", reason="low_confidence"))

        result = PerceptionPipeline(screen, board, ammo).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertIsNotNone(result.board)
        self.assertIsNone(result.ammo)
        self.assertEqual(result.issues, ("ammo:low_confidence",))

    def test_invalid_board_shape_blocks_policy(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((64,)), source="pieces:v1"))
        ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))

        result = PerceptionPipeline(screen, board, ammo).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertEqual(result.issues, ("board:invalid_shape_or_values",))


if __name__ == "__main__":
    unittest.main()
