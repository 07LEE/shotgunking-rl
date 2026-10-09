"""Offline checks for independently replaceable perception readers."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.perception import (
    AmmoReading,
    CombatStatsReading,
    PerceptionPipeline,
    Prediction,
    ScreenKind,
)


VALID_COMBAT_STATS = CombatStatsReading(
    attack=4,
    range_min=3,
    range_max=5,
    spread_degrees=65,
    penetration_percent=0,
    knockback_percent=0,
    grenades_remaining=0,
    grenade_damage=0,
    melee_damage=0,
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
        combat = FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source="combat:v1"))

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertTrue(result.ready_for_policy)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.combat_stats, VALID_COMBAT_STATS)
        self.assertEqual(result.sources, ("screen:v1", "pieces:v1", "ammo:v1", "combat:v1"))

    def test_non_gameplay_screen_skips_board_and_ammo_readers(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.CARD_SELECTION, source="screen:v1"))
        board = FakeReader(Prediction.unknown(source="pieces:v1", reason="not run"))
        ammo = FakeReader(Prediction.unknown(source="ammo:v1", reason="not run"))
        combat = FakeReader(Prediction.unknown(source="combat:v1", reason="not run"))

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertEqual(result.screen_kind, ScreenKind.CARD_SELECTION)
        self.assertEqual(result.issues, ("screen:card_selection",))
        self.assertEqual(board.calls, 0)
        self.assertEqual(ammo.calls, 0)
        self.assertEqual(combat.calls, 0)

    def test_unknown_ammo_blocks_policy_without_discarding_valid_board(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
        ammo = FakeReader(Prediction.unknown(source="ammo:v1", reason="low_confidence"))
        combat = FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source="combat:v1"))

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertIsNotNone(result.board)
        self.assertIsNone(result.ammo)
        self.assertEqual(result.issues, ("ammo:low_confidence",))

    def test_invalid_board_shape_blocks_policy(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((64,)), source="pieces:v1"))
        ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))
        combat = FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source="combat:v1"))

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertEqual(result.issues, ("board:invalid_shape_or_values",))

    def test_unknown_combat_stats_block_policy(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
        ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))
        combat = FakeReader(Prediction.unknown(source="combat:v1", reason="unreadable"))

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertEqual(result.issues, ("combat_stats:unreadable",))

    def test_invalid_combat_stats_block_policy(self):
        screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
        board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
        ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))
        combat = FakeReader(
            Prediction.accepted(
                CombatStatsReading(
                    attack=4,
                    range_min=6,
                    range_max=5,
                    spread_degrees=65,
                    penetration_percent=0,
                    knockback_percent=0,
                    grenades_remaining=0,
                    grenade_damage=0,
                    melee_damage=0,
                ),
                source="combat:v1",
            )
        )

        result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertEqual(result.issues, ("combat_stats:invalid_values",))

    def test_combat_stats_reject_invalid_numeric_ranges(self):
        invalid_stats = (
            replace(VALID_COMBAT_STATS, attack=-1),
            replace(VALID_COMBAT_STATS, spread_degrees=361),
            replace(VALID_COMBAT_STATS, penetration_percent=101),
            replace(VALID_COMBAT_STATS, knockback_percent=101),
            replace(VALID_COMBAT_STATS, grenades_remaining=-1),
            replace(VALID_COMBAT_STATS, grenades_remaining=0.5),
            replace(VALID_COMBAT_STATS, grenade_damage=-1),
            replace(VALID_COMBAT_STATS, melee_damage=-1),
            replace(VALID_COMBAT_STATS, attack=float("nan")),
            replace(VALID_COMBAT_STATS, attack=True),
        )

        for stats in invalid_stats:
            with self.subTest(stats=stats):
                screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:v1"))
                board = FakeReader(Prediction.accepted(np.zeros((8, 8)), source="pieces:v1"))
                ammo = FakeReader(Prediction.accepted(AmmoReading(loaded=1, reserve=5), source="ammo:v1"))
                combat = FakeReader(Prediction.accepted(stats, source="combat:v1"))

                result = PerceptionPipeline(screen, board, ammo, combat).observe(object())

                self.assertFalse(result.ready_for_policy)
                self.assertEqual(result.issues, ("combat_stats:invalid_values",))


if __name__ == "__main__":
    unittest.main()
