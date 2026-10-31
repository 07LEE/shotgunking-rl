"""Offline checks for the limited live-assistance scope."""

import unittest
from unittest.mock import patch

import train
from shotgun_king_rl.runtime.scope import (
    assess_assistance_scope,
    format_assistance_scope,
)


class RuntimeScopeTests(unittest.TestCase):
    def test_registered_weapon_is_supported(self):
        scope = assess_assistance_scope(1, 0, "Solomon", 1)

        self.assertTrue(scope.supported)
        self.assertEqual(scope.issues, ())
        self.assertFalse(scope.floor_verified)

    def test_other_registered_weapon_is_supported(self):
        scope = assess_assistance_scope(2, 1, "Victoria", 3)

        self.assertTrue(scope.supported)
        self.assertEqual(scope.issues, ())

    def test_unknown_weapon_is_reported_as_unsupported(self):
        scope = assess_assistance_scope(
            2, 99, "Unknown weapon", 3, weapon_registered=False
        )

        self.assertFalse(scope.supported)
        self.assertEqual(scope.issues, ("unknown_weapon_type:99",))

    def test_later_rank_uses_configuration_without_blocking(self):
        scope = assess_assistance_scope(20, 0, "Solomon", 8)

        self.assertTrue(scope.supported)
        self.assertEqual(scope.rank, 20)

    def test_later_floor_remains_supported_metadata(self):
        scope = assess_assistance_scope(1, 0, "Solomon", 12)

        self.assertTrue(scope.supported)
        self.assertEqual(scope.floor, 12)

    def test_display_requires_operator_floor_confirmation(self):
        scope = assess_assistance_scope(1, 0, "Solomon", 1)

        message = format_assistance_scope(scope)

        self.assertIn("Rank 1", message)
        self.assertIn("Solomon", message)
        self.assertIn("weapon 0", message)
        self.assertIn("Floor 1", message)
        self.assertIn("does not restrict support", message)
        self.assertIn("confirm it on the game screen", message)
        self.assertIn("Rank scaling uses configuration", message)
        self.assertIn("Card and buff effects", message)

    def test_unsupported_scope_stops_before_environment_creation(self):
        with patch.object(train, "ShotgunKingEnv") as environment:
            started = train.train_dqn(weapon_type=99)

        self.assertFalse(started)
        environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
