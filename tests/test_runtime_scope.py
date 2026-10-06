"""Offline checks for the limited live-assistance scope."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import train
from shotgun_king_rl.runtime.scope import (
    assess_assistance_scope,
    format_assistance_scope,
)


class RuntimeScopeTests(unittest.TestCase):
    def test_rank_one_basic_weapon_first_floor_is_supported(self):
        scope = assess_assistance_scope(1, 0, "Solomon", 1)

        self.assertTrue(scope.supported)
        self.assertEqual(scope.issues, ())
        self.assertFalse(scope.floor_verified)

    def test_other_conditions_are_reported_as_unsupported(self):
        scope = assess_assistance_scope(2, 1, "Victoria", 3)

        self.assertFalse(scope.supported)
        self.assertEqual(scope.issues, ("rank:2", "weapon_type:1", "floor:3"))

    def test_display_requires_operator_floor_confirmation(self):
        scope = assess_assistance_scope(1, 0, "Solomon", 1)

        message = format_assistance_scope(scope)

        self.assertIn("Rank 1", message)
        self.assertIn("Solomon", message)
        self.assertIn("basic", message)
        self.assertIn("Floor 1", message)
        self.assertIn("confirm it on the game screen", message)

    def test_unsupported_scope_stops_before_environment_creation(self):
        with patch.object(train, "ShotgunKingEnv") as environment:
            started = train.train_dqn(rank=2)

        self.assertFalse(started)
        environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
