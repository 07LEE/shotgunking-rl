"""Offline checks for stale pre-action observation rejection."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.runtime.consistency import compare_policy_observation


def observation():
    state = np.zeros(281, dtype=np.float32)
    state[7 * 8 + 4] = 1
    state[128:130] = (2, 6)
    return state


class RuntimeConsistencyTests(unittest.TestCase):
    def test_equal_policy_fields_are_consistent(self):
        result = compare_policy_observation(observation(), observation())

        self.assertTrue(result.consistent)
        self.assertEqual(result.reason, "match")

    def test_board_change_reports_squares(self):
        expected = observation()
        actual = observation()
        actual[7 * 8 + 4] = 0
        actual[6 * 8 + 4] = 1

        result = compare_policy_observation(expected, actual)

        self.assertFalse(result.consistent)
        self.assertEqual(result.reason, "board_changed")
        self.assertEqual(
            [cell["square"] for cell in result.as_info()["changed_cells"]],
            ["e2", "e1"],
        )

    def test_ammo_change_reports_expected_and_actual_counts(self):
        expected = observation()
        actual = observation()
        actual[128:130] = (1, 5)

        result = compare_policy_observation(expected, actual)

        self.assertFalse(result.consistent)
        self.assertEqual(result.reason, "ammo_changed")
        self.assertEqual(result.as_info()["expected_ammo"], {"loaded": 2, "reserve": 6})
        self.assertEqual(result.as_info()["actual_ammo"], {"loaded": 1, "reserve": 5})


if __name__ == "__main__":
    unittest.main()
