"""Offline checks for stale pre-action observation rejection."""

import unittest

import numpy as np

from shotgun_king_rl.runtime.consistency import compare_policy_observation


def observation(size=281):
    state = np.zeros(size, dtype=np.float32)
    state[7 * 8 + 4] = 1
    state[128:130] = (2, 6)
    return state


class RuntimeConsistencyTests(unittest.TestCase):
    def test_observation_layouts_must_match(self):
        cases = ((281, 281, True, 'match'), (345, 345, True, 'match'),
                 (281, 345, False, 'observation_shape_invalid'))
        for expected_size, actual_size, consistent, reason in cases:
            with self.subTest(expected_size=expected_size, actual_size=actual_size):
                result = compare_policy_observation(observation(expected_size), observation(actual_size))
                self.assertEqual(result.consistent, consistent)
                self.assertEqual(result.reason, reason)

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
