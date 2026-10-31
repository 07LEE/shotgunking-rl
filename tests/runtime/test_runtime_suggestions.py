"""Offline checks for inspectable suggest-mode output."""

import unittest

import numpy as np

from shotgun_king_rl.runtime.suggestions import (
    build_suggestion_view,
    format_suggestion_view,
)


class RuntimeSuggestionTests(unittest.TestCase):
    def test_view_shows_board_ammo_valid_actions_and_reasons(self):
        observation = np.zeros(281, dtype=np.float32)
        board = observation[:64].reshape(8, 8)
        board[4, 4] = 1
        board[4, 6] = 3
        mask = np.zeros(10, dtype=np.float32)
        mask[[4, 8, 9]] = 1
        q_values = np.arange(10, dtype=np.float32)

        view = build_suggestion_view(
            observation,
            mask,
            q_values,
            loaded_ammo=1,
            max_ammo=2,
            reserve_ammo=4,
            range_limit=5,
        )
        rendered = format_suggestion_view(view)

        self.assertIn("K", rendered)
        self.assertIn("N", rendered)
        self.assertIn("Ammo: loaded 1/2 | reserve 4", rendered)
        self.assertIn("safe empty square f4", rendered)
        self.assertIn("reload 1 shell(s) from reserve", rendered)
        self.assertIn("target knight on g4", rendered)

    def test_recommendations_rank_only_valid_actions_by_q_value(self):
        observation = np.zeros(281, dtype=np.float32)
        observation[4 * 8 + 4] = 1
        mask = np.zeros(10, dtype=np.float32)
        mask[[3, 4, 8]] = 1
        q_values = np.array([99, 98, 97, 0.2, 0.7, 96, 95, 94, 0.5, 93])

        view = build_suggestion_view(
            observation,
            mask,
            q_values,
            loaded_ammo=1,
            max_ammo=2,
            reserve_ammo=4,
            range_limit=5,
        )

        self.assertEqual([item.action for item in view.recommendations], [4, 8, 3])

    def test_output_labels_q_values_without_confidence_claim(self):
        observation = np.zeros(281, dtype=np.float32)
        observation[4 * 8 + 4] = 1
        mask = np.zeros(10, dtype=np.float32)
        mask[4] = 1

        rendered = format_suggestion_view(
            build_suggestion_view(
                observation,
                mask,
                np.ones(10),
                loaded_ammo=2,
                max_ammo=2,
                reserve_ammo=4,
                range_limit=5,
            )
        )

        self.assertIn("model Q-values, not probabilities", rendered)
        self.assertIn("validity and targeting rules", rendered)
        self.assertNotIn("Confidence", rendered)


if __name__ == "__main__":
    unittest.main()
