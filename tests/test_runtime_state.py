"""Offline checks for pure runtime state calculations."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shotgun_king_rl.runtime.state import (
    build_observation,
    build_threat_matrix,
    update_enemy_turns,
)
from train import remember_manual_defeat


class RuntimeStateTests(unittest.TestCase):
    def test_threat_rays_stop_after_first_occupied_cell(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[2, 2] = 5
        board[2, 4] = 2

        threat = build_threat_matrix(board)

        self.assertEqual(threat[2, 3], 1)
        self.assertEqual(threat[2, 4], 1)
        self.assertEqual(threat[2, 5], 0)

    def test_observation_layout_stays_281_values(self):
        board = np.arange(64, dtype=np.float32).reshape(8, 8)
        threat = np.ones((8, 8), dtype=np.float32)
        hp = np.full((8, 8), 0.5, dtype=np.float32)
        turns = np.full((8, 8), 0.25, dtype=np.float32)

        result = build_observation(
            board=board,
            threat=threat,
            loaded_ammo=1,
            reserve_ammo=6,
            damage=4,
            range_limit=5,
            spread=55,
            extra_turn_active=True,
            hp_matrix=hp,
            turn_matrix=turns,
        )

        self.assertEqual(result.shape, (281,))
        np.testing.assert_array_equal(result[:64], board.reshape(-1))
        np.testing.assert_array_equal(result[128:133], [1, 6, 4, 5, 55])
        self.assertEqual(result[133], 1)
        np.testing.assert_array_equal(result[153:217], hp.reshape(-1))
        np.testing.assert_array_equal(result[217:281], turns.reshape(-1))

    def test_enemy_turns_decrement_or_reset_after_movement(self):
        specs = {2: {"turn": 5.0}, 3: {"turn": 3.0}}
        previous = np.zeros((8, 8), dtype=np.float32)
        current = np.zeros((8, 8), dtype=np.float32)
        previous[1, 1] = 2
        previous[2, 2] = 3
        current[1, 1] = 2
        current[3, 2] = 3

        result = update_enemy_turns(
            previous_board=previous,
            current_board=current,
            previous_turns={(1, 1): 2.0, (2, 2): 1.0},
            enemy_specs=specs,
        )

        self.assertEqual(result[(1, 1)], 1.0)
        self.assertEqual(result[(3, 2)], 3.0)

    def test_manual_defeat_records_complete_transition(self):
        agent = Mock()
        state = np.zeros(281, dtype=np.float32)
        mask = np.ones(10, dtype=np.float32)

        remember_manual_defeat(agent, state, 4, mask)

        agent.remember.assert_called_once_with(
            state,
            4,
            -15.0,
            state,
            mask,
            mask,
            True,
        )


if __name__ == "__main__":
    unittest.main()
