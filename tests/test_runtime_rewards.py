"""Offline checks for pure runtime reward calculations."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.runtime.actions import ShotTarget
from shotgun_king_rl.runtime.rewards import calculate_step_reward


class RuntimeRewardTests(unittest.TestCase):
    def setUp(self):
        self.board = np.zeros((8, 8), dtype=np.float32)
        self.board[4, 4] = 1
        self.board[4, 6] = 2
        self.threat = np.zeros((8, 8), dtype=np.float32)
        self.target = ShotTarget(
            direction=(0, 1),
            distance=2,
            piece=2,
            threatens_king=False,
        )

    def calculate(self, **overrides):
        values = {
            "original_action": 9,
            "executed_action": 9,
            "previous_enemy_count": 1,
            "shot_board": self.board,
            "current_board": self.board,
            "current_threat": self.threat,
            "king_position": (4, 4),
            "shot_target": self.target,
            "valid_shot_target": True,
            "shot_was_melee": False,
            "damage": 4.0,
            "melee_damage": 0.0,
            "melee_kill_extra_turn": False,
            "falloff_start": 3,
            "range_limit": 5,
            "spread": 55.0,
            "pierce_chance": 0.0,
            "knockback_chance": 0.0,
            "target_hp": 0.5,
            "promotion_detected": False,
        }
        values.update(overrides)
        return calculate_step_reward(**values)

    def test_valid_shot_without_kill_uses_expected_damage(self):
        result = self.calculate()

        self.assertAlmostEqual(result.expected_damage, 3.6333333333)
        self.assertAlmostEqual(result.reward, 0.99)
        self.assertEqual(result.killed_enemies, 0)

    def test_invalid_guarded_shot_gets_attempt_penalty(self):
        result = self.calculate(
            executed_action=4,
            shot_target=None,
            valid_shot_target=False,
        )

        self.assertAlmostEqual(result.reward, -0.9)
        self.assertTrue(result.invalid_shot_attempt)

    def test_confirmed_melee_kill_controls_extra_turn_fact(self):
        cleared_board = self.board.copy()
        cleared_board[4, 6] = 0

        result = self.calculate(
            current_board=cleared_board,
            shot_was_melee=True,
            melee_damage=3.0,
            melee_kill_extra_turn=True,
        )

        self.assertTrue(result.melee_kill)
        self.assertAlmostEqual(result.expected_damage, 4.5)
        self.assertAlmostEqual(result.reward, 3.25)

    def test_threat_and_promotion_penalties_stack(self):
        threatened = self.threat.copy()
        threatened[4, 4] = 1

        result = self.calculate(
            original_action=4,
            executed_action=4,
            shot_target=None,
            valid_shot_target=False,
            current_threat=threatened,
            promotion_detected=True,
        )

        self.assertAlmostEqual(result.reward, -4.6)
        self.assertTrue(result.threatened)


if __name__ == "__main__":
    unittest.main()
