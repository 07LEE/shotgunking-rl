"""Offline checks for action and terminal transition calculations."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.runtime.actions import (
    action_destination,
    available_moves,
    detect_promotion,
    select_shot_target,
)
from shotgun_king_rl.runtime.state import build_action_mask
from shotgun_king_rl.runtime.transitions import (
    apply_terminal_rules,
    update_countdown,
)


class RuntimeActionTests(unittest.TestCase):
    def test_shot_target_prioritizes_threat_over_nearer_piece(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[4, 6] = 3
        board[1, 4] = 5

        target = select_shot_target(board, (4, 4))

        self.assertEqual(target.direction, (-1, 0))
        self.assertEqual(target.distance, 3)
        self.assertTrue(target.threatens_king)

    def test_royal_guard_skips_boss_while_knight_exists(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[2, 4] = 9
        board[4, 6] = 3

        target = select_shot_target(
            board,
            (4, 4),
            royal_guard_active=True,
        )

        self.assertEqual(target.piece, 3)
        self.assertEqual(target.direction, (0, 1))

    def test_target_selection_ignores_out_of_range_priority(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[0, 4] = 5
        board[4, 6] = 3

        target = select_shot_target(board, (4, 4), range_limit=3)

        self.assertEqual(target.piece, 3)
        self.assertEqual(target.distance, 2)

    def test_available_moves_share_action_destinations(self):
        board = np.zeros((8, 8), dtype=np.float32)
        threat = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[3, 4] = 2
        threat[4, 5] = 1

        valid, safe = available_moves(
            board,
            threat,
            (4, 4),
            move_range_bonus=1,
        )

        self.assertNotIn(1, valid)
        self.assertNotIn(11, valid)
        self.assertIn(4, valid)
        self.assertNotIn(4, safe)
        self.assertEqual(action_destination((4, 4), 14), (4, 6))

    def test_action_mask_uses_valid_moves_when_none_are_safe(self):
        board = np.full((8, 8), 2, dtype=np.float32)
        threat = np.ones((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[4, 5] = 0
        observation = np.zeros(281, dtype=np.float32)
        observation[:64] = board.reshape(-1)
        observation[64:128] = threat.reshape(-1)

        mask = build_action_mask(
            observation,
            action_count=10,
            tracked_king=(4, 4),
            loaded_ammo=0,
            max_ammo=2,
            reserve_ammo=0,
            range_limit=5,
        )

        self.assertEqual(np.flatnonzero(mask).tolist(), [4])

    def test_action_mask_allows_adjacent_melee_without_ammo(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[4, 5] = 2
        observation = np.zeros(281, dtype=np.float32)
        observation[:64] = board.reshape(-1)

        mask = build_action_mask(
            observation,
            action_count=10,
            tracked_king=(4, 4),
            loaded_ammo=0,
            max_ammo=2,
            reserve_ammo=0,
            range_limit=5,
            melee_damage=3,
        )

        self.assertEqual(mask[9], 1)

    def test_promotion_uses_adjacent_previous_pawn(self):
        previous = np.zeros((8, 8), dtype=np.float32)
        current = np.zeros((8, 8), dtype=np.float32)
        previous[6, 2] = 2
        current[7, 3] = 5

        self.assertTrue(detect_promotion(previous, current))

    def test_no_melee_damage_preserves_movement_reload_and_loaded_shots(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[4, 5] = 2
        observation = np.zeros(281, dtype=np.float32)
        observation[:64] = board.reshape(-1)
        for loaded_ammo in (0, 1):
            with self.subTest(loaded_ammo=loaded_ammo):
                mask = build_action_mask(
                    observation, action_count=10, tracked_king=(4, 4),
                    loaded_ammo=loaded_ammo, max_ammo=2, reserve_ammo=4,
                    range_limit=5, melee_damage=0,
                )
                self.assertEqual(mask[9], float(loaded_ammo > 0))
                self.assertEqual(mask[8], 1)
                self.assertEqual(mask[3], 1)
                self.assertEqual(mask[4], 0)

    def test_terminal_rules_preserve_existing_evaluation_order(self):
        result = apply_terminal_rules(
            reward=-0.1,
            enemy_count=0,
            king_present=True,
            popup_detected=False,
            card_selection_detected=False,
            countdown_expired=True,
        )

        self.assertEqual(result.outcome, "board_victory")
        self.assertEqual(result.reward, -5.0)
        self.assertTrue(result.terminated)

    def test_popup_defeat_overrides_other_terminal_rewards(self):
        result = apply_terminal_rules(
            reward=4.0,
            enemy_count=3,
            king_present=False,
            popup_detected=True,
            card_selection_detected=True,
            countdown_expired=True,
        )

        self.assertEqual(result.outcome, "retry_popup_defeat")
        self.assertEqual(result.reward, -18.0)

    def test_countdown_trigger_and_expiry_are_deterministic(self):
        trigger, expired = update_countdown(True, None, 5, 6)
        self.assertEqual(trigger, 5)
        self.assertFalse(expired)
        self.assertEqual(update_countdown(True, trigger, 17, 4), (5, True))


if __name__ == "__main__":
    unittest.main()
