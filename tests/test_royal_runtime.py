"""Offline royal rules, observations and completed action transitions."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import env
from specs import ENEMY_SPECS, get_enemy_specs_matrices
from context_fixtures import NormalContextReader
from shotgun_king_rl.perception import AmmoReading, PerceptionPipeline, Prediction, ScreenKind
from shotgun_king_rl.runtime.actions import detect_promotion, select_shot_target
from shotgun_king_rl.runtime.io import CaptureResult, IOStatus, InputResult
from shotgun_king_rl.runtime.state import build_threat_matrix, update_enemy_turns
from shotgun_king_rl.runtime.suggestions import build_suggestion_view, format_suggestion_view
from test_perception_pipeline import FakeReader, VALID_COMBAT_STATS


class RoyalRuntimeTests(unittest.TestCase):
    def test_white_king_threat_is_exactly_eight_adjacent_squares(self):
        board = np.zeros((8, 8), dtype=int)
        board[3, 3] = 9
        expected = np.zeros((8, 8), dtype=int)
        expected[2:5, 2:5] = 1
        expected[3, 3] = 0
        np.testing.assert_array_equal(build_threat_matrix(board), expected)
        board[:] = 0
        board[0, 0] = 9
        self.assertEqual(np.argwhere(build_threat_matrix(board)).tolist(), [[0, 1], [1, 0], [1, 1]])

    def test_queen_threat_uses_both_ray_types_and_stops_at_blockers(self):
        board = np.zeros((8, 8), dtype=int)
        board[3, 3] = 8
        expected = np.zeros((8, 8), dtype=int)
        for row in range(8):
            for col in range(8):
                if (row, col) != (3, 3) and (row == 3 or col == 3 or abs(row - 3) == abs(col - 3)):
                    expected[row, col] = 1
        np.testing.assert_array_equal(build_threat_matrix(board), expected)
        # The player king is a blocker without adding another enemy threat.
        board[3, 5] = 1
        threat = build_threat_matrix(board)
        self.assertEqual(threat[3, 5], 1)
        self.assertEqual(threat[3, 6], 0)
        self.assertEqual(threat[3, 7], 0)

    def test_distant_white_king_has_no_queen_target_priority(self):
        for royal, expected in ((8, 8), (9, 3)):
            with self.subTest(royal=royal):
                board = np.zeros((8, 8), dtype=int)
                board[4, 4], board[4, 5], board[1, 4] = 1, 3, royal
                self.assertEqual(select_shot_target(board, (4, 4)).piece, expected)
        board[:] = 0
        board[4, 4], board[3, 4] = 1, 9
        self.assertTrue(select_shot_target(board, (4, 4)).threatens_king)

    def test_royal_guard_protects_white_king_but_not_queen(self):
        for royal in (8, 9):
            with self.subTest(royal=royal):
                board = np.zeros((8, 8), dtype=int)
                board[4, 4], board[2, 4], board[0, 1] = 1, royal, 3
                target = select_shot_target(board, (4, 4), royal_guard_active=True)
                if royal == 9:
                    self.assertIsNone(target)
                else:
                    self.assertEqual(target.piece, 8)
                board[0, 1] = 0
                self.assertEqual(select_shot_target(board, (4, 4), royal_guard_active=True).piece, royal)

    def test_promotion_can_create_queen_but_not_white_king(self):
        previous = np.zeros((8, 8), dtype=int)
        previous[6, 2] = 2
        for royal in (8, 9):
            with self.subTest(royal=royal):
                current = np.zeros((8, 8), dtype=int)
                current[7, 3] = royal
                self.assertEqual(detect_promotion(previous, current), royal == 8)

    def test_royals_keep_nonzero_estimates_and_separate_turn_tracks(self):
        board = np.zeros((8, 8), dtype=int)
        board[2, 2], board[3, 3] = 8, 9
        hp, turns = get_enemy_specs_matrices(board, {(2, 2): 1, (3, 3): 3})
        self.assertGreater(hp[2, 2], 0)
        self.assertGreater(hp[3, 3], 0)
        self.assertAlmostEqual(turns[2, 2], .2)
        self.assertAlmostEqual(turns[3, 3], .6)
        updated = update_enemy_turns(board, board, {(2, 2): 1, (3, 3): 3}, ENEMY_SPECS)
        self.assertEqual(updated, {(2, 2): 0, (3, 3): 2})
        self.assertEqual(set(update_enemy_turns(board, board, {}, ENEMY_SPECS)), {(2, 2), (3, 3)})

    def test_live_observation_masks_and_rewards_keep_distinct_royal_ids(self):
        for killed, surviving in ((8, 9), (9, 8)):
            with self.subTest(killed=killed):
                board = np.zeros((8, 8), dtype=int)
                board[4, 4], board[4, 6], board[0, 0] = 1, killed, surviving
                pieces = FakeReader(Prediction.accepted(board, source='pieces'))
                ammo = FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo'))
                pipeline = PerceptionPipeline(
                    FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen')),
                    pieces, ammo, FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source='combat')),
                    NormalContextReader(lambda: pieces.prediction.value),
                )
                game = env.ShotgunKingEnv(perception_pipeline=pipeline)

                def after_shot():
                    after = board.copy()
                    after[4, 6] = 0
                    pieces.prediction = Prediction.accepted(after, source='pieces')
                    ammo.prediction = Prediction.accepted(AmmoReading(0, 4), source='ammo')
                    return game._get_obs()

                with patch.object(game, '_check_emergency_stop'), patch.object(env, 'capture_screen', return_value=CaptureResult(IOStatus.SUCCEEDED, 'data/screenshot.png')), patch.object(env.cv2, 'imread', return_value=np.zeros((720, 1280, 3), dtype=np.uint8)), patch.object(env.time, 'sleep'), patch.object(env, 'click_relative_in_window', return_value=InputResult(IOStatus.SUCCEEDED, 'click')) as click, patch.object(game, '_wait_for_equilibrium', side_effect=after_shot), patch.object(env, 'calculate_step_reward', wraps=env.calculate_step_reward) as reward:
                    game.current_state = game._get_obs()
                    self.assertTrue(game.last_perception.ready_for_policy)
                    np.testing.assert_array_equal(game.current_state[:64], board.flatten())
                    self.assertEqual(game.current_state.shape, (345,))
                    self.assertEqual(game.get_action_mask()[9], 1)
                    view = build_suggestion_view(game.current_state, game.get_action_mask(), None, loaded_ammo=1, max_ammo=2, reserve_ammo=4, range_limit=5)
                    rendered = format_suggestion_view(view)
                    self.assertIn('Q', rendered)
                    self.assertIn('W', rendered)
                    self.assertIn('white king' if killed == 9 else 'queen', next(item.reason for item in view.valid_actions if item.action == 9))
                    obs, _, terminated, _, info = game.step(9)
                click.assert_called_once()
                self.assertEqual(reward.call_args.kwargs['shot_target'].piece, killed)
                self.assertEqual(reward.call_args.kwargs['previous_enemy_count'], 2)
                self.assertTrue(info['transition_ready'])
                # A missing king alone cannot establish victory with unmodeled cards.
                self.assertFalse(terminated)
                self.assertEqual(int(obs[0]), surviving)
                self.assertIn((0, 0), game.enemy_turns)


if __name__ == '__main__':
    unittest.main()
