"""Explicit side/moat state must reach policy, movement and input guards."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import env
from shotgun_king_rl.perception import (
    AmmoReading, BoardContextReading, PerceptionPipeline, Prediction, ScreenKind,
)
from shotgun_king_rl.perception.context import terrain_grid
from shotgun_king_rl.runtime.actions import available_moves, select_shot_target
from shotgun_king_rl.runtime.io import CaptureResult, IOStatus, InputResult
from shotgun_king_rl.runtime.state import build_threat_matrix
from test_perception_pipeline import FakeReader, VALID_COMBAT_STATS


def context_for(board, ally=None, moat=True):
    sides = np.where(board == 0, 0, np.where(board == 1, 1, 2)).astype(np.int8)
    if ally is not None:
        sides[ally] = 1
    return BoardContextReading(sides, moat)


class BoardContextTests(unittest.TestCase):
    def setUp(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[6, 4], self.board[5, 4], self.board[2, 4], self.board[0, 0] = 1, 5, 5, 9
        self.context = context_for(self.board, (5, 4))
        self.pieces = FakeReader(Prediction.accepted(self.board, source='pieces'))
        self.context_reader = FakeReader(Prediction.accepted(self.context, source='context'))
        self.pipeline = PerceptionPipeline(
            FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen')),
            self.pieces, FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo')),
            FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source='combat')), self.context_reader,
        )

    def test_accepted_context_encodes_allies_without_mutating_reader_values(self):
        result = self.pipeline.observe(object())
        self.assertTrue(result.ready_for_policy)
        self.assertEqual(result.board[5, 4], -5)
        self.assertEqual(result.board[2, 4], 5)
        self.assertEqual(result.board[6, 4], 1)
        self.assertEqual(self.board[5, 4], 5)
        np.testing.assert_array_equal(terrain_grid(result.board_context)[4], np.ones(8))
        self.context.piece_sides[5, 4] = 2
        self.assertEqual(result.board_context.piece_sides[5, 4], 1)

    def test_missing_unknown_or_invalid_context_never_defaults_to_enemy_normal(self):
        invalid = [None] + [replace(self.context, moat_active=value) for value in (None, 0, 'false')]
        for position, value in (((5, 4), 0), ((5, 4), 3), ((6, 4), 2), ((7, 7), 1)):
            sides = self.context.piece_sides.copy()
            sides[position] = value
            invalid.append(replace(self.context, piece_sides=sides))
        invalid.extend(replace(self.context, piece_sides=sides) for sides in (
            np.ones((8, 8), dtype=bool), np.ones((7, 8), dtype=int), self.context.piece_sides.astype(float),
        ))
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                self.context_reader.prediction = Prediction.accepted(value, source='context')
                result = self.pipeline.observe(object())
                self.assertFalse(result.ready_for_policy)
                self.assertEqual(len(result.issues), 1)
                self.assertIn('board_context:', result.issues[0])
                self.assertIsNone(result.board_context)
        self.context_reader.prediction = Prediction.unknown(source='context', reason='hidden')
        self.assertEqual(self.pipeline.observe(object()).issues, ('board_context:hidden',))
        self.pipeline.context_reader = None
        self.assertEqual(self.pipeline.observe(object()).issues, ('board_context:reader_unavailable',))

    def test_context_is_not_reused_when_the_next_frame_is_unknown(self):
        self.assertTrue(self.pipeline.observe(object()).ready_for_policy)
        self.context_reader.prediction = Prediction.unknown(source='context', reason='unreadable')
        result = self.pipeline.observe(object())
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.board_context)

    def test_allies_block_rays_but_are_not_threats_or_shot_targets(self):
        board = self.pipeline.observe(object()).board
        threat = build_threat_matrix(board)
        self.assertEqual(threat[5, 4], 1)
        self.assertEqual(threat[6, 4], 0)
        self.assertEqual(threat[5, 7], 0)
        self.assertIsNone(select_shot_target(board, (6, 4)))
        moves, _ = available_moves(board, threat, (6, 4))
        self.assertNotIn(1, moves)

    def test_moat_requires_stop_on_entry_but_allows_knights_and_moves_from_water(self):
        terrain = terrain_grid(self.context)
        threat = np.zeros((8, 8))
        for origin, up_jump_allowed in (((5, 4), False), ((6, 4), True), ((4, 4), True)):
            with self.subTest(origin=origin):
                board = np.zeros((8, 8), dtype=int)
                board[origin] = 1
                moves, _ = available_moves(board, threat, origin, move_range_bonus=1, terrain=terrain)
                self.assertEqual(11 in moves, up_jump_allowed)
                self.assertIn(1, moves)
        board[:] = 0
        board[3, 3] = 3
        self.assertEqual(build_threat_matrix(board, terrain)[5, 4], 1)

    def test_sliding_threats_stop_at_water_but_shots_still_cross(self):
        terrain = terrain_grid(self.context)
        for piece in (5, 8):
            with self.subTest(piece=piece):
                board = np.zeros((8, 8), dtype=int)
                board[2, 4] = piece
                threat = build_threat_matrix(board, terrain)
                self.assertEqual(threat[4, 4], 1)
                self.assertEqual(threat[5, 4], 0)
                board[:] = 0
                board[4, 4] = piece
                self.assertEqual(build_threat_matrix(board, terrain)[7, 4], 1)
                self.assertEqual(build_threat_matrix(board, terrain)[4, 7], 1)
        board[:] = 0
        board[6, 4], board[2, 4], board[6, 5] = 1, 5, 3
        self.assertEqual(select_shot_target(board, (6, 4)).piece, 5)
        self.assertEqual(select_shot_target(board, (6, 4), terrain=terrain).piece, 3)
        board[6, 5] = 0
        target = select_shot_target(board, (6, 4), terrain=terrain)
        self.assertEqual(target.piece, 5)
        self.assertFalse(target.threatens_king)

    def game_patches(self, game):
        patches = [patch.object(game, '_check_emergency_stop'),
                   patch.object(env, 'capture_screen', return_value=CaptureResult(IOStatus.SUCCEEDED, 'data/screenshot.png')),
                   patch.object(env.cv2, 'imread', return_value=np.zeros((720, 1280, 3), dtype=np.uint8)),
                   patch.object(env.time, 'sleep'),
                   patch.object(env, 'click_relative_in_window', return_value=InputResult(IOStatus.SUCCEEDED, 'click')),
                   patch.object(env, 'press_key')]
        mocks = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)
        return mocks[-2:]

    def test_live_observation_contains_sides_and_terrain_and_blocks_allied_ray(self):
        game = env.ShotgunKingEnv(perception_pipeline=self.pipeline)
        click, key = self.game_patches(game)
        game.current_state = game._get_obs()
        self.assertEqual(game.current_state.shape, (345,))
        self.assertTrue(game.observation_space.contains(game.current_state))
        self.assertEqual(game.current_state[5 * 8 + 4], -5)
        np.testing.assert_array_equal(game.current_state[281:].reshape(8, 8)[4], np.ones(8))
        self.assertEqual(game.get_action_mask()[9], 0)
        self.assertEqual(game.get_action_mask()[4], 1)
        _, _, _, _, info = game.step(9)
        self.assertEqual(info['action_issue'], 'not_permitted_by_action_mask')
        click.assert_not_called()
        key.assert_not_called()

    def test_side_or_moat_changes_block_input_before_action(self):
        for changed in ('side', 'moat', 'unknown'):
            with self.subTest(changed=changed):
                self.context_reader.prediction = Prediction.accepted(self.context, source='context')
                game = env.ShotgunKingEnv(perception_pipeline=self.pipeline)
                click, key = self.game_patches(game)
                game.current_state = game._get_obs()
                if changed == 'side':
                    sides = self.context.piece_sides.copy()
                    sides[5, 4] = 2
                    prediction = Prediction.accepted(replace(self.context, piece_sides=sides), source='context')
                elif changed == 'moat':
                    prediction = Prediction.accepted(replace(self.context, moat_active=False), source='context')
                else:
                    prediction = Prediction.unknown(source='context', reason='hidden')
                self.context_reader.prediction = prediction
                _, _, _, _, info = game.step(4)
                self.assertFalse(info['transition_ready'])
                if changed != 'unknown':
                    self.assertEqual(info['observation_consistency']['reason'], 'board_changed' if changed == 'side' else 'terrain_changed')
                else:
                    self.assertIn('board_context:hidden', info['screen_state']['reason'])
                click.assert_not_called()
                key.assert_not_called()

    def test_live_shot_does_not_reward_allied_loss_or_track_allied_turns(self):
        self.board[5, 4], self.board[6, 5] = 0, 5
        self.context = context_for(self.board, (6, 5))
        self.context_reader.prediction = Prediction.accepted(self.context, source='context')
        game = env.ShotgunKingEnv(perception_pipeline=self.pipeline)
        self.game_patches(game)
        game.current_state = game._get_obs()

        def after_shot():
            after = self.board.copy()
            after[6, 5] = 0
            self.pieces.prediction = Prediction.accepted(after, source='pieces')
            self.context_reader.prediction = Prediction.accepted(context_for(after), source='context')
            self.pipeline.ammo_reader.prediction = Prediction.accepted(AmmoReading(0, 4), source='ammo')
            return game._get_obs()

        with patch.object(game, '_wait_for_equilibrium', side_effect=after_shot), patch.object(env, 'calculate_step_reward', wraps=env.calculate_step_reward) as reward:
            _, value, terminated, _, info = game.step(9)
        self.assertEqual(reward.call_args.kwargs['previous_enemy_count'], 2)
        self.assertLess(value, 2)  # No enemy-kill bonus for losing the ally.
        self.assertEqual(set(game.enemy_turns), {(0, 0), (2, 4)})
        self.assertTrue(info['transition_ready'])
        self.assertFalse(terminated)


if __name__ == '__main__':
    unittest.main()
