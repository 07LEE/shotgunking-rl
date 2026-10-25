"""Exercise the real environment observation and hover recovery without desktop I/O."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import env
from shotgun_king_rl.perception import AmmoReading, PerceptionPipeline, Prediction, ScreenKind
from shotgun_king_rl.runtime.io import CaptureResult, IOStatus, InputResult
from shotgun_king_rl.runtime.transitions import RuntimeScreenState
from test_perception_pipeline import FakeReader, VALID_COMBAT_STATS


class LivePerceptionTests(unittest.TestCase):
    def setUp(self):
        self.board = np.zeros((8, 8), dtype=np.float32)
        self.board[4, 4] = 1
        self.board[0, 0] = 2
        self.stats = replace(VALID_COMBAT_STATS, attack=8, range_min=2, range_max=6, spread_degrees=40, penetration_percent=25, knockback_percent=50, melee_damage=2)
        self.screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen'))
        self.combat = FakeReader(Prediction.accepted(self.stats, source='combat'))
        self.pipeline = PerceptionPipeline(self.screen, FakeReader(Prediction.accepted(self.board, source='board')), FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo')), self.combat)
        self.game = env.ShotgunKingEnv(perception_pipeline=self.pipeline, hover_clear_position=(45, 50), hover_clear_delay=0.25)
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.capture_result = CaptureResult(IOStatus.SUCCEEDED, 'data/screenshot.png')
        self.patches = [
            patch.object(self.game, '_check_emergency_stop'),
            patch.object(env, 'capture_screen', return_value=self.capture_result),
            patch.object(env.cv2, 'imread', return_value=self.image),
            patch.object(env, 'move_relative_in_window', return_value=InputResult(IOStatus.SUCCEEDED, 'move')),
            patch.object(env.time, 'sleep'),
        ]
        self.stop, self.capture, self.imread, self.move, self.sleep = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def test_observation_mask_and_runtime_stats_use_read_values(self):
        obs = self.game._get_obs(force_inject=True)
        self.assertEqual(obs.shape, (281,))
        np.testing.assert_array_equal(obs[:64], self.board.flatten())
        np.testing.assert_array_equal(obs[128:133], [1, 4, 8, 6, 40])
        self.assertEqual(self.game.falloff_start, 2)
        self.assertEqual(self.game.pierce_chance, 0.25)
        self.assertEqual(self.game.knockback_chance, 0.5)
        self.assertEqual(self.game.melee_damage, 2)
        self.game.current_state = obs
        self.assertTrue(np.any(self.game.get_action_mask()))
        self.assertEqual(self.game._assess_screen(obs).state, RuntimeScreenState.GAMEPLAY)
        self.move.assert_not_called()

    def test_real_environment_calls_recovery_and_recaptures(self):
        calls = []
        def read(image):
            calls.append(image)
            return (Prediction.unknown(source='combat', reason='obscured') if len(calls) == 1 else Prediction.accepted(self.stats, source='combat'))
        self.combat.read = read
        second = self.image.copy()
        self.imread.side_effect = [self.image, second]
        obs = self.game._get_obs()
        self.assertEqual(obs[130], 8)
        self.assertIs(calls[1], second)
        self.assertEqual(self.capture.call_count, 2)
        self.move.assert_called_once_with('Shotgun King', 45, 50)
        self.sleep.assert_called_once_with(0.25)

    def test_failed_hud_blocks_step_and_mask_without_fallback(self):
        self.game.current_state = self.game._get_obs()
        self.combat.prediction = Prediction.unknown(source='combat', reason='missing')
        with patch.object(env, 'click_relative_in_window') as click, patch.object(env, 'press_key') as key:
            obs, reward, term, trunc, info = self.game.step(4)
        click.assert_not_called()
        key.assert_not_called()
        np.testing.assert_array_equal(obs, np.zeros(281))
        np.testing.assert_array_equal(self.game.get_action_mask(), np.zeros(10))
        self.assertFalse(info['transition_ready'])
        self.assertEqual(info['screen_state']['state'], 'observation_failure')
        self.assertIn('combat_stats:missing', info['screen_state']['reason'])
        self.assertEqual(reward, 0)
        self.assertFalse(term)

    def test_failed_capture_never_reuses_previous_frame(self):
        self.game._get_obs()
        self.capture.return_value = CaptureResult(IOStatus.FAILED, 'data/screenshot.png')
        self.imread.reset_mock()
        obs = self.game._get_obs()
        np.testing.assert_array_equal(obs, np.zeros(281))
        self.imread.assert_not_called()
        self.assertEqual(self.game._assess_screen(obs).state, RuntimeScreenState.OBSERVATION_FAILURE)

    def test_confirmed_no_melee_blocks_empty_weapon_attack_without_input(self):
        self.board[0, 0], self.board[4, 5] = 0, 2
        self.combat.prediction = Prediction.accepted(replace(self.stats, melee_damage=0), source='combat')
        self.pipeline.ammo_reader.prediction = Prediction.accepted(AmmoReading(0, 4), source='ammo')
        self.game.current_state = self.game._get_obs()
        self.assertTrue(self.game.last_perception.ready_for_policy)
        self.assertEqual(self.game.melee_damage, 0)
        mask = self.game.get_action_mask()
        self.assertEqual(mask[9], 0)
        self.assertEqual(mask[3], 1)
        self.assertEqual(mask[8], 1)
        with patch.object(env, 'click_relative_in_window') as click, patch.object(env, 'press_key') as key:
            _, reward, _, _, info = self.game.step(9)
        click.assert_not_called()
        key.assert_not_called()
        self.assertEqual(info['action_issue'], 'not_permitted_by_action_mask')
        self.assertFalse(info['transition_ready'])
        self.assertEqual(reward, 0)
        self.move.assert_not_called()

    def test_confirmed_no_melee_uses_loaded_ranged_shot_at_adjacent_enemy(self):
        self.board[0, 0], self.board[4, 5] = 0, 2
        self.combat.prediction = Prediction.accepted(replace(self.stats, melee_damage=0), source='combat')
        self.game.current_state = self.game._get_obs()
        self.assertEqual(self.game.get_action_mask()[9], 1)

        def after_shot():
            self.pipeline.ammo_reader.prediction = Prediction.accepted(AmmoReading(0, 4), source='ammo')
            return self.game._get_obs()

        with patch.object(self.game, '_wait_for_equilibrium', side_effect=after_shot), patch.object(env, 'click_relative_in_window', return_value=InputResult(IOStatus.SUCCEEDED, 'click')) as click, patch.object(env, 'calculate_step_reward', wraps=env.calculate_step_reward) as reward:
            _, _, _, _, info = self.game.step(9)
        # Aim past the adjacent enemy, so the click is a ranged shot.
        click.assert_called_once_with('Shotgun King', 796, 408)
        self.assertFalse(reward.call_args.kwargs['shot_was_melee'])
        self.assertEqual(reward.call_args.kwargs['melee_damage'], 0)
        self.assertTrue(info['transition_ready'])

    def test_change_outside_legacy_features_blocks_action(self):
        self.game.current_state = self.game._get_obs()
        self.combat.prediction = Prediction.accepted(replace(self.stats, penetration_percent=75), source='combat')
        with patch.object(env, 'click_relative_in_window') as click:
            _, _, _, _, info = self.game.step(4)
        click.assert_not_called()
        self.assertEqual(info['observation_consistency']['reason'], 'combat_stats_changed')
        self.assertFalse(info['transition_ready'])

    def test_new_screen_classification_preserves_terminal_screens(self):
        for kind, expected in [(ScreenKind.CARD_SELECTION, RuntimeScreenState.VICTORY), (ScreenKind.RETRY, RuntimeScreenState.DEFEAT)]:
            self.screen.prediction = Prediction.accepted(kind, source='screen')
            obs = self.game._get_obs()
            self.assertEqual(self.game._assess_screen(obs).state, expected)
        self.move.assert_not_called()

    def test_default_environment_builds_pipeline_once_and_passes_references(self):
        game = env.ShotgunKingEnv(perception_collection='/collection', melee_title_annotation='/title')
        with patch.object(game, '_check_emergency_stop'), patch.object(env, 'build_live_pipeline', return_value=(self.pipeline, {'readers': {}})) as factory:
            game._get_obs()
            game._get_obs()
        factory.assert_called_once_with('/collection', '/title')
        self.assertIs(game.perception_pipeline, self.pipeline)

    def test_missing_king_is_never_force_injected(self):
        self.pipeline.board_reader.prediction = Prediction.rejected(np.zeros((8, 8)), source='board', reason='player_king_count:0')
        obs = self.game._get_obs(force_inject=True)
        self.assertFalse(np.any(obs[:64] == 1))
        self.assertFalse(self.game.last_perception.ready_for_policy)

    def test_reward_uses_combat_stats_from_before_action(self):
        self.game.current_state = self.game._get_obs()

        def after_action():
            moved = self.board.copy()
            moved[4, 4] = 0
            moved[4, 5] = 1
            self.pipeline.board_reader.prediction = Prediction.accepted(moved, source='board')
            self.combat.prediction = Prediction.accepted(replace(self.stats, attack=3), source='combat')
            return self.game._get_obs()

        with patch.object(self.game, '_wait_for_equilibrium', side_effect=after_action), patch.object(env, 'click_relative_in_window', return_value=InputResult(IOStatus.SUCCEEDED, 'click')), patch.object(env, 'calculate_step_reward', wraps=env.calculate_step_reward) as reward:
            _, _, _, _, info = self.game.step(4)
        self.assertTrue(info['transition_ready'])
        self.assertEqual(reward.call_args.kwargs['damage'], 8)
        self.assertEqual(self.game.damage, 3)
