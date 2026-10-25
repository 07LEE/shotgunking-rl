"""Offline checks for bounded hover recovery with fresh observations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import input as game_input
from shotgun_king_rl.perception import AmmoReading, CombatStatsReading, PerceptionPipeline, Prediction, ScreenKind
from context_fixtures import NormalContextReader


class HoverRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.first = object()
        self.second = object()
        self.board = np.zeros((8, 8))
        self.board[7, 4] = 1
        self.stats = CombatStatsReading(attack=4, range_min=3, range_max=5, spread_degrees=65, penetration_percent=0, knockback_percent=0, grenades_remaining=0, grenade_damage=0, melee_damage=0)
        self.screen = Mock()
        self.screen.read.return_value = Prediction.accepted(ScreenKind.GAMEPLAY, source='screen')
        self.pieces = Mock()
        self.pieces.read.return_value = Prediction.accepted(self.board, source='board')
        self.ammo = Mock()
        self.ammo.read.return_value = Prediction.accepted(AmmoReading(1, 5), source='ammo')
        self.combat = Mock()
        self.combat.read.side_effect = [Prediction.unknown(source='combat', reason='obscured'), Prediction.accepted(self.stats, source='combat')]
        # Mock must expose only the ordinary reader interface.
        for reader in (self.screen, self.pieces, self.ammo, self.combat):
            reader.read_prepared = None
        self.context = NormalContextReader(lambda: self.board)
        self.pipeline = PerceptionPipeline(self.screen, self.pieces, self.ammo, self.combat, self.context)
        self.events = []
        self.capture = Mock(side_effect=[self.first, self.second])
        self.clear = Mock(side_effect=lambda: self.events.append('move') or True)
        self.wait = Mock(side_effect=lambda: self.events.append('wait'))

    def observe(self):
        return self.pipeline.observe_live(self.capture, self.clear, self.wait)

    def test_recovery_revalidates_every_reader_from_new_image(self):
        fresh_board = self.board.copy()
        fresh_board[0, 0] = 2
        boards = iter((self.board, fresh_board))
        def read(image):
            self.board = next(boards)
            return Prediction.accepted(self.board, source='board')
        self.pieces.read.side_effect = read
        result = self.observe()
        self.assertTrue(result.ready_for_policy)
        np.testing.assert_array_equal(result.board, fresh_board)
        self.assertIsNot(result.board, fresh_board)
        self.assertEqual(self.events, ['move', 'wait'])
        for reader in (self.screen, self.pieces, self.ammo, self.combat):
            self.assertEqual([call.args[0] for call in reader.read.call_args_list], [self.first, self.second])

    def test_no_recovery_for_ready_or_non_gameplay_screen(self):
        self.combat.read.side_effect = None
        self.combat.read.return_value = Prediction.accepted(self.stats, source='combat')
        self.assertTrue(self.observe().ready_for_policy)
        self.clear.assert_not_called()
        self.wait.assert_not_called()
        self.capture.assert_called_once()
        self.capture.reset_mock(side_effect=True)
        self.capture.return_value = self.first
        self.screen.read.return_value = Prediction.accepted(ScreenKind.CARD_SELECTION, source='screen')
        self.assertFalse(self.observe().ready_for_policy)
        self.clear.assert_not_called()

    def test_repeated_unknown_does_not_loop_or_reuse_stats(self):
        self.combat.read.side_effect = None
        self.combat.read.return_value = Prediction.unknown(source='combat', reason='missing')
        result = self.observe()
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertEqual(self.capture.call_count, 2)
        self.clear.assert_called_once()

    def test_second_frame_screen_change_is_not_gameplay(self):
        self.screen.read.side_effect = [Prediction.accepted(ScreenKind.GAMEPLAY, source='screen'), Prediction.accepted(ScreenKind.RETRY, source='screen')]
        result = self.observe()
        self.assertEqual(result.screen_kind, ScreenKind.RETRY)
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.board)

    def test_move_failure_stops_before_wait_and_recapture(self):
        self.clear.side_effect = None
        self.clear.return_value = False
        result = self.observe()
        self.assertIn('recovery:hover_clear_failed', result.issues)
        self.assertIsNone(result.board)
        self.capture.assert_called_once()
        self.wait.assert_not_called()

    def test_capture_failures_never_use_stale_observations(self):
        self.capture.side_effect = [None]
        result = self.observe()
        self.assertIn('recovery:capture_failed', result.issues)
        self.clear.assert_not_called()
        self.capture.side_effect = [self.first, None]
        result = self.observe()
        self.assertIn('recovery:recapture_failed', result.issues)
        self.assertIsNone(result.board)
        self.assertIsNone(result.ammo)
        self.assertFalse(result.ready_for_policy)

    def test_pointer_move_never_clicks_and_checks_bounds(self):
        desktop = Mock()
        rect = dict(left=100, top=200, width=640, height=360)
        with patch.object(game_input, 'find_window_geometry', return_value=rect), patch.object(game_input, 'pyautogui', desktop):
            self.assertTrue(game_input.move_relative_in_window('game', 10, 20))
            desktop.moveTo.assert_called_once_with(110, 220, duration=0.1)
            desktop.mouseDown.assert_not_called()
            desktop.mouseUp.assert_not_called()
            desktop.reset_mock()
            self.assertFalse(game_input.move_relative_in_window('game', 640, 20))
            desktop.moveTo.assert_not_called()
            desktop.moveTo.side_effect = RuntimeError('failed')
            self.assertFalse(game_input.move_relative_in_window('game', 10, 20))
        with patch.object(game_input, 'find_window_geometry', return_value=None), patch.object(game_input, 'pyautogui', desktop):
            desktop.reset_mock()
            self.assertFalse(game_input.move_relative_in_window('game', 10, 20))
            desktop.moveTo.assert_not_called()
