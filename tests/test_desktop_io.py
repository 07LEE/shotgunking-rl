"""Offline checks for structured desktop capture and input results."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import capture
import env
import input as game_input
from shotgun_king_rl.runtime.io import CaptureResult, IOStatus, InputResult


class DesktopIOTests(unittest.TestCase):
    def test_capture_refuses_full_monitor_fallback_when_window_is_missing(self):
        screen = Mock()
        screen.monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]
        context = Mock()
        context.__enter__ = Mock(return_value=screen)
        context.__exit__ = Mock(return_value=False)

        with patch.object(capture, "find_window_geometry", return_value=None), patch.object(capture.mss, "mss", return_value=context):
            result = capture.capture_screen("unused.png", "Shotgun King")

        self.assertEqual(result.status, IOStatus.FAILED)
        self.assertEqual(result.reason, "window_not_found")
        screen.grab.assert_not_called()

    def test_relative_click_reports_missing_window(self):
        with patch.object(game_input, "find_window_geometry", return_value=None):
            result = game_input.click_relative_in_window("Shotgun King", 10, 20)

        self.assertEqual(result.status, IOStatus.FAILED)
        self.assertEqual(result.reason, "window_not_found")
        self.assertEqual(result.target, (10, 20))

    def test_failed_move_does_not_commit_position_or_ammunition(self):
        game = env.ShotgunKingEnv(max_steps=10)
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        state = np.zeros(281, dtype=np.float32)
        state[:64] = board.reshape(-1)
        game.current_state = state
        loaded_before = game.loaded_ammo
        reserve_before = game.reserve_ammo
        failure = InputResult(IOStatus.FAILED, "click", "window_not_found", (733, 408))

        with patch.object(game, "_check_emergency_stop"), patch.object(game, "_get_obs", return_value=state), patch.object(env, "click_relative_in_window", return_value=failure):
            result, reward, terminated, truncated, info = game.step(4)

        np.testing.assert_array_equal(result, state)
        self.assertEqual((game.king_row, game.king_col), (4, 4))
        self.assertEqual(game.loaded_ammo, loaded_before)
        self.assertEqual(game.reserve_ammo, reserve_before)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["input_succeeded"])
        self.assertEqual(info["input"]["reason"], "window_not_found")

    def test_failed_click_releases_a_pressed_mouse_button(self):
        pointer = Mock()
        pointer.mouseUp.side_effect = [RuntimeError("release failed"), None]

        with patch.object(game_input, "pyautogui", pointer), patch.object(game_input.time, "sleep"):
            result = game_input.click_at(10, 20)

        self.assertEqual(result.status, IOStatus.FAILED)
        self.assertEqual(pointer.mouseUp.call_count, 2)

    def test_reset_stops_when_game_window_cannot_be_captured(self):
        game = env.ShotgunKingEnv()
        failure = CaptureResult(IOStatus.FAILED, "data/screenshot.png", "window_not_found")

        with patch.object(game, "_check_emergency_stop"), patch.object(env, "capture_screen", return_value=failure), patch.object(env, "click_relative_in_window") as click, patch.object(env.time, "sleep"):
            observation, info = game.reset()

        np.testing.assert_array_equal(observation, np.zeros(281, dtype=np.float32))
        self.assertFalse(info["reset_ready"])
        self.assertEqual(info["capture"]["reason"], "window_not_found")
        click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
