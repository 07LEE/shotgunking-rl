"""Offline regression tests for legacy environment state handling."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import env


def observation(board, threat=None, marker=0.0):
    result = np.zeros(281, dtype=np.float32)
    result[:64] = np.asarray(board, dtype=np.float32).reshape(-1)
    if threat is not None:
        result[64:128] = np.asarray(threat, dtype=np.float32).reshape(-1)
    result[128] = marker
    return result


class EnvironmentStateTests(unittest.TestCase):
    def test_failed_capture_returns_empty_observation(self):
        game = env.ShotgunKingEnv()
        with patch.object(game, "_check_emergency_stop"), patch.object(
            env, "capture_screen", return_value=False
        ), patch.object(env.cv2, "imread") as imread:
            result = game._get_obs()
        np.testing.assert_array_equal(result, np.zeros(281, dtype=np.float32))
        imread.assert_not_called()

    def test_action_mask_uses_board_threat_ammo_and_range(self):
        game = env.ShotgunKingEnv()
        board = np.zeros((8, 8), dtype=np.float32)
        threat = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[3, 4] = 2
        board[4, 3] = 2
        threat[3, 5] = 1
        game.current_state = observation(board, threat)
        game.loaded_ammo = 1
        game.reserve_ammo = 3

        mask = game.get_action_mask()

        self.assertEqual(mask.shape, (10,))
        self.assertEqual(mask[1], 0)  # occupied
        self.assertEqual(mask[2], 0)  # threatened
        self.assertEqual(mask[3], 0)  # occupied
        self.assertEqual(mask[4], 1)  # clear and safe
        self.assertEqual(mask[8], 1)  # reload available
        self.assertEqual(mask[9], 1)  # visible target in range

    def test_retry_popup_is_handled_once_and_returns_final_state(self):
        game = env.ShotgunKingEnv(max_steps=10)
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1
        board[0, 0] = 2
        before = observation(board, marker=1)
        equilibrium = observation(board, marker=2)
        real = observation(board, marker=3)
        final = observation(board, marker=4)
        game.current_state = before

        with patch.object(game, "_check_emergency_stop"), patch.object(
            game, "_get_obs", side_effect=[before, real, final]
        ), patch.object(
            game, "_wait_for_equilibrium", return_value=equilibrium
        ), patch.object(
            env, "click_relative_in_window"
        ) as click, patch.object(
            env, "check_retry_popup", return_value=True
        ), patch.object(
            env.os.path, "exists", return_value=True
        ), patch.object(
            env.cv2, "imread", return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
        ), patch.object(env.time, "sleep"):
            result, reward, terminated, truncated, info = game.step(4)

        np.testing.assert_array_equal(result, final)
        np.testing.assert_array_equal(game.current_state, final)
        self.assertEqual(reward, -16.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], 4)
        self.assertEqual(click.call_count, 2)  # move click and one retry click


if __name__ == "__main__":
    unittest.main()
