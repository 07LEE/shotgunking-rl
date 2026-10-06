"""Offline checks for action completion evidence."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.runtime.execution import CompletionStatus, verify_action_completion


class RuntimeExecutionTests(unittest.TestCase):
    def test_move_requires_the_king_at_the_requested_destination(self):
        before = np.zeros((8, 8), dtype=np.float32)
        after = np.zeros((8, 8), dtype=np.float32)
        before[4, 4] = 1
        after[4, 5] = 1

        result = verify_action_completion(4, before, after, (2, 6), (2, 6), destination=(4, 5))

        self.assertTrue(result.confirmed)
        self.assertEqual(result.reason, "king_reached_destination")

    def test_unchanged_board_does_not_confirm_move(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1

        result = verify_action_completion(4, board, board.copy(), (2, 6), (2, 6), destination=(4, 5))

        self.assertEqual(result.status, CompletionStatus.UNCHANGED)

    def test_reload_and_shot_require_visible_ammunition_changes(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[4, 4] = 1

        reload_result = verify_action_completion(8, board, board.copy(), (0, 6), (2, 4))
        shot_result = verify_action_completion(9, board, board.copy(), (2, 4), (1, 4))

        self.assertTrue(reload_result.confirmed)
        self.assertTrue(shot_result.confirmed)


if __name__ == "__main__":
    unittest.main()
