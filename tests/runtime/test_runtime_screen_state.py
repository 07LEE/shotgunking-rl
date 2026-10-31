"""Offline checks for explicit runtime screen classification."""

import unittest

from shotgun_king_rl.runtime.transitions import (
    RuntimeScreenState,
    apply_screen_state_rules,
    classify_runtime_screen,
)


class RuntimeScreenStateTests(unittest.TestCase):
    def test_explicit_overlays_classify_terminal_states(self):
        defeat = classify_runtime_screen(
            capture_succeeded=True,
            king_count=0,
            enemy_count=3,
            retry_detected=True,
        )
        victory = classify_runtime_screen(
            capture_succeeded=True,
            king_count=0,
            enemy_count=0,
            card_selection_detected=True,
        )

        self.assertEqual(defeat.state, RuntimeScreenState.DEFEAT)
        self.assertEqual(victory.state, RuntimeScreenState.VICTORY)

    def test_missing_king_waits_before_becoming_observation_failure(self):
        waiting = classify_runtime_screen(
            capture_succeeded=True,
            king_count=0,
            enemy_count=4,
        )
        exhausted = classify_runtime_screen(
            capture_succeeded=True,
            king_count=0,
            enemy_count=4,
            transition_exhausted=True,
        )

        self.assertEqual(waiting.state, RuntimeScreenState.TRANSITION)
        self.assertEqual(exhausted.state, RuntimeScreenState.OBSERVATION_FAILURE)

    def test_capture_failure_is_not_a_defeat(self):
        assessment = classify_runtime_screen(
            capture_succeeded=False,
            king_count=0,
            enemy_count=0,
        )
        terminal = apply_screen_state_rules(-2.0, 0, assessment)

        self.assertEqual(assessment.state, RuntimeScreenState.OBSERVATION_FAILURE)
        self.assertFalse(terminal.terminated)
        self.assertEqual(terminal.reward, -2.0)

    def test_ready_and_cleared_boards_are_distinct(self):
        gameplay = classify_runtime_screen(
            capture_succeeded=True,
            king_count=1,
            enemy_count=2,
        )
        victory = classify_runtime_screen(
            capture_succeeded=True,
            king_count=1,
            enemy_count=0,
        )

        self.assertEqual(gameplay.state, RuntimeScreenState.GAMEPLAY)
        self.assertEqual(victory.state, RuntimeScreenState.VICTORY)


if __name__ == "__main__":
    unittest.main()
