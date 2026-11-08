"""Static advice respects perception gates and newly exposed attack rays."""

from dataclasses import replace
import unittest

import numpy as np

from shotgun_king_rl.perception.types import AmmoReading, BoardContextReading, CombatStatsReading, FusedPerception, ScreenKind
from shotgun_king_rl.runtime.advice import build_manual_advice, format_manual_advice


def observed(board, *, loaded=1):
    sides = np.where(board == 0, 0, 2).astype(np.int32)
    sides[board == 1] = 1
    return FusedPerception(
        ScreenKind.GAMEPLAY, board, AmmoReading(loaded, 5),
        CombatStatsReading(4, 3, 5, 65, 0, 0, 0, 0, 0), True, (), (),
        BoardContextReading(sides, False))


class ManualAdviceTests(unittest.TestCase):
    def test_unknown_or_non_gameplay_observation_has_no_candidates(self):
        board = np.zeros((8, 8), int)
        board[4, 4] = 1
        for state in (replace(observed(board), ready_for_policy=False, issues=('combat_stats:unknown',)),
                      replace(observed(board), screen_kind=ScreenKind.CARD_SELECTION),
                      replace(observed(board), board_context=None),
                      replace(observed(board), combat_stats=None)):
            advice = build_manual_advice(state)
            self.assertTrue(advice['blocked'])
            self.assertEqual(advice['moves'], [])
            self.assertIsNone(advice['shot'])
            self.assertIn('추천 보류', format_manual_advice(advice))

    def test_missing_combat_only_permits_movement_and_reload_without_filling_stats(self):
        board = np.zeros((8, 8), int)
        board[4, 4] = 1
        state = replace(observed(board, loaded=0), combat_stats=None, ready_for_policy=False,
                        issues=('combat_stats:grenade_damage:unknown',))
        result = build_manual_advice(state, max_ammo=2)
        self.assertFalse(result['blocked'])
        self.assertTrue(result['moves'])
        self.assertEqual(result['reload'], {'shells': 2, 'priority': True})
        self.assertIsNone(result['shot'])
        self.assertIn('사격 보류', format_manual_advice(result))
        self.assertFalse(state.ready_for_policy)
        self.assertIsNone(state.combat_stats)
        for extra in ('board:unknown', 'board_context:unknown', 'ammo:unknown', 'recovery:capture_failed'):
            invalid = replace(state, issues=(*state.issues, extra))
            self.assertTrue(build_manual_advice(invalid)['blocked'])

    def test_occlusion_message_names_the_square_and_refresh_condition(self):
        board = np.zeros((8, 8), int)
        board[4, 4] = 1
        state = replace(observed(board), ready_for_policy=False, board_context=None, combat_stats=None,
                        issues=('board_context:piece_sides:indicator_occlusion:5,2', 'combat_stats:unknown'))
        result = build_manual_advice(state)
        self.assertTrue(result['blocked'])
        message = format_manual_advice(result)
        self.assertIn('c3:', message)
        self.assertIn('표시가 사라진 뒤 다시 확인', message)
        self.assertNotIn('전투 수치 확인 필요', message)
        self.assertEqual(result['moves'], [])

    def test_vacating_king_reveals_rook_ray_and_does_not_mutate_input(self):
        board = np.zeros((8, 8), int)
        board[4, 4], board[4, 0] = 1, 5
        original = board.copy()
        result = build_manual_advice(observed(board))
        self.assertNotIn('f4', [row['square'] for row in result['moves']])
        self.assertTrue(result['current_threat'])
        self.assertIsNone(result['reload'])
        np.testing.assert_array_equal(board, original)

    def test_reload_needs_confirmed_capacity_and_unthreatened_position(self):
        board = np.zeros((8, 8), int)
        board[4, 4] = 1
        state = observed(board, loaded=0)
        self.assertIsNone(build_manual_advice(state)['reload'])
        result = build_manual_advice(state, max_ammo=2)
        self.assertEqual(result['reload'], {'shells': 2, 'priority': True})
        self.assertIsNone(result['shot'])
        with self.assertRaises(ValueError):
            build_manual_advice(state, max_ammo=0)

    def test_shot_is_separate_from_ranked_moves_and_never_claims_a_kill(self):
        board = np.zeros((8, 8), int)
        board[4, 4], board[4, 6] = 1, 2
        result = build_manual_advice(observed(board))
        self.assertEqual(result['shot']['square'], 'g4')
        self.assertTrue(all(row['action'] < 8 for row in result['moves']))
        self.assertIn('미평가', format_manual_advice(result))


if __name__ == '__main__':
    unittest.main()
