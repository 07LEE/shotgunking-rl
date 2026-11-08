"""Offline composed-state sequences; no captured game or desktop input."""

import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import env
from shotgun_king_rl.perception import AmmoReading, PerceptionPipeline, Prediction, ScreenKind
from shotgun_king_rl.perception.combat import combat_state_contract
from shotgun_king_rl.perception.grenade_damage import RetainedGrenadeDamageReader
from shotgun_king_rl.perception.hud import stat_patch
from shotgun_king_rl.perception.optional_zero import ZeroWhenNotDisplayedReader
from shotgun_king_rl.runtime.perception import build_live_pipeline
from tests.perception.test_visibility_proposals import hud
from tests.support.perception import FakeReader, NormalContextReader, make_reader


class DamageSource(FakeReader):
    def has_visible_title(self, image):
        return False


class CombatStateSequenceTests(unittest.TestCase):
    def setUp(self):
        image = hud()
        self.titles = [(name, stat_patch(image, offset)[:30]) for name, offset in
                       (('attack', 0), ('range', 64), ('spread_degrees', 128))]
        self.clean = image.copy()
        self.clean[380:650, 270:354] = 0
        self.occluded = self.clean.copy()
        for x in (280, 300, 320):
            cv2.rectangle(self.occluded, (x, 350), (x+11, 361), (80, 0, 255), -1)
        self.damage = DamageSource(Prediction.unknown(source='direct:test', reason='grenade_damage_no_match'))
        self.retained = RetainedGrenadeDamageReader(self.damage, visibility_titles=self.titles)
        self.combat = make_reader()
        self.combat.grenade_damage_reader = self.retained
        for field, name in (('penetration_percent', 'penetration_reader'),
                            ('knockback_percent', 'knockback_reader'), ('melee_damage', 'melee_reader')):
            setattr(self.combat, name, ZeroWhenNotDisplayedReader(
                getattr(self.combat, name), field, visibility_titles=self.titles))
        board = np.zeros((8, 8), np.int8)
        board[7, 3] = 1
        self.screen = FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen:test'))
        self.pipeline = PerceptionPipeline(self.screen,
            FakeReader(Prediction.accepted(board, source='board:test')),
            FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo:test')),
            self.combat, NormalContextReader(board))
        # Only board-layout recognition is stubbed. The actual visibility
        # classifier checks primary titles, blank optional rows and hearts.
        check = patch('shotgun_king_rl.perception.visibility_proposals.check_gameplay_board', return_value=True)
        check.start()
        self.addCleanup(check.stop)

    def accept_damage(self, value):
        self.damage.prediction = Prediction.accepted(value, source='direct:test', confidence=.99)
        return self.pipeline.observe(self.clean)

    def hide_damage(self):
        self.damage.prediction = Prediction.unknown(source='direct:test', reason='grenade_damage_no_match')

    def test_cold_start_observe_retain_card_retry_and_new_observation(self):
        self.assertFalse(self.pipeline.observe(self.clean).ready_for_policy)
        self.assertTrue(self.accept_damage(2).ready_for_policy)
        self.hide_damage()
        remembered = self.pipeline.observe(self.clean)
        self.assertTrue(remembered.ready_for_policy)
        self.assertEqual(remembered.combat_stats.grenade_damage, 2)
        self.assertTrue(self.pipeline.memory_info()['used_memory'])
        for screen in (ScreenKind.CARD_SELECTION, ScreenKind.RETRY):
            self.screen.prediction = Prediction.accepted(screen, source='screen:test')
            self.assertFalse(self.pipeline.observe(self.clean).ready_for_policy)
            self.assertIsNone(self.pipeline.memory_info()['value'])
            self.screen.prediction = Prediction.accepted(ScreenKind.GAMEPLAY, source='screen:test')
            self.assertFalse(self.pipeline.observe(self.clean).ready_for_policy)
            self.assertTrue(self.accept_damage(3).ready_for_policy)
            self.hide_damage()
            self.assertEqual(self.pipeline.observe(self.clean).combat_stats.grenade_damage, 3)

    def test_occlusion_recaptures_once_then_uses_only_verified_hidden_damage(self):
        self.assertTrue(self.accept_damage(2).ready_for_policy)
        self.hide_damage()
        capture = Mock(side_effect=[self.occluded, self.clean])
        clear, wait = Mock(return_value=True), Mock()
        result = self.pipeline.observe_live(capture, clear, wait)
        self.assertTrue(result.ready_for_policy)
        self.assertEqual(result.combat_stats.grenade_damage, 2)
        self.assertEqual(capture.call_count, 2)
        clear.assert_called_once()
        wait.assert_called_once()

    def test_other_unknown_after_recapture_cannot_borrow_memory_or_old_values(self):
        self.accept_damage(2)
        self.hide_damage()
        self.combat.knockback_reader.reader.prediction = Prediction.unknown(source='knockback:test', reason='no_state_evidence')
        capture = Mock(side_effect=[self.occluded, self.clean])
        result = self.pipeline.observe_live(capture, Mock(return_value=True), Mock())
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertIn('knockback_percent:no_state_evidence', ';'.join(result.issues))
        self.assertTrue(self.pipeline.memory_info()['used_memory'])
        self.assertEqual(capture.call_count, 2)

    def test_missing_capture_or_references_blocks_complete_state(self):
        self.accept_damage(2)
        self.hide_damage()
        result = self.pipeline.observe_live(Mock(side_effect=[self.occluded, None]), Mock(return_value=True), Mock())
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertIn('recovery:recapture_failed', result.issues)
        self.retained.visibility_titles = ()
        self.assertFalse(self.pipeline.observe(self.clean).ready_for_policy)
        self.assertFalse(self.pipeline.memory_info()['used_memory'])

    def test_contract_and_reference_metadata_survive_environment_info(self):
        with tempfile.TemporaryDirectory() as root:
            _, metadata = build_live_pipeline(root)
        self.assertEqual(metadata['combat_state_contract'], combat_state_contract())
        game = env.ShotgunKingEnv(perception_pipeline=self.pipeline)
        game.perception_metadata = metadata
        game.last_perception = self.accept_damage(2)
        info = game._perception_info()
        self.assertEqual(info['reader_metadata']['combat_state_contract']['retained_fields'], ['grenade_damage'])
        self.assertEqual(info['grenade_damage_memory']['observed_source'], 'direct:test')
        changed = combat_state_contract()
        changed['required_fields'].clear()
        self.assertEqual(len(combat_state_contract()['required_fields']), 9)
