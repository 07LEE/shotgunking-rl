"""User-confirmed absent-zero semantics must never apply to an occluded panel."""

import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shotgun_king_rl.perception import Prediction, PredictionStatus
from shotgun_king_rl.perception.hud import stat_patch
from shotgun_king_rl.perception.optional_zero import ZeroWhenNotDisplayedReader, ZERO_WHEN_NOT_DISPLAYED
from shotgun_king_rl.runtime.perception import build_live_pipeline
from shotgun_king_rl.collection.combat_evaluation import build_combat_reader
from tests.perception.test_visibility_proposals import hud
from tests.support.perception import FakeReader


class OptionalZeroReaderTests(unittest.TestCase):
    def setUp(self):
        self.original = hud()
        self.titles = [(name, stat_patch(self.original, offset)[:30]) for name, offset in
                       (('attack', 0), ('range', 64), ('spread_degrees', 128))]
        self.clean = self.original.copy()
        self.clean[380:650, 270:354] = 0
        gate = patch('shotgun_king_rl.perception.visibility_proposals.check_gameplay_board', return_value=True)
        gate.start()
        self.addCleanup(gate.stop)

    def reader(self, field, result=None):
        result = result or Prediction.unknown(source='raw:test', reason=field.removesuffix('_percent') + '_no_match')
        return ZeroWhenNotDisplayedReader(FakeReader(result), field, visibility_titles=self.titles)

    def test_absent_optional_rows_on_readable_panel_are_zero_for_only_three_fields(self):
        for field in ZERO_WHEN_NOT_DISPLAYED:
            with self.subTest(field=field):
                result = self.reader(field).read(self.clean)
                self.assertTrue(result.is_accepted)
                self.assertEqual(result.value, 0)
                self.assertEqual(result.source, field + ':clean-panel-zero-v1')
        for field in ('grenade_damage', 'grenades_remaining', 'attack'):
            with self.assertRaises(ValueError):
                self.reader(field)

    def test_piece_tooltip_covering_attack_is_unknown_not_zero(self):
        obscured = self.clean.copy()
        obscured[180:380, 270:354] = 0
        hearts = self.clean.copy()
        for x in (280, 300, 320):
            cv2.rectangle(hearts, (x, 350), (x+11, 361), (80, 0, 255), -1)
        for image in (obscured, hearts, np.zeros_like(self.clean), None):
            for field in ZERO_WHEN_NOT_DISPLAYED:
                with self.subTest(field=field, invalid=image is None):
                    self.assertFalse(self.reader(field).read(image).is_accepted)

    def test_unknown_optional_title_and_missing_title_bank_do_not_authorize_zero(self):
        for field in ZERO_WHEN_NOT_DISPLAYED:
            self.assertFalse(self.reader(field).read(self.original).is_accepted)
            subject = self.reader(field)
            subject.visibility_titles = ()
            self.assertFalse(subject.read(self.clean).is_accepted)

    def test_displayed_but_unreadable_never_becomes_zero(self):
        for field in ZERO_WHEN_NOT_DISPLAYED:
            with self.subTest(field=field):
                subject = self.reader(field)
                subject.visibility_titles += ((field, stat_patch(self.original, 192)[:30]),)
                result = subject.read(self.original)
                self.assertFalse(result.is_accepted)
                self.assertIsNone(result.value)

    def test_direct_positive_is_preserved_and_conflicting_absence_rejected(self):
        for field in ZERO_WHEN_NOT_DISPLAYED:
            direct = Prediction.accepted(40, source='raw:test', confidence=.99)
            subject = self.reader(field, direct)
            subject.visibility_titles += ((field, stat_patch(self.original, 192)[:30]),)
            self.assertEqual(subject.read(self.original), direct)
            self.assertEqual(subject.read(self.clean).status, PredictionStatus.REJECTED)

    def test_rejected_ambiguous_and_unknown_frames_do_not_reuse_prior_zero(self):
        for field in ZERO_WHEN_NOT_DISPLAYED:
            subject = self.reader(field)
            self.assertTrue(subject.read(self.clean).is_accepted)
            self.assertFalse(subject.read(self.original).is_accepted)
            subject.reader.prediction = Prediction.unknown(source='raw:test', reason='ambiguous')
            self.assertFalse(subject.read(self.clean).is_accepted)
            subject.reader.prediction = Prediction.rejected(None, source='raw:test', reason='conflict')
            self.assertEqual(subject.read(self.clean).status, PredictionStatus.REJECTED)

    def test_live_factory_only_and_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            offline, _ = build_combat_reader(root)
            live, metadata = build_live_pipeline(root)
        self.assertNotIsInstance(offline.penetration_reader, ZeroWhenNotDisplayedReader)
        self.assertIsInstance(live.combat_stats_reader.penetration_reader, ZeroWhenNotDisplayedReader)
        self.assertIsInstance(live.combat_stats_reader.knockback_reader, ZeroWhenNotDisplayedReader)
        self.assertIsInstance(live.combat_stats_reader.melee_reader, ZeroWhenNotDisplayedReader)
        contract = metadata['combat_state_contract']
        self.assertEqual(contract['zero_when_not_displayed_fields'], list(ZERO_WHEN_NOT_DISPLAYED))
        self.assertFalse(contract['occluded_or_displayed_unreadable_means_zero'])
        self.assertEqual('hud-clean-panel-optional-zero-v1', metadata['readers']['melee_reader']['zero_rule_source'])
