"""Diagnostics separate image targets, state gaps and held-out data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shotgun_king_rl.collection.combat_diagnostics import diagnose_combat, main
from shotgun_king_rl.perception import Prediction
from shotgun_king_rl.perception.visibility import COMBAT_FIELDS
from tests.support.perception import make_reader


class CombatDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'annotations').mkdir()
        (self.root / 'originals').mkdir()
        cv2.imwrite(str(self.root / 'originals/run-01__screen.png'), np.zeros((720, 1280, 3), dtype=np.uint8))
        self.reader = make_reader()
        self.data = {'split': 'train', 'image': '../originals/run-01__screen.png',
                     'screen_state': {'screen_kind': 'gameplay',
                                      'reviews': {'screen_kind': True, 'combat_stats': True},
                                      'combat_stats': {'attack': 4, 'range_min': 3, 'range_max': 5, 'spread_degrees': 65},
                                      'combat_visibility': {'attack': 'visible', 'range_min': 'visible', 'range_max': 'visible', 'spread_degrees': 'visible', 'penetration_percent': 'not_displayed'}}}
        self.path = self.root / 'annotations/run-01__screen.json'
        self.path.write_text(json.dumps(self.data))
        for name, value in [('build_combat_reader', (self.reader, {})),
                            ('combat_visibility_titles', []),
                            ('propose_combat_visibility', dict.fromkeys(COMBAT_FIELDS, 'unknown'))]:
            mock = patch(f'shotgun_king_rl.collection.combat_diagnostics.{name}', return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_reports_missing_state_without_zero_and_preserves_data(self):
        self.reader.penetration_reader.prediction = Prediction.unknown(source='test', reason='no_match')
        raw = self.path.read_bytes()
        report = diagnose_combat(self.root)
        detail = report['samples'][0]['fields']['penetration_percent']
        self.assertEqual(detail['category'], 'not_displayed_requires_state_evidence')
        self.assertIsNone(detail['predicted'])
        self.assertIsNone(detail['expected_visual'])
        self.assertEqual(report['composed']['train']['samples'], 1)
        self.assertEqual(report['composed']['train']['accepted'], 0)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertTrue(report['support_contract']['grenade_damage_required_at_zero_count'])

    def test_field_accuracy_and_range_labels_are_separate_from_unlabeled_acceptance(self):
        result = self.reader.attack_reader.prediction
        self.data['screen_state']['combat_stats']['attack'] = result.value + 1
        self.path.write_text(json.dumps(self.data))
        report = diagnose_combat(self.root)
        self.assertEqual(report['counts']['train:attack']['incorrect'], 1)
        self.assertEqual(report['counts']['train:range_min']['correct'], 1)
        self.assertEqual(report['counts']['train:range_max']['correct'], 1)
        self.assertEqual(report['counts']['train:knockback_percent']['accepted_without_visual_target'], 1)

    def test_test_and_unreviewed_images_are_never_opened(self):
        for split, reviewed in [('test', True), ('val', False)]:
            data = json.loads(json.dumps(self.data))
            data['split'] = split
            data['image'] = '../originals/missing.png'
            data['screen_state']['reviews']['combat_stats'] = reviewed
            (self.root / f'annotations/{split}__screen.json').write_text(json.dumps(data))
        self.assertEqual(len(diagnose_combat(self.root)['samples']), 1)
        for splits in [('test',), (), ('val', 'val')]:
            with self.assertRaises(ValueError):
                diagnose_combat(self.root, splits=splits)

    def test_cli_refuses_existing_report_before_evaluation(self):
        output = self.root / 'existing.json'
        output.write_text('preserve')
        with patch('sys.argv', ['diagnose', '--output', str(output)]), self.assertRaises(SystemExit):
            main()
        self.assertEqual(output.read_text(), 'preserve')

    def test_state_rule_targets_are_not_counted_as_visible_zero_digits(self):
        from shotgun_king_rl.perception.optional_zero import ZeroWhenNotDisplayedReader
        self.reader.penetration_reader = ZeroWhenNotDisplayedReader(
            self.reader.penetration_reader, 'penetration_percent')
        with patch('shotgun_king_rl.collection.combat_diagnostics.connect_optional_zero_readers'), patch(
                'shotgun_king_rl.perception.optional_zero.propose_combat_visibility',
                return_value={'penetration_percent': 'not_displayed'}):
            report = diagnose_combat(self.root, state_contract=True)
        detail = report['samples'][0]['fields']['penetration_percent']
        self.assertEqual(detail['predicted'], 0)
        self.assertEqual(detail['expected_state'], 0)
        self.assertTrue(detail['state_correct'])
        self.assertIsNone(detail['expected_visual'])
        self.assertIsNone(detail['correct'])
        self.assertEqual('hud-clean-panel-optional-zero-v1', detail['state_target_source'])
        self.assertEqual(report['counts']['train:penetration_percent']['rule_derived_zero'], 1)
        self.assertNotIn('correct', report['counts']['train:penetration_percent'])
