"""Refreshing drafts must preserve labels, approvals, and concurrent edits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shotgun_king_rl.collection.refresh import fill_unknowns, refresh_session


class RefreshCollectionTests(unittest.TestCase):
    def proposal(self):
        return {'floor': 2, 'screen_state': {'screen_kind': 'gameplay', 'ammo': {'loaded': 0, 'reserve': 8}, 'combat_stats': {'attack': 4}, 'souls': [], 'cards': {'left': None, 'right': None}}, 'prediction': {'method': 'fixture'}}

    def test_only_blank_unreviewed_values_change(self):
        data = {'confirmed': True, 'floor': None, 'board': [['ambiguous_royal']], 'screen_state': {'ammo': {'loaded': None, 'reserve': 4}, 'combat_stats': {'attack': None}, 'souls': None, 'reviews': {'combat_stats': True}}}
        changed = fill_unknowns(data, self.proposal())
        self.assertEqual(data['screen_state']['ammo'], {'loaded': 0, 'reserve': 4})
        self.assertIsNone(data['screen_state']['combat_stats']['attack'])
        self.assertIsNone(data['floor'])
        self.assertTrue(data['confirmed'])
        self.assertEqual(data['board'], [['ambiguous_royal']])
        self.assertNotIn('floor', changed)
        self.assertEqual(data['screen_state']['souls'], [])

    def fixture(self, root):
        path = root / 'sessions/ref/annotations/screen.json'
        path.parent.mkdir(parents=True)
        (path.parent.parent / 'originals').mkdir()
        cv2.imwrite(str(path.parent.parent / 'originals/screen.png'), np.zeros((720, 1280, 3), np.uint8))
        path.write_text(json.dumps({'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'confirmed': False, 'screen_state': {}}))
        return path

    def test_dry_run_is_read_only_and_existing_report_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            before = path.read_bytes()
            report = root / 'report.json'
            with patch('shotgun_king_rl.collection.refresh.propose', return_value=self.proposal()):
                result = refresh_session(root, 'ref', report)
                self.assertTrue(result['fields_filled'])
                self.assertEqual(path.read_bytes(), before)
                with self.assertRaises(FileExistsError):
                    refresh_session(root, 'ref', report, apply=True)
                self.assertEqual(path.read_bytes(), before)

    def test_apply_is_idempotent_and_detects_concurrent_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            with patch('shotgun_king_rl.collection.refresh.propose', return_value=self.proposal()):
                self.assertTrue(refresh_session(root, 'ref', root / 'one.json', apply=True)['fields_filled'])
                after = path.read_bytes()
                self.assertEqual(refresh_session(root, 'ref', root / 'two.json', apply=True)['fields_filled'], 0)
                self.assertEqual(path.read_bytes(), after)
            def change(*args, **kwargs):
                path.write_text('external update')
                return self.proposal()
            with patch('shotgun_king_rl.collection.refresh.propose', side_effect=change):
                result = refresh_session(root, 'ref', root / 'three.json', apply=True)
            self.assertEqual(len(result['errors']), 1)
            self.assertEqual(path.read_text(), 'external update')
