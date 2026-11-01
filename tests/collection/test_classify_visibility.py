"""Batch drafts preserve numeric labels, manual visibility, approvals and test data."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2

from shotgun_king_rl.collection.classify_visibility import classify_visibility
from tests.support.collection import game_screen


class Classifier:
    source = 'visibility:test'

    def propose(self, image):
        return {'attack': 'visible', 'grenade_damage': 'occluded', 'melee_damage': 'unknown'}


class ClassifyVisibilityTests(unittest.TestCase):
    def test_preview_apply_and_repeat_do_not_modify_reviewed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for split in ('train', 'val', 'test'):
                session = root / 'sessions' / split
                (session / 'annotations').mkdir(parents=True)
                (session / 'originals').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), game_screen())
                path = session / 'annotations/screen.json'
                path.write_text(json.dumps({'split': split, 'image': '../originals/screen.png',
                                            'game_crop': [0, 0, 1280, 720], 'screen_state': {
                                                'screen_kind': 'gameplay', 'combat_stats': {'grenade_damage': 2},
                                                'combat_visibility': {'attack': 'unreadable'},
                                                'reviews': {'combat_stats': True}},
                                            'prediction': {'other': 'preserved'}}))
                paths.append(path)
            original = {path: path.read_bytes() for path in paths}
            preview = classify_visibility(root, classifier=Classifier())
            self.assertEqual(preview['test_skipped'], 1)
            self.assertTrue(all(path.read_bytes() == original[path] for path in paths))
            applied = classify_visibility(root, apply=True, classifier=Classifier())
            self.assertFalse(applied['errors'])
            for path in paths[:2]:
                before, after = json.loads(original[path]), json.loads(path.read_text())
                self.assertEqual(after['screen_state'], before['screen_state'])
                self.assertEqual(after['prediction']['other'], 'preserved')
                self.assertEqual(after['prediction']['combat_visibility'], {'grenade_damage': 'occluded'})
                self.assertTrue(after['prediction']['combat_visibility_provenance']['human_review_required'])
            self.assertEqual(paths[2].read_bytes(), original[paths[2]])
            repeat = classify_visibility(root, apply=True, classifier=Classifier())
            self.assertTrue(all(not row['changed'] for row in repeat['results']))
