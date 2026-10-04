"""Recognition rejects unsupported and contradictory evidence."""
from pathlib import Path
import sys
import tempfile
import json
import cv2
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.collection.screen_state import match, propose


class ScreenStateRecognitionTests(unittest.TestCase):
    def test_matches_supported_value_but_rejects_conflict(self):
        pattern = np.random.default_rng(10).integers(0, 255, (30, 40), dtype=np.uint8)
        self.assertEqual(match(pattern, [(4, pattern)])[0], 4)
        self.assertIsNone(match(pattern, [(4, pattern), (5, pattern)]))
        unrelated = np.random.default_rng(20).integers(0, 255, pattern.shape, dtype=np.uint8)
        self.assertIsNone(match(pattern, [(4, unrelated)]))
        self.assertIsNone(match(np.zeros_like(pattern), [(4, pattern)]))

    def test_no_examples_means_unknown_not_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            result = propose(np.zeros((720, 1280, 3), dtype=np.uint8), Path(directory))
        self.assertTrue(all(value is None for value in result['screen_state']['stats'].values()))
        self.assertEqual(result['screen_state']['cards'], {'left': None, 'right': None})
        self.assertFalse(result['screen_state']['reviewed'])
        self.assertEqual(result['prediction']['references'], [])

    def test_reviewed_train_reference_and_self_exclusion(self):
        screen = np.random.default_rng(30).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            session = collection / 'sessions/run-01'
            (session / 'originals').mkdir(parents=True)
            (session / 'annotations').mkdir()
            cv2.imwrite(str(session / 'originals/screen.png'), screen)
            annotation = session / 'annotations/screen.json'
            data = {'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'split': 'train', 'screen_state': {'reviewed': True, 'stats': {'attack': 4}, 'cards': {'left': ['known_card']}}}
            annotation.write_text(json.dumps(data))
            result = propose(screen, collection)
            self.assertEqual(result['screen_state']['stats']['attack'], 4)
            self.assertEqual(result['screen_state']['cards']['left'][0], 'known_card')
            self.assertIsNone(propose(screen, collection, exclude=annotation)['screen_state']['stats']['attack'])
            data['split'] = 'test'
            annotation.write_text(json.dumps(data))
            self.assertIsNone(propose(screen, collection)['screen_state']['stats']['attack'])
            data['split'] = 'train'
            data['screen_state']['reviewed'] = False
            annotation.write_text(json.dumps(data))
            self.assertIsNone(propose(screen, collection)['screen_state']['stats']['attack'])
