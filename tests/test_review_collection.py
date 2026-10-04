"""Offline preparation and review persistence checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shotgun_king_rl.collection.images import detect_crop
from shotgun_king_rl.collection.prepare import prepare_session
from shotgun_king_rl.review.server import load_item, save_item


@unittest.skipUnless((ROOT / 'data/test_fixtures/board_01.png').exists() and (ROOT / 'assets/pieces').is_dir(), 'Private fixtures required')
class ReviewCollectionTests(unittest.TestCase):
    def test_prepare_preserves_review_and_checks_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            inbox = collection / 'inbox'
            inbox.mkdir()
            image = cv2.imread(str(ROOT / 'data/test_fixtures/board_01.png'))
            cv2.imwrite(str(inbox / 'screen.png'), image)
            result = prepare_session(collection, 'test-session')
            self.assertEqual(result['created'], 1)
            self.assertEqual(result['errors'], [])
            relative = 'sessions/test-session/annotations/screen.png.json'
            path, data, _, revision = load_item(collection, relative)
            payload = dict(data, id=relative, revision=revision, floor=1, confirmed=True)
            payload['board'][0][6] = 'special_knight'
            payload['screen_state'] = {'stats': {'attack': 4}, 'cards': {'left': ['unknown']}, 'locked_cells': [[1, 2]], 'reviewed': True}
            saved = save_item(collection, payload)
            self.assertTrue(saved['confirmed'])
            self.assertEqual(json.loads(path.read_text())['screen_state']['stats']['attack'], 4)
            self.assertEqual(json.loads(path.read_text())['board'][0][6], 'special_knight')
            raw = path.read_bytes()
            self.assertEqual(prepare_session(collection, 'test-session')['created'], 0)
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, 'changed'):
                save_item(collection, payload)
            payload['revision'] = saved['revision']
            payload['confirmed'] = False
            payload['exclude_cells'] = [[0, 3]]
            save_item(collection, payload)
            self.assertFalse(json.loads(path.read_text())['confirmed'])
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                load_item(collection, '../../escape.json')

    def test_missing_floor_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            (collection / 'inbox').mkdir()
            image = cv2.imread(str(ROOT / 'data/test_fixtures/board_01.png'))
            cv2.imwrite(str(collection / 'inbox/screen.png'), image)
            prepare_session(collection, 'test-session')
            relative = 'sessions/test-session/annotations/screen.png.json'
            _, data, _, revision = load_item(collection, relative)
            with self.assertRaisesRegex(ValueError, 'floor'):
                save_item(collection, dict(data, id=relative, revision=revision, confirmed=True))


class ScreenStateTests(unittest.TestCase):
    def test_unknown_and_empty_cards_are_distinct(self):
        from shotgun_king_rl.review.server import validate_screen_state
        unknown = validate_screen_state({})
        self.assertIsNone(unknown['stats']['attack'])
        self.assertIsNone(unknown['cards']['left'])
        reviewed = validate_screen_state({'stats': {'attack': 4, 'range_min': 3, 'range_max': 5}, 'cards': {'left': ['unknown'], 'right': []}, 'locked_cells': [[1, 2]], 'reviewed': True})
        self.assertEqual(reviewed['locked_cells'], [[1, 2]])
        self.assertEqual(reviewed['cards']['right'], [])

    def test_invalid_state_is_rejected(self):
        from shotgun_king_rl.review.server import validate_screen_state
        for value in ({'stats': {'attack': -1}}, {'stats': {'attack': True}}, {'stats': {'range_min': 5, 'range_max': 3}}, {'cards': {'left': 'knight'}}, {'locked_cells': [[8, 2]]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_screen_state(value)
