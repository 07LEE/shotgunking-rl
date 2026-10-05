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
            payload['screen_state'] = {
                'screen_kind': 'gameplay',
                'ammo': {'loaded': 2, 'reserve': 6},
                'combat_stats': {'attack': 4},
                'cards': {'left': ['unknown']},
                'locked_cells': [[1, 2]],
                'reviews': {'screen_kind': True, 'ammo': True, 'combat_stats': True, 'cards': True},
            }
            saved = save_item(collection, payload)
            self.assertTrue(saved['confirmed'])
            stored_state = json.loads(path.read_text())['screen_state']
            self.assertEqual(stored_state['combat_stats']['attack'], 4)
            self.assertEqual(stored_state['ammo'], {'loaded': 2, 'reserve': 6})
            self.assertTrue(all(stored_state['reviews'].values()))
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
        self.assertIsNone(unknown['combat_stats']['attack'])
        self.assertIsNone(unknown['ammo']['loaded'])
        self.assertIsNone(unknown['cards']['left'])
        reviewed = validate_screen_state({'screen_kind': 'gameplay', 'ammo': {'loaded': 2, 'reserve': 6}, 'combat_stats': {'attack': 4, 'range_min': 3, 'range_max': 5}, 'cards': {'left': ['unknown'], 'right': []}, 'locked_cells': [[1, 2]], 'reviews': {'screen_kind': True, 'ammo': True, 'combat_stats': True, 'cards': True}})
        self.assertEqual(reviewed['locked_cells'], [[1, 2]])
        self.assertEqual(reviewed['cards']['right'], [])
        self.assertEqual(reviewed['screen_kind'], 'gameplay')
        self.assertTrue(all(reviewed['reviews'].values()))

    def test_legacy_review_flag_maps_without_inventing_screen_kind(self):
        from shotgun_king_rl.review.server import validate_screen_state
        state = validate_screen_state({'stats': {'ammo_loaded': 1, 'ammo_reserve': 4, 'attack': 3}, 'reviewed': True})
        self.assertEqual(state['ammo'], {'loaded': 1, 'reserve': 4})
        self.assertEqual(state['combat_stats']['attack'], 3)
        self.assertFalse(state['reviews']['screen_kind'])
        self.assertTrue(state['reviews']['ammo'])
        self.assertTrue(state['reviews']['combat_stats'])
        self.assertTrue(state['reviews']['cards'])

    def test_invalid_state_is_rejected(self):
        from shotgun_king_rl.review.server import validate_screen_state
        for value in ({'combat_stats': {'attack': -1}}, {'combat_stats': {'attack': True}}, {'combat_stats': {'range_min': 5, 'range_max': 3}}, {'ammo': {'loaded': -1}}, {'cards': {'left': 'knight'}}, {'locked_cells': [[8, 2]]}, {'screen_kind': 'menu'}, {'reviews': {'screen_kind': True}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_screen_state(value)
