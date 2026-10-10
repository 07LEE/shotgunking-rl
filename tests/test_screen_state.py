"""Recognition rejects unsupported and contradictory evidence."""
from pathlib import Path
import sys
import tempfile
import json
import cv2
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.collection.screen_state import match, propose, propose_screen_kind


class ScreenStateRecognitionTests(unittest.TestCase):
    def test_screen_kind_prefers_overlays_and_requires_one_player_king(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        board[7, 4] = 1
        with patch('shotgun_king_rl.collection.screen_state.screens.check_retry_popup', return_value=True), patch('shotgun_king_rl.collection.screen_state.screens.check_card_selection_screen', return_value=True), patch('shotgun_king_rl.collection.screen_state.analyzer.get_state_matrix', return_value=board):
            self.assertEqual(propose_screen_kind(image), 'retry')
        with patch('shotgun_king_rl.collection.screen_state.screens.check_retry_popup', return_value=False), patch('shotgun_king_rl.collection.screen_state.screens.check_card_selection_screen', return_value=True), patch('shotgun_king_rl.collection.screen_state.analyzer.get_state_matrix', return_value=board):
            self.assertEqual(propose_screen_kind(image), 'card_selection')
        with patch('shotgun_king_rl.collection.screen_state.screens.check_retry_popup', return_value=False), patch('shotgun_king_rl.collection.screen_state.screens.check_card_selection_screen', return_value=False), patch('shotgun_king_rl.collection.screen_state.analyzer.get_state_matrix', return_value=board):
            self.assertEqual(propose_screen_kind(image), 'gameplay')
        board[7, 4] = 0
        with patch('shotgun_king_rl.collection.screen_state.screens.check_retry_popup', return_value=False), patch('shotgun_king_rl.collection.screen_state.screens.check_card_selection_screen', return_value=False), patch('shotgun_king_rl.collection.screen_state.analyzer.get_state_matrix', return_value=board):
            self.assertIsNone(propose_screen_kind(image))

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
        self.assertIsNone(result['floor'])
        self.assertTrue(all(value is None for value in result['screen_state']['ammo'].values()))
        self.assertTrue(all(value is None for value in result['screen_state']['combat_stats'].values()))
        self.assertEqual(result['screen_state']['cards'], {'left': None, 'right': None})
        self.assertIsNone(result['screen_state']['souls'])
        self.assertFalse(any(result['screen_state']['reviews'].values()))
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
            data = {'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'split': 'train', 'screen_state': {'ammo': {'loaded': 2, 'reserve': 6}, 'combat_stats': {'attack': 4}, 'cards': {'left': ['known_card']}, 'souls': ['knight'], 'reviews': {'ammo': True, 'combat_stats': True, 'cards': True, 'soul': True}}}
            annotation.write_text(json.dumps(data))
            result = propose(screen, collection)
            self.assertEqual(result['screen_state']['ammo']['loaded'], 2)
            self.assertEqual(result['screen_state']['combat_stats']['attack'], 4)
            self.assertEqual(result['screen_state']['cards']['left'][0], 'known_card')
            self.assertEqual(result['screen_state']['souls'], ['knight'])
            self.assertIsNone(propose(screen, collection, exclude=annotation)['screen_state']['combat_stats']['attack'])
            self.assertIsNone(propose(screen, collection, exclude=annotation)['screen_state']['souls'])
            data['split'] = 'test'
            annotation.write_text(json.dumps(data))
            self.assertIsNone(propose(screen, collection)['screen_state']['combat_stats']['attack'])
            data['split'] = 'train'
            data['screen_state']['reviews'] = {'ammo': False, 'combat_stats': False, 'cards': False, 'soul': False}
            annotation.write_text(json.dumps(data))
            self.assertIsNone(propose(screen, collection)['screen_state']['combat_stats']['attack'])

    def test_confirmed_train_floor_is_proposed(self):
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(screen, '3', (685, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            session = collection / 'sessions/run-floor'
            (session / 'originals').mkdir(parents=True)
            (session / 'annotations').mkdir()
            cv2.imwrite(str(session / 'originals/screen.png'), screen)
            annotation = session / 'annotations/screen.json'
            annotation.write_text(json.dumps({'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'split': 'train', 'confirmed': True, 'floor': 3, 'screen_state': {}}))

            result = propose(screen, collection)

            self.assertEqual(result['floor'], 3)
            self.assertIsNone(propose(screen, collection, exclude=annotation)['floor'])
