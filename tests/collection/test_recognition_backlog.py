"""Backlog export is read-only and keeps held-out visual groups separate."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from shotgun_king_rl.collection.backlog import build_backlog


class RecognitionBacklogTests(unittest.TestCase):
    def source(self, root, name, split, kind):
        path = root / f'sessions/{name}/annotations/screen.json'
        path.parent.mkdir(parents=True)
        (path.parent.parent / 'originals').mkdir()
        image = np.random.default_rng(32).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(str(path.parent.parent / 'originals/screen.png'), image)
        board = [['empty'] * 8 for _ in range(8)]
        board[0][0] = 'ambiguous_royal'
        path.write_text(json.dumps({'split': split, 'confirmed': False, 'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'board': board, 'screen_state': {'screen_kind': kind, 'reviews': {}}}))
        return path

    def test_export_preserves_annotations_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.source(root, 'one', 'train', 'gameplay')
            before = path.read_bytes()
            with patch('shotgun_king_rl.collection.backlog.analyzer.crop_chessboard', return_value=np.full((520, 520, 3), 100, np.uint8)):
                result = build_backlog(root, root / 'report')
            self.assertEqual(result['summary']['ambiguous_royal_cells'], 1)
            self.assertTrue((root / 'report' / result['royal_cells'][0]['crop']).is_file())
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                build_backlog(root, root / 'report')

    def test_card_visual_groups_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.source(root, 'one', 'train', 'card_selection')
            self.source(root, 'two', 'test', 'card_selection')
            result = build_backlog(root, root / 'report')
            self.assertEqual(result['summary']['card_choices_needing_review'], 8)
            self.assertEqual({group['split'] for group in result['visual_groups']}, {'train', 'test'})
            for group in result['visual_groups']:
                sessions = {item['annotation'].split('/')[1] for item in group['occurrences']}
                self.assertEqual(len(sessions), 1)
