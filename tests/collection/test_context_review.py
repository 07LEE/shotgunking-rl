"""Offline review exports all failures and preserves source evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from shotgun_king_rl.collection.context_review import export_context_review, source_cell_box
from shotgun_king_rl.perception import BoardContextReader, BoardContextReading, Prediction, ScreenKind
from shotgun_king_rl.perception.frame import PreparedFrame


class ContextReviewTests(unittest.TestCase):
    def test_exports_more_than_first_failure_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'originals').mkdir()
            (root / 'annotations').mkdir()
            image = np.random.default_rng(91).integers(0, 256, (720, 1280, 3), np.uint8)
            reference = PreparedFrame(image, normalized=True).board_image
            sides = np.zeros((8, 8), np.int8)
            sides[7, 3] = 1
            context = BoardContextReader([(reference, BoardContextReading(sides, False))])
            # Each normalized board cell maps to 64 source pixels.
            image[119:183, 383:447] = 0
            image[503:567, 639:703] = 0
            original = root / 'originals/session__frame.png'
            cv2.imwrite(str(original), image)
            annotation = root / 'annotations/session__frame.png.json'
            annotation.write_text(json.dumps({'image': '../originals/session__frame.png', 'session_id': 'session',
                'game_crop': [0, 0, 1280, 720], 'board': [['empty'] * 8 for _ in range(8)]}))
            before = (original.read_bytes(), annotation.read_bytes())
            board = np.zeros((8, 8), int)
            board[7, 3] = 1
            pieces = Mock()
            pieces.read_prepared.return_value = Prediction.accepted(board, source='test')
            screen = Mock()
            screen.read_prepared.return_value = Prediction.accepted(ScreenKind.GAMEPLAY, source='test')
            with patch('shotgun_king_rl.collection.context_review.BoardContextReader.from_collection', return_value=context), patch(
                'shotgun_king_rl.collection.context_review.TemplatePieceReader.from_collection', return_value=pieces
            ), patch('shotgun_king_rl.collection.context_review.OverlayScreenReader', return_value=screen):
                report = export_context_review(root, 'session', root / 'review')
            cells = report['records'][0]['cells']
            self.assertTrue({'a8', 'e2'}.issubset({cell['square'] for cell in cells}))
            self.assertEqual(before, (original.read_bytes(), annotation.read_bytes()))
            for cell in cells:
                self.assertTrue((root / 'review' / cell['patch']).is_file())
                self.assertTrue((root / 'review' / cell['neighborhood']).is_file())
            self.assertIn('data:image/png;base64,', (root / 'review/index.html').read_text())
            self.assertTrue((root / 'review/all-cells.png').is_file())
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                export_context_review(root, 'session', root / 'review')

    def test_source_coordinates_respect_crop_offset_and_scale(self):
        self.assertEqual(source_cell_box([0, 0, 1280, 720], 0, 0), [383, 119, 447, 183])
        self.assertEqual(source_cell_box([51, 124, 1330, 844], 0, 0), [433, 243, 498, 307])
