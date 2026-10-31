"""Allied pieces and terrain must survive review and dataset export."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.board_metadata import board_metadata, MOAT_MOVEMENT, terrain_from_moat, terrain_source
from shotgun_king_rl.collection.dataset import build_dataset
from shotgun_king_rl.review.server import load_item, save_item


class BoardMetadataTests(unittest.TestCase):
    def test_card_state_derives_fixed_row_and_keeps_unknown_distinct(self):
        for active in (True, False, None):
            with self.subTest(active=active):
                data = {'board': self.board(), 'moat_active': active,
                        'terrain': [['moat'] * 8 for _ in range(8)]}
                _, terrain = board_metadata(data)
                self.assertEqual(terrain, terrain_from_moat(active))
                self.assertEqual(sum(cell == 'moat' for row in terrain for cell in row), 8 if active is True else 0)
                self.assertEqual(terrain[0][0], 'unknown' if active is None else 'normal')
                self.assertEqual(terrain_source(data), 'unknown' if active is None else 'moat_card')
        for invalid in (0, 1, 'false', [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                terrain_from_moat(invalid)

    def board(self):
        board = [['empty'] * 8 for _ in range(8)]
        board[7][3] = 'player_king'
        board[6][4] = 'rook'
        return board

    def test_legacy_inference_preserves_unknown_terrain(self):
        sides, terrain = board_metadata({'board': self.board()})
        self.assertEqual(sides[7][3], 'ally')
        self.assertEqual(sides[6][4], 'enemy')
        self.assertIsNone(sides[0][0])
        self.assertEqual(terrain[4], ['unknown'] * 8)

    def test_invalid_metadata_and_enemy_controlled_king_are_rejected(self):
        board = self.board()
        sides, terrain = board_metadata({'board': board})
        for field, bad in [('terrain', []), ('piece_sides', [['ally'] * 8] * 8), ('terrain', [['water'] * 8] * 8)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                board_metadata({'board': board, field: bad})
        sides[7][3] = 'enemy'
        with self.assertRaisesRegex(ValueError, 'king must be allied'):
            board_metadata({'board': board, 'piece_sides': sides})

    def test_allied_rook_and_moat_survive_save_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            session = collection / 'sessions/run-01'
            (session / 'annotations').mkdir(parents=True)
            (session / 'originals').mkdir()
            cv2.imwrite(str(session / 'originals/screen.png'), np.zeros((720, 1280, 3), dtype=np.uint8))
            annotation = session / 'annotations/screen.json'
            annotation.write_text(json.dumps({
                'schema': 'piece-cells-v1', 'session_id': 'run-01', 'split': 'train', 'floor': 1,
                'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720],
                'confirmed': False, 'exclude_cells': [], 'board': self.board(),
                'screen_state': {'screen_kind': 'gameplay'},
            }))
            relative = 'sessions/run-01/annotations/screen.json'
            _, data, _, revision = load_item(collection, relative)
            sides, terrain = board_metadata(data)
            sides[6][4] = 'ally'
            terrain[4] = ['moat'] * 8
            data.update(id=relative, revision=revision, piece_sides=sides, terrain=terrain, confirmed=True, piece_reviewed=True)
            result = save_item(collection, data)
            stored = json.loads(annotation.read_text())
            self.assertEqual(stored['board'][6][4], 'rook')
            self.assertEqual(stored['piece_sides'][6][4], 'ally')
            self.assertEqual(stored['terrain'][4], ['moat'] * 8)
            output = collection / 'dataset'
            build_dataset(collection / 'sessions', output)
            records = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            rook = next(row for row in records if row['row'] == 6 and row['col'] == 4)
            self.assertEqual(rook['piece_side'], 'ally')
            self.assertFalse(rook['controlled'])
            self.assertEqual(next(row for row in records if row['row'] == 4)['terrain'], 'moat')
            self.assertEqual(MOAT_MOVEMENT['other_pieces'], 'stop_in_moat_before_continuing')
            # Metadata edits cannot silently retain an old board approval.
            data.update(revision=result['revision'], piece_reviewed=False)
            data['terrain'][4][0] = 'normal'
            self.assertFalse(save_item(collection, data)['confirmed'])
            # The server, not a stale client grid, determines card-derived terrain.
            _, saved_data, _, revision = load_item(collection, relative)
            saved_data.update(id=relative, revision=revision, moat_active=True, confirmed=True, piece_reviewed=True)
            saved_data['terrain'] = [['normal'] * 8 for _ in range(8)]
            saved = save_item(collection, saved_data)
            derived = json.loads(annotation.read_text())
            self.assertEqual(derived['terrain'], terrain_from_moat(True))
            self.assertEqual(derived['terrain_source'], 'moat_card')
            build_dataset(collection / 'sessions', collection / 'derived_dataset')
            records = [json.loads(line) for line in (collection / 'derived_dataset/manifest.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['moat_active'] and row['terrain_source'] == 'moat_card' for row in records))
            saved_data.update(revision=saved['revision'], moat_active=False, piece_reviewed=False)
            self.assertFalse(save_item(collection, saved_data)['confirmed'])
            self.assertEqual(json.loads(annotation.read_text())['terrain'], terrain_from_moat(False))


if __name__ == '__main__':
    unittest.main()
