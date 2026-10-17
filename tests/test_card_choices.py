"""Offline extraction checks for unlabeled card-choice crops."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.collection.card_choices import CHOICE_BOXES, extract_card_choices


def source(root, reviewed=True, split='test', layout=True):
    session = root / 'sessions/run-01'
    (session / 'originals').mkdir(parents=True)
    (session / 'annotations').mkdir()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    if layout:
        for i, (_, _, (left, top, right, bottom)) in enumerate(CHOICE_BOXES):
            image[top:bottom, left:right] = 230 + i * 5
    cv2.imwrite(str(session / 'originals/screen.png'), image)
    data = {
        'session_id': 'run-01', 'split': split,
        'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720],
        'screen_state': {'screen_kind': 'card_selection', 'reviews': {'screen_kind': reviewed}},
    }
    annotation = session / 'annotations/screen.json'
    annotation.write_text(json.dumps(data))
    return session, annotation


class CardChoiceExtractionTests(unittest.TestCase):
    def test_extracts_four_slots_with_provenance_without_altering_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, annotation = source(root)
            original = session / 'originals/screen.png'
            before = (original.read_bytes(), annotation.read_bytes())
            output = root / 'cards'
            summary = extract_card_choices(root / 'sessions', output)
            rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(summary['card_images'], 4)
            for index, row in enumerate(rows):
                crop = cv2.imread(str(output / row['image']))
                self.assertEqual(crop.shape, (120, 88, 3))
                self.assertTrue(np.all(crop == 230 + index * 5))
                self.assertEqual((row['choice'], row['side']), CHOICE_BOXES[index][:2])
                self.assertEqual(row['split'], 'test')
                self.assertEqual(row['source_image_sha256'], hashlib.sha256(before[0]).hexdigest())
                self.assertIsNone(row['card_key'])
                self.assertTrue(row['human_review_required'])
            self.assertEqual(before, (original.read_bytes(), annotation.read_bytes()))

    def test_unreviewed_screens_require_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root, reviewed=False)
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                extract_card_choices(root / 'sessions', root / 'cards')
            result = extract_card_choices(root / 'sessions', root / 'cards', include_unreviewed=True)
            self.assertEqual(result['unreviewed_screens'], 1)

    def test_invalid_layout_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root, layout=False)
            with self.assertRaisesRegex(ValueError, 'layout not verified'):
                extract_card_choices(root / 'sessions', root / 'cards')
            self.assertFalse((root / 'cards').exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root)
            output = root / 'cards'
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                extract_card_choices(root / 'sessions', output)
            self.assertEqual(marker.read_text(), 'keep')

    def test_explicit_layout_override_retains_unverified_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root, layout=False)
            output = root / 'cards'
            result = extract_card_choices(root / 'sessions', output, allow_unverified_layout=True)
            self.assertEqual(result['unverified_layout_screens'], 1)
            rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            self.assertTrue(all(not row['automatic_layout_verified'] for row in rows))
            self.assertTrue(all(row['human_review_required'] and row['card_key'] is None for row in rows))

    def test_one_session_cannot_span_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, annotation = source(root)
            data = json.loads(annotation.read_text())
            data['split'] = 'train'
            annotation.with_name('other.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                extract_card_choices(root / 'sessions', root / 'cards')
            self.assertFalse((root / 'cards').exists())


if __name__ == '__main__':
    unittest.main()
