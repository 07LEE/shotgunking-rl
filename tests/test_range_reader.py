"""Offline checks for paired range readings from reviewed HUD references."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.perception import RangeReader, RangeReading
from shotgun_king_rl.perception.range import range_patch


def screen(y=250, seed=7, digits='3-5'):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(image, '4', (305, y + 20), 0, .8, (220, 220, 220), 2)
    title = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    image[y + 29:y + 54, 270:354] = title
    cv2.putText(image, digits, (285, y + 84), 0, .7, (220, 220, 220), 2)
    return image


class RangeReaderTests(unittest.TestCase):
    def test_returns_both_limits_after_vertical_hud_shift(self):
        reader = RangeReader([(RangeReading(3, 5), range_patch(screen()))])
        result = reader.read(screen(y=290))
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value, RangeReading(3, 5))

    def test_conflicting_pairs_are_unknown(self):
        template = range_patch(screen())
        reader = RangeReader([(RangeReading(3, 5), template), (RangeReading(3, 6), template)])
        self.assertEqual(reader.read(screen()).reason, 'range_ambiguous')

    def test_matching_digits_require_matching_title_context(self):
        reader = RangeReader([(RangeReading(3, 5), range_patch(screen()))])
        self.assertFalse(reader.read(screen(seed=11)).is_accepted)

    def test_unseen_pair_is_not_replaced_by_known_pair(self):
        reader = RangeReader([(RangeReading(3, 5), range_patch(screen()))])
        self.assertFalse(reader.read(screen(digits='2-8')).is_accepted)

    def test_missing_references_and_invalid_images_are_unknown(self):
        reader = RangeReader()
        self.assertEqual(reader.read(screen()).reason, 'range_no_match')
        for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
            self.assertFalse(reader.read(image).is_accepted)

    def test_loads_only_valid_reviewed_train_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            cases = [
                ('valid', 'train', True, 'gameplay', 3, 5),
                ('val', 'val', True, 'gameplay', 3, 6),
                ('test', 'test', True, 'gameplay', 3, 7),
                ('unreviewed', 'train', False, 'gameplay', 3, 8),
                ('card', 'train', True, 'card_selection', 3, 9),
                ('reversed', 'train', True, 'gameplay', 5, 3),
                ('missing', 'train', True, 'gameplay', 3, None),
                ('boolean', 'train', True, 'gameplay', True, 5),
            ]
            for name, split, reviewed, kind, minimum, maximum in cases:
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), screen())
                annotation = {
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720],
                    'screen_state': {
                        'screen_kind': kind,
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                        'combat_stats': {'range_min': minimum, 'range_max': maximum},
                    },
                }
                (session / 'annotations/screen.json').write_text(json.dumps(annotation))
            reader = RangeReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(reader.read(screen()).value, RangeReading(3, 5))


if __name__ == '__main__':
    unittest.main()
