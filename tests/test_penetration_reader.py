"""Small offline checks for the partial penetration HUD reader."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.perception.penetration import PenetrationReader, penetration_patch
from shotgun_king_rl.perception.types import PredictionStatus
from shotgun_king_rl.perception.hud import stat_context_patch


def screen(y=250, seed=7, digits="40%"):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    title = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    image[y + 157:y + 182, 270:354] = title
    cv2.putText(image, '4', (305, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
    cv2.putText(image, digits, (285, y + 212), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 220, 220), 2)
    return image


class PenetrationReaderTests(unittest.TestCase):
    def test_accepts_known_penetration_after_vertical_hud_shift(self):
        reader = PenetrationReader([(40, penetration_patch(screen(250)))])
        result = reader.read(screen(290))
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value, 40)

    def test_rejects_conflicting_labels_for_same_patch(self):
        template = penetration_patch(screen())
        result = PenetrationReader([(40, template), (75, template)]).read(screen())
        self.assertEqual(result.status, PredictionStatus.UNKNOWN)
        self.assertEqual(result.reason, 'penetration_ambiguous')

    def test_requires_title_context_not_just_matching_digit(self):
        reader = PenetrationReader([(40, penetration_patch(screen()))])
        self.assertFalse(reader.read(screen(seed=11)).is_accepted)

    def test_unseen_percentage_is_unknown(self):
        reader = PenetrationReader([(40, penetration_patch(screen()))])
        self.assertFalse(reader.read(screen(digits="25%")).is_accepted)

    def test_absent_row_is_unknown_rather_than_zero(self):
        image = screen()
        image[400:600, 270:354] = 0
        reader = PenetrationReader([(40, penetration_patch(screen()))])
        result = reader.read(image)
        self.assertFalse(result.is_accepted)
        self.assertIsNone(result.value)

    def test_reviewed_zero_requires_both_known_row_and_panel(self):
        absent = screen()
        absent[400:600, 270:354] = 0
        reader = PenetrationReader(zero_examples=[(stat_context_patch(absent), penetration_patch(absent))])
        self.assertTrue(reader.read(absent).is_accepted)
        self.assertEqual(reader.read(absent).value, 0)
        self.assertFalse(reader.read(screen(digits="75%")).is_accepted)
        different = absent.copy()
        different[210:235, 270:354] = np.random.default_rng(77).integers(0, 100, (25, 84, 3), dtype=np.uint8)
        self.assertFalse(reader.read(different).is_accepted)
        self.assertFalse(reader.read(np.zeros_like(absent)).is_accepted)

    def test_reviewed_zero_conflicting_with_positive_is_ambiguous(self):
        image = screen()
        reader = PenetrationReader([(40, penetration_patch(image))],
                                [(stat_context_patch(image), penetration_patch(image))])
        self.assertEqual(reader.read(image).reason, 'penetration_ambiguous')

    def test_missing_reference_is_unknown(self):
        self.assertEqual(PenetrationReader().read(screen()).reason, 'penetration_no_match')

    def test_invalid_and_blank_images_are_unknown(self):
        reader = PenetrationReader()
        for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
            with self.subTest(image_is_none=image is None):
                self.assertEqual(reader.read(image).status, PredictionStatus.UNKNOWN)

    def test_only_reviewed_gameplay_train_references_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, split, reviewed, kind, value in (
                ('train', 'train', True, 'gameplay', 40),
                ('val', 'val', True, 'gameplay', 8),
                ('test', 'test', True, 'gameplay', 6),
                ('unreviewed', 'train', False, 'gameplay', 3),
                ('card', 'train', True, 'card_selection', 2),
                ('negative', 'train', True, 'gameplay', -1),
                ('zero', 'train', True, 'gameplay', 0),
                ('val_zero', 'val', True, 'gameplay', 0),
                ('unreviewed_zero', 'train', False, 'gameplay', 0),
                ('missing', 'train', True, 'gameplay', None),
                ('overflow', 'train', True, 'gameplay', 101),
                ('boolean', 'train', True, 'gameplay', True),
            ):
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                sample = screen()
                if type(value) is int and value == 0:
                    sample[400:600, 270:354] = 0
                cv2.imwrite(str(session / 'originals/screen.png'), sample)
                data = {
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720],
                    'screen_state': {
                        'screen_kind': kind, 'combat_stats': {'penetration_percent': value},
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                    },
                }
                (session / 'annotations/screen.json').write_text(json.dumps(data))
            reader = PenetrationReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(len(reader.zero_examples), 1)
            absent = screen()
            absent[400:600, 270:354] = 0
            self.assertEqual(reader.read(absent).value, 0)
            self.assertEqual(reader.read(screen()).value, 40)


if __name__ == '__main__':
    unittest.main()
