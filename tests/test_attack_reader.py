"""Small offline checks for the partial attack HUD reader."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.perception.attack import AttackReader, attack_patch
from shotgun_king_rl.perception.types import PredictionStatus


def screen(y=270, seed=7):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    title = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    image[y - 35:y - 10, 270:354] = title
    cv2.putText(image, '4', (305, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
    return image


class AttackReaderTests(unittest.TestCase):
    def test_accepts_known_attack_after_vertical_hud_shift(self):
        reader = AttackReader([(4, attack_patch(screen(260)))])
        result = reader.read(screen(300))
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value, 4)

    def test_rejects_conflicting_labels_for_same_patch(self):
        template = attack_patch(screen())
        result = AttackReader([(4, template), (8, template)]).read(screen())
        self.assertEqual(result.status, PredictionStatus.UNKNOWN)
        self.assertEqual(result.reason, 'attack_ambiguous')

    def test_requires_title_context_not_just_matching_digit(self):
        reader = AttackReader([(4, attack_patch(screen()))])
        self.assertFalse(reader.read(screen(seed=11)).is_accepted)

    def test_missing_reference_is_unknown(self):
        self.assertEqual(AttackReader().read(screen()).reason, 'attack_no_match')

    def test_invalid_and_blank_images_are_unknown(self):
        reader = AttackReader()
        for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
            with self.subTest(image_is_none=image is None):
                self.assertEqual(reader.read(image).status, PredictionStatus.UNKNOWN)

    def test_only_reviewed_gameplay_train_references_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, split, reviewed, kind, value in (
                ('train', 'train', True, 'gameplay', 4),
                ('val', 'val', True, 'gameplay', 8),
                ('test', 'test', True, 'gameplay', 6),
                ('unreviewed', 'train', False, 'gameplay', 3),
                ('card', 'train', True, 'card_selection', 2),
            ):
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), screen())
                data = {
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720],
                    'screen_state': {
                        'screen_kind': kind, 'combat_stats': {'attack': value},
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                    },
                }
                (session / 'annotations/screen.json').write_text(json.dumps(data))
            reader = AttackReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(reader.read(screen()).value, 4)


if __name__ == '__main__':
    unittest.main()
