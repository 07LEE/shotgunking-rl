"""Small offline checks for the partial grenade damage HUD reader."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.perception.grenade_damage import GrenadeDamageReader, grenade_damage_patch
from shotgun_king_rl.perception.types import PredictionStatus


def screen(y=250, seed=7, digits="2"):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    title = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    image[y + 221:y + 246, 270:354] = title
    cv2.putText(image, '4', (305, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
    cv2.putText(image, digits, (285, y + 276), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 220, 220), 2)
    return image


class GrenadeDamageReaderTests(unittest.TestCase):
    def test_accepts_known_grenade_damage_after_vertical_hud_shift(self):
        reader = GrenadeDamageReader([(2, grenade_damage_patch(screen(250)))])
        result = reader.read(screen(290))
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value, 2)

    def test_rejects_conflicting_labels_for_same_patch(self):
        template = grenade_damage_patch(screen())
        result = GrenadeDamageReader([(2, template), (3, template)]).read(screen())
        self.assertEqual(result.status, PredictionStatus.UNKNOWN)
        self.assertEqual(result.reason, 'grenade_damage_ambiguous')

    def test_requires_title_context_not_just_matching_digit(self):
        reader = GrenadeDamageReader([(2, grenade_damage_patch(screen()))])
        self.assertFalse(reader.read(screen(seed=11)).is_accepted)

    def test_unseen_damage_is_unknown(self):
        reader = GrenadeDamageReader([(2, grenade_damage_patch(screen()))])
        self.assertFalse(reader.read(screen(digits="8")).is_accepted)

    def test_absent_row_is_unknown_rather_than_zero(self):
        image = screen()
        image[450:650, 270:354] = 0
        reader = GrenadeDamageReader([(2, grenade_damage_patch(screen()))])
        result = reader.read(image)
        self.assertFalse(result.is_accepted)
        self.assertIsNone(result.value)

    def test_missing_reference_is_unknown(self):
        self.assertEqual(GrenadeDamageReader().read(screen()).reason, 'grenade_damage_no_match')

    def test_invalid_and_blank_images_are_unknown(self):
        reader = GrenadeDamageReader()
        for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
            with self.subTest(image_is_none=image is None):
                self.assertEqual(reader.read(image).status, PredictionStatus.UNKNOWN)

    def test_only_reviewed_gameplay_train_references_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, split, reviewed, kind, value in (
                ('train', 'train', True, 'gameplay', 2),
                ('val', 'val', True, 'gameplay', 8),
                ('test', 'test', True, 'gameplay', 6),
                ('unreviewed', 'train', False, 'gameplay', 3),
                ('card', 'train', True, 'card_selection', 2),
                ('negative', 'train', True, 'gameplay', -1),
                ('absent', 'train', True, 'gameplay', None),
                ('boolean', 'train', True, 'gameplay', True),
            ):
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), screen())
                data = {
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720],
                    'screen_state': {
                        'screen_kind': kind, 'combat_stats': {'grenade_damage': value, 'grenades_remaining': 0},
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                    },
                }
                (session / 'annotations/screen.json').write_text(json.dumps(data))
            reader = GrenadeDamageReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(reader.read(screen()).value, 2)


if __name__ == '__main__':
    unittest.main()
