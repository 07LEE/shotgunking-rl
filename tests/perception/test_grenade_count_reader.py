"""Small offline checks for reviewed grenade HUD slot counts."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.perception import GrenadeCountReader
from shotgun_king_rl.perception.grenade_count import grenade_count_patch, grenade_context_patch


def screen(count):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for index, x in enumerate((598, 622)):
        active = index < count
        color = (200, 200, 220) if active else (50, 50, 60)
        cv2.rectangle(image, (x, 90), (x + 10, 104), color, -1 if active else 1)
        cv2.rectangle(image, (x + 3, 85), (x + 7, 89), color, 1)
    return image


class GrenadeCountReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = GrenadeCountReader([
            (count, grenade_count_patch(screen(count))) for count in (0, 1, 2)
        ])

    def test_distinguishes_active_and_depleted_slot_counts(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                self.assertEqual(self.reader.read(screen(count)).value, count)

    def test_absent_hud_is_not_assumed_zero(self):
        result = self.reader.read(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertFalse(result.is_accepted)
        self.assertIsNone(result.value)

    def test_other_hud_values_do_not_determine_grenade_count(self):
        image = screen(1)
        image[78:112, 384:575] = (0, 0, 255)
        image[300:600, 270:354] = 200
        self.assertEqual(self.reader.read(image).value, 1)

    def test_conflicting_count_labels_are_unknown(self):
        template = grenade_count_patch(screen(1))
        result = GrenadeCountReader([(1, template), (2, template)]).read(screen(1))
        self.assertEqual(result.reason, 'grenades_remaining_ambiguous')

    def test_missing_references_and_invalid_images_are_unknown(self):
        self.assertEqual(GrenadeCountReader().read(screen(1)).reason, 'grenades_remaining_no_match')
        self.assertFalse(self.reader.read(None).is_accepted)

    def test_unrelated_shapes_are_unknown(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(image, (592, 78), (641, 111), (220, 220, 220), 3)
        self.assertFalse(self.reader.read(image).is_accepted)

    def test_featureless_zero_needs_reviewed_slots_and_visible_context(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (390, 40), (410, 60), (0, 0, 255), -1)
        slots, context = grenade_count_patch(image), grenade_context_patch(image)
        reader = GrenadeCountReader(zero_examples=[(context, slots)])
        self.assertEqual(reader.read(image).value, 0)
        self.assertFalse(GrenadeCountReader().read(image).is_accepted)
        self.assertFalse(reader.read(np.zeros_like(image)).is_accepted)
        changed = image.copy()
        changed[85:90, 600:605] = 100
        self.assertFalse(reader.read(changed).is_accepted)
        image[:] = 0  # Reader owns copies of both reference crops.
        self.assertGreater(reader.zero_examples[0][0].std(), 5)

    def test_featureless_zero_references_use_only_reviewed_train_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[40:60, 390:410] = (0, 0, 255)
            cases = (("train", True, 0), ("val", True, 0), ("test", True, 0),
                     ("train", False, 0), ("train", True, 1))
            for index, (split, reviewed, value) in enumerate(cases):
                session = root / "sessions" / str(index)
                (session / "annotations").mkdir(parents=True)
                (session / "originals").mkdir()
                cv2.imwrite(str(session / "originals/screen.png"), image)
                data = {"split": split, "image": "../originals/screen.png",
                        "game_crop": [0, 0, 1280, 720], "screen_state": {
                            "screen_kind": "gameplay", "reviews": {"screen_kind": True, "combat_stats": reviewed},
                            "combat_stats": {"grenades_remaining": value}}}
                (session / "annotations/screen.json").write_text(json.dumps(data))
            reader = GrenadeCountReader.from_collection(root)
            self.assertEqual(len(reader.examples), 0)
            self.assertEqual(len(reader.zero_examples), 1)
            self.assertEqual(reader.read(image).value, 0)

    def test_loads_only_reviewed_gameplay_train_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, split, reviewed, kind, value in (
                ('valid', 'train', True, 'gameplay', 2),
                ('val', 'val', True, 'gameplay', 1),
                ('test', 'test', True, 'gameplay', 0),
                ('unreviewed', 'train', False, 'gameplay', 1),
                ('card', 'train', True, 'card_selection', 1),
                ('missing', 'train', True, 'gameplay', None),
                ('negative', 'train', True, 'gameplay', -1),
                ('boolean', 'train', True, 'gameplay', True),
            ):
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), screen(2))
                data = {
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720],
                    'screen_state': {
                        'screen_kind': kind,
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                        'combat_stats': {'grenades_remaining': value, 'grenade_damage': 99},
                    },
                }
                (session / 'annotations/screen.json').write_text(json.dumps(data))
            reader = GrenadeCountReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(reader.read(screen(2)).value, 2)


if __name__ == '__main__':
    unittest.main()
