"""Grenade rows can move and share numeric evidence with reviewed attacks."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.perception import GrenadeDamageReader
from shotgun_king_rl.perception.hud import stat_patch
from shotgun_king_rl.perception.numeric import numeric_token


def hud(damage=2, offset=256, *, seed=18, anchor=230, attack=3):
    image = np.zeros((720, 1280, 3), np.uint8)
    cv2.putText(image, str(attack), (300, anchor + 20), 0, .8, (220, 220, 220), 2)
    image[anchor - 35 + offset:anchor - 20 + offset, 270:285] = np.random.default_rng(seed).integers(
        0, 100, (15, 15, 3), dtype=np.uint8,
    )
    cv2.putText(image, str(damage), (300, anchor + offset + 20), 0, .8, (220, 220, 220), 2)
    return image


def reader():
    reference = hud()
    return GrenadeDamageReader(
        [(2, stat_patch(reference, 256))],
        titles=[('grenade_damage', stat_patch(reference, 256)[:30])],
        integers=[(2, numeric_token(reference, 256)), (3, numeric_token(reference))],
    )


class GrenadeDamageLayoutTests(unittest.TestCase):
    def test_known_whole_row_moves_between_optional_slots(self):
        reference = hud()
        subject = GrenadeDamageReader([(2, stat_patch(reference, 256))])
        for offset in (192, 256, 320):
            with self.subTest(offset=offset):
                self.assertEqual(subject.read(hud(offset=offset)).value, 2)

    def test_attack_integer_transfers_only_with_matching_damage_title(self):
        subject = reader()
        for offset in (192, 256, 320):
            with self.subTest(offset=offset):
                self.assertEqual(subject.read(hud(3, offset)).value, 3)
        self.assertFalse(subject.read(hud(3, 192, seed=25)).is_accepted)
        self.assertFalse(subject.read(hud(8, 192)).is_accepted)

    def test_missing_row_and_unseen_blank_panel_stay_unknown(self):
        image = hud()
        image[300:, 270:354] = 0
        for query in (None, image, np.zeros_like(image)):
            self.assertIsNone(reader().read(query).value)

    def test_conflicting_integer_labels_cannot_force_a_value(self):
        image = hud()
        subject = GrenadeDamageReader(
            titles=[('grenade_damage', stat_patch(image, 256)[:30])],
            integers=[(3, numeric_token(image)), (8, numeric_token(image))],
        )
        self.assertFalse(subject.read(hud(3, 192)).is_accepted)

    def test_conflicting_rows_override_a_whole_row_match(self):
        image = hud(3, 192)
        second = hud(2, 256)
        image[451:516, 270:354] = second[451:516, 270:354]
        self.assertEqual(reader().read(image).reason, 'grenade_damage_ambiguous')

    def test_collection_uses_only_reviewed_train_values_and_honors_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            reference_path = None
            for name, split, reviewed, kind in (
                ('known', 'train', True, 'gameplay'),
                ('val', 'val', True, 'gameplay'),
                ('test', 'test', True, 'gameplay'),
                ('draft', 'train', False, 'gameplay'),
                ('card', 'train', True, 'card_selection'),
            ):
                session = collection / 'sessions' / name
                (session / 'annotations').mkdir(parents=True)
                (session / 'originals').mkdir()
                if name == 'known':
                    cv2.imwrite(str(session / 'originals/screen.png'), hud())
                # Forbidden sources deliberately have no readable image.
                path = session / 'annotations/screen.json'
                path.write_text(json.dumps({
                    'split': split, 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720], 'screen_state': {
                        'screen_kind': kind, 'combat_stats': {'attack': 3, 'grenade_damage': 2},
                        'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                    },
                }))
                if name == 'known':
                    reference_path = path
            subject = GrenadeDamageReader.from_collection(collection)
            self.assertEqual(len(subject.examples), 1)
            self.assertEqual(subject.read(hud(3, 192)).value, 3)
            excluded = GrenadeDamageReader.from_collection(collection, exclude=reference_path)
            self.assertEqual(excluded.integers, ())
            self.assertIsNone(excluded.read(hud(3, 192)).value)
