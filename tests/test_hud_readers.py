"""Shared HUD contracts with field-specific layouts, values and label bounds."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.perception import AttackReader, RangeReader, RangeReading, SpreadReader, GrenadeDamageReader, PredictionStatus
from shotgun_king_rl.perception.attack import attack_patch
from shotgun_king_rl.perception.range import range_patch
from shotgun_king_rl.perception.spread import spread_patch
from shotgun_king_rl.perception.grenade_damage import grenade_damage_patch


@dataclass(frozen=True)
class HudCase:
    name: str
    reader: type
    patch: object
    offset: int
    value: object
    conflicting: object
    digits: str
    unseen: str | None
    labels: dict
    invalid_labels: tuple = ()


CASES = (
    HudCase('attack', AttackReader, attack_patch, 0, 4, 8, '4', None, {'attack': 4}),
    HudCase('range', RangeReader, range_patch, 64, RangeReading(3, 5), RangeReading(3, 6), '3-5', '2-8',
            {'range_min': 3, 'range_max': 5},
            ({'range_min': 5, 'range_max': 3}, {'range_max': None}, {'range_min': True})),
    HudCase('spread', SpreadReader, spread_patch, 128, 65, 95, '65', '180', {'spread_degrees': 65},
            tuple({'spread_degrees': value} for value in (-1, 361, True))),
    HudCase('grenade_damage', GrenadeDamageReader, grenade_damage_patch, 256, 2, 3, '2', '8',
            {'grenade_damage': 2, 'grenades_remaining': 0},
            tuple({'grenade_damage': value} for value in (-1, None, True))),
)


def screen(case, y=250, seed=7, digits=None):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    digits = case.digits if digits is None else digits
    top = y + case.offset - 35
    image[top:top+25, 270:354] = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    cv2.putText(image, digits if case.offset == 0 else '4', (305, y+20), 0, .8, (220, 220, 220), 2)
    if case.offset:
        cv2.putText(image, digits, (285, y+case.offset+20), 0, .7, (220, 220, 220), 2)
    return image


class HudReaderTests(unittest.TestCase):
    def test_known_value_survives_vertical_panel_shift(self):
        for case in CASES:
            with self.subTest(reader=case.name):
                reader = case.reader([(case.value, case.patch(screen(case)))])
                result = reader.read(screen(case, y=290))
                self.assertTrue(result.is_accepted)
                self.assertEqual(result.value, case.value)

    def test_conflicting_labels_are_unknown(self):
        for case in CASES:
            with self.subTest(reader=case.name):
                template = case.patch(screen(case))
                result = case.reader([(case.value, template), (case.conflicting, template)]).read(screen(case))
                self.assertEqual(result.status, PredictionStatus.UNKNOWN)
                self.assertEqual(result.reason, f'{case.name}_ambiguous')

    def test_matching_digits_require_title_context(self):
        for case in CASES:
            with self.subTest(reader=case.name):
                reader = case.reader([(case.value, case.patch(screen(case)))])
                self.assertFalse(reader.read(screen(case, seed=11)).is_accepted)

    def test_unseen_values_are_not_replaced_by_known_values(self):
        for case in CASES:
            if case.unseen is None:
                continue
            with self.subTest(reader=case.name):
                reader = case.reader([(case.value, case.patch(screen(case)))])
                self.assertFalse(reader.read(screen(case, digits=case.unseen)).is_accepted)

    def test_missing_references_and_invalid_images_are_unknown(self):
        for case in CASES:
            with self.subTest(reader=case.name):
                reader = case.reader()
                self.assertEqual(reader.read(screen(case)).reason, f'{case.name}_no_match')
                for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
                    self.assertEqual(reader.read(image).status, PredictionStatus.UNKNOWN)

    def test_absent_grenade_damage_is_not_zero(self):
        case = next(case for case in CASES if case.name == 'grenade_damage')
        reader = case.reader([(case.value, case.patch(screen(case)))])
        image = screen(case)
        image[450:650, 270:354] = 0
        result = reader.read(image)
        self.assertFalse(result.is_accepted)
        self.assertIsNone(result.value)

    def test_reference_filters_preserve_field_specific_bounds(self):
        for case in CASES:
            with self.subTest(reader=case.name), tempfile.TemporaryDirectory() as directory:
                collection = Path(directory)
                sources = [
                    ('train', 'train', True, 'gameplay', {}),
                    ('val', 'val', True, 'gameplay', {}),
                    ('test', 'test', True, 'gameplay', {}),
                    ('unreviewed', 'train', False, 'gameplay', {}),
                    ('card', 'train', True, 'card_selection', {}),
                ] + [(f'invalid-{i}', 'train', True, 'gameplay', values)
                     for i, values in enumerate(case.invalid_labels)]
                for name, split, reviewed, kind, overrides in sources:
                    session = collection / 'sessions' / name
                    (session / 'originals').mkdir(parents=True)
                    (session / 'annotations').mkdir()
                    cv2.imwrite(str(session / 'originals/screen.png'), screen(case))
                    annotation = {'split': split, 'image': '../originals/screen.png',
                                  'game_crop': [0, 0, 1280, 720], 'screen_state': {
                                      'screen_kind': kind, 'combat_stats': case.labels | overrides,
                                      'reviews': {'screen_kind': True, 'combat_stats': reviewed}}}
                    (session / 'annotations/screen.json').write_text(json.dumps(annotation))
                reader = case.reader.from_collection(collection)
                self.assertEqual(len(reader.examples), 1)
                self.assertEqual(reader.read(screen(case)).value, case.value)


if __name__ == '__main__':
    unittest.main()
