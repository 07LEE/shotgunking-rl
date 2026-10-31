"""Shared HUD contracts with field-specific layouts, values and label bounds."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.perception import AttackReader, RangeReader, RangeReading, SpreadReader, GrenadeDamageReader, MeleeDamageReader, PredictionStatus

from tests.support.hud import CASES, screen


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

    def test_melee_requires_visible_title_reference_and_filters_other_titles(self):
        case = next(case for case in CASES if case.name == 'melee_damage')
        self.assertEqual(MeleeDamageReader.source, 'melee-damage-reviewed-templates:v2')
        for value in (0, -1, True, None):
            self.assertFalse(MeleeDamageReader([(value, case.patch(screen(case)))]).read(screen(case)).is_accepted)
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, seed in (('melee', 7), ('other-title', 11)):
                session = collection / 'sessions' / name
                (session / 'originals').mkdir(parents=True)
                (session / 'annotations').mkdir()
                cv2.imwrite(str(session / 'originals/screen.png'), screen(case, seed=seed))
                (session / 'annotations/screen.json').write_text(json.dumps({
                    'split': 'train', 'image': '../originals/screen.png',
                    'game_crop': [0, 0, 1280, 720], 'screen_state': {
                        'screen_kind': 'gameplay', 'combat_stats': {'melee_damage': 2 if seed == 7 else 40},
                        'reviews': {'screen_kind': True, 'combat_stats': True}}}))
            reader = MeleeDamageReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 0)
            self.assertFalse(reader.read(screen(case)).is_accepted)
            for title in (np.zeros((30, 84), dtype=np.uint8), np.ones((5, 5), dtype=np.uint8)):
                with self.subTest(title_shape=getattr(title, 'shape', None)), self.assertRaises(ValueError):
                    MeleeDamageReader.from_collection(collection, title_reference=title)
            reader = MeleeDamageReader.from_collection(collection, title_reference=case.patch(screen(case))[:30])
            self.assertEqual(len(reader.examples), 1)
            self.assertEqual(reader.read(screen(case)).value, 2)
            self.assertFalse(reader.read(screen(case, seed=11)).is_accepted)
            image = screen(case)
            image[420:520, 270:354] = 0
            self.assertFalse(reader.read(image).is_accepted)

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
                kwargs = ({'title_reference': case.patch(screen(case))[:30]}
                          if case.name == 'melee_damage' else {})
                reader = case.reader.from_collection(collection, **kwargs)
                self.assertEqual(len(reader.examples), 1)
                self.assertEqual(reader.read(screen(case)).value, case.value)


if __name__ == '__main__':
    unittest.main()
